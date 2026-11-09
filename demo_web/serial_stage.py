"""Direct USB/UART implementation of STM32/robot_stage_protocol.md.

One reader routes replies; the command lock covers ACK retries, never the DONE
wait. This lets STOP interrupt motion without another reader stealing replies.
"""
import fcntl
import secrets
import struct
import termios
import threading
import time

from .stage import DeviceOwner
from .store import Problem


def crc16(data):
    crc = 0xffff
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xffff
    return crc


def packet(kind, seq, payload=b''):
    body = bytes((1, kind, seq, len(payload))) + payload
    return b'\xaa\x55' + body + struct.pack('<H', crc16(body))


class SerialStage:
    mode = 'hardware'
    transport = 'serial'
    LENGTHS = {0x81: 1, 0x82: 2, 0x83: 12, 0x84: 20, 0x85: 4, 0x86: 16, 0x90: 9}

    def __init__(self, path=None, port=None, ack_timeout=.1):
        # ttyACM number changes on re-enumeration; the ST-LINK by-id link does not.
        if path is None:
            import glob
            links = sorted(glob.glob('/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_*-if02'))
            path = links[0] if links else '/dev/ttyACM0'
        self.path, self.port = path, port
        self.owner = None
        self.condition = threading.Condition()
        self.command_lock = threading.Lock()
        self.sequence = secrets.randbelow(256)
        self.pending = None
        self.motion = None
        self.done = None
        self.error = None
        self.closed = False
        self.reader = None
        self.ack_timeout = ack_timeout

    def _open(self):
        if self.reader:
            return
        try:
            if self.port is None:
                import serial
                self.owner = DeviceOwner()
                self.port = serial.Serial(None, 115200, timeout=.02, write_timeout=1, exclusive=True)
                self.port.dtr = False
                self.port.rts = False
                self.port.port = self.path
                self.port.open()
                discipline = bytearray(4)
                fcntl.ioctl(self.port.fileno(), termios.TIOCGETD, discipline, True)
                if struct.unpack('i', discipline)[0] != 0:
                    raise Problem('USB 포트를 커널 드라이버가 사용 중입니다. 드라이버 연결을 선택하세요.', 409)
            self.reader = threading.Thread(target=self._read, name='stm32-rx', daemon=True)
            self.reader.start()
        except Exception as exc:
            self.close()
            if isinstance(exc, Problem):
                raise
            if isinstance(exc, OSError) and getattr(exc, 'errno', None) == 2 or 'No such file' in str(exc):
                raise Problem(f'{self.path} 포트가 없습니다. STM32 USB 연결을 확인하세요.', 503) from exc
            if 'Resource busy' in str(exc) or 'exclusive' in str(exc).lower():
                raise Problem('USB 포트를 다른 프로그램(ldattach 등)이 사용 중입니다. 연결 방식을 커널 드라이버로 선택하세요.', 409) from exc
            raise Problem('STM32 USB 연결 실패. 포트와 사용 권한을 확인하세요.', 503) from exc

    def _read(self):
        buffer = bytearray()
        last_byte = time.monotonic()
        try:
            while not self.closed:
                chunk = self.port.read(max(1, self.port.in_waiting))
                if not chunk:
                    if time.monotonic() - last_byte > .02:
                        buffer.clear()
                    continue
                last_byte = time.monotonic()
                buffer.extend(chunk)
                while len(buffer) >= 6:
                    if buffer[:2] != b'\xaa\x55' or buffer[2] != 1 or buffer[5] > 32:
                        del buffer[0]
                        continue
                    size = 8 + buffer[5]
                    if len(buffer) < size:
                        break
                    frame = bytes(buffer[:size])
                    if crc16(frame[2:-2]) != struct.unpack('<H', frame[-2:])[0]:
                        del buffer[0]
                        continue
                    del buffer[:size]
                    kind, seq, payload = frame[3], frame[4], frame[6:-2]
                    if self.LENGTHS.get(kind) == len(payload):
                        self._receive(kind, seq, payload)
        except Exception:
            with self.condition:
                if not self.closed:
                    self.error = 'STM32 USB 연결이 끊겼습니다. 장치 상태를 확인하세요.'
                self.condition.notify_all()

    def _receive(self, kind, seq, payload):
        with self.condition:
            if kind == 0x90 and payload[0] == 3:
                self.error = 'STM32가 재부팅되었습니다. 다시 연결하고 원점을 설정하세요.'
            elif kind == 0x83 and self.motion == (seq, payload[1]):
                self.done = dict(result=payload[0], cmd_type=payload[1],
                                 position=struct.unpack('<ii', payload[4:]))
                if self.pending and self.pending['seq'] == seq and self.pending['cmd'] == payload[1]:
                    self.pending['response'] = (0x81, bytes((payload[1],)))
            elif self.pending and self.pending['seq'] == seq:
                cmd = self.pending['cmd']
                expected = {1: 0x85, 6: 0x84, 8: 0x86}.get(cmd, 0x81)
                if ((kind == 0x82 and payload[0] == cmd)
                        or (kind == 0x81 and expected == 0x81 and payload[0] == cmd)
                        or (kind == expected and kind not in (0x81, 0x82))):
                    self.pending['response'] = (kind, payload)
            self.condition.notify_all()

    def _exchange(self, cmd, payload=b''):
        with self.command_lock:
            self._open()
            with self.condition:
                if self.closed or self.error:
                    raise Problem(self.error or '장치 연결이 닫혔습니다.', 503)
                self.sequence = (self.sequence + 1) % 256
                seq = self.sequence
                self.pending = dict(seq=seq, cmd=cmd, response=None)
                if cmd in (2, 3, 4):
                    self.motion, self.done = (seq, cmd), None
            frame = packet(cmd, seq, payload)
            try:
                for _ in range(4):
                    with self.condition:
                        if self.error or self.closed:
                            raise Problem(self.error or '장치 연결이 닫혔습니다.', 503)
                        if self.pending['response']:
                            break
                    self.port.write(frame)
                    with self.condition:
                        self.condition.wait_for(lambda: self.pending['response'] or self.error or self.closed,
                                                timeout=self.ack_timeout)
                        if self.pending['response']:
                            break
                with self.condition:
                    if self.error or self.closed:
                        raise Problem(self.error or '장치 연결이 닫혔습니다.', 503)
                    response = self.pending['response']
                    if not response:
                        raise Problem('STM32 응답 시간 초과. 실제 이동 여부를 확인하세요.', 503)
                    kind, result = response
                    if kind == 0x82:
                        reasons = {3: '원점 미설정', 4: '허용 범위 초과', 5: '이동 중', 6: '장치 고장', 7: '잘못된 값'}
                        raise Problem('STM32 명령 거부: ' + reasons.get(result[1], str(result[1])), 409)
                    return result
            except (OSError, TimeoutError) as exc:
                raise Problem('STM32 USB 전송 실패. 실제 이동 여부를 확인하세요.', 503) from exc
            finally:
                with self.condition:
                    self.pending = None

    def status(self):
        state, flags, y, t, ys, ts = struct.unpack('<BB2xiiii', self._exchange(6))
        return dict(state=state, flags=flags, position=(y, t), steps=(ys, ts))

    def limits(self):
        return struct.unpack('<iiii', self._exchange(8))

    def ping(self):
        return struct.unpack('<BBBB', self._exchange(1))

    def speed(self, y, t):
        self._exchange(7, struct.pack('<II', y, t))

    def start(self, y, t, relative=False, allow_t=False):
        if t != 0 and not allow_t:
            raise Problem('T축 모터가 비활성 상태입니다. T 명령은 0이어야 합니다.')
        self._exchange(3 if relative else 4, struct.pack('<ii', y, t))

    def home_y(self):
        self._exchange(2, b'\x01')

    def wait_done(self, timeout_ms=30000):
        with self.condition:
            if self.motion is None:
                raise Problem('대기 중인 이동이 없습니다.', 409)
            self.condition.wait_for(lambda: self.done is not None or self.error or self.closed,
                                    timeout=timeout_ms / 1000)
            if self.error or self.closed:
                raise Problem(self.error or '장치 연결이 닫혔습니다.', 503)
            if self.done is not None:
                done, self.done, self.motion = self.done, None, None
                return done
        try:
            self.status()
        except Problem:
            pass
        raise Problem('STM32 이동 완료 응답 시간 초과. 이동 명령은 재전송하지 않았습니다.', 503)

    def stop(self):
        self._exchange(5)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.reader and self.reader is not threading.current_thread():
            self.reader.join(timeout=.5)
        if self.port:
            self.port.close()
        if self.owner:
            self.owner.close()
            self.owner = None
