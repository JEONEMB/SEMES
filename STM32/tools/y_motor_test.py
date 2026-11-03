"""PC-only commissioning client. Requires pyserial; never retries a jog.

python tools/y_motor_test.py --port COM4 jog --dir high --pulses 160 --hz 160
"""
import argparse
import random
import struct
import sys
import time

SOF = b'\xaa\x55'
STATES = ('IDLE', 'RUNNING', 'DONE', 'STOPPED', 'ABORTED', 'LIMIT')
REASONS = {1: 'UNKNOWN_TYPE (old firmware?)', 2: 'BAD_LEN', 3: 'NOT_HOMED',
           4: 'OUT_OF_RANGE', 5: 'BUSY', 6: 'FAULT (check limits/output/clock)',
           7: 'BAD_PARAM'}


def crc16(data):
    crc = 0xffff
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xffff
    return crc


def packet(kind, seq, payload=b''):
    body = bytes((1, kind, seq, len(payload))) + payload
    return SOF + body + struct.pack('<H', crc16(body))


class Link:
    def __init__(self, port):
        self.port = port
        self.buffer = bytearray()

    def send(self, kind, seq, payload=b''):
        self.port.write(packet(kind, seq, payload))
        self.port.flush()

    def receive(self, deadline):
        while time.monotonic() < deadline:
            self.buffer.extend(self.port.read(max(1, self.port.in_waiting)))
            while len(self.buffer) >= 2:
                if self.buffer[:2] != SOF:
                    del self.buffer[0]
                    continue
                if len(self.buffer) < 6:
                    break
                size = self.buffer[5] + 8
                if self.buffer[2] != 1 or self.buffer[5] > 32:
                    del self.buffer[0]
                    continue
                if len(self.buffer) < size:
                    break
                data = bytes(self.buffer[:size])
                if crc16(data[2:-2]) != struct.unpack('<H', data[-2:])[0]:
                    del self.buffer[0]
                    continue
                del self.buffer[:size]
                return data[3], data[4], data[6:-2]
        raise TimeoutError('No matching response before deadline; motion may have occurred. Do not blindly resend.')

    def wait(self, seq, kinds, timeout):
        deadline = time.monotonic() + timeout
        while True:
            kind, received_seq, payload = self.receive(deadline)
            if received_seq != seq:
                continue
            if kind == 0x82 and len(payload) == 2:
                raise RuntimeError('NAK: ' + REASONS.get(payload[1], str(payload[1])))
            if kind in kinds:
                return kind, payload


def report(payload):
    if len(payload) != 20:
        raise RuntimeError('Unsupported test report layout')
    state, direction, exact, enabled, requested, completed, hz, ready, allowed, cached, last = struct.unpack('<BBBBIIIBBBB', payload)
    name = STATES[state] if state < len(STATES) else f'UNKNOWN({state})'
    print(f'{name}: pulses={completed}/{requested}, hz<={hz}, DIR={direction}, '
          f'EN={enabled}, count_exact={exact}, ready={ready}, output_allowed={allowed}')
    return state, ready, allowed, cached, last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='ST-LINK VCP, e.g. COM4')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('ports')
    sub.add_parser('status')
    sub.add_parser('stop')
    jog = sub.add_parser('jog')
    jog.add_argument('--dir', choices=('low', 'high'), required=True)
    jog.add_argument('--pulses', type=int, default=160)
    jog.add_argument('--hz', type=int, default=160)
    args = parser.parse_args()
    import serial
    from serial.tools import list_ports
    if args.command == 'ports':
        for item in list_ports.comports():
            print(item.device, item.description)
        return
    if not args.port:
        parser.error('--port is required')
    if args.command == 'jog' and not (1 <= args.pulses <= 0xffffffff and 16 <= args.hz <= 3200):
        parser.error('pulses: 1..4294967295 (uint32); hz: 16..3200')
    with serial.Serial(args.port, 115200, timeout=0.05, write_timeout=1) as port:
        link = Link(port)
        seq = random.randrange(256)
        if args.command == 'stop':
            link.send(5, seq)
            _, ack = link.wait(seq, {0x81}, 2)
            if ack != b'\x05':
                raise RuntimeError('Unexpected STOP ACK')
            print('STOP acknowledged. Use status to check pulse count.')
            return
        link.send(0x31, seq)
        _, payload = link.wait(seq, {0xB0}, 2)
        state, ready, allowed, cached, last = report(payload)
        link.send(6, (seq + 1) % 256)
        _, inputs = link.wait((seq + 1) % 256, {0x84}, 2)
        if len(inputs) != 20:
            raise RuntimeError('Unsupported input status layout')
        home, limit = (inputs[1] >> 2) & 1, (inputs[1] >> 3) & 1
        print(f'Y_HOME={home}, Y_LIMIT={limit} (input levels only; firmware controls limit enforcement)')
        if args.command == 'status':
            return
        if state == 1 or not ready or not allowed:
            raise RuntimeError('Motor busy, not ready, or output locked')
        # Avoid the last accepted jog SEQ even across separate invocations.
        seq = (last + 1) % 256 if cached else (seq + 1) % 256
        started = False
        try:
            started = True  # a write failure can still have delivered bytes
            link.send(0x30, seq, struct.pack('<BII', args.dir == 'high', args.pulses, args.hz))
            kind, payload = link.wait(seq, {0x81, 0xB0}, 2)
            if kind == 0x81:
                if payload != b'\x30':
                    raise RuntimeError('Unexpected jog ACK')
                print('ACK: pulse burst accepted; waiting for completion (Ctrl+C sends STOP).')
                _, payload = link.wait(seq, {0xB0}, args.pulses / args.hz * 2 + 5)
            state, *_ = report(payload)
            started = False
            if state != 2:
                raise RuntimeError('Burst ended without normal DONE; inspect status/limits')
        finally:
            if started:
                # Best effort only: cable loss can prevent STOP delivery.
                try:
                    link.send(5, (seq + 1) % 256)
                    link.wait((seq + 1) % 256, {0x81}, 2)
                    print('STOP acknowledged after interruption/error.')
                except Exception as error:
                    print(f'STOP not confirmed: {error}', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f'Error: {error or "interrupted"}', file=sys.stderr)
        sys.exit(1)
