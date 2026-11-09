"""Firmware frames through an in-memory serial port; no physical motor I/O."""
import struct
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from demo_web.serial_stage import SerialStage, crc16, packet
from demo_web.store import Problem


class Port:
    def __init__(self):
        self.condition = threading.Condition()
        self.buffer = bytearray()
        self.writes = []
        self.reply = lambda frame: None

    @property
    def in_waiting(self):
        with self.condition:
            return len(self.buffer)

    def read(self, size):
        with self.condition:
            self.condition.wait_for(lambda: self.buffer, timeout=.005)
            result = bytes(self.buffer[:size])
            del self.buffer[:size]
            return result

    def feed(self, data):
        with self.condition:
            self.buffer.extend(data)
            self.condition.notify_all()

    def write(self, frame):
        self.writes.append(frame)
        self.reply(frame)
        return len(frame)

    def close(self):
        pass


class SerialTests(unittest.TestCase):
    def setUp(self):
        self.port = Port()
        self.stage = SerialStage(port=self.port, ack_timeout=.02)

    def tearDown(self):
        self.stage.close()

    def test_crc_and_readonly_status_config(self):
        self.assertEqual(crc16(b'123456789'), 0x29b1)
        def reply(frame):
            cmd, seq = frame[3:5]
            if cmd == 6:
                self.port.feed(packet(0x84, seq, struct.pack('<BB2xiiii',0,1,123,0,0,0)))
            elif cmd == 8:
                self.port.feed(packet(0x86, seq, struct.pack('<iiii',0,500000,0,0)))
        self.port.reply = reply
        self.assertEqual(self.stage.status(), dict(state=0, flags=1, position=(123,0), steps=(0,0)))
        self.assertEqual(self.stage.limits(), (0,500000,0,0))
        self.assertEqual([f[3] for f in self.port.writes], [6,8])

    def test_retries_use_identical_sequence_and_done_recovers_lost_ack(self):
        def reply(frame):
            if len(self.port.writes) == 2:
                self.port.feed(packet(0x83, frame[4], struct.pack('<BB2xii',0,3,100,0)))
        self.port.reply = reply
        self.stage.start(100,0,True)
        self.assertEqual(self.stage.wait_done()['position'], (100,0))
        self.assertEqual(len(self.port.writes),2)
        self.assertEqual(self.port.writes[0],self.port.writes[1])

    def test_stop_interrupts_done_wait_with_original_move_sequence(self):
        move = {}
        def reply(frame):
            cmd, seq = frame[3:5]
            self.port.feed(packet(0x81,seq,bytes([cmd])))
            if cmd == 3:
                move['seq'] = seq
            elif cmd == 5:
                self.port.feed(packet(0x83,move['seq'],struct.pack('<BB2xii',1,3,40,0)))
        self.port.reply = reply
        self.stage.start(100,0,True)
        with ThreadPoolExecutor() as pool:
            result = pool.submit(self.stage.wait_done,1000)
            self.stage.stop()
            self.assertEqual(result.result(2),dict(result=1,cmd_type=3,position=(40,0)))
        self.assertEqual([f[3] for f in self.port.writes],[3,5])

    def test_done_timeout_queries_status_never_resends_motion(self):
        def reply(frame):
            cmd, seq = frame[3:5]
            if cmd == 3:
                self.port.feed(packet(0x81,seq,b'\x03'))
            elif cmd == 6:
                self.port.feed(packet(0x84,seq,struct.pack('<BB2xiiii',1,1,0,0,0,0)))
        self.port.reply = reply
        self.stage.start(100,0,True)
        with self.assertRaisesRegex(Problem,'완료 응답 시간 초과'):
            self.stage.wait_done(5)
        self.assertEqual([f[3] for f in self.port.writes],[3,6])

    def test_bad_crc_wrong_seq_wrong_command_are_ignored(self):
        def reply(frame):
            seq = frame[4]
            good = packet(0x81,seq,b'\x03')
            bad = good[:-1]+bytes([good[-1]^1])
            self.port.feed(b'noise'+bad+packet(0x81,(seq+1)%256,b'\x03')+packet(0x81,seq,b'\x04'))
        self.port.reply = reply
        with self.assertRaisesRegex(Problem,'응답 시간 초과'):
            self.stage.start(100,0,True)
        self.assertEqual(len(self.port.writes),4)
        self.assertEqual(len(set(self.port.writes)),1)

    def test_query_does_not_accept_ack(self):
        self.port.reply = lambda frame: self.port.feed(packet(0x81,frame[4],b'\x06'))
        with self.assertRaisesRegex(Problem,'응답 시간 초과'):
            self.stage.status()

    def test_nak_and_boot_stop_execution(self):
        self.port.reply = lambda frame: self.port.feed(packet(0x82,frame[4],b'\x03\x03'))
        with self.assertRaisesRegex(Problem,'원점 미설정'):
            self.stage.start(100,0,True)
        self.port.reply = lambda frame: self.port.feed(packet(0x90,0,struct.pack('<Bii',3,0,0)))
        with self.assertRaisesRegex(Problem,'재부팅'):
            self.stage.status()
        count = len(self.port.writes)
        with self.assertRaisesRegex(Problem,'재부팅'):
            self.stage.home_y()
        self.assertEqual(len(self.port.writes),count)

    def test_y_home_mask_and_theta_guard(self):
        self.port.reply = lambda frame: self.port.feed(packet(0x83,frame[4],struct.pack('<BB2xii',0,2,0,0)))
        self.stage.home_y()
        self.assertEqual(self.port.writes[0][3],2)
        self.assertEqual(self.port.writes[0][6:-2],b'\x01')
        self.assertEqual(self.stage.wait_done()['cmd_type'],2)
        with self.assertRaisesRegex(Problem,'T축'):
            self.stage.start(0,1)
        self.assertEqual(len(self.port.writes),1)


if __name__ == '__main__':
    unittest.main()
