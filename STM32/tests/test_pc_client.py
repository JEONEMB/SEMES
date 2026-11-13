"""Host framing tests; no serial device or motor used."""
import importlib.util
from pathlib import Path
import struct
import unittest
from unittest import mock
from types import SimpleNamespace
import sys

spec = importlib.util.spec_from_file_location('client', Path(__file__).resolve().parents[1] / 'tools/y_motor_test.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class FakePort:
    def __init__(self, data):
        self.data = bytearray(data)
        self.sent = bytearray()

    @property
    def in_waiting(self):
        return 1 if self.data else 0

    def read(self, _):
        result = self.data[:1]
        del self.data[:1]
        return result

    def write(self, data):
        self.sent.extend(data)

    def flush(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class ClientTests(unittest.TestCase):
    def test_ping_known_crc(self):
        self.assertEqual(client.packet(1, 1).hex(), 'aa550101010075f6')
        self.assertEqual(client.crc16(b'123456789'), 0x29b1)

    def test_fragmentation_noise_crc_and_sequence(self):
        bad = bytearray(client.packet(0x81, 8, b'\x30'))
        bad[-1] ^= 1
        data = b'noise' + bad + client.packet(0x90, 1, b'\x03') + client.packet(0x81, 9, b'\x30')
        link = client.Link(FakePort(data))
        self.assertEqual(link.wait(9, {0x81}, 1), (0x81, b'\x30'))

    def test_nak_and_timeout(self):
        with self.assertRaisesRegex(RuntimeError, 'BUSY'):
            client.Link(FakePort(client.packet(0x82, 9, b'\x30\x05'))).wait(9, {0x81}, 1)
        with self.assertRaises(TimeoutError):
            client.Link(FakePort(b'')).wait(9, {0x81}, 0.001)

    def test_report_and_jog_encoding(self):
        payload = struct.pack('<BBBBIIIBBBB', 2, 1, 1, 0, 160, 160, 160, 1, 1, 1, 9)
        self.assertEqual(client.report(payload), (2, 1, 1, 1, 9))
        link = client.Link(FakePort(b''))
        link.send(0x30, 10, struct.pack('<BII', 1, 160, 160))
        self.assertEqual(len(link.port.sent), 17)
        self.assertEqual(link.port.sent[6:15], bytes.fromhex('01a0000000a0000000'))

    def run_jog(self, active_limit, pulses=160):
        initial = struct.pack('<BBBBIIIBBBB', 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 9)
        done = struct.pack('<BBBBIIIBBBB', 2, 1, 1, 0, pulses, pulses, 160, 1, 1, 1, 10)
        inputs = bytes((3, 4 if active_limit else 0)) + bytes(18)
        port = FakePort(client.packet(0xB0, 4, initial) + client.packet(0x84, 5, inputs) +
                        client.packet(0x81, 10, b'\x30') + client.packet(0xB0, 10, done))
        serial = SimpleNamespace(Serial=lambda *args, **kwargs: port)
        modules = {'serial': serial, 'serial.tools': SimpleNamespace(list_ports=None)}
        argv = ['client', '--port', 'FAKE', 'jog', '--dir', 'high', '--pulses', str(pulses)]
        with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, 'argv', argv), \
             mock.patch.object(client.random, 'randrange', return_value=4):
            client.main()
        expected = client.packet(0x31, 4) + client.packet(6, 5)
        expected += client.packet(0x30, 10, struct.pack('<BII', 1, pulses, 160))
        self.assertEqual(port.sent, expected)

    def test_jog_end_to_end(self):
        self.run_jog(False)

    def test_high_input_does_not_block_client(self):
        self.run_jog(True)

    def test_large_counts(self):
        for pulses in (1601, 16000, 0xffffffff):
            self.run_jog(False, pulses)

    def test_invalid_count_never_opens_port(self):
        for pulses in (0, -1, 0x100000000):
            open_port = mock.Mock()
            modules = {'serial': SimpleNamespace(Serial=open_port),
                       'serial.tools': SimpleNamespace(list_ports=None)}
            argv = ['client', '--port', 'FAKE', 'jog', '--dir', 'high', '--pulses', str(pulses)]
            with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, 'argv', argv):
                with self.assertRaises(SystemExit):
                    client.main()
            open_port.assert_not_called()


if __name__ == '__main__':
    unittest.main()
