"""Compile production C with a mock HAL and execute on an emulated ARM CPU.

python -m pip install -r tests/requirements.txt
python tests/run_tests.py --gcc /path/to/arm-none-eabi-gcc
No board or serial port is opened.
"""
import argparse
from pathlib import Path
import struct
import subprocess

from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB, UC_MODE_MCLASS
from unicorn.arm_const import UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_R0

root = Path(__file__).resolve().parents[1]
args = argparse.ArgumentParser()
args.add_argument('--gcc', default='arm-none-eabi-gcc')
args.add_argument('--motor-enabled', action='store_true', help='Enable pulse API in the emulator only')
args.add_argument('--with-switches', action='store_true', help='Test restored NC switch enforcement')
options = args.parse_args()
out = root / 'build' / 'protocol-tests'
out.mkdir(parents=True, exist_ok=True)
elf = out / 'tests.elf'
subprocess.run([
    options.gcc, '-std=c11', '-mcpu=cortex-m4', '-mthumb', '-mfloat-abi=soft',
    '-O0', '-g', '-Wall', '-Wextra', '-Werror', '-ffreestanding', '-fno-builtin',
    '-nostdlib', '-I.', '-ICore/Inc', '-Itests', '-include', 'tests/mock_hal.h',
    '-DY_MOTOR_OUTPUT_ALLOWED=' + ('1' if options.motor_enabled else '0'),
    '-DY_MOTOR_USE_LIMITS=' + ('1' if options.with_switches else '0'),
    'Core/Src/robot_stage_transport.c', 'Core/Src/robot_stage_service.c',
    'Core/Src/app.c', 'Core/Src/y_motor.c', 'Core/Src/y_motor_tim.c', 'Core/Src/y_test.c',
    'tests/test_protocol.c', 'tests/test_y_motor.c', '-Ttests/test.ld',
    '-o', str(elf),
], cwd=root, check=True)
nm = options.gcc.replace('gcc', 'nm')
symbols = {}
for line in subprocess.check_output([nm, str(elf)], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3:
        symbols[fields[2]] = int(fields[0], 16)
blob = elf.read_bytes()
phoff = struct.unpack_from('<I', blob, 28)[0]
phsize, phnum = struct.unpack_from('<HH', blob, 42)
for name in ['TestTransport', 'TestService', 'TestApp', 'TestYMotor', 'TestYTest', 'TestPCJog']:
    cpu = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
    cpu.mem_map(0x10000, 0x100000)
    cpu.mem_map(0x20000000, 0x20000)
    for index in range(phnum):
        kind, offset, address, _, size, _, _, _ = struct.unpack_from(
            '<8I', blob, phoff + index * phsize)
        if kind == 1 and size:
            cpu.mem_write(address, blob[offset:offset + size])
    sentinel = 0x10FFF0
    cpu.reg_write(UC_ARM_REG_SP, 0x2001FFF0)
    cpu.reg_write(UC_ARM_REG_LR, sentinel | 1)
    cpu.emu_start(symbols[name] | 1, sentinel, timeout=10_000_000)
    result = cpu.reg_read(UC_ARM_REG_R0)
    from unicorn.arm_const import UC_ARM_REG_PC
    if cpu.reg_read(UC_ARM_REG_PC) != sentinel:
        raise RuntimeError(f'{name}: emulator timed out')
    if result:
        raise AssertionError(f'{name}: C test source line {result}')
    print(f'PASS {name}')
