# A507 STM32 펌웨어

NUCLEO-F446RE에서 Pi의 바이너리 명령을 받는 펌웨어다.
통신 계약은 [robot_stage_protocol.md](robot_stage_protocol.md), 하드웨어 배경과
검증 기록은 [A507_STM32_Context.md](A507_STM32_Context.md)를 참고한다.

## 현재 구현 단계

2026-11-03: 요청당 1,600펄스 시험 상한을 제거했다.
Debug/Release 빌드 및 자동 테스트를 통과했다. 2026-11-03 보드 재연결 후 최신 Debug ELF를
다운로드하고 플래시 검증·리셋·COM4 상태 조회에 성공했다. IDLE, ready=1,
output_allowed=1을 확인했으며 실제 이동 명령은 보내지 않았다.
변경된 펌웨어와 PC 스크립트는
이제 **1~4,294,967,295펄스(uint32)**를 받는다. 이는 데이터 형식의 범위이며 기계의
허용 이동거리가 아니다. 사용자가 확인한 환산은 약 1,600펄스/mm다.
예: `--pulses 16000 --hz 160`은 약 10mm, 약 100초다. 실제 남은 이동거리 안에서 요청한다.
주파수 16~3200Hz, DIR high/low, 동시 실행 1건 제한은 유지한다.
가감속·원점복귀·누적 소프트 리미트·통신 단절 자동 정지는 구현하지 않았다.
PC는 접수 응답을 최대 2초, 완료를 `펄스 수/Hz × 2 + 5초` 동안 기다리고
타임아웃/Ctrl+C에 STOP을 시도한다. USB 단절 시 STOP 전달을 보장하지 않는다.

2026-11-02: 사용자 요청으로 **스위치 제한을 비활성화**했다.
`Core/Inc/y_motor.h`의 `Y_MOTOR_USE_LIMITS=0`이 현재 기본값이다.
Y_HOME/Y_LIMIT가 1이어도 시작하며 이동 중 입력 변화로 멈추지 않는다.
스위치 기반 LIMIT_HIT 이벤트도 보내지 않는다. 입력값 조회는 유지한다.
PC 스크립트도 입력값으로 이동을 차단하지 않는다. 당시 유지했던 1~1600펄스 제한은
2026-11-03 제거했으며 STOP은 유지한다.
끝단 자동 정지가 없으므로 이동 여유가 있는 위치에서 짧게 시험한다.
스위치 설치 후 `Y_MOTOR_USE_LIMITS=1`로 변경하고 다시 빌드·다운로드하면 제한이 복구된다.
Debug/Release 빌드와 스위치 제한 비활성/활성·출력 잠금 구성의 모의 테스트를 통과했다.
2026-11-02 보드 재연결 후 이번 Debug ELF 다운로드·플래시 검증·리셋에 성공했다.
COM4 조회에서 IDLE, ready=1, output_allowed=1을 확인했다. Y_HOME/Y_LIMIT는 둘 다 1이지만
현재 빌드에서는 이동을 제한하지 않는다. 실제 이동 명령은 전송하지 않았다.
이후 2026-11-02 사용자가 PC jog 시험에서 실제 모터가 정상 동작한다고 확인했다.
정량적인 이동 거리·정밀도·전류·발열은 측정하지 않았다.

2026-10-30: **PC USB 명령으로 Y축 펄스 시험 가능**. 기존 코드에 시험 명령과
PC 스크립트를 추가했다. Y축 출력은 허용되어 있으며 명시적인 `jog` 명령으로만
시작한다. 부팅·RESET·USER 버튼·PING은 모터를 움직이지 않는다.
Pi의 MOVE/HOME은 좌표·원점 관리가 구현되기 전까지 NAK(FAULT)를 반환한다.
PC 시험 명령은 아래에 정의한 별도 확장으로, 원본 Pi 통신 계약을 변경하지 않는다.
2026-10-30 시험 확장을 포함한 Debug ELF를 다운로드하고 플래시 검증·재시작·COM4
상태 조회를 확인했다. `ready=1`, `output_allowed=1`, `IDLE`, Enable 해제 상태였다.
당시 `Y_HOME=1`, `Y_LIMIT=1`이므로 실제 이동 요청은 보내지 않았다.

| 파일 | 역할 |
| --- | --- |
| `Core/Src/y_motor.c` | 지정 펄스 수 관리, 방향/Enable, 정지, 상태 조회 |
| `Core/Src/y_motor_tim.c` | TIM3 CH1 단발 PWM2 출력, GPIO, 인터럽트 어댑터 |
| `Core/Inc/y_motor.h` | 펌웨어 내부 호출 API 및 펄스 시험 한도 |
| `Core/Src/app.c` | 초기화·스위치 감시 및 기존 비차단 UART 처리 |
| `Core/Src/y_test.c` | PC 시험 명령·ACK·완료 보고·최근 요청 중복 방지 |
| `tools/y_motor_test.py` | PC에서 상태 조회·짧은 이동·정지, 바이너리 패킷 자동 처리 |
| `tests/test_y_motor.c` | 실제 드라이버 C 코드와 모의 타이머 테스트 |

## 배선 기준

| STM32 | 보드 표시 | 연결 |
| --- | --- | --- |
| PB4 / TIM3_CH1 | D5 | TMC2209 STEP |
| PB5 | D4 | TMC2209 DIR |
| PB10 | D6 | TMC2209 EN/ENN, Low=활성 |
| 3V3 / GND | 3V3 / GND | 모듈 VIO / 공통 GND |
| PA0 | A0 | 외장 KW-10 원점, COM–NC 접점 회로 |
| PA1 | A1 | 외장 KW-10 끝 리미트, COM–NC 접점 회로 |

모터 전원은 TMC2209 VM에 별도 공급하고 전원 GND를 공통으로 연결한다.
모터 코일 두 쌍은 각각 OA1/OA2, OB1/OB2에 연결한다. 모듈 실크 표기와
코일 쌍을 확인한다. 외장 NC 스위치는 평상시 Low, 눌림/열림 시 High다.
LSM-NK223 설명서의 내장 NO 센서는 이 논리와 다르다.

현재 스위치는 연결하지 않아도 된다. 나중에 `Y_MOTOR_USE_LIMITS=1`로 복구하면
PA0 또는 PA1 하나라도 High일 때 시작 거절·이동 중 중단이 적용된다.
미연결 풀업도 High다. 스위치에서 이탈하는 HOME 절차는 다음 단계다.

## PC에서 실제 펄스 시험하기

STM32 ST-LINK USB(CN1)를 PC에 연결한다. 파이는 필요 없다. 모터·TMC2209·외부
모터 전원은 별도로 연결하고 전류·마이크로스텝 설정을 확인한다. Θ축 드라이버는
Enable 극성이 미확정이므로 이번 Y축 시험에서 전원을 공급하지 않는다.
현재 빌드는 스위치 미연결 시험용이며 NC 스위치 배선은 이후 설치 시 적용한다.

VS Code에서 프로젝트 폴더를 열고 **터미널 → 새 터미널 → PowerShell**을 선택한다.
기존 PowerShell `$port`나 다른 시리얼 프로그램이 COM 포트를 열고 있다면 먼저 닫는다.
아래는 프로젝트 루트에서 실행한다. Python이 PATH에 없다면 설치된 Python 경로를 사용한다.

```powershell
# 처음 한 번: PC용 환경과 시리얼 라이브러리 설치 (보드에는 설치하지 않음)
python -m venv build/pc-venv
.\build\pc-venv\Scripts\python.exe -m pip install -r tools/requirements.txt

# STLink Virtual COM Port 번호 확인
.\build\pc-venv\Scripts\python.exe tools/y_motor_test.py ports

# COM4는 위에서 확인한 포트로 교체
.\build\pc-venv\Scripts\python.exe tools/y_motor_test.py --port COM4 status
```

`IDLE`, `ready=1`, `output_allowed=1`이면 펌웨어 펄스 시험 기능이 준비된 것이다.
함께 표시되는 Y_HOME/Y_LIMIT는 입력값 정보다. 현재 빌드에서는 1이어도 이동한다.
스위치 제한을 켠 펌웨어를 사용하면 스크립트가 명령을 보내더라도 펌웨어가 제한한다.
이는 모터 배선·전류 설정의 정상 여부를 검증한 결과는 아니다.
구형 펌웨어는 `UNKNOWN_TYPE (old firmware?)`를 반환하므로 새 ELF 다운로드가 필요하다.

배선과 이동 여유를 확인한 뒤 다음 명령을 **한 번** 실행한다.

```powershell
# DIR High, 160펄스, 최대 160Hz: 약 1초의 짧은 시험
.\build\pc-venv\Scripts\python.exe tools/y_motor_test.py --port COM4 jog --dir high --pulses 160 --hz 160

# 위 시험 종료 후, 반대 DIR 레벨로 시험할 때만 별도로 실행
.\build\pc-venv\Scripts\python.exe tools/y_motor_test.py --port COM4 jog --dir low --pulses 160 --hz 160
```

1/16 분주·리드 2mm·200스텝/회전이면 각각 명목상 0.1mm다. `high`/`low`는
전기적 DIR 레벨이며 실제 +Y/−Y 방향은 실물 확인 전이다. 동작 중에는
`ACK: pulse burst accepted`가 나오고 정상 완료 시 `DONE: pulses=160/160`이 나온다.
모터 실측 위치가 아닌 출력 펄스 집계다. 매 요청은 끝나면 Enable을 해제한다.

- 실행 중 **Ctrl+C**: 스크립트가 STOP을 전송하고 포트를 닫는다. USB가 끊기면
  STOP 전달을 보장하지 않으며 이미 접수한 유한 펄스 출력은 계속될 수 있다.
- `NAK: FAULT`: 출력/클록 설정을 확인한다. 스위치 제한을 켠 빌드에서는 입력 High도 원인이다.
- `NAK: BUSY`: 앞선 이동과 완료 보고가 끝난 뒤 다시 요청한다.
- `LIMIT`/`ABORTED`, `count_exact=0`: 정상 완료가 아니다. 일부 펄스가 잘렸을 수 있다.
- 타임아웃이면 자동으로 이동을 재전송하지 않는다. STOP 전달을 시도하고 오류로 종료한다.
  연결을 복구한 뒤 상태를 조회한다. 반복 명령으로 전체 이동 한계를 보장하지는 않는다.
- 포트는 한 프로그램만 사용한다. 실행 중인 jog의 중단은 같은 터미널의 Ctrl+C를 사용한다.

포트가 열려 있지 않을 때 별도로 정지·조회할 수도 있다.

```powershell
.\build\pc-venv\Scripts\python.exe tools/y_motor_test.py --port COM4 stop
.\build\pc-venv\Scripts\python.exe tools/y_motor_test.py --port COM4 status
```

USER 버튼은 이번 기능에 연결하지 않았다. RESET은 MCU 재시작이며 모터 전원 차단이 아니다.

## PC 시험 확장 패킷

프레임/CRC는 기존 `AA 55 / VER / TYPE / SEQ / LEN / payload / CRC16`을 사용한다.
다중 바이트 값은 little-endian이다. 아래 TYPE은 **이 펌웨어의 시험 확장**이며 Pi v1 계약의
MOVE_REL 또는 DONE을 대체하지 않는다. Pi 통합 때 별도 합의가 필요하다.

| TYPE | LEN | payload / 동작 |
| --- | --- | --- |
| `0x30 Y_TEST_JOG` | 9 | `dir:u8`, `pulses:u32`, `hz:u32`. DIR 0/1, 1~4,294,967,295펄스, 16~3200Hz |
| `0x31 Y_TEST_GET` | 0 | 현재 또는 마지막 펄스 상태 조회 |
| `0x81 ACK` | 1 | cmd_type=`0x30`. 접수 후 비동기 실행 |
| `0x82 NAK` | 2 | 기존 cmd_type/reason. BAD_LEN, BAD_PARAM, BUSY, FAULT |
| `0xB0 Y_TEST_REPORT` | 20 | 조회 응답 또는 접수한 시험의 완료/중단 보고 |
| `0x05 STOP` | 0 | 기존 STOP. 즉시 중단 후 ACK; 접수된 시험은 별도 `0xB0` 보고 |

보고 payload: offset 0 state(u8), 1 dir(u8), 2 count_exact(u8), 3 enabled(u8),
4 requested_pulses(u32), 8 completed_pulses(u32), 12 frequency_hz(u32),
16 ready(u8), 17 output_allowed(u8), 18 cached(u8), 19 last_jog_seq(u8).
state는 0 IDLE / 1 RUNNING / 2 DONE / 3 STOPPED / 4 ABORTED / 5 LIMIT.
조회 보고의 SEQ는 조회 요청, 완료 보고의 SEQ는 원래 jog 요청과 같다.

마지막으로 접수한 jog의 SEQ/payload만 보관한다. 같은 요청은 재실행하지 않고
기존 ACK 또는 완료 보고를 재전송하며, 같은 SEQ의 다른 payload는 BAD_PARAM이다.
PING/조회/STOP은 이 캐시를 지우지 않는다. 새 jog가 접수되거나 리셋되면 이전 요청의
중복 방지 보장은 사라진다. 스크립트는 조회한 마지막 jog SEQ를 피하고 자동 재시도하지 않는다.
한 송신자, 한 outstanding jog를 전제로 한다. 절대좌표·원점·소프트 리미트·가감속은 없다.
기존 GET_STATUS는 잠금 상태의 Pi 서비스 정보를 반환하므로 시험 상태는 `status`로 조회한다.

## 내부 호출 API

기본 빌드는 `Y_MOTOR_OUTPUT_ALLOWED=1`이다. 다시 전체 펄스 출력을 잠그려면
`y_motor.h`의 값을 0으로 바꾸거나 모든 관련 번역 단위에
`-DY_MOTOR_OUTPUT_ALLOWED=0`을 적용한다. Pi MOVE/HOME의 `APP_MOTION_LOCKED`와는
별개이며, 이 값만으로 자동 이동이 시작되지 않는다.
전류·배선·움직일 여유가 확인된 시험 진입점에서 **한 번만** 호출한다.

```c
#include "y_motor.h"

/* App_Init()이 GPIO/TIM3 초기화 후 YMotor_Init()을 이미 호출한다.
 * 물리적 +방향이 아닌 DIR High 레벨로 160펄스를 최대 160Hz로 요청한다.
 * 이 코드를 while(1)에서 반복 호출하지 않는다. */
YMotorResult result = YMotor_Start(1U, 160U, 160U);
/* OK면 비동기 실행, LOCKED/LIMIT_ACTIVE/BUSY 등은 시작 거절. */

/* 이후 메인 루프에서 상태 조회. 완료를 기다리는 busy-wait는 하지 않는다. */
YMotorStatus status;
YMotor_GetStatus(&status);
/* state == Y_MOTOR_DONE이면 requested_pulses == completed_pulses */
```

설계 기준인 1.8도·리드 2mm·1/16 분주가 실제 구성과 일치하면 1,600펄스/mm이며,
위의 160펄스는 명목상 0.1mm다. 이 계층은 **펄스만 취급**하며 µm 환산,
절대좌표, 원점, 가감속은 아직 없다. DIR High의 실제 이동 방향은 실물 확인이
필요하다. 전류·마이크로스텝은 모듈에 설정해야 하며 코드가 UART로 설정하지 않는다.

| API | 의미 |
| --- | --- |
| `YMotor_Start(dir_high, pulses, hz)` | 자동 Enable 후 비동기 출력 시작. 완료 시 Disable |
| `YMotor_GetStatus(&status)` | 마지막 요청의 펄스 수·완료 수·상태·Enable·집계 유효성 조회 |
| `YMotor_Stop()` | 현재 단발 펄스를 끝낸 뒤 추가 펄스 없이 정지·Disable |
| `YMotor_Abort()` | 즉시 출력 중단·Disable. 잘린 펄스가 있으면 `count_exact=0` |
| `YMotor_SetEnabled(0/1)` | 정지 중 Enable 제어. 실행 중에는 BUSY |
| `YMotor_Poll()` | 스위치 제한 활성 시 원시 리미트 입력 감시. App_Poll에서 호출 |

- 요청당 **1~4,294,967,295펄스**, **16~3,200Hz**를 받는다. 0 또는 범위 밖은 거절한다.
- 펄스 High는 5µs, 첫 상승 에지 전 DIR/Enable 준비시간은 최소 2ms다.
- TIM3는 1MHz 카운터, PWM2 + One Pulse Mode다. 하드웨어가 한 펄스마다
  자동 정지하므로 인터럽트 지연 중 펄스가 계속 발생하지 않는다.
- 다음 펄스는 IRQ에서 시작한다. 요청 주파수는 상한이며 ISR 지연만큼 실제
  주기는 길어지고 흔들릴 수 있다. 정밀한 동시축 타이밍용 구현은 아니다.
- `completed_pulses`는 완전히 끝난 전기적 펄스 수다. 실측 이동량/엔코더 값이
  아니다. 강제 중단된 펄스가 드라이버에 이미 인식됐는지는 보장하지 않는다.
- 일반 완료/Stop은 EN을 비활성화하므로 정지 유지 토크와 위치 유지도 보장하지
  않는다. 다음 단계의 위치 제어에서 홀드 정책을 별도로 구현해야 한다.
- 스위치 제한 활성 시 원시 입력은 메인 루프와 펄스 완료 IRQ에서 확인한다. 감지 전 이미 시작된
  펄스가 있을 수 있다. 보드 레벨의 독립 비상정지 회로는 아니다.
- CubeMX 재생성 후 TIM3 IRQ의 USER CODE 연결과 App_Init/App_Poll 연결을
  확인한다. TIM3를 이 모듈과 HAL PWM Start/Stop에서 동시에 제어하지 않는다.
- 현재 APB1=HCLK, TIMPRE=0 설정만 지원한다. 클록이 바뀌면 ready=0으로
  출력 시작을 거절하며, 클록 계산을 갱신해야 한다.

## 빌드와 검증

STM32 ARM GCC, CMake, Ninja가 PATH에 있는 환경에서 프로젝트 루트를 연다.

```text
cmake --preset Debug
cmake --build --preset Debug
cmake --preset Release
cmake --build --preset Release
```

ARM 에뮬레이터 테스트:

```text
python -m pip install -r tests/requirements.txt
python tests/run_tests.py --gcc arm-none-eabi-gcc
python tests/run_tests.py --gcc arm-none-eabi-gcc --motor-enabled
python tests/run_tests.py --gcc arm-none-eabi-gcc --motor-enabled --with-switches
python tests/test_pc_client.py
```

테스트 기본 실행은 출력을 잠근 모의 빌드, `--motor-enabled`는 출력을 허용한 모의 빌드다.
둘 다 보드에 다운로드하지 않는다. 일반 펌웨어 빌드의 기본값은 출력 허용이다.
타이머 모델로 정상 펄스 수·High 폭, IRQ 지연, 중복 시작 거절, graceful stop,
강제정지 집계, 기본 빌드에서 양 입력 High인 채 정상 완료, 스위치 활성 빌드의 중단,
잘못된 클록 거절, 펄스 실행 중 UART 응답을 검사한다.
실제 타이머 파형·모터 방향·이동거리·전류·발열은 실물 측정 전이다.
PC 시험 명령의 중복 방지·BUSY·완료/중단 보고, UART를 통한 JOG→ACK→STOP→보고와
동작 중 PING을 검증한다. PC 클라이언트는 CRC·분할 수신·잡음·SEQ·NAK·타임아웃을 검사한다.

설계 참고: [STM32F446 RM0390](https://www.st.com/resource/en/reference_manual/dm00135183.pdf),
[TMC2209 데이터시트](https://www.analog.com/media/en/technical-documentation/data-sheets/TMC2209_datasheet_rev1.09.pdf).
