# A507 STM32 펌웨어

NUCLEO-F446RE에서 Pi의 바이너리 명령을 받는 펌웨어다.
통신 계약은 [robot_stage_protocol.md](robot_stage_protocol.md), 하드웨어 배경과
검증 기록은 [A507_STM32_Context.md](A507_STM32_Context.md)를 참고한다.

## 현재 구현 단계

### 2026-11-04 Y축 스위치 원점복귀 구현

사용자 확인: **DIR High=오른쪽(+Y), Low=왼쪽**, 외부 NC 스위치 사용,
**파이 HOME 명령을 받을 때만** 원점복귀한다. 전원 ON/RESET/USER/PING은 이동을 시작하지 않는다.
`app_config.h`는 `APP_Y_SWITCH_HOME=1`, `APP_Y_MANUAL_HOME=0`,
`y_motor.h`는 `Y_MOTOR_USE_LIMITS=1`이다. 이전 스위치 없는 시험 펌웨어와 기본값이 다르다.

| 스위치 | STM32 핀 / 보드 표시 | 연결 |
| --- | --- | --- |
| 1: Y 왼쪽 원점 | PA0 / A0 | 스위치 COM→GND, NC→PA0 |
| 2: Y 오른쪽 끝 | PA1 / A1 | 스위치 COM→GND, NC→PA1 |
| 3: Θ 원점 | PA4 / A2 | 스위치 COM→GND, NC→PA4. Θ 제어는 아직 비활성 |

GPIO 내부 풀업을 사용한다. NO 단자는 연결하지 않는다. 평상시 COM–NC가 이어져 0,
눌리면 접점이 열려 1이다. 선이 끊기거나 미연결이어도 1이므로 눌림과 구별할 수 없다.
양쪽 Y 입력이 모두 1이면 HOME/jog/이동을 거절하거나 중단한다. 실제 스위치 설치·논리값은
아직 보드에서 검증하지 않았다. 스위치는 기계 끝에 충돌하기 전에 작동하도록 설치한다.

`HOME(axis_mask=1)` 접수 시 ACK를 보내고 다음을 비차단 실행한다.

1. 이미 왼쪽 스위치가 눌렸다면 오른쪽으로 움직여 먼저 접점을 해제한다.
2. 왼쪽으로 800Hz(약 0.5mm/s) 탐색하다 스위치 1을 만나면 즉시 펄스를 멈춘다.
3. 오른쪽으로 320Hz(약 0.2mm/s) 이탈한다.
4. 왼쪽으로 160Hz(약 0.1mm/s) 다시 접근해 접촉점을 원점 0으로 설정한다.
5. 원래 HOME SEQ로 DONE(OK)을 보내고 Y 원점 플래그를 켠다.

접점 변화 후 10ms 안정 시간을 확인한다. 탐색은 최대 83,200펄스(52mm), 이탈/재접근은
각 최대 3,200펄스(2mm), 전체 제한 시간은 120초다. 접점을 찾지 못하거나 해제하지 못하면
DONE(FAULT), 양쪽 동시 감지 또는 이탈 중 오른쪽 끝 감지는 DONE(LIMIT)로 종료한다.
이 탐색 한도는 센서 고장 때 기계 충돌을 방지하는 거리 보장이 아니다.
HOME 중 STOP도 ACK 후 정지하고 원래 HOME SEQ에 DONE(STOPPED)를 보낸다.
실패·중단한 HOME은 원점을 설정하지 않으며 다시 HOME해야 한다.

요청한 **소프트웨어 범위는 0~500mm(0~500000µm)**다. 실제 장비는 여전히 50mm 모델이며
500mm 이동이 가능하다는 뜻은 아니다. 표준 MOVE의 소프트 리밋보다 실제 끝 스위치 정지가
우선한다. 원점에서 왼쪽 스위치가 눌린 채로 오른쪽 이동을 시작할 수 있다.
일반 이동도 진행 방향의 끝 스위치만 제한하며, 양쪽이 동시에 눌리면 양방향을 제한한다.
PC raw jog는 원점·소프트 리밋을 사용하지 않지만 이 스위치 제한은 적용된다.

Debug/Release 빌드와 ARM 모의 시험을 통과했다. **2026-11-04 새 Debug 펌웨어 다운로드,
플래시 검증 및 리셋을 완료했다.** COM4에서 PONG fw 0.2.0, CONFIG(0,500000,0,0),
STATUS(IDLE, flags=0x0C)를 확인했다. 두 Y 스위치 입력이 모두 1이며 원점은 미설정이다.
스위치 배선·실물 HOME·파이 연동 시험은 아직 하지 않았다. 이동 명령은 보내지 않았다.

이후 사용자 실물 시험: PC jog 1600펄스/1600Hz의 실제 이동을 확인했다.
파이 `rsctl home 1`은 DONE(FAULT) 또는 DONE(LIMIT)가 발생한다고 보고했다.
특히 오른쪽으로 조금 이동한 뒤 FAULT가 자주 발생하며, 원인은 아직 확정하지 않았다.
따라서 실물 HOME·파이 표준 이동 전체 시험은 성공으로 검증된 상태가 아니다.
STATUS의 FAULT는 이전 HOME 실패 또는 대기 중 스위치 눌림으로 저장될 수 있으며,
스위치를 놓거나 STOP을 보내는 것만으로 해제되지 않는다. HOME 성공 또는 리셋으로 초기화한다.

### 2026-11-04 통신 계약 대조 및 추가 구현

새로 첨부한 계약은 저장된 `robot_stage_protocol.md`와 동일하다. 바이트 형식/VER는
변경하지 않았다. 아래 구현은 fw 0.2.0에 포함되며 위 다운로드 기록대로 보드에 적용했다.

| 계약 항목 | 현재 소스 및 확인 범위 |
| --- | --- |
| 프레임·CRC·잘못된 VER/LEN·20ms 수신 중단 | 구현 및 모의 시험 통과 |
| PING/STATUS/CONFIG | 상태와 무관하게 해당 조회 응답. CONFIG는 16바이트 |
| Y 이동·속도·좌표·ACK/DONE | 구현, 원점복귀 후 표준 이동까지 모의 시험 통과 |
| 표준 STOP 감속 정지 | 추가 구현. ACK 후 감속, 원래 이동 SEQ로 DONE(STOPPED) |
| 재전송 | 계약대로 동일 SEQ/TYPE이면 기존 응답 재전송, payload가 달라도 재실행하지 않음 |
| 대기 중 고장·리미트 | 타이머 준비 실패 시 FAULT EVENT 1회. 활성 스위치 감지 시 LIMIT_HIT EVENT |
| 이동 중 고장·리미트 | EVENT 대신 원래 명령의 DONE(FAULT/LIMIT). 감속 중 리미트도 STOPPED보다 우선 |
| 원점·Y 범위 | NC 스위치 탐색 HOME, CONFIG Y 0~500000µm. 실물 스위치 시험 대기 |
| Θ축 | 구동·원점·변환 상수·범위 미확정, 비활성. Θ 이동/속도 요청과 HOME mask=2/3 거절 |
| 수신 후 20ms 응답 | 비차단 처리 구현. 실물 최악 응답시간 및 파이 연동은 미측정 |

표준 STOP은 감속률 8000펄스/s²(약 5mm/s²)로 다음 STEP 주기를 늘린다.
진행 중인 펄스를 자르지 않고 원래 목표를 넘지 않으며, 이미 완료된 이동은 STOP을
뒤늦게 받아도 OK로 보고한다. 중복 STOP은 감속을 다시 시작하지 않는다.
3200Hz에서 최대 약 641개의 추가 펄스(약 0.401mm)가 나올 수 있다.
목표까지 남은 거리가 짧으면 목표에서 먼저 멈춘다. 이 값은 전기적 펄스 모델의
상한이며 실물 관성·실제 감속 성능의 검증값이 아니다. 일반 출발 가속은 아직 없다.
PC raw jog의 STOP은 기존 즉시 중단 동작을 유지한다. 하드웨어 비상정지를 대체하지 않는다.

STATUS의 스위치 비트는 활성화된 Y 스위치만 보고한다. 스위치 기능을 끈 상태의
미연결 풀업 High를 '눌림'으로 보고하지 않는다. Θ 비트는 비활성으로 0이다.
PC 도구의 `Y_HOME_flag`/`Y_LIMIT_flag`는 GPIO 원시 전압이 아닌 이 STATUS 비트다.

Y축 명령은 연결했지만 Θ축이 미구현이므로 2축 계약 전체 구현 완료는 아니다.
모의 시험은 실제 NC HOME 구성과 별도 수동 원점/출력 잠금/스위치 비활성 구성도 검사한다.

### 파이 표준 Y축 명령 사용

fw 0.2.0 소스에 표준 명령 연결을 추가했다. Python은 필수가 아니며 파이 드라이버가
기존 바이너리 패킷을 그대로 보내면 된다. **2026-11-04 보드 다운로드를 완료했다.**
현재 `app_config.h`는 스위치 HOME 및 Y 0~500000µm를 사용한다.
HOME 성공 전 MOVE는 NOT_HOMED로 거절한다. 부팅·리셋·PC raw jog 이후에는 다시 HOME한다.

| 명령 | 새 구현 |
| --- | --- |
| GET_CONFIG | 상태와 무관하게 CONFIG 16바이트, 설정된 Y 최소/최대와 Θ 0/0 반환 |
| HOME | axis_mask=1만 지원. ACK→왼쪽 스위치 탐색·이탈·재접촉→DONE |
| SET_SPEED | Y 10~2000µm/s → 16~3200Hz. Y=0은 유지. 기본 100µm/s(160Hz). Θ는 0만 허용 |
| MOVE_REL / MOVE_ABS | 원점 설정 후 Y µm 좌표로 이동, ACK→DONE 12바이트 |
| GET_STATUS | IDLE/MOVING/HOMING/FAULT, 원점 플래그, 펄스 기반 추정 위치와 signed 스텝 수 |
| STOP | ACK 후 표준 이동을 감속 정지하고 DONE(STOPPED). PC raw jog는 즉시 중단 |

Θ 이동/속도 값이 0이 아니거나 HOME mask가 1이 아니면 BAD_PARAM이다. Θ 원점 비트는
항상 0이며 Θ CONFIG 0/0은 축 비활성 표시다. 파이에서 HOME mask=3을 보내면 거절된다.
좌표의 +방향은 사용자 확인대로 DIR High/오른쪽이다. 1600펄스/mm 환산이며 목표 절대좌표를 기준으로
반올림해 작은 상대 이동을 반복해도 반올림 오차가 누적되지 않게 했다.

표준 이동은 설정 범위를 검사한다. 기존 PC raw jog는 거리 제한을 적용하지 않으며,
raw jog가 접수되면 표준 Y 원점 플래그를 지운다. 두 경로의 동시 시작은 BUSY로 거절한다.
강제정지로 펄스 집계가 불확실해지면 추정 좌표만 반환하고 원점을 해제하며 FAULT로 전환한다.
오류 후 HOME으로 원점을 다시 찾는다. 이동 좌표는 엔코더 측정값이 아닌 출력 펄스 기반 추정값이다.
HOME 진행·실패·중단 시 좌표는 이전 추정값이며 유효한 원점 좌표가 아니다.

마지막 접수 HOME/MOVE는 조회·STOP을 사이에 끼워도 재실행하지 않는다. 실행 중에는
기존 ACK, 완료 후에는 기존 DONE을 반환한다. 동일 SEQ/TYPE이면 payload가 달라도 기존 응답을 반환한다.
다음 접수 HOME/MOVE 또는 256개 새 요청 후 완료 캐시는 교체/만료된다. 리셋 전 요청 재생은
지원하지 않는다. 조회 응답의 직전 요청 캐시와 독립적으로 관리한다.

표준 순서: CONFIG 조회 → HOME(Y=1)의 ACK/DONE 대기 →
SET_SPEED → MOVE_REL/MOVE_ABS의 ACK/DONE 대기. 수신 응답은 TYPE뿐 아니라 SEQ로 구분한다.
HOME/MOVE 응답이 빠르면 ACK와 DONE이 같은 UART 수신 묶음에 들어올 수 있다.

예: HOME payload=`01`, SET_SPEED payload=`E8 03 00 00 00 00 00 00`(Y 1000µm/s),
MOVE_REL payload=`E8 03 00 00 00 00 00 00`(Y +1000µm=오른쪽 1mm, Θ 0).
각 payload에 계약의 헤더·새 SEQ·CRC를 붙인다. HOME은 ACK 대기와 DONE 대기를 구분하며
DONE까지 최대 120초 이상을 허용한다. 100ms 재전송 타이머를 동작 완료 타이머로 사용하지 않는다.

아래의 10월/이전 11월 기록은 당시 펌웨어의 이력이며 새 표준 명령 구현보다 우선하지 않는다.

2026-11-03: 요청당 1,600펄스 시험 상한을 제거했다.
Debug/Release 빌드 및 자동 테스트를 통과했다. 2026-11-03 보드 재연결 후 최신 Debug ELF를
다운로드하고 플래시 검증·리셋·COM4 상태 조회에 성공했다. IDLE, ready=1,
output_allowed=1을 확인했으며 실제 이동 명령은 보내지 않았다.
변경된 펌웨어와 PC 스크립트는
이제 **1~4,294,967,295펄스(uint32)**를 받는다. 이는 데이터 형식의 범위이며 기계의
허용 이동거리가 아니다. 사용자가 확인한 환산은 약 1,600펄스/mm다.
예: `--pulses 16000 --hz 160`은 약 10mm, 약 100초다. 실제 남은 이동거리 안에서 요청한다.
주파수 16~3200Hz, DIR high/low, 동시 실행 1건 제한은 유지한다.
이 raw jog 경로에는 가감속·원점복귀·누적 소프트 리미트·통신 단절 자동 정지가 없다.
PC는 접수 응답을 최대 2초, 완료를 `펄스 수/Hz × 2 + 5초` 동안 기다리고
타임아웃/Ctrl+C에 STOP을 시도한다. USB 단절 시 STOP 전달을 보장하지 않는다.

2026-11-02: 사용자 요청으로 **스위치 제한을 비활성화**했다.
당시 `Core/Inc/y_motor.h`의 기본값은 `Y_MOTOR_USE_LIMITS=0`이었다. 현재 소스는 1이다.
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
| `Core/Src/y_home.c` | 왼쪽 스위치 탐색·이탈·재접촉, 접점 안정 확인 및 시간/거리 제한 |
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
| PA4 | A2 | 외장 Θ 원점 스위치 입력. Θ 제어는 비활성 |

모터 전원은 TMC2209 VM에 별도 공급하고 전원 GND를 공통으로 연결한다.
모터 코일 두 쌍은 각각 OA1/OA2, OB1/OB2에 연결한다. 모듈 실크 표기와
코일 쌍을 확인한다. 외장 NC 스위치는 평상시 Low, 눌림/열림 시 High다.
LSM-NK223 설명서의 내장 NO 센서는 이 논리와 다르다.

현재 소스는 `Y_MOTOR_USE_LIMITS=1`이며 Y 양 끝 스위치가 필요하다.
왼쪽 High는 왼쪽 이동, 오른쪽 High는 오른쪽 이동을 제한한다. 양쪽 High는 양방향을 제한한다.
누른 스위치에서 반대로 이탈할 수 있다. 미연결 풀업도 High다.

## PC에서 실제 펄스 시험하기

STM32 ST-LINK USB(CN1)를 PC에 연결한다. 파이는 필요 없다. 모터·TMC2209·외부
모터 전원은 별도로 연결하고 전류·마이크로스텝 설정을 확인한다. Θ축 드라이버는
Enable 극성이 미확정이므로 이번 Y축 시험에서 전원을 공급하지 않는다.
현재 빌드는 NC 스위치 사용 버전이다. 위의 양 끝 스위치 배선을 먼저 완료한다.

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
함께 표시되는 Y_HOME_flag/Y_LIMIT_flag는 STATUS의 스위치 정보다.
펌웨어가 진행 방향의 스위치와 양쪽 동시 눌림 조건을 검사한다.
이는 모터 배선·전류 설정의 정상 여부를 검증한 결과는 아니다.
구형 펌웨어는 `UNKNOWN_TYPE (old firmware?)`를 반환하므로 새 ELF 다운로드가 필요하다.

배선과 이동 여유를 확인한 뒤 다음 명령을 **한 번** 실행한다.

```powershell
# DIR High, 160펄스, 최대 160Hz: 약 1초의 짧은 시험
.\build\pc-venv\Scripts\python.exe tools/y_motor_test.py --port COM4 jog --dir high --pulses 160 --hz 160

# 위 시험 종료 후, 반대 DIR 레벨로 시험할 때만 별도로 실행
.\build\pc-venv\Scripts\python.exe tools/y_motor_test.py --port COM4 jog --dir low --pulses 160 --hz 160
```

사용자 확인 환산은 약 1600펄스/mm이므로 위 160펄스는 약 0.1mm다.
`high`는 오른쪽(+Y), `low`는 왼쪽(−Y)이다. 동작 중에는
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
state는 0 IDLE / 1 RUNNING / 2 DONE / 3 STOPPED / 4 ABORTED / 5 LIMIT / 6 SWITCH(원점복귀 중 접점 정지).
조회 보고의 SEQ는 조회 요청, 완료 보고의 SEQ는 원래 jog 요청과 같다.

마지막으로 접수한 jog의 SEQ/payload만 보관한다. 같은 요청은 재실행하지 않고
기존 ACK 또는 완료 보고를 재전송하며, 같은 SEQ의 다른 payload는 BAD_PARAM이다.
PING/조회/STOP은 이 캐시를 지우지 않는다. 새 jog가 접수되거나 리셋되면 이전 요청의
중복 방지 보장은 사라진다. 스크립트는 조회한 마지막 jog SEQ를 피하고 자동 재시도하지 않는다.
한 송신자, 한 outstanding jog를 전제로 한다. 절대좌표·원점·소프트 리미트·가감속은 없다.
GET_STATUS는 표준 좌표/원점 상태를 반환한다. raw jog 펄스 집계는 `status`가 조회하는 0x31 보고를 사용한다.

## 내부 호출 API

기본 빌드는 `Y_MOTOR_OUTPUT_ALLOWED=1`이다. 다시 전체 펄스 출력을 잠그려면
`y_motor.h`의 값을 0으로 바꾸거나 모든 관련 번역 단위에
`-DY_MOTOR_OUTPUT_ALLOWED=0`을 적용한다. 이 값만으로 자동 이동이 시작되지 않는다.
전류·배선·움직일 여유가 확인된 시험 진입점에서 **한 번만** 호출한다.

```c
#include "y_motor.h"

/* App_Init()이 GPIO/TIM3 초기화 후 YMotor_Init()을 이미 호출한다.
 * DIR High(오른쪽)로 160펄스를 최대 160Hz로 요청한다.
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
절대좌표와 원점은 상위 표준 서비스와 y_home에서 관리한다. 출발 가속은 아직 없다.
DIR High는 오른쪽이다. 전류·마이크로스텝은 모듈에 설정하며 코드가 UART로 설정하지 않는다.

| API | 의미 |
| --- | --- |
| `YMotor_Start(dir_high, pulses, hz)` | 자동 Enable 후 비동기 출력 시작. 완료 시 Disable |
| `YMotor_GetStatus(&status)` | 마지막 요청의 펄스 수·완료 수·상태·Enable·집계 유효성 조회 |
| `YMotor_Stop()` | 현재 단발 펄스를 끝낸 뒤 추가 펄스 없이 정지·Disable |
| `YMotor_Decelerate()` | 표준 STOP용 감속 정지. 원래 요청 펄스 수를 넘지 않음 |
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
python tests/run_tests.py --gcc arm-none-eabi-gcc --motor-enabled --unconfigured
python tests/run_tests.py --gcc arm-none-eabi-gcc --motor-enabled --physical-home
python tests/test_pc_client.py
```

테스트 기본 실행은 출력을 잠근 모의 빌드, `--motor-enabled`는 출력을 허용한 모의 빌드다.
둘 다 보드에 다운로드하지 않는다. 일반 펌웨어 빌드의 기본값은 출력 허용이다.
타이머 모델로 정상 펄스 수·High 폭, IRQ 지연, 중복 시작 거절, graceful stop,
강제정지 집계, 스위치 비활성 시험 구성에서 양 입력 High인 채 정상 완료, 스위치 활성 빌드의 중단,
잘못된 클록 거절, 펄스 실행 중 UART 응답을 검사한다.
실제 타이머 파형·모터 방향·이동거리·전류·발열은 실물 측정 전이다.
PC 시험 명령의 중복 방지·BUSY·완료/중단 보고, UART를 통한 JOG→ACK→STOP→보고와
동작 중 PING을 검증한다. PC 클라이언트는 CRC·분할 수신·잡음·SEQ·NAK·타임아웃을 검사한다.

`--physical-home`은 실제 기본 설정인 NC HOME/0~500mm를 사용한다. 탐색·이탈·재접촉,
스위치 튐, 접점 미검출·해제 실패·양쪽 감지·시간 초과·STOP을 모의 입력으로 검사한다.
UART HOME→ACK→DONE→STATUS(IDLE/원점)→MOVE_REL→DONE과 원점 접촉 상태에서
이탈도 검증했다. 다른 구성은 회귀 시험용이며 `--unconfigured`는 HOME을 명시적으로 끈다.
2026-11-04 Debug: FLASH 26,996B / RAM 5,448B, Release: FLASH 14,604B / RAM 5,448B.
모의 시험 5개 구성 및 PC 클라이언트 8개 테스트 통과. 실물 스위치·파이 연동은 미검증이다.

설계 참고: [STM32F446 RM0390](https://www.st.com/resource/en/reference_manual/dm00135183.pdf),
[TMC2209 데이터시트](https://www.analog.com/media/en/technical-documentation/data-sheets/TMC2209_datasheet_rev1.09.pdf).
