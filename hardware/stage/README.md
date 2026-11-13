# stage — 스테이지 API (stage_api)

보정 앱이 STM32 스테이지를 움직일 때 쓰는 C++ 라이브러리. `/dev/robot_stage` ioctl을 감싼다.
앱은 ioctl 번호, 구조체, errno를 몰라도 된다.

```cpp
#include "stage/stage.hpp"

stage::Stage st;                                   // /dev/robot_stage 열기
st.home();                                         // 원점 복귀, DONE 까지 기다림
auto r = st.moveTo(stage::Position::fromMmDeg(1.5, 0));
if (!r.ok()) { /* r.result: LIMIT / FAULT / STOPPED */ }
```

| 파일 | 하는 일 |
|---|---|
| `include/stage/stage.hpp` | 공개 API |
| `src/stage.cpp` | ioctl 호출, 오류 변환 |
| `tools/stagectl.cpp` | 시험 도구 (mm, ° 단위) |

드라이버와 같은 헤더(`../driver/include`)를 쓴다. 프로토콜이 바뀌면 함께 다시 빌드한다.

## 빌드

```bash
cd hardware/stage
cmake -S . -B build -G Ninja
cmake --build build
```

## 실행 전 준비 (드라이버)

```bash
cd hardware/driver && make
sudo insmod robot_stage.ko
sudo ldattach -s 115200 -8 -n -1 29 /dev/ttyACM0   # STM32 USB 연결 후
```

## API

| 함수 | 하는 일 | 기다림 |
|---|---|---|
| `ping()`, `status()`, `limits()` | 조회 | 응답까지 |
| `home(axes)` | 원점 복귀 | DONE 까지 (기본 60 s) |
| `moveTo(pos)`, `moveBy(delta)` | 절대 / 상대 이동 | DONE 까지 (기본 20 s) |
| `startHome`, `startMoveTo`, `startMoveBy` + `waitDone(t)` | 보내고 나중에 기다리기 | 나눠서 |
| `stop()`, `setSpeed()` | 정지, 속도 | 응답까지 |
| `homeIntact()` | 원점이 유지되고 있는지 STATUS 로 확인 | 응답까지 |

단위는 프로토콜 그대로 Y = µm, Θ = 0.001°이다. mm, °에서 바꿀 때는 `Position::fromMmDeg()`를 쓴다 (반올림).

### 결과와 오류

| 상황 | 앱이 받는 것 |
|---|---|
| 이동 완료 | `MoveResult.result = Ok` |
| 이동 중 리밋, 고장, STOP | `MoveResult.result = Limit / Fault / Stopped` (예외 아님 → 앱이 판단) |
| STM32 거부 (범위 밖, 바쁨 등) | `StageError`, `kind() == Nak`, `nak()`에 사유 |
| 원점 없음 (NAK NOT_HOMED) | `StageError`, `kind() == HomeLost` → `home()` 다시 |
| DONE 이 안 옴 + 원점 플래그 사라짐 | `StageError`, `kind() == HomeLost` (STM32 재부팅 의심) |
| DONE 이 안 옴 | `StageError`, `kind() == Timeout` (이동 중일 수 있음 → `status()`로 확인) |
| 장치 없음, line discipline 안 붙음 | `StageError`, `kind() == NoDevice` |

## STM32 이벤트 (BOOT, LIMIT_HIT)

드라이버는 아직 EVENT 프레임을 앱에 넘기지 않는다 (`rs_cmd.c`의 `TODO: read/poll로 앱에 전달`).
그래서 이 API는 아래 방법으로 대신 알아챈다.

- **STM32 재부팅(BOOT):** 원점이 지워지므로 다음 이동이 NAK(NOT_HOMED)가 되거나, 이동 중이었다면 DONE 이 오지 않는다. 둘 다 `HomeLost`로 알린다. 이동 전에 확인하려면 `homeIntact()`를 쓴다.
- **대기 중 리밋 감지(LIMIT_HIT):** `status()`의 `yLimitNeg()`, `yLimitPos()`로 확인한다.

드라이버에 이벤트 전달(`read`/`poll` 또는 ioctl)이 생기면 즉시 알 수 있게 바꾼다.

## stagectl

```bash
sudo ./build/stagectl ping
sudo ./build/stagectl status
sudo ./build/stagectl config
sudo ./build/stagectl home
sudo ./build/stagectl move 1.5          # Y 1.5 mm 로
sudo ./build/stagectl rel -0.05         # Y −0.05 mm 만큼
sudo ./build/stagectl cycle 1 3 10      # 1 mm ↔ 3 mm 10번 왕복, 이동 시간·도착 위치
```

`cycle`의 위치는 STM32가 스텝 수로 계산한 값이다. 실제로 그 자리에 갔는지는 카메라로 확인해야 한다 (③ 측정 모드).

## 확인한 것 (2026-11-05)

- 빌드 (경고 없음), 장치가 없을 때 오류 메시지.
- **STM32 연결 시험은 아직 안 함** (STM32 미연결).
