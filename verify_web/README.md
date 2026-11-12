# verify_web — 알고리즘 동작 확인 화면

실제 장비에서 알고리즘이 제대로 보정하는지 확인하는 화면이다. 발표용 화면은 `demo_web`(모터값을 직접 지정)이고, 이 앱은 그 코드를 고치지 않는다.

- 회차마다 **흔들림 대기 → 촬영 → 추론 → 보정 이동**을 최대 20회 반복하고, 지금 몇 회차·어느 단계인지 보여 준다.
- 매 회차 **알고리즘 추론값**과 **스테이지 실제 오차**(STATUS 위치 − 센터)를 나란히 기록한다. 둘의 차이가 알고리즘 오차다.
- 모터 이동 중인지 아닌지(ACK~DONE 구간), 스테이지 위치, 원점·리밋 상태를 보여 준다.
- 실시간 영상: 왜곡 보정 + SEMES 화각 크롭 + 확대(1·2·4배, 화면 중심/플레이트 중심)를 변환표 하나로 처리한 1280×720 흑백, 약 10fps.
- **상단 오른쪽 `데모 웹으로 →` 버튼**: 누르면 카메라·STM32를 놓고 데모 웹을 띄워 그 화면으로 넘어간다. 이 앱 주소를 다시 열면 데모 웹을 끄고 장치를 다시 잡는다.

## 실행 (저장소 루트)

```bash
# 1) 알고리즘 실행기 빌드 (OpenCV 3.4, 저장소 src/alignment.cpp)
cmake -S verify_web/native -B verify_web/native/build -G Ninja
cmake --build verify_web/native/build

# 2) 장치 없이 화면·흐름 시험
demo_web/.venv/bin/python -m verify_web serve --mode mock --camera off --algo mock

# 3) 실물 (robot_stage 드라이버 + 카메라 + 실제 알고리즘)
demo_web/.venv/bin/python -m verify_web serve --mode hardware --calib <캘리브레이션.yml> --host 0.0.0.0
```

- 화면: `http://<파이 IP>:8090` (기본은 파이 안에서만 `127.0.0.1`). `--host 0.0.0.0` 이면 같은 망 누구나 조작할 수 있다.
- 데모 웹은 `데모 웹으로` 버튼을 누를 때 `python -m demo_web serve --host <같은 host> --port 8080 --mode hardware` 로 뜬다. 바꾸려면 `--demo-args "--mode hardware --transport serial"`, `--demo-port`.
- 데모 웹이 따로 이미 떠 있으면(8080 사용 중) 넘기지 않고 오류를 보여 준다.

## 주요 옵션

| 옵션 | 기본 | |
|---|---|---|
| `--mode` | `mock` | 스테이지 `mock` / `hardware` |
| `--transport` | `driver` | `driver` = `/dev/robot_stage`, `serial` = USB 직접 (데모 웹 `demo_web.stage` 를 그대로 사용) |
| `--camera` | `picamera2` | `off` = 격자 무늬 가짜 영상 |
| `--calib` | `hardware/camera_calib/camera_calib.yml` (있으면) → `demo_web/calibration/camera_calib.yml` | 데이터셋을 찍은 캘리브레이션과 같은 파일을 쓴다. `roll_deg`(기울기 보정)도 읽는다 |
| `--zoom` | 1.0 | 알고리즘 사진 배율. 데이터셋 촬영(`capture_dataset --zoom`)과 같게 |
| `--exposure`, `--gain`, `--fps` | 8333, 2.0, 10 | 데이터셋 촬영 카메라 설정과 같음 (AWB 끔, 색 게인 3.3/1.4) |
| `--algo` | `worker` | `worker` = `native/build/align_worker`, `mock` = 실제 오차 + 잡음 |
| `--algo-output` | `offset` | `offset` = 센터 대비 위치(명령 = 부호 반대, 김성현 방식), `command` = 보낼 명령(이승재 방식) |
| `--algo-gray` | 끔 | 기본은 SEMES BMP 를 `imread` 한 것과 같은 BGR 3채널로 넘김 |

화면의 **설정**에서 허용 오차(기본 Y ±0.045 mm, Θ ±0.01°), 최대 회차, 한 번 이동 한도(넘으면 움직이지 않고 멈춤), 랜덤 시작 범위, 접근 거리, 평균 장수, 안정 판정, Θ 명령 전송, 센터 좌표를 바꾼다. `verify_web/instance/settings.json` 에 저장된다.

## 검증 순서

1. **Y 원점 복귀** → 플레이트를 센터에 맞춘 뒤 **현재 위치를 센터로** (또는 설정에 데이터셋 촬영 `--center` 값 입력).
2. **랜덤 시작 위치** 또는 Y/Θ 를 넣고 **지정 위치로**. 시작 위치는 데이터셋과 같이 `접근 거리`만큼 −쪽에서 들어온다.
3. **자동 보정** (끝까지) / **한 단계** (한 회차씩) / **측정만** (이동 없이 추론만).
4. 결과: 추론이 허용 안으로 들어오면 완료. 이때 실제 오차가 허용 밖이면 `실제는 허용 밖!` 으로 표시한다.

기록은 `verify_web/instance/runs/<시각>/` 에 남는다: `trials.csv`, `run.json`, 회차 사진 `trial_NN.bmp`(3840×2160, 알고리즘 입력 그대로).

## 구조

| 파일 | |
|---|---|
| `geometry.py` | 센서 원본 → 알고리즘 3840×2160 / 화면 1280×720 변환표. 알고리즘 쪽은 `hardware/camera` 의 C++ `Rectifier` 와 같은 식 |
| `camera.py` | Picamera2 4056×3040 YUV420 의 Y. 화면 JPEG, 추론용 프레임, 흔들림 판정(위상 상관) |
| `algorithm.py` | `align_worker` 자식 프로세스(멈추거나 죽으면 다음 요청에 다시 띄움) / 모의 추론 |
| `runner.py` | 검증 루프와 화면 상태 |
| `devices.py` | 장치 소유, 데모 웹 띄우기·끄기 (SIGTERM = 데모 웹 정상 종료) |
| `native/` | `align_worker` (저장소 `alignment(cv::Mat&)` 호출). 알고리즘 소스가 바뀌면 `-DALGO_SOURCES=...;...` |

`native/linux_prelude.hpp`: 팀 헤더 `include/alignment.hpp` 의 `typedef double error_t;` 가 리눅스 glibc 의 `error_t` 와 이름이 겹쳐 빌드가 안 되는 것을 우회한다. 헤더의 이름을 바꾸면 필요 없다.

## 시험

```bash
demo_web/.venv/bin/python -m unittest discover -s verify_web/tests -t .
```

장치 없이 돈다 (모의 스테이지 + 가짜 카메라 + 모의 추론, 데모 웹 대신 `http.server`).
