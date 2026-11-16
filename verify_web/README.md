# verify_web — 알고리즘 동작 확인 화면

실제 장비에서 알고리즘이 제대로 보정하는지 확인하는 화면이다. 발표용 화면은 `demo_web`(모터값을 직접 지정)이고, 이 앱은 그 코드를 고치지 않는다.

- **흔들림 대기 → 촬영 → 추론 → 보정 이동**을 최대 20번 반복하고, 지금 보정 이동 몇 번째·어느 단계인지 보여 준다.
  횟수는 **보정 이동** 기준이다. 측정은 이동 + 1 번 (시작 위치에서 한 번, 이동마다 뒤에서 한 번 확인).
- 측정마다 **알고리즘 추론값**과 **스테이지 실제 오차**(STATUS 위치 − 센터)를 나란히 기록한다. 둘의 차이가 알고리즘 오차다.
- 모터 이동 중인지 아닌지(ACK~DONE 구간), 스테이지 위치, 원점·리밋 상태를 보여 준다.
- 실시간 영상: 왜곡 보정 + SEMES 화각 크롭 + 확대(1·2·4배, 화면 중심/플레이트 중심)를 변환표 하나로 처리한 1280×720 흑백, 약 10fps.
- **상단 오른쪽 `데모 웹으로 →` 버튼**: 누르면 카메라·STM32를 놓고 데모 웹을 띄워 그 화면으로 넘어간다. 이 앱 주소를 다시 열면 데모 웹을 끄고 장치를 다시 잡는다.

## 실행 (저장소 루트)

```bash
# 1) 알고리즘 실행기 빌드 (OpenCV 3.4, 저장소 src/alignment.cpp)
cmake -S verify_web/native -B verify_web/native/build -G Ninja
cmake --build verify_web/native/build

# 1-1) 김성현 A507 알고리즘으로 빌드 (영점 등록 포함). Code 폴더 = src/ include/ config/
cmake -S verify_web/native -B verify_web/native/build_ksh -G Ninja -DALGO_DIR=/home/pi/KSH_Algorithm/Code
cmake --build verify_web/native/build_ksh

# 2) 장치 없이 화면·흐름 시험
demo_web/.venv/bin/python -m verify_web serve --mode mock --camera off --algo mock

# 3) 실물 (robot_stage 드라이버 + 카메라 + 김성현 알고리즘)
demo_web/.venv/bin/python -m verify_web serve --mode hardware --host 0.0.0.0 \
    --algo-path verify_web/native/build_ksh/align_worker \
    --algo-config /home/pi/KSH_Algorithm/Code/config/alignment.yml
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

화면의 **설정**에서 허용 오차(기본 Y ±0.045 mm, Θ ±0.01°), 최대 보정 이동 횟수, 한 번 이동 한도(넘으면 움직이지 않고 멈춤), 랜덤 시작 범위, 접근 거리, 평균 장수, 안정 판정, Θ 명령 전송, 센터 좌표를 바꾼다. `verify_web/instance/settings.json` 에 저장된다.

## 알고리즘 고르기 (성현 / 혜원 / 윤석)

`verify_web/instance/algorithms.json` 이 있으면 화면 위쪽에서 알고리즘을 고를 수 있다 (형식: `algorithms.example.json`).
바꾸면 실행기를 다시 띄우고 센터 기준·지난 기록을 지운다 (영점이 알고리즘마다 달라서). 고른 것은 설정에 남는다.

```bash
# 성현님 (영점 등록 가능, ALGR)
cmake -S verify_web/native -B verify_web/native/build_ksh -G Ninja -DALGO_DIR=$HOME/KSH_Algorithm/Code
# 혜원님 SSAFY 패키지 (chore/hye/merge-ssafy 2390aed~, 학습 범위 밖 보정 허용). --config = data/model
#   ALGR: set_zero_from_photos, 영점 사진 점검에 실패하면 set_reference_from_photos 로 기준을 다시 등록한 뒤 다시 영점
cmake -S verify_web/native -B verify_web/native/build_hye2 -G Ninja -DHYE2_DIR=$HOME/HYE_Algorithm
# (예전) 혜원님 모듈 (initialize_joint, 기본은 assets 의 절대 영점, set_joint_zero 로 그날 센터를 영점). HYE_DIR = 저장소 형태 폴더, HYE_MODULE = 모듈 폴더
cmake -S verify_web/native -B verify_web/native/build_hye -G Ninja -DHYE_DIR=$HOME/HYE_Algorithm \
      -DHYE_MODULE=$HOME/HYE_Algorithm/experiments/hye/00000_final_Y_T_ssafy
# 신윤석님 QR 전용 Rig 모델 (feature/shin/QRonlyMultipleExe). 센터 기준 사진(ALGR) 대비 Y·Θ.
#   --config = models/rig-reverse-fixed.xml.gz (kind qr_affine_joint, 979c1d75~ 비전 상대 보정: QR 아핀 값 18개)
#           또는 models/rig-cancellation.xml.gz (qr::Cancellation, 값 4개)
cmake -S verify_web/native -B verify_web/native/build_shin_rig -G Ninja -DQR_RIG_DIR=$HOME/SHIN_Algorithm/experiments/shin/qr_only_multiple_exe/native
# (예전) 신윤석님 단일 프레임 qr_pose::estimate: -DQR_POSE_DIR=$HOME/SHIN_Algorithm/experiments/shin/ssafy_qr_pose
cmake --build verify_web/native/build_ksh && cmake --build verify_web/native/build_hye && cmake --build verify_web/native/build_shin_rig
```

- 실행기는 시작할 때 `READY ref roi` (영점 등록·위치 돌려줌) 또는 `READY` 를 알린다.
- 영점 등록이 없는 알고리즘은 '센터 기준' 때 화면 윤곽·차이 영상용 사진만 찍고 알고리즘의 고정 영점을 그대로 쓴다.
- 혜원님 모듈은 `feature/hye/ssafy-reference` 16e5d32e 부터 Pi 에서 그대로 빌드된다 (모듈 안내: `PORTING_PI.md`).
  `include/joint_reference.hpp` 가 있으면 영점 등록(ALGR = `set_joint_zero`, 측정이 안 되면 `set_joint_reference` 뒤 다시)을 켠다.
  Pi 확인 (2026-11-13): rig_grid_rev 센터 영점 Y 0.281931, T −0.0442237 (Windows 와 1e-6 안), 한 장 약 19 ms.

## 보정 순서 (Y → Θ)

설정 `sequential_axes` (화면의 '보정 순서'). 켜면 한 번 측정에서 나온 두 축 보정량으로
Y 를 먼저 옮기고 이어서 Θ 를 옮긴다 (다시 재지 않음). 안 움직이는 축은 접근점도 거치지 않는다.
끄면 예전처럼 두 축을 함께 움직인다.

## 장비 조건 (2026-11-13)

- 카메라 게인 기본 3.0, 노출 8333 µs — 데이터셋 촬영과 같다. 알고리즘이 그 밝기로 학습됐다.
- **모든 이동은 −쪽 접근점(Y −0.2 mm, Θ −0.1°)을 거쳐 +방향으로 들어간다.** Θ 는 방향을 바꾸면 약 0.023° 빈 구간(백래시)이 있고,
  같은 방향으로 접근하면 반복 정밀도가 0.002° 다 (`~/data/theta_test_20261113/REPORT.md`). 데이터셋도 이 방향으로 찍었다.
- 보정 이동은 상대 이동이 아니라 "STATUS 위치 + 명령" 을 절대 목표로 보낸다 (접근점을 계산하려고).
- 장치를 잡을 때 Y 속도를 1 mm/s 로 맞춘다 (STM32 리셋 뒤 기본 0.1 mm/s).
- **센터 기준 잡기**: 센터로 가서 찍은 사진을 알고리즘 영점으로 등록한다 (`alignment_set_reference`). 센터는 몇 시간에 0.01~0.02° 씩 밀리므로
  **자동 보정·한 단계·측정만을 시작할 때 기준이 없거나 60분이 지났으면 자동으로 먼저 잡는다** (설정 `auto_reference`, `reference_max_age_min`).
  한 단계·측정만은 지금 자리에서 해야 하므로 센터에서 기준을 잡은 뒤 원래 자리로 (−쪽에서 접근해) 돌아온다.
  기준 없이 쓰면 알고리즘 설정 파일의 영점(학습 데이터 센터)을 쓰는데, 2026-11-13 에는 그 영점과 센터가 Y 0.2 mm, Θ 0.02° 달랐다.
  **STM32 가 리셋돼 원점을 다시 잡은 날은 센터 좌표가 0.4 mm 쯤 바뀔 수 있으니, 센터 좌표를 먼저 다시 찾은 뒤에 기준을 잡는다.**
- 측정 실패(QR·나사를 못 찾음)는 NaN 이 아니라 오류로 돌아오고, 그 회차에서 멈춘다.

## 검증 순서

1. **Y 원점 복귀** → 플레이트를 센터에 맞춘 뒤 **현재 위치를 센터로** (또는 설정에 센터 좌표 입력) → **센터 기준 잡기**.
   STM32 가 리셋되면 원점이 0.4 mm 쯤 달라질 수 있으니, 센터 좌표는 알고리즘으로 다시 찾는다 (센터에서 "측정만" → 그만큼 옮겨 다시 측정).
2. **랜덤 시작 위치** 또는 Y/Θ 를 넣고 **지정 위치로**. 시작 위치는 데이터셋과 같이 `접근 거리`만큼 −쪽에서 들어온다.
3. **자동 보정** (끝까지) / **한 단계** (측정 + 보정 이동 한 번씩) / **측정만** (이동 없이 추론만).
4. 결과: 추론이 허용 안으로 들어오면 완료. 이때 실제 오차가 허용 밖이면 `실제는 허용 밖!` 으로 표시한다.

기록은 `verify_web/instance/runs/<시각>/` 에 남는다: `trials.csv`, `run.json`, 측정 사진 `trial_NN.bmp`(NN = 측정 순번, 01 = 시작 위치, 3840×2160, 알고리즘 입력 그대로).

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
