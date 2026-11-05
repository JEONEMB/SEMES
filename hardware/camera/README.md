# camera — 촬영 모듈 (stage_camera)

보정 앱이 부르는 촬영 함수. Raspberry Pi HQ 카메라(IMX477)로 찍어 **알고리즘 입력 규격(3840×2160, 8bit 흑백, SEMES와 같은 화각)** 사진을 만든다.

```
카메라 4056×3040 (전체 센서, 노출·게인·색 게인 고정)
  → [안정 대기] 이동 직후 흔들림이 멎을 때까지 프레임 비교
  → [촬영] 함수를 부른 뒤 노출된 새 프레임 1장 또는 N장 평균
  → [보정] 왜곡 펴기 + SEMES 화각 크롭 + 3840×2160 (변환표 하나로 remap)
  → cv::Mat 3840×2160 CV_8UC1
```

| 파일 | 하는 일 |
|---|---|
| `include/stage_camera/camera.hpp` | 공개 API `Camera` — `capture()`, `captureSettled()`, `waitSettled()` |
| `include/stage_camera/rectifier.hpp` | `Rectifier` — 원본 → SEMES 규격 변환. 카메라·libcamera 없이 사용 가능 |
| `src/libcamera_source.*` | libcamera C++ 로 연속 촬영, 가장 최근 프레임 보관 (내부용) |
| `tools/cam_capture.cpp` | 시험용 CLI |

## 빌드 (Pi)

OpenCV 3.4.15 (`/usr/local`), `libcamera-dev` 필요. libcamera 헤더 때문에 C++20.

```bash
cd hardware/camera
cmake -S . -B build -G Ninja
cmake --build build
```

`stage_rectifier`(변환만)와 `stage_camera`(촬영 + 변환) 두 정적 라이브러리가 만들어진다.

## 사용 (앱)

```cpp
#include "stage_camera/camera.hpp"

stage_camera::CameraOptions opt;
opt.calib_path = "../camera_calib/camera_calib.yml";   // hardware/camera_calib 의 결과
stage_camera::Camera cam(opt);         // 실패 시 std::runtime_error

auto ref = cam.captureSettled(4);      // 기준(center) 사진: 안정 후 4장 평균
alignment_set_reference(ref.image);

// 이동 명령 → DONE 대기 후
auto s = cam.captureSettled();
if (!s.settle.settled) { /* 시간 초과. 재시도할지는 앱이 결정 */ }
Alignment_t r = alignment(s.image);
```

- 사진은 **함수를 부른 뒤 노출이 시작된 프레임**만 쓴다. 마지막으로 받은 프레임이 S면 S+2부터 쓴다 (S+1은 이미 노출 중이었을 수 있음).
- `Shot::raw`는 센서 원본(4056×3040)이다. 기록이나 재캘리브레이션용으로 저장할 수 있다.
- HQ 카메라는 소프트웨어로 초점을 바꿀 수 없다. 초점·조리개는 렌즈 링으로 고정한다 (바꾸면 캘리브레이션을 다시 해야 함).

### 고정하는 카메라 설정 (`CameraOptions`)

| 항목 | 기본값 | 이유 |
|---|---|---|
| 노출 | 8333 µs | 60Hz 조명 깜빡임 → 8333 µs의 배수만 사용 |
| 아날로그 게인 | 2.0 | 캘리브레이션 사진 때 자동 노출 값 (약 1.98) |
| 색 게인 (R, B) | 3.3, 1.4 | 흑백 밝기도 색 게인 영향을 받으므로 고정. 캘리브레이션 사진 때 자동 값 |
| 프레임 속도 | 10 fps | 전체 센서 모드 최대 |
| 센서 모드 | 4056×3040, 12bit | 캘리브레이션과 같은 화각. 다른 모드는 화각·중심이 달라짐 |

### 안정 대기 (`SettleOptions`)

1. 함수를 부른 뒤의 첫 프레임을 기준으로 잡는다.
2. 다음 프레임마다 기준과의 이동량을 위상 상관(`phaseCorrelate`, 1/4 축소)으로 잰다.
3. 이동량이 `max_shift_px`(0.5 원본 px) 이하이면 안정 횟수를 센다. 넘으면 그 프레임을 새 기준으로 삼고 처음부터 센다.
4. `stable_frames`(3)장 연속으로 안정하고 `min_wait_ms`(200 ms)가 지났으면 통과한다.
5. `timeout_ms`(5000 ms) 안에 통과하지 못하면 `settled = false`로 마지막 프레임을 반환한다.

연속 프레임끼리 비교하지 않고 기준 프레임과 비교하므로, 천천히 흘러가는 움직임도 걸러진다.
진동 주파수가 정확히 10Hz의 배수이면 매 프레임이 같은 위상에서 찍혀 놓칠 수 있다. 그래서 최소 대기와 연속 횟수를 함께 쓴다.
정지 상태에서도 측정값이 0.1~0.5 px 흔들린다 (흰 종이가 대부분이라 무늬가 적음). 0.3 px로는 10프레임 연속 조건에서 8번 중 3번 시간 초과가 나서 0.5 px로 정했다.
스테이지를 움직인 직후 이동량이 줄어드는 모습을 기록해서 임계값과 최소 대기를 다시 정한다.

## 시험 도구

```bash
# 카메라 없이: 저장된 원본 사진 변환
./build/cam_capture --calib ../camera_calib/camera_calib.yml --rectify calib_001.png -o out.png

# 한 장
./build/cam_capture --calib ../camera_calib/camera_calib.yml -o shot.png --raw raw.png

# 안정 대기 후 1초 간격 5장 (shot_001.png ...)
./build/cam_capture --calib ../camera_calib/camera_calib.yml -o shot.png --settle --count 5 --interval 1000

# 기준 사진
./build/cam_capture --calib ../camera_calib/camera_calib.yml -o center.bmp --settle --average 4
```

옵션: `--exposure US`, `--gain G`, `--zoom Z`, `--settle-shift PX`, `--settle-frames N`, `--settle-timeout MS`

## 확인한 것 (2026-11-05)

- 저장된 캘리브레이션 사진 `calib_001.png` 변환: 변환표 생성 105 ms, 변환 **18.8 ms**. 원본 밖을 가리키는 픽셀 0%.
- `calibrate.cpp --undistort` 결과(`*_semes3840.png`)와 비교: 평균 밝기 차 0.82/255, 전체 이동 약 (−0.5, −0.9) px로 일정.
  네 구석과 중앙의 이동이 같아 스케일·왜곡 처리는 일치한다. 남은 차이는 기존 도구가 크롭 시작점을 정수로 반올림해서 생긴다.
  이 모듈은 렌즈 중심을 정확히 사진 가운데(1919.5, 1079.5)에 둔다.
- 실촬영 (스테이지 정지 상태, QR 타깃 앞):
  - 카메라 열기 0.7~1.2 s. 수동 노출이 적용됨 (메타데이터: 노출 8317 µs = 센서 줄 단위로 반올림, 게인 2.00)
  - 약 9.6 fps (5초에 48~49프레임)
  - 안정 대기 (0.5 px, 3프레임): 8/8 통과, 장당 약 0.5 s. 엄격하게 10프레임으로 해도 5/5 통과, 1.2~2.1 s
  - 기준 사진 (안정 대기 + 4장 평균): 0.84 s
- libcamera 로그는 `LIBCAMERA_LOG_LEVELS`가 정해져 있지 않으면 오류만 보이게 한다.

## 남은 일

- 스테이지 이동 직후의 흔들림 기록 → `max_shift_px`, `min_wait_ms` 다시 결정 (③ 포팅 앱 측정 모드에서)
- 조명: 사진 위쪽에 형광등 반사가 하얗게 날아감. 장비 조명과 가림막이 정해지면 노출·게인·색 게인을 다시 정한다
- 장비 장착 후 px/mm 확인 → 필요하면 `zoom` 조정 (hardware/camera_calib/GUIDE.md 4-2)
- 캘리브레이션 RMS 1.68 px (잠정값). 더 평평한 보드로 다시 하면 `camera_calib.yml`만 바꾸면 됨
