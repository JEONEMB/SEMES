# 실제 장치 (hardware/device)

서비스 흐름은 [include/hardware.hpp](../../include/hardware.hpp)의 `ICamera`·`IRobot`만 호출한다. 이 폴더는 그 계약을 실제 장치로 구현하는 곳이다.

## 현재 상태 (2026-10-28)

| 장치 | 파일 | 상태 |
|---|---|---|
| 카메라 | [device_camera.hpp](include/device_camera.hpp), [device_camera.cpp](src/device_camera.cpp) | **미연결.** `open()`·`capture()`는 `HardwareError(camera, "Camera not connected")`를 던진다 |
| 로봇 | 없음 | 미작성. 이동 단위·부호(A2), 한 번 최대 이동량(A4) 확정 후 |

`semes_device` 라이브러리는 아직 루트 CMakeLists에 연결하지 않았다.

## 카메라를 연결할 때 할 일

`DeviceCamera`의 `TODO(hardware)` 네 곳을 채운다.

1. `open()`: 장치 열기, 3840×2160 흑백 8비트로 설정, 성공하면 `connected_ = true`
2. `capture()`: 한 장 촬영 → 8비트 흑백 `cv::Mat`으로 변환 → 크기 확인 → `Frame` 반환
3. `close()`: 장치 해제
4. 실패는 `HardwareError(HardwareFault::camera, 이유)`로 던진다. 빈 영상이나 가짜 사진을 반환하지 않는다

`Frame` 계약

| 필드 | 값 |
|---|---|
| `image` | 3840×2160, `CV_8UC1` (joint·control 입력과 같음) |
| `source` | 장치 이름 |
| `simulated` | `false` |
| `note` | 촬영 시각 등 |

연결 방식 후보

- **MIL10** (명세서의 필수 환경 "OpenCV 3.4.15/MIL10"): `MappAlloc` → `MsysAlloc` → `MdigAlloc` → `MdigGrab` → 버퍼를 `cv::Mat`으로 복사
- **OpenCV `cv::VideoCapture`**: USB·일반 카메라일 때. 4K 흑백 설정이 되는지 확인 필요

## 하드웨어 팀에 확인할 것

- 카메라 종류와 SDK, 해상도·비트 깊이가 기존 데이터(3840×2160)와 같은지
- 촬영 시점: 로봇 이동이 끝나고 몇 ms 뒤에 찍어야 흔들림이 없는지
- 조명·노출 고정 여부 (기준 사진 `center.bmp`와 밝기가 다르면 정합이 흔들릴 수 있음)
