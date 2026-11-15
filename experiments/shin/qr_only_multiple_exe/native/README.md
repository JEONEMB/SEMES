# QR 자세 추정·보정 C++14 포팅

납품용과 실증용 모두 `qr_pose` 정적 라이브러리를 사용한다. 런타임과 실행 예시는 OpenCV만 사용한다. Windows x64에서 MSVC v141·OpenCV 3.4.15로 빌드했다. 학습 결과는 `models/*.xml.gz`에 있다.

## Windows에서 SSAFY 사진 처리

이 폴더에서 실행한다. 설치 위치가 다르면 CMake와 PATH의 경로를 바꾼다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 `
  -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" `
  -DOpenCV_DIR="C:/opencv/build"
cmake --build build --config Release
$env:PATH = "C:\opencv\build\x64\vc15\bin;$env:PATH"
.\build\Release\qr_pose_cli.exe ssafy --config config/ssafy.json `
  --direct models/ssafy-direct.xml.gz image.bmp
```

입력은 동일 촬영 환경의 왜곡 보정된 3840×2160 BMP다. `config/ssafy.json`의 `y_mm`, `theta_deg`를 변경하면 해당 목표점에 대한 잔차를 출력한다. 출력은 입력 파일 순서의 JSONL이다. `method`는 `feature_fit` 또는 `joint_pixel_refined`다. `inference_ms`는 파일 읽기·디코딩을 제외한다. `--direct`를 생략하면 기존 QR 특징 모델을 사용한다. 기준 영상 패치는 파일에 저장되어 있다.

라이브러리에는 `#include "qr_pose.hpp"`와 `target_link_libraries(<target> PRIVATE qr_pose)`로 연결한다. `qr::FeaturePose(model).predict(gray)`는 캘리브레이션 원점 기준이다. 목표 보정이 필요하면 결과에서 `qr::Pose(y_mm, theta_deg)`를 빼고, 이동 코드에서 반대 부호를 사용한다. `qr::DirectPose`에서 `initial()`, `ssafy_supported()`, `refine()`를 이어 호출해 선택 적용할 수 있다. 단일 스레드 추론은 실행 초기의 `cv::setNumThreads(1)`로 설정한다.

## Rig 사진 점검과 C++ 제어 재생

```powershell
.\build\Release\qr_pose_cli.exe rig-frame --reference center.bmp `
  --model models/rig-cancellation.xml.gz --direct models/rig-direct.xml.gz image.bmp
.\build\Release\qr_pose_replay.exe validation/rig-surface.csv models/rig-cancellation.xml.gz
```

`rig-frame`은 이동 없이 기준 사진 대비 QR 수평 변위와 Y·θ 추정값을 출력한다. `qr_pose_replay`는 두 촬영에서 생성한 QR 격자 자료를 C++ 보정 루프로 재생한다. 첫 명령 뒤 숨긴 Y 오프셋을 9개 조건으로 고정했고, 명령대로 이동한다고 가정했다. 실기 복귀율과 제어기 보고 좌표는 독립 계측으로 구분한다.

## Pi에서 모터와 카메라 연결

원본 S15P31A507 체크아웃 경로를 CMake의 `QR_STAGE_REPO`에 지정한다. 원본의 `hardware/stage`와 `hardware/camera`를 사용한다. 기존 카메라 모듈의 libcamera 빌드에는 C++20을 사용할 수 있는 컴파일러가 필요하다. QR 추정기와 보정 루프는 C++14다. 준비된 보정 파일, 같은 화각의 기준 영상, 기준 영상의 절대 스테이지 좌표, 자체 장치 권한이 필요하다.

```sh
cmake -S . -B build -DQR_STAGE_REPO=/path/to/S15P31A507 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/qr_rig camera_calib.yml center.bmp models/rig-cancellation.xml.gz \
  0 0 1 run-001.jsonl models/rig-direct.xml.gz
```

마지막 공동 정합 모델은 진단용 선택 인수다. 앞의 `0 0 1`은 절대 목표 Y(mm), θ(°), 줌을 뜻한다. 기존 스테이지가 설정한 소프트 리밋을 사용하고, 웹과 같은 스테이지 잠금 파일을 획득한다. 첫 Y 절대 명령 이후 QR 수평 잔차를 줄이고, 상쇄선에서는 QR 형상 기반 Y·θ로 함께 보정한다. 최종 센터 후보의 종료 코드는 2다. Pi 대상 장치의 빌드 및 실제 모터 구동은 인수 시험에서 확인해야 한다.

원본 스테이지를 직접 연결하려면 `qr::Stage`의 `position()`, `limits()`, `move_absolute()`에 장치 API를 매핑한다. 이동 단위는 µm와 mdeg이며, `move_absolute()`는 DONE 이후 보고값을 반환한다. 카메라 콜백은 움직임이 가라앉은 뒤 정류된 새 흑백 프레임을 공급한다.

## 확인한 결과

기본 추정은 SSAFY 575장에서 이전 구현과 일치했다. 공동 정합을 적용할 수 있었던 자세 보류 사진 123장에서는 두 축 동시 통과가 62장에서 98장으로 늘었다. 이는 장비 기록과 비교한 프레임 단위 결과다. C++ Rig 재생에서는 같은 시작점 3,360개를 Y 잔차 9개 조건에서 평가했으며, 두 축을 최대 세 번 보정한 모드에서 30,240건이 모두 재생 허용 범위에 들었다. 실제 장비의 반복 구동은 인수 시험 항목이다.

[프로젝트 전체 보고서와 포팅 절차](../../../../docs/shin/qr_only_multiple_exe/qr-pose-correction-and-porting.pdf)
