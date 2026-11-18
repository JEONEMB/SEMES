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

### SSAFY 상쇄선 정렬

기존 단일 사진 추정으로 첫 이동을 정하고, 멈춘 뒤 새로 찍은 영상에서 다음 상대 이동을 계산한다.

```powershell
.\build\Release\qr_pose_cli.exe ssafy-align-step --config config/ssafy.json `
  --alignment-model models/ssafy-alignment.xml.gz image-after-move.bmp
```

반복 제어기는 `qr::run_ssafy_alignment`에 정지 후 촬영 콜백과 상대 이동 콜백을 전달한다. 촬영 콜백에서 `aligner.observe(gray)`를 호출하고, 이동 콜백은 상대 µm·mdeg 명령이 완료될 때까지 기다린 뒤 성공 여부를 반환한다. 이동 가능 범위는 장비 어댑터에서 검사한다. 첫 이동에 사용한 명령 수를 20에서 빼서 정렬기에 남은 횟수를 전달한다. 새 영상의 QR 수평 잔차가 0.30 px 이내이면 QR 세 개의 간격·폭·높이를 반영한 Y·θ 추정으로 센터를 확인하고, 허용 범위 밖이면 두 축을 최대 3회 보정한다. 허용 범위에 든 추정은 `qr_center_candidate`로 종료한다.

동일 SSAFY 카메라와 기준 패치에 맞춘 보정 모델이다. 외부 목표는 학습 좌표계의 Y(mm), θ(°)로 지정한다. 모의 재생은 SSAFY 첫 사진 이후의 관측을 Rig 정·역방향 격자에서 빌려 사용한다. 실기에서는 최종 위치를 독립 계측해 센터 판정을 확인한다.

### SSAFY에 Rig 이동 루프 연결

`qr::run_ssafy_rig_alignment(stage, settings, aligner, visual_target, capture)`는 Rig의
`run_correction`을 호출한다. SSAFY 관측의 수평 감도를 Rig 제어기의 부호와 단위로
환산하며, 두 축 잔차는 SSAFY 영상으로 적합한 `ssafy-alignment.xml.gz`에서 얻는다.
`settings.center`에는 기준 사진 자세의 절대 스테이지 좌표(µm·mdeg)를 넣고,
`visual_target`에는 같은 목표의 SSAFY 캘리브레이션 좌표(mm·°)를 넣는다.
`capture`는 명령 완료 뒤 정지한 새 영상에서 `aligner.observe(gray)`를 반환한다.
두 축의 기준 좌표와 이동 방향은 실기에서 확인한다. 복귀 재생 결과는 첫 Y 절대 이동 뒤 남는 잔차에 민감하다.

## Rig와 SSAFY의 비전 상대 이동

`--vision-joint`는 첫 사진과 매 이동 뒤 새 사진에서 QR 세 개의 변위와 형상을 읽고
Y와 θ를 동시에 추정한다. 영상 잔차의 −0.8배를 두 축 상대 명령으로 보내며, 한 번에
Y 0.45 mm, θ 0.30°까지 이동한다. 두 추정 잔차가 각각 ±0.008 mm와 ±0.002° 안에
들면 센터 후보로 종료한다. 화면에서 수평 변위가 상쇄되는 자세도 같은 추정기를 사용한다.
제어기는 스테이지의 보고 좌표를 사용하지 않는다. 최대 이동 횟수는 20회다.

Rig는 역순 격자로 학습한 `models/rig-reverse-fixed.xml.gz`를 두 촬영에 공통으로
사용한다. 기준 영상은 해당 촬영의 센터 사진이다.

```sh
./build/qr_rig --vision-joint camera_calib.yml /path/to/rig_grid_rev/center.bmp models/rig-reverse-fixed.xml.gz 1 run-reverse.jsonl
./build/qr_rig --vision-joint camera_calib.yml /path/to/rig_grid/center.bmp models/rig-reverse-fixed.xml.gz 1 run-forward.jsonl
```

기준 사진은 각 촬영에서 목표 위치에 맞춰 확보한다. 같은 QR 계수에 기준 사진 대비
관측 차이를 입력한다.

SSAFY는 해당 카메라의 정류 설정, 목표 사진 `center1.bmp`,
`models/ssafy-vision-joint.xml.gz`를 지정한다. 영상과 QR 배치는 모델을
피팅한 해상도와 화각을 따른다. `qr_pose_cli vision-joint-step --reference /path/to/rig_grid/center.bmp
--model models/rig-reverse-fixed.xml.gz image.bmp`는 장비 연결 없이 한 프레임의
추정과 이동 명령을 출력한다.

기록 영상과 내삽 관측을 사용한 재생에서 역순 1,680개와 정순 1,678개가 각각
Y ±0.045 mm, θ ±0.010° 범위에서 센터 판정으로 정지했다. 장비 적용 전 독립
계측으로 센터 후보의 실제 위치와 반복 정밀도를 확인해야 한다.

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
