# QRwithBar Lanczos-3 추정 모듈

QR 내부630점과 허용된 외부 상·하 수평 경계에 Lanczos-3 패치 정합을 적용한다. 세 QR의 평균 x/y 이동6개,두 선의 위치·기울기4개를 합친10특징을 사용한다. C++17/OpenCV3.4.15/MSVC v141로 빌드하며 추론·검증·모델 내보내기에 Python이 필요하지 않다.

## 호출과 입출력

```cpp
#include "qrwithbar_lanczos.hpp"
qrwithbar::Estimator model("qrwithbar_assets"); // 한 번 초기화하고 재사용
auto estimate = model.estimate(image);
// estimate.y_mm, estimate.yaw_deg: 현재 자세의 추정값, 이동 명령이 아님
```

루트 API `Alignment_t alignment(cv::Mat&)`도 이 모듈로 실제 Y/T를 계산한다. 명시적 초기화는 `initialize_lanczos_alignment(assetsDirectory)`이며, 생략하면 실행파일 옆의 `qrwithbar_assets` 또는 저장소 상대 assets를 찾는다. 기존 서명·필드명은 유지한다.

입력은3840×2160,uint8,1/3/4채널이다. 색 영상은 회색조로 변환하므로 이전 정밀 검증의 바이트 조건을 재현하려면 `IMREAD_GRAYSCALE`로 디코딩한다. 원본 Mat는 수정하지 않는다. 실패는 예외로 전달하며0,0으로 숨기지 않는다. 출력 Y는mm,T는 기존 검증의yaw각도 라벨 관례다. 실제 장치 부호·명령 변환·이동·재촬영은 호출 측 책임이다.

직접 모듈을 사용하면 측정10특징,QR점 수,선 유효 표본 수,학습 범위 밖 추정 여부도 받을 수 있다. 유효성이나범위 진단은 자세 정확도 보증이 아니다. 현재 보정 범위 바깥의 예측을 임의로0이나경계값으로 자르지 않는다.

## 추론 흐름과 파라미터

1. 기준 QR템플릿 NCC로 초기 위치를 찾고 affine ECC를 수행한다. ECC 가우시안커널은1이며,원본 영상에 GaussianBlur/Sobel/Canny를 적용하는 경계 추출은 없다.
2. 기존 원본 밝기 모델로 QR·선의 초기 서브픽셀 위치를 구한다. 이것은 공통 초기화이며최종 채택 측정값은다음 단계의정합 결과다.
3. 기준9×9 원본 QR패치와 현재 패치를 Lanczos-3로 정합한다. affine 초기 형태를 유지하며x/y 이동,gain,offset을 제곱오차로 추정한다.
4. 상·하 선의15×7 패치를 수직 이동,gain,offset과soft-L1 척도3으로 정합한다. 수평 텍스처 이동 변수를 추가하지 않는다. 동일 허용 영역·표본 수·강건 직선 적합을 유지한다.
5.10특징을 이 방식의 학습 분할에서 계산한 평균·표준편차와회귀계수로 변환해 Y/T를 추정한다.

커널은 `sinc(x)*sinc(x/3)`,반경3,6×6 지지 영역이며합을정규화한다. 밝기overshoot를clipping하지 않는다. 해석 미분을 사용한다. QR이동 범위±3.6px,선±4px,gain0.2~5,offset±255,기존 감쇠Gauss–Newton과종료 기준을 유지했다. 수동 설정값의 최적성을 이번에 새로 증명한 것은 아니다.

기준 영상/QR패턴/카메라 설치가 달라지면 현재 보정값을 그대로 적용해도 정확하다고 보장하지 않는다. 임의의 새 QR내용을 자동 보정하는 범용 모델이 아니다.

## 모델과 평가의 분리

`assets/model.json`은 Lanczos 측정값으로 전체57장을 학습한 **배포 모델**이다. `method=lanczos3`와10특징 행렬 크기를 검사하여 과거 밝기 모델의 계수를 실수로 섞지 않도록 했다. 정규화 강도는 좌표 단위 내부 검증에서 선택한다.

`tests/fixtures/lanczos_validation_models.json`의19개 모델은 좌표 보류 재현용이며 **일반 추론에서 사용하지 않는다.** 영상 파일명·정답·fold는추론 API의 인자가 아니다. 검증 도구만 별도 메타데이터로 시험 모델을 선택한다. 배포 모델로57장 자체를 다시 예측한 값은 재대입이며 OOF수치와 구분한다.

승인된 실험의 개발용19좌표 보류 기준은 Y평균/최대절대오차0.044733/0.156571mm,T0.008000/0.027060,동시±1틱34/57이다. 이미 개발에 노출된57장 성적이며새 장비·세션·QR·실제 반복 복귀를 검증한 결과가 아니다. 비교 수치와 최종 실행 코드의 대응은 `results/validation.json`에 남긴다.

## 빌드와 실행

저장소 루트 PowerShell:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
cmake --build build --config Release --parallel 4
.\build\Release\main.exe "IMAGE.bmp"
ctest --test-dir build -C Release --output-on-failure
```

빌드가 실행파일 옆에 `qrwithbar_assets`를 복사한다. 저장소 밖으로 전달할 때는 main.exe,OpenCV DLL,이 assets폴더를 함께 전달한다. 소스의 개발자 절대 경로는 추론에 필요하지 않다. `--assets DIRECTORY`로 명시적인 경로도 사용할 수 있다.

기존 판 검출/디코딩 시각화는 `--detect-only`에서 보존한다. `--gui`·`--output`·`--refinement corner|model|fourier`는 예전 디버그 기능이다. 그 디버그 화면의 Fourier/CornerSubPix 점이나 판 검출용 엣지를 Lanczos 자세 추정의 실제 측정값으로 해석하지 않는다. 기본 실행은 GUI를 열거나디버그 판 검출기를 통과해야만 자세를 반환하는 구조가 아니다.

## 원본 검증 재현

```powershell
.\build\modules\qrwithbar_lanczos\Release\qrwithbar_validate.exe modules/qrwithbar_lanczos "IMAGE_ROOT" modules/qrwithbar_lanczos/results
```

IMAGE_ROOT는manifest의상대경로57장이 있는폴더다. 원본해시·입력불변·10특징·19좌표보류출력의대응을 검사하고,일반배포모델 출력은별도열로저장한다. 영상데이터는검증입력으로별도제공한다.

같은보정계수를다시만들려면(개발용):

```powershell
.\build\modules\qrwithbar_lanczos\Release\qrwithbar_train.exe modules/qrwithbar_lanczos "IMAGE_ROOT"
```

이도구는포함된Lanczos측정캐시57장과라벨로모델을계산한다. 새로운영상을학습시키는범용훈련기가아니다. 기준BMP의해시를검사하고무손실회색조PNG를생성한다. `assets/`는필수전달자료이며원본57장·TXT·검증전용계수는일반추론에필요하지않다.

과거 `tool/qr_common_c/`는C17 밝기모델실험기록으로보존했으며현재루트추론에링크하지않는다.
