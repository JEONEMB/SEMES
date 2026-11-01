# QR + 공통 수평 경계로 Y·yaw 추정

요청한 **QR 이동 6개 + 공통 배경 경계 4개** 구성을 구현한다. QR 교점과 경계의 원본 밝기 적합, 강건 직선 적합, 최종 회귀는 C17이다. 이미지 읽기·템플릿 검색·초기 affine ECC 정합만 OpenCV C++ 어댑터를 쓴다. Canny나 Sobel은 사용하지 않는다. 실행 시 Python을 호출하지 않는다.

## 결과

기존 두 번째 데이터의 혼합 이미지 **57장, 19개 좌표**를 평가했다. 같은 좌표의 반복 3장을 함께 보류했다. 정답은 장비 Actual 기록에서 원점 Actual 평균을 뺀 값이다. 원본 이미지와 라벨은 수정하지 않았다.

| 검증 경로 | Y RMSE (mm) | yaw RMSE (°) | 동시 ±1틱 |
|---|---:|---:|---:|
| 기존 Python | 0.075169 | 0.014019 | 52.6% |
| 동일 10특징 → C 회귀 | 0.075169 | 0.014019 | 52.6% |
| 원본 BMP → C 특징 추출 → C 회귀 | 0.074458 | 0.013847 | 52.6% |

이미지 경로의 작은 차이는 최적화기와 OpenCV 버전 등이 다른 구현의 수치 차이다. 정확도가 향상됐다고 해석하지 않는다. 동일 특징을 사용한 C 회귀의 최대 Python 차이는 Y `3.3e-14 mm`, yaw `7.2e-15°` 미만이다. 이미지 경로의 Python 대비 출력 차이 RMS는 Y 0.00706mm, yaw 0.00134°다.

최종 산출물: [Y–THETA 좌표 대응 보고서](results/report.html), [PNG](results/y_theta_pairs.png), [SVG](results/y_theta_pairs.svg), [57장별 예측 CSV](results/image_predictions.csv), [검증 기록](results/validation.json).

파란 원은 입력 이미지의 정답 좌표, 주황 ×는 예측 좌표이며 같은 이미지의 두 점을 선으로 연결했다. 보고서와 그래프는 **좌표 보류 예측**이다. 배포 모델로 학습 사진 자체를 예측한 점수와 구분한다.

## 빌드 (Windows x64, 검증 환경)

Visual Studio Build Tools 2026 / MSVC 19.51, CMake, OpenCV 3.4.15의 vc15 바이너리를 사용했다. 원본 분석은 같은 S15P31A507 저장소의 `docs/offaxis-eda/`에 있다. `tool/qr_common_c/`는 독립 CMake 프로젝트이며 아래 명령은 이 디렉터리에서 실행한다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DOpenCV_DIR=C:/opencv/build -DOpenCV_RUNTIME=vc15
cmake --build build --config Release
.\build\Release\qr_common_predict.exe "C:/data/example.bmp" .\calibration
```

출력 JSON의 `Y_mm`, `yaw_deg`는 기준 원점에 대한 **현재 자세 추정값**이며 이동 명령이 아니다. 단위와 모터 방향을 검증한 뒤 제어 계층에서 보정 명령을 결정해야 한다. 프로그램은 장비를 제어하지 않는다.

OpenCV 3.4 Windows 배포본은 최신 MSVC를 자동 인식하지 못해 `OpenCV_RUNTIME=vc15`를 명시한다. C 런타임·OpenCV DLL을 실행 환경에서 찾을 수 있어야 한다. 현재 CMake는 제공된 OpenCV DLL을 실행 파일 옆에 복사한다. 다른 OS·컴파일러는 이번에 빌드 검증하지 않았다.

## C API

```c
#include "qr_common.h"

qc_result result;
int status = qc_predict_image("frame.bmp", "calibration", -1, &result);
if (status == QC_OK) {
    double y_mm = result.pose.y_mm;
    double yaw_deg = result.pose.yaw_deg;
    /* 제어 계층에서 단위/방향을 적용한다. */
}
```

헤더는 C에서 호출 가능하다. 이미지 API는 `qr_common_image`와 `qr_common_core`를 링크한다. 이미 특징 10개가 있으면 `qc_predict_features(features, -1, &pose)`로 `qr_common_core`만 사용한다. 참 좌표나 파일명에서 라벨을 파싱하는 코드, 프레임별 특징 캐시는 추론 경로에 없다. 원본 영상은 3840×2160 8비트 그레이로 디코딩한다. 다른 해상도는 오류로 반환한다.

특징 순서:

1. `L_dx, L_dy, C_dx, C_dy, R_dx, R_dy`: QR별 L 교점 평균 이동(px), 기준 center1 대비.
2. `upper_position, upper_slope, lower_position, lower_slope`: 기준 원점 대비 배경 상·하 경계 위치와 기울기 계수(px). 기울기의 x는 `(x-W/2)/(W/2)`로 정규화한다. 기울기는 각도 단위가 아니다.

경계 후보는 QR 크기·위치에 상대적인 넓은 띠에서 원본 밝기 PSF 계단 모델을 직접 비교하여 고른다. 좌우 선 일관성으로 초기화하고 15×7 픽셀에 연속 파라미터 적합을 수행한다. 이후 기준 경계와 QR 이동을 약한 추적 선호도로 사용한다. 임의의 고정 y 행으로 잘라 찾지 않는다. 다만 탐색 띠 배수, 대비 방향, 수평성에 대한 구조 가정은 있다.

`calibration/`에는 QR L 교점 630개의 패턴·기하, 세 QR 템플릿, 기준 원점의 점과 경계, 10개 특징의 학습 계수가 포함된다. **현재 촬영 세팅용 캘리브레이션**이다. 다른 장비·QR 패턴·카메라 위치에서도 그대로 정확하다고 보장하지 않는다. 기준 자료는 이미지에서 나온 교정 상수이며 평가 이미지별 정답/측정값 표가 아니다.

`fold=-1`은 57장 전체로 학습한 일반 추론 모델이다. `fold=0..18`은 재현 평가에만 쓴다. 평가기는 정답 좌표 묶음을 알고 보류 모델을 선택한다. 실사용에서 이미지로 fold를 고르거나 평가 모델을 자동 선택하면 안 된다.

## 검증 재현

```powershell
python tools/validate.py --analysis-root C:/Users/SSAFY/Documents/GitHub/S15P31A507
```

이 스크립트는 C DLL을 직접 호출한다. 저장된 57개 특징에 대한 회귀 일치, 잘못된 fold·NaN·NULL 거부, 원본 BMP 57장 전체 추출·예측 성공, 630개 QR 점 생성 여부를 확인한다. 결과 JSON/CSV와 그래프를 다시 생성한다. Python 분석 의존성이 S15P31A507의 `.analysis_packages`에 설치된 환경을 사용한다. 그 경로는 빌드나 배포 추론에는 필요하지 않다.

캘리브레이션을 기존 분석과 동일하게 다시 내보내려면:

```powershell
python tools/export_calibration.py --analysis-root C:/Users/SSAFY/Documents/GitHub/S15P31A507
```

원본 이미지 해시와 분석 파일 해시는 `calibration/provenance.json`, 전체 원본 추출 검증 해시는 `results/source_image_hashes.json`에 기록한다. 학습 계수와 평가용 fold 계수는 서로 구분해 저장한다.

## 현재 범위

한 틱은 Y 0.05mm, yaw 0.01°다. 57장 중 30장이 두 축 모두 한 틱 이내였다. 큰 잔차는 남아 있다. 같은 촬영 환경 안의 좌표 보류 평가이며, 새 장비/촬영 세션에 대한 독립 검증·원점 반복 수렴·축 전용 전환·실시간 처리 성능은 확인하지 않았다. 세션과 좌표의 상관관계도 남아 있을 수 있다. 이 모듈은 한 장의 자세 추정까지 구현한 것이다.
