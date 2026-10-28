# QR 교점 subpixel 정밀화 비교 및 로컬 통합

2026-10-19. 기준 브랜치 `refactor/shin/QR-Detection`, 기준 커밋 `28f6759`.
비교 실험은 커밋/push 보류 상태에서 수행했다. 이후 사용자의 push 요청으로 [S15P31A507-25](https://ssafy.atlassian.net/browse/S15P31A507-25) (활성 스프린트 59016, Story Points 5)에 연결했다.

## 판단

현재 영상과 고정된 순차 추정 평가에서는 평균 밝기를 제거한 Fourier 변형이 Y/theta RMSE 기준으로 우세하여 기본값으로 연결했다. 영상 형성 모델과 기존 cornerSubPix도 선택 가능하게 남겼다. 이것은 명목 구동값에 대한 상대 비교이며 실측 정확도 인증이 아니다. 최종 Y 목표 ±0.09 mm는 아직 달성하지 못했다.

중요: 최초 공학 검증에서 보류 구간 결과를 확인한 뒤, 제약 경계에서의 최적화 수렴 판정 오류를 수정했다. 설정 선택은 계속 내부 검증 구간만 사용했지만, 아래 보류 결과는 같은 구간을 재사용한 것이므로 완전히 새로운 블라인드 시험으로 취급하지 않는다. 추가 촬영이 불가능한 현재 조건에서 이 한계를 유지한다.

## 논문과 재현 범위

- Krüger & Wöhler (2011), Accurate chequerboard corner localisation for camera calibration: https://doi.org/10.1016/j.patrec.2011.04.002
  - 초록/공개 본문 일부의 7변수 영상 형성 모델을 바탕으로 구현했다. 전문의 근사식은 확보하지 못했다.
  - 위치 2, 경계 방향 2, Gaussian PSF 폭 1, gain/offset 2를 적합한다.
  - 비직교 경계에서도 Gaussian convolution을 처리하도록 이변량 정규분포의 상관 적분을 12점 Gauss–Legendre로 계산했다. 단순 erf 곱을 일반 교차각에서 정확한 convolution이라고 취급하지 않는다.
  - 제약을 포함한 감쇠 Gauss–Newton/LM 변형이다. 논문의 상세 근사식까지 검증한 완전 재현으로 부르지 않는다.
- Spitschan & Ostermann, Robust Fourier-based Checkerboard Corner Detection for Camera Calibration: https://www.tnt.uni-hannover.de/papers/data/1335/main.pdf
  - 전문 식 (23)의 Cartesian 격자 합을 사용한다. 영상을 극좌표로 재보간하지 않는다.
  - 홀수 성분 n=1,3,5의 실수/허수 잔차를 최소화한다. 논문의 무한 합을 유한 절단한 구현이다.
  - 원문 형태의 밝기 원본 변형과 고정 패치 평균을 뺀 변형을 모두 비교했다. 최종 선택은 후자이므로 이를 논문 그대로의 구현과 구분한다.
  - 원문 합성 실험의 81x81 창 및 ±0.5 px 범위를 그대로 쓰지 않았다. 우리 QR의 모듈 간격은 약 8 px이므로 작은 창과 기존 후보 이동 허용 범위를 사용했다.

## 공통 조건과 파라미터

플레이트/ROI/디코딩/이상적 모듈 재생성/체커보드 후보 생성/ID는 유지했다. 모든 방식은 같은 원본 grayscale, 같은 투영 초기값에서 출발한다. cornerSubPix 정밀화 좌표를 새 알고리즘의 초기값으로 사용하지 않았다. (payload, feature ID)로 대응하며 누락점을 0으로 채우지 않는다.

비교 설정은 영상 형성 모델 반경 3/4/5, Fourier alpha 1.2/1.8/2.4와 평균 제거 유무, 기존 cornerSubPix 및 반경 3/4/5 변형이다. Fourier 패치 반경은 5 (11x11)로 고정했다. 실제 영상 설정 선택은 내부 검증 성능만 사용했다.

영상 형성 모델: 방향은 투영 모델의 두 경계 법선으로 초기화하며 각 방향 ±0.25 rad, PSF 폭 0.18~2.5 px, gain -1~1, offset -0.1~1.1 범위를 사용했다. 두 방법 모두 위치 이동 범위는 각 좌표 ±0.45 모듈 간격이다. 위치가 경계에 걸리거나 미수렴하거나 열 정규화 Jacobian 조건수가 1e5를 넘으면 탈락시켰다. 흐림/방향의 제약 경계는 KKT 활성 집합으로 처리하고 별도 경계 마스크를 기록했다. nuisance 변수의 경계 도달만으로 위치를 무조건 버리지는 않았다.

최대 반복은 60이다. 반복을 늘려 개선을 주장하지 않았으며 해석적 Jacobian과 제약 최적화 동작을 직접 검사했다. CSV의 refinement residual은 서로 다른 목적함수의 잔차이므로 두 방식 간 수치 크기를 직접 비교하지 않는다.

## 데이터와 분할

원본 BMP 281장: 기준 1, Y 80, theta 200. 원본은 수정하지 않았다. 파일명 값은 독립 측정 정답이 아니다. 여섯 그룹은 촬영 세션 후보이며 확정 촬영 로그가 아니다.

각 그룹을 명목값 순서로 정렬하여 중간 1/3을 최종 보류로 정하고, 나머지의 매 세 번째를 내부 검증으로 정했다. 내부 학습 120, 내부 검증 66, 최종 보류 94장이다. 기준 영상은 공유한다. 1차 theta 순차 추정기를 고정하고, θ는 관측 평균을 뺀 상대 변형에서 추정한 뒤 전체 theta 이동을 복원해 Y를 구했다. 그룹별 영점 offset은 학습 구간에서만 적합했으며 평가 시 그룹을 알고 있다고 가정한다. 실제 혼합 이동이나 미지 세션의 일반화 성능이 아니다.

주 비교에서는 기준 영상의 기존 채택 ID와 프레임별 기존 채택 마스크를 공통 적용한 뒤, 각 정밀화의 수렴/조건수 마스크를 적용한다. 점별 모델은 내부 학습의 70% 이상 및 최소 48회 관측, 설계 행렬 full rank를 요구한다. 최소 24점과 세 QR가 모두 있어야 추정 성공이다. 이로 인한 점 수 차이를 아래에 함께 표시한다. 최종 통합 경로에서는 새 좌표로 국소/기하 검증을 다시 수행했다.

설정은 기존 baseline, 조정 baseline r5, 영상 형성 r3, Fourier alpha2.4 + 평균 제거가 선택됐다. 설정 선택 점수는 (Y RMSE/0.09)^2 + (theta RMSE/0.09)^2와 추정 실패 페널티이다. 내부 검증에서 조정 baseline과 Fourier의 점수 차이는 작았으므로 압도적 우위라고 해석하지 않는다.

## 보류 94장: 공통 기존 채택 마스크

| 방법 | Y RMSE (mm) | theta RMSE (deg) | 두 축 ±0.09 동시 충족 | 사용 점 중앙값 |
|---|---:|---:|---:|---:|
| cornerSubPix | 1.025537 | 0.183038 | 5/94 | 102 |
| cornerSubPix_r5 | 0.345074 | 0.061506 | 20/94 | 102 |
| model_r3 | 0.341769 | 0.060866 | 22/94 | 51 |
| fourier_a24_centered | 0.247964 | 0.044185 | 28/94 | 100 |

## 보류 94장: 선택한 네 방식 모두에 남는 동일 ID만 사용

| 방법 | Y RMSE (mm) | theta RMSE (deg) | 두 축 ±0.09 동시 충족 | 사용 점 중앙값 |
|---|---:|---:|---:|---:|
| cornerSubPix | 1.080748 | 0.192870 | 10/94 | 50 |
| cornerSubPix_r5 | 0.399109 | 0.071182 | 10/94 | 50 |
| model_r3 | 0.349763 | 0.062304 | 24/94 | 50 |
| fourier_a24_centered | 0.303004 | 0.053999 | 19/94 | 50 |

동일 ID 비교에서도 Fourier의 RMSE가 가장 낮지만, 동시 허용 오차 충족 수는 영상 형성 모델 쪽이 더 높다. 어떤 지표에서도 항상 우월한 방법으로 주장하지 않는다.

## 실제 통합 경로의 전수 검사

- fourier: 플레이트 281/281, 세 QR 디코딩 281/281, 세 QR 특징점 검증 281/281 (QR 843/843). 채택점 총 34622.
  - 같은 보류 구간의 실제 새 채택 마스크: Y RMSE 0.230780 mm, theta RMSE 0.041132 deg, 동시 기준 충족 27/94. 비교표와 점 집합이 달라 별도 수치로 기록한다.
  - 4프로세스, 프로세스당 OpenCV 1스레드, 전수 실행 38.1초. 실시간 단일 프레임 지연을 뜻하지 않는다.
- model: 플레이트 281/281, 세 QR 디코딩 281/281, 세 QR 특징점 검증 281/281 (QR 843/843). 채택점 총 27507.
  - 같은 보류 구간의 실제 새 채택 마스크: Y RMSE 0.288576 mm, theta RMSE 0.051375 deg, 동시 기준 충족 23/94. 비교표와 점 집합이 달라 별도 수치로 기록한다.
  - 4프로세스, 프로세스당 OpenCV 1스레드, 전수 실행 40.9초. 실시간 단일 프레임 지연을 뜻하지 않는다.

Fourier 통합 좌표와 비교 도구 좌표의 차이는 최대 0.000137 px였다. 영상 형성 모델은 최대 0.00582 px였다. 후자는 비교 도구에서 투영 H를 초기점들로 복원하는 차이와 방향 초기화 민감도가 포함되어 있으므로 비트 단위 동일 구현 경로라고 주장하지 않는다.

현재 운영 alignment()의 물리적 Y/theta 반환은 계속 NaN이다. 위 물리량은 오프라인 연구 평가기의 결과이다.

## 수치 검증과 합성 시험

- 해석적 미분 64개 무작위 파라미터/패치 검사: 영상 모델 최대 차이 3.50e-10 미만, Fourier 5.90e-11 미만 (중앙 유한차분).
- 교차각별 중심값의 arcsin 항등식과 직교 경계의 erf 곱 항등식 차이: 2.23e-16 미만.
- 최적점이 nuisance 변수 하한에 있는 제약 문제에서 수렴 및 위치 채택 검사 통과.
- 144개 합성 사례: 32배 해상도 이진 패턴을 discrete Gaussian으로 흐린 후 정수 위치에서 샘플링하고, 8bit 양자화/독립 잡음을 추가했다. 피팅용 이변량 적분식이나 영상 이동 보간을 생성에 재사용하지 않았다. blur 0.45/0.8/1.3 px, 교차각 75/90/105도, 밝기 얼룩 유무를 포함한다.
- 합성 시험은 고립 교점 대상이며 불규칙한 주변 QR 모듈까지 모두 재현하지 않는다. 센서 픽셀 면적 적분과 실제 인쇄 결함도 포함하지 않는다.

| 설정 | 얼룩 없음 위치 RMSE (px) | 얼룩 있음 위치 RMSE (px) |
|---|---:|---:|
| cornerSubPix | 0.138733 | 0.200238 |
| cornerSubPix_r5 | 0.044888 | 0.089816 |
| model_r3 | 0.011024 | 0.045946 |
| model_r5 | 0.008077 | 0.033328 |
| fourier_a24_centered | 0.020936 | 0.042392 |

합성에서는 큰 창의 영상 형성 모델이 더 좋지만 실제 영상에서는 같은 설정이 유리하지 않았다. 합성 순위만으로 최종 선택하지 않은 이유이다.

## 오차 전파와 한계

학습한 평균 감도는 Y 약 14.375 px/mm, theta 약 80.546 px/deg이며 theta 오차의 Y 전파 계수는 약 5.603 mm/deg이다. θ RMSE 0.04도도 Y에 약 0.22mm 규모로 전파될 수 있다. 평균 제거가 독립 잡음 자체를 증폭한다는 설명과는 구분한다.

정지 반복 영상, 독립 실측 정답, 실제 혼합 운동 영상, K/D와 회전축 기구 정보가 없다. 현재 차이는 실제 구동·교점 편향·모델 오차가 섞인 명목값 기준 결과다. 재투영 잔차나 검출 성공만으로 정밀도를 인증하지 않는다. 다음 과제는 남은 Y 오차의 원인 분석이며 추가 촬영 없이 가능한 검증 범위에 한계가 있다.

## 사용 및 재현

기본 모드는 Fourier 평균 제거 변형이다. 기존 방식과 영상 형성 모델은 명시적으로 선택할 수 있다.

```powershell
.\build\Release\main.exe <image.bmp> --no-gui --refinement fourier
.\build\Release\main.exe <image.bmp> --no-gui --refinement model
.\build\Release\main.exe <image.bmp> --no-gui --refinement corner

cmake -S tool/qr-subpixel -B tool/qr-subpixel/build -DOpenCV_DIR=C:/opencv/build -DOpenCV_RUNTIME=vc15
cmake --build tool/qr-subpixel/build --config Release
.\tool\qr-subpixel\build\Release\test_refiners.exe
python tool/qr-subpixel/run.py
python tool/qr-subpixel/evaluate.py
python tool/qr-subpixel/run.py --pipeline fourier
python tool/qr-subpixel/run.py --pipeline model
python tool/qr-subpixel/evaluate_pipeline.py
```

평가 Python은 numpy가 필요하다. 실행에 사용한 환경은 Codex bundled Python이며 일반 시스템 Python에는 numpy가 설치되어 있지 않았다. run.py는 4개의 native worker subprocess를 실행하고 각 worker가 OpenCV 1스레드로 동작한다. 생성 결과는 tool/qr-subpixel/results와 pipeline-*에 남으며 도구의 .gitignore가 제외한다.

정리된 summary.json, selection.json, pipeline-summary.json, split.json, numerics.json과 보류 예측 CSV는 이 보고서 옆에 보존한다. 전체 원시 CSV는 docs/validation-runs/qr-subpixel/ 아래에 보존하며 기존 저장소 규칙에 따라 Git 추적에서 제외된다. 최초 제약 최적화 오류의 결과도 engineering-run-1에 별도 보존한다.
