# alignment — A507 장비용 Y·Θ 정렬 (교보재 제출 구조 그대로)

`OutPut_KSH/Code` 의 제출 구조(`main` + `alignment` + `ksh` 모듈 + `config/*.yml`)를 그대로 따르고,
측정 알고리즘만 A507 스테이지에 맞게 바꾼 버전이다.

## 1. 폴더

```
Code/
  config/alignment.yml      설정 (QR 찾기, 나사 ROI, 영점, 허용 범위, 학습 조건)
  config/model_s1d2.yml     보정식 (align_train 이 만든 것)
  include/alignment.hpp     제출 형식 API : Alignment_t alignment(cv::Mat &src)
  include/ksh/*.hpp         모듈
  src/main.cpp              사용 예 (교보재와 같은 사용법)
  src/alignment.cpp         진입점
  src/modules/*.cpp         ① qr_locate ② screw_locate ③ features ④ pose_model, estimator, align_config,
                            geometry(서브픽셀 공통), image_util, dataset, batch, evaluation
  tools/align_train.cpp     보정식 학습 + 정확도 평가
  data/features_*.csv       2026-11-12 촬영 1,960장에서 잰 특징 (사진 없이 다시 학습할 때)
  results/                  학습 보고서 (report.md, holdout/report.md)와 사진별 결과
  img/center_sample.bmp     설치 점검용 사진 1장 (알고리즘은 사진 파일을 쓰지 않음)
  CMakeLists.txt
```

## 2. 교보재 코드에서 바꾼 것

| 항목 | 교보재 (OutPut_KSH) | A507 버전 |
|---|---|---|
| 기준 사진 | `img/center.bmp` + `roi_template.bmp` 필수 | **없음** — 설정 yml 두 개만 |
| ① | Coarse : 플레이트 창 위상 상관 | QR 띠(1/4 축소, 어두운 덩어리) → 흰 사각형 3개 |
| ② | Fine : QR·배경 ROI 마다 IC-GN (기준 대비 이동) | QR 사각형 4변 서브픽셀 직선 → 중심. 나사 ROI 는 QR 이동량으로 위치를 추정해 두고 머리 테두리 원 맞춤 |
| ③ s | QR 3개 평균 **이동** (기준 대비) | QR 3개 중심 **가로 위치** (사진 좌표) |
| ③ d | s − 배경(W2·W3) 평균 이동 | s − 나사(왼쪽 아래·오른쪽 위) 가로 위치 평균 |
| ④ 보정식 | `model_s1d1.yml` (s, d 모두 Θ 1차) | `model_s1d2.yml` (s Θ 1차, d Θ 2차) — 형식·역산 코드(pose_model.cpp)는 그대로 |
| 기준 사진 교체 | `alignment_set_reference` → ROI 재배치 | `alignment_set_reference` → **영점**(센터에서 잰 값)만 기억. `save=true` 면 yml 의 `zero:` 에 기록 |
| 부호 | SEMES : +Y·+Θ 에 플레이트 오른쪽 | A507 : +Y·+Θ 에 플레이트 **왼쪽**. 출력은 A507 스테이지 명령 방향 (보정 이동 = (−y, −t)) |
| 허용 범위 | ±90 µm, ±0.01° | ±90 µm, **±0.09°** (권장 멈춤 기준은 그 절반) |
| 리눅스 빌드 | `typedef double error_t` 가 glibc 의 `error_t(int)` 와 충돌 | 리눅스에서만 이름을 바꿔 피함 (윈도우는 교보재와 같음) |
| 결과 창 | 항상 띄움 | `--no-window` 로 끔 (라즈베리파이처럼 화면이 없을 때) |

그대로 둔 것 : `alignment()` / `alignment_init()` / `alignment_last_info()` / `alignment_reference_info()` 이름과 쓰는 법,
NaN 실패 규칙, `pose_model.cpp`(학습·역산), `evaluation.cpp`(채점), `dataset.cpp`(파일 이름 = 라벨, `center_after_NNNN` 만 추가), `image_util.cpp`(`to_gray8` 추가).

## 3. 원리

| 측정 | Y 감도 | Θ 감도 |
|---|---|---|
| QR 3개 중심 (앞판) | −14.5 px/mm | −86.1 px/° |
| 나사 2개 (뒤판, 카메라에서 더 멂) | −10.9 px/mm | −86.0 px/° |

Θ 에는 둘이 거의 똑같이, Y 에는 다르게 움직인다 → `d = s − 나사 평균` 이 거의 Y 만 보여 준다(−3.58 px/mm). 보정식 :

```
s = 1908.1774 − 14.494194·Y − 86.144083·Θ                     (σ 0.64 px)
d = 9.869607  − 3.5814673·Y − 0.1002733·Θ + 0.075759323·Θ²    (σ 0.072 px)
→ 측정한 (s, d) 를 동시에 만족하는 (Y, Θ) 를 가우스–뉴턴으로 구함 (pose_model.cpp, 교보재와 같은 코드)
```

## 4. 정확도 (2026-11-12 A507 촬영, 기준 Y ±90 µm · Θ ±0.09°)

| 평가 | 장 | 기준 안 | Y rms | Y 최대 | Θ rms | Θ 최대 |
|---|---|---|---|---|---|---|
| grid 로 학습 → **set1 검증** (학습에 안 쓴 사진) | 280 | **280 / 280** | 35.0 µm | 80.7 µm | 0.0089° | 0.019° |
| 최종 모델 (grid + set1 1,960장) 블록 교차검증 | 1,960 | **1,960 / 1,960** | 20.5 µm | 81.0 µm | 0.0077° | 0.022° |

- 보고서 : `results/report.md` (최종), `results/holdout/report.md` (set1 검증).
- 순수 측정 흔들림은 Y 약 14 µm, Θ 약 0.0023° (1σ). Θ 오차의 대부분은 스테이지 재현성·백래시.
- Θ 차수 4조합(s1d1, s1d2, s2d1, s2d2) 비교 : s1d1 은 교차검증 3장이 Y 기준 밖(최대 98.5 µm), s1d2 가 가장 좋았다.

## 5. 빌드

OpenCV (교보재와 같은 3.4.15 또는 4.x) + C++17.

```bash
# 리눅스 · 라즈베리파이 5 (Raspberry Pi OS : sudo apt install libopencv-dev cmake g++)
mkdir build && cd build && cmake .. && make -j4
# 윈도우 (Visual Studio) : CMake 로 열거나, 교보재 프로젝트에 src/ include/ 를 바꿔 넣고 /utf-8 옵션
```
이 컨테이너에서 GCC 13 + OpenCV 4.6 으로 빌드·실행 확인. OpenCV 3.4.15 · MSVC · 라즈베리파이 실기에서는 아직 빌드해 보지 않았다.

## 6. 실행 (작업 폴더 = Code)

```bash
./alignment                                 # 설치 점검 : img/center_sample.bmp 추정 + 결과 창
./alignment <사진> [센터 사진] [--no-window]  # 한 장 추정. 센터 사진을 주면 그 사진을 영점으로 삼아 비교
./alignment --zero <센터 사진>               # 영점을 config/alignment.yml 의 zero: 에 기록 (원본 .bak)
./align_train                               # 보정식 다시 학습 (config 의 training: 사용)
```

장비 코드에서는 교보재와 같이 :
```cpp
alignment_init("config/alignment.yml");
Alignment_t r = alignment(frame);          // frame : 보정 프레임 3840×2160
if (!std::isnan(r.y)) stage_move(-r.y, -r.t);
```

- 입력은 `capture_dataset` 이 저장하는 것과 같은 보정 프레임(왜곡 보정·화각 맞춤 3840×2160)이어야 한다.
- 영점은 정렬할 때와 같은 방향(−쪽에서 지나쳐 접근)으로 센터에 놓고 잡는다. 반대 방향이면 백래시만큼 Θ 가 밀린다
  (촬영 데이터의 센터 사진은 Θ +0.015 ~ +0.03° 로 읽힘).
- 나사 2개(왼쪽 아래·오른쪽 위)는 그대로 두고 가리지 않는다. 바꾸면 사진을 다시 찍어 `align_train` 으로 재학습.

## 7. 다시 학습하기

`config/alignment.yml` 의 `training:` 에서 세트를 정한다. `feature_csv` 파일이 있으면 사진 없이 그 값으로 학습하고,
지우거나 없으면 `image_dir` 의 사진을 재서 `results/<이름>_features.csv` 로 저장한다. 사진 측정은 장당 약 15~30 ms (PC).
학습에 안 쓰고 채점만 할 세트는 `test:` 에 넣는다.
