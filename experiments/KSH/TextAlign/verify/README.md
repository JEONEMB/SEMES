# OutPut_Text — 교보재 Y·Θ 정렬 (기준 사진 없이 텍스트 설정만으로)

`OutPut`(교보재 제출판)은 **기준 사진(center.bmp) + ROI 템플릿 사진**과 비교해 이동량을 쟀다.
이 버전은 `PJT/OutPut_A507` 의 구조(진입점 + ksh 모듈 + yml 두 개)를 그대로 따르면서, 측정을 **사진 한 장 안의 절대 위치**로 바꿔
실행에 필요한 것을 **텍스트 파일 두 개**(`config/alignment.yml`, `config/model_s1d1.yml`)로 줄였다.

- 사진 파일은 실행에 쓰지 않는다. 센터 사진은 **처음 세팅(ROI·영점 기록)** 때만 쓰고, 결과는 숫자로만 남긴다.
- 보정식 : s·d 모두 Θ 1차 (`model_s1d1.yml`, 계수 6개). 학습 = 1차 + 4차, 검증 = 2차 + 3차.
- 영점(판정 기준) : **모터 엔코더 영점** (Y_Actual −1.300 mm, T_Actual −1.670°). center1 사진 자세는 엔코더 영점에서 Y −35.3 µm, Θ −0.0030° 떨어져 있어 `center_to_encoder` 로 보정한다.
- git(`PJT`)에는 아직 옮기지 않았다. 검토 후 옮길 것.

## 1. 폴더

```
OutPut_Text/
  config/alignment.yml        설정 — ROI 7개 절대 좌표(QR 3 + 뒷벽 홀 4), 측정 파라미터, 영점(zero), 허용 범위, 학습 조건
  config/model_s1d1.yml       보정식 (align_train 이 만든 것)
  include/alignment.hpp       제출 형식 API : Alignment_t alignment(cv::Mat &src)  (+ alignment_setup, alignment_set_reference)
  include/ksh/*.hpp           모듈
  src/main.cpp                사용 예 + 초기 세팅 (--setup, --zero)
  src/alignment.cpp           진입점
  src/modules/                ① qr_locate ② hole_locate ③ features ④ pose_model · estimator · setup · align_config
                              · dataset · batch · evaluation · image_util
  tools/align_train.cpp       보정식 학습 + 검증
  tests/test_alignment.cpp    단위 테스트 (사진 없이, 31개)
  tests/test_real_photos.cpp  실사진 점검 (사진이 있을 때, 7개)
  data/features_*.csv         교보재 사진 719장에서 잰 특징 (사진 없이 다시 학습할 때)
  results/                    학습·검증 보고서(report.md), 사진별 결과(predictions.csv), 검증보고서.md
  img/center1.bmp             설치 점검용 사진 1장 (알고리즘은 사진 파일을 쓰지 않음)
  OutPut_Text.vcxproj         Visual Studio : alignment.exe (main)
  OutPut_TextTrain.vcxproj    Visual Studio : align_train.exe
  OutPut_TextTest.vcxproj     Visual Studio : align_tests.exe (단위 테스트)
  OutPut_TextPhoto.vcxproj    Visual Studio : align_photo_check.exe (실사진 점검)
  OutPut_Text.slnx            이 폴더만 여는 솔루션
  CMakeLists.txt              CMake (위 3개 + align_photo_check)
```

## 2. 기존 교보재판(OutPut)·A507 판과 비교

| 항목 | OutPut (교보재, 기존) | OutPut_A507 | **OutPut_Text (이번)** |
|---|---|---|---|
| 실행에 필요한 파일 | yml + center.bmp + roi_template.bmp | yml 2개 | **yml 2개** |
| ① 대략 위치 | 플레이트 창 위상 상관 | QR 띠 → 흰 사각형 3개 | QR 사이 어두운 틈 2곳 (닫힘 연산 후 열 평균) |
| ② QR | 기준 대비 IC-GN 이동 | 사각형 4변 직선 → 중심 | 사각형 4변 직선 → 중심 (행·열마다 서브픽셀, 강건 직선) |
| ② 배경 | W2·W3 IC-GN 이동 | 나사 2개 원 맞춤 | **뒷벽 홀 4곳 방사 대칭 중심** (Parthasarathy 2012) |
| s / d | 기준 대비 이동 | 절대 위치 | 절대 위치 (s = QR x 평균, d = s − 홀 x 평균) |
| 영점 | 기준 사진 | zero: {y_mm, t_deg} | zero: {y_mm, t_deg} — `--setup`/`--zero` 가 기록 |
| ROI 좌표 | 템플릿 사진 위 좌표 + 재배치 | QR 기준 상대 | **사진 절대 좌표**, `--setup` 이 센터 사진으로 다시 맞춤 |
| 학습 | 단일 세트 | 세트 합쳐서 | **세트마다 상수항 따로 + 세트 균형 가중** (촬영 날 차이 흡수) |
| 부호 | +Y·+Θ → 플레이트 오른쪽 | 반대 | 교보재와 같음 |

## 3. 처음 세팅 (설비에 넣을 때 1회)

```bash
# 작업 폴더 = OutPut_Text
alignment --setup <센터1.bmp> [센터2.bmp 센터3.bmp] --no-window --overlay setup_check.png
```

1. 지금 박스로 센터 사진을 잰다 — 박스가 수십 px 어긋나 있어도 ① 대략 위치(±350 px)와 박스 여유(40 px)로 찾는다.
2. 찾은 중심으로 **박스 7개의 cx_px·cy_px 를 고쳐 쓴다** (hw 는 그대로).
3. 새 박스로 다시 재서 **센터 사진들의 평균을 영점**(zero:)으로 쓴다. 원본 설정은 `config/alignment.yml.bak`.
4. 검사 — 경고가 있으면 종료 코드 2 :
   - 센터 사진끼리 s 차이 > 0.3 px, d 차이 > 0.1 px (로봇이 같은 자리에 섰는지)
   - QR 간격 차이 > 2 px (카메라 기울어짐 · QR 판)
5. `--overlay` 그림으로 박스와 찾은 중심(빨간 +)을 눈으로 한 번 확인한다.

- 영점만 다시 잡을 때 : `alignment --zero <센터.bmp>` (박스는 그대로).
- 센터에 놓을 때는 정렬할 때와 **같은 방향으로 접근**해서 찍는다 (반대면 백래시만큼 밀림).
- 지금 들어 있는 값 = center1.bmp 로 세팅한 박스 (QR 1643/1901/2159) 와 엔코더 기준 영점 (y −0.08413 mm, t +0.00985°).
- `--setup` · `--zero` 는 영점을 **새 센터 측정값 + center_to_encoder** 로 기록한다 → 센터를 다시 찍어도 엔코더 기준이 유지된다.
  (센터를 다시 찍지 않는다면 `--setup` 은 필요 없다. 지금 설정 그대로 쓰면 됨)

## 4. 실행

```bash
alignment                             # 설치 점검 : img/center1.bmp → (+0.0353 mm, +0.0030°) 이면 정상 (= center1 이 엔코더 영점에서 떨어진 만큼)
alignment <사진> [센터 사진]           # 한 장 추정 (센터를 주면 그 사진을 영점으로, 파일에는 안 씀)
alignment <사진> --no-window --overlay out.png
```

장비 코드에서는 :

```cpp
alignment_init("config/alignment.yml");
Alignment_t r = alignment(frame);              // frame : 3840×2160
if (!std::isnan(r.y)) stage_move(-r.y, -r.t);  // 실패하면 NaN, 이유는 alignment_last_info().message
```

실패(NaN)로 처리하는 경우 : QR 사이 틈을 못 찾음, 이동이 ±350 px 밖, 박스가 사진 밖, 사각형 변을 못 찾음(가림·조명),
홀이 안 보임(명암 차 < 10), 홀이 둥글지 않음(중심 차 > 12 px), 홀 4곳 위치가 서로 안 맞음(폭 > 5 px).

## 5. 보정식 다시 학습

```bash
align_train                 # config/alignment.yml 의 training: 사용
```

- `feature_csv` 가 있으면 사진 없이 그 값으로 학습한다 (지금 `data/features_*.csv` 로 재현됨).
  지우거나 경로를 바꾸면 `image_dir` 의 사진을 다시 잰다 (교보재 사진 배치 그대로 : `1st/Y`, `1st/Theta`, `2nd/<Y+a T+b>`, `3rd/…`, `4th`).
- 학습 세트(`train`)는 **세트마다 상수항을 따로** 두고 기울기만 공통으로 맞춘다 → 촬영 날마다 다른 영점 차이가 기울기로 새지 않음.
- 검증 세트(`test`)는 `zero_ref`(같은 날 센터 사진)로 영점을 잡고 채점한다 — 실제 사용과 같은 방식.
- ROI 를 바꾸거나 홀을 바꾸면 반드시 다시 학습.

## 6. 빌드

```bash
# 리눅스 (OpenCV 4.x) — 이 환경에서 GCC 13 + OpenCV 4.6 으로 빌드·테스트함
cmake -S . -B build && cmake --build build -j
cd <OutPut_Text> && build/align_tests && build/align_photo_check img/center1.bmp
```

Visual Studio (2026, 툴셋 v145) — 두 가지 방법 :
- **이 폴더만** : `OutPut_Text.slnx` 를 연다.
- **기존 솔루션** : `QR_Image.slnx` 의 `/5_텍스트설정/` 폴더에 같은 4개 프로젝트가 들어 있다.

| 프로젝트 | 실행 파일 | 하는 일 | 실행 인자 (.vcxproj.user) |
|---|---|---|---|
| `OutPut_Text` | alignment.exe | 사진 한 장 추정 · `--setup` · `--zero` | 없음 → `img/center1.bmp` 추정 + 결과 창 |
| `OutPut_TextTrain` | align_train.exe | 보정식 학습 + 검증 보고서 | 없음 → `config/alignment.yml` |
| `OutPut_TextTest` | align_tests.exe | 단위 테스트 31개 | 없음 |
| `OutPut_TextPhoto` | align_photo_check.exe | 실사진 점검 7개 | `img\center1.bmp` |

1. 실행할 프로젝트를 오른쪽 클릭 → **시작 프로젝트로 설정**, 구성 **x64** (Debug 또는 Release).
2. 작업 폴더는 모두 이 폴더(`$(ProjectDir)`)로 잡혀 있다 — `config/alignment.yml` 을 상대 경로로 찾음.
3. OpenCV 설정은 기존 `OutPut.vcxproj` 와 같다 : `C:\opencv\build\include`, `C:\opencv\build\x64\vc15\lib`,
   `opencv_world3415(d).lib`, 빌드 후 DLL 을 실행 파일 옆으로 복사. C++17, `/utf-8`, `NOMINMAX`.
4. 실행 파일·중간 파일 : `x64\<구성>\` (프로젝트마다 `obj_<이름>` 으로 나눔).
5. 다른 인자로 실행하려면 프로젝트 속성 → 디버깅 → 명령 인수 (예 `--setup img\center1.bmp --no-window`).

**MSVC · OpenCV 3.4.15 확인 (2026-11-17)** : 단위 테스트 31 / 31, 실사진 점검 7 / 7, center1 의 s = 1901.123 px · d = −14.110 px 로 리눅스(GCC 13 + OpenCV 4.6)와 같은 값.

## 7. 검증 결과 요약 (자세히는 `results/검증보고서.md`)

| 검증 | 결과 |
|---|---|
| 단위 테스트 (합성 영상 · 보정식 · 설정 파일 · 전체 흐름) | 31 / 31 통과 (리눅스 · Windows MSVC 모두) |
| 실사진 점검 (센터, 반복, 밝기 · 잡음, 이동, 가림) | 7 / 7 통과 (리눅스 · Windows MSVC 모두) |
| Python 검토판과 같은 계산인지 (718장, parabola 모드) | s · d 최대 차이 2 × 10⁻¹¹ px |
| 학습 1·4차 → 검증 2·3차 291장, **엔코더 기준** | 명세 282 / 291, 권장 기준 265 / 291 (Θ ±1° 안 277 / 279) |
| 정밀도 (2·3차 흩어짐) | Y σ 13.7 µm, Θ σ 0.0035° |
| 엔코더 보정값 교차 확인 (2차로 정해 3차 채점 / 반대) | 3차 232 / 234, 2차 48 / 57 — Θ 영점이 세트마다 ±0.0033° 달라짐 |
| 처리 시간 | 약 4 ms / 장 (리눅스), 14.9 ms (Windows, 구성 미확인) |

- 엔코더 보정값(center_to_encoder)은 2·3차 사진 자체로 정했으므로 위 282 / 291 은 그 값에 대해서는 재대입이다. 교차 확인 행을 함께 볼 것.

## 8. 알려진 한계

1. **홀 근처(약 30 px 안)에 새로 생긴 먼지·점** : 크고 진한 것은 실패로 잡히지만(14/24),
   옅은 점(밝기 140)이나 W1 근처 점은 실패로 안 잡히고 Y 를 30 ~ 285 µm 바꿀 수 있다 (10/24). 40 px 밖의 점은 영향 없음.
   → 세팅 후 홀 주변을 깨끗이 유지, 주기적으로 `alignment <센터 사진>` 으로 (0, 0) 근처인지 확인.
2. **Θ 영점 불확실성** : 같은 날 세트끼리도 엔코더 대비 Θ 영점이 ±0.0033° 달라졌다 (로봇 Θ 백래시 크기). 엔코더로 판정하면 Θ 오차에 이만큼이 더해질 수 있다.
3. Θ 1° 를 넘는 범위는 학습 데이터가 없다 (1차식이라 Θ 2° 에서 Y 약 −30 µm, Θ 약 +0.005° 치우침 — 폐루프 다음 단계에서 고쳐짐).
4. 이동 한계 : 플레이트가 ±350 px (Y 약 ±24 mm 또는 Θ 약 ±4.4°) 를 넘으면 실패. 실사진 확인은 Θ 2° · Y 1.6 mm 까지.
