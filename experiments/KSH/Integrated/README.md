# Integrated — 통합 정렬 (김성현 ksh 추정 + 신윤석 판 검출·QR 검증)

사진 한 장으로 기준(center) 대비 Y [mm]·Θ [°] 어긋남을 구하고, 서로 독립인 두 번째 측정으로 그 값을 검증한다.
팀 공통 인터페이스 `include/alignment.hpp` 의 `alignment(cv::Mat&)` 가 이제 실제 y·t 를 돌려준다.

근거 : [`docs/integrated_comparison.html`](docs/integrated_comparison.html) · 채점 원자료 `docs/comparison_all.csv` (2026-11-01, 같은 사진·같은 규칙으로 두 분석 비교)

## 역할 분담

| 단계 | 담당 | 내용 | 근거 |
|---|---|---|---|
| ① Coarse | ksh | 플레이트 창 위상 상관 | |
| ② Fine | ksh | QR 3개 + 하단 홀 W2·W3(56×60 px) IC-GN · O-MOMS | 실험계획 2026-10-30 |
| ③ 특징 | ksh | s = QR 평균 가로 이동, d = s − 홀 평균 (깊이 차 시차) | 판(QR)만으로는 Y·Θ 구별 불가 — shin 식별성 분석(조건수 799)과 같은 결론 |
| ④ 추정 | ksh | s, d = c + aY + bΘ 역변환 | 2차 좌표 보류에서 Y RMS 13 µm (QR6 역회귀 148 µm, +배경 가장자리 67 µm) |
| ⑤ 검증 | shin | 판 검출 → QR payload 해독 → 모듈 재생성 일치 → Fourier 코너 | 판·주소 확인, 판 중심 이동 ≈ Coarse (±30 px), 코너 QR 이동 ≈ IC-GN (±0.1 px) |

검증 결과는 세 가지다.

- **검증 통과** : 판·QR 을 찾았고 값이 서로 맞음
- **검증 모순** : 찾았는데 값이 다름 → `reject_on_guard: 1` 이면 NaN (오정합, 다른 판·주소)
- **검증 못 함** : 판·QR 을 못 찾음 → 기본은 값 유지 (`reject_unverified: 0`). 575장 중 4장(0.7 %), 그 4장도 추정은 허용 안

## 폴더

```
src/Integrated/
  CMakeLists.txt                     ksh_align · shin_detect · integrated 라이브러리 + 실행 파일 2개
  include/integrated/integrated.hpp  Aligner, Settings, Result
  src/integrated.cpp                 추정 → 검증 → 판정, 새 center 채택 검사
  src/team_alignment.cpp             팀 alignment() 구현 (shin src/alignment.cpp 대신 링크)
  apps/integrated_main.cpp           사진 한 장 / --center-check
  apps/integrated_eval.cpp           라벨 폴더 정확도 평가
  config/integrated.yml              ROI·정렬·모델·검증 한계 (값은 모두 여기서)
  config/model_integrated_s1d1.yml   위 ROI 로 1차 240장 학습한 보정식
```

ksh 소스는 `src/KSH/Alignment/ksh`, shin 소스는 `src/*.cpp`·`include/` 를 그대로 가져다 쓴다 (복사하지 않음).

## 빌드 (저장소 루트, Windows x64)

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build --config Release --target integrated_main integrated_eval
```

루트 `main` (shin 검출 예제) 도 그대로 빌드된다.

## 실행 (작업 폴더 = src/Integrated)

| 명령 | 동작 |
|---|---|
| `..\..\build\src\Integrated\Release\integrated_main.exe "..\..\data\img\3rd\Y+0.05 T+0.40\Y+0.05 T+0.40 (1).bmp"` | 한 장 추정 + 검증 |
| `integrated_main.exe <사진> --no-verify` | ksh 추정만 (약 35 ms) |
| `integrated_main.exe --center-check <새 center>` | 새 기준 사진 채택 검사 (±21 µm · ±0.0055°) |
| `integrated_eval.exe --dir ..\..\data\img\3rd` | 폴더 평가 (하위 폴더까지, 기준 = 설정의 center1) |

## 검증 결과 (리눅스 OpenCV 4.6 빌드, 2026-11-01)

| 세트 | 허용 안 (±45 µm · ±0.005°) | 과제 허용 (±90 µm · ±0.01°) | Y RMS | Θ RMS | 검증 모순 | 검증 못 함 | 시간 (추정 / 검증) |
|---|---|---|---|---|---|---|---|
| 1차 교차검증 (Y8+Θ8, 240장) | 219 | 240 | 12.9 µm | 0.0029° | — | — | 30 ms |
| 2차 57장 (Θ 2° 12장 포함) | 36 | 42 | 28.9 µm | 0.0096° | 0 | 0 | 41 / 725 ms |
| 3rd 234장 | 229 | 234 | 10.8 µm | 0.0025° | 0 | 4 | 36 / 669 ms |

- C++ 결과가 파이썬 실험계획 결과(1차 CV J 0.651, 3rd J 0.552)와 같다.
- 검증 최대 차 : 판 중심 − Coarse 10.6 px, 코너 − IC-GN QR 이동 0.018 px (한계 30 px, 0.1 px).
- Windows(MSVC v141, OpenCV 3.4.15) 빌드는 아직 확인 전.

## 운용 규칙 (Center 정의)

1. 측정 직전에 center 를 찍고 `--center-check` 로 채택 검사 → 통과하면 `reference_path` 교체
2. center 와 작업 이동은 같은 방향에서 접근 (ISO 230-2 단방향)
3. 보정식의 영점 상수 c 는 유지 (center 사진 ↔ 지령 0 차이 Y +35.5 µm, 두 날 같음)
