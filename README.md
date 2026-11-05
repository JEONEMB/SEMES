# S15P31A507 — Y·Θ 정렬 알고리즘 (김성현, ksh)

사진 한 장으로 기준(center) 대비 어긋남 (Y mm, Θ °) 을 추정한다. 로봇은 (−Y, −Θ) 만큼 보정 이동을 반복해 20회 안에 영점을 맞춘다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `include/alignment.hpp` | 제출 형식 헤더 — `Alignment_t alignment(cv::Mat &src)` |
| `include/ksh/*.hpp` | 알고리즘 모듈 선언 (정렬 · 특징 · 보정식 · 평가, 함수 단위) |
| `src/alignment.cpp` | 진입점 `alignment()` (ksh 모듈을 감싼 얇은 층) |
| `src/main.cpp` | 예제 실행 파일 (사진 한 장 추정 + 결과 창) |
| `src/modules/*.cpp` | 알고리즘 모듈 구현 |
| `tool/ksh/` | `align_train` (보정식 학습) · `align_eval` (정확도 평가) |
| `config/` | `alignment.yml` (모든 설정값) · `model_s1d1.yml`, `model_s3d2.yml` (학습된 보정식) |
| `data/img/1st·2nd·3rd/` | 팀 공통 사진 (저장소에 올리지 않음) |
| `docs/KSH/` | 알고리즘 설명 (`alignment/README.md` : 처리 흐름 · 모듈 · 검증 결과) |
| `experiments/KSH/` | 실험 기록 — EDA, 운동 모델, 가설 1~5, 실험계획(DOE), 분석, MFC 측정 앱, 시뮬레이터, 통합 정렬기 (빌드에 포함하지 않음) |
| `legacy/team_pipeline/` | 이전 루트 `src`·`include` 의 팀 고정 파이프라인 (2026-11-04 옮김, 빌드에 포함하지 않음) |

## 빌드 (Windows x64, PowerShell, 저장소 루트)

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR="C:/opencv/build"
cmake --build build --config Release
```

## 실행 (작업 폴더 = 저장소 루트)

| 명령 | 동작 |
|---|---|
| `.\build\Release\alignment.exe` | `config/alignment.yml` 의 `demo.image` 한 장 추정 |
| `.\build\Release\alignment.exe <사진> [기준 사진]` | 지정한 사진 한 장 추정 |
| `.\build\Release\align_train.exe` | 1차 축 사진으로 보정식 학습 → `results/model_s1d1.yml` |
| `.\build\Release\align_eval.exe [--dir data/img/3rd]` | 라벨 사진 평가 → `results/eval_photos*.csv · .md` |

- 사진은 `data/img/` 아래 원본 배치(1st/Y, 1st/Theta, 2nd/기준 위치, 2nd/Y+a T+b, 3rd/…) 그대로 둔다. 하위 폴더까지 찾고, 한글 경로도 읽는다.
- `align_eval` 기본 폴더(`data/img`)는 3rd 세트까지 함께 평가한다. 세트별로 보려면 `--dir data/img/2nd` 처럼 지정한다.
- 통합 정렬기(ksh + 팀 판 검출)는 `-DKSH_BUILD_INTEGRATED=ON` 일 때만 함께 빌드한다.
