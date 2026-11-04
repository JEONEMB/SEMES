# tool/hw/eda

데이터 묶음 하나를 분석하고 발표자료(pptx)까지 만드는 C++ 프로그램 `hw_eda`.

## 흐름

```
data/hw/data-automation  →  hw_eda <묶음>  →  docs/validation-runs/hw/eda/<묶음>/SEMES_EDA_<묶음>.pptx
 (zip → data/hw/bmp)        (측정 → 분석 → 슬라이드)
```

## 빌드

프로젝트 루트에서 [Windows 개발환경 설정](../../../../../../../../20261020_EDA_도구_hw/결과%20정리/docs/windows-setup.md)과 같은 옵션으로 빌드한다.

```powershell
cmake -S tool/hw/eda -B build/hw/eda -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build/hw/eda --config Release
```

## 실행

프로젝트 루트에서 실행한다.

```powershell
.\build\hw\eda\Release\hw_eda.exe                          # 묶음을 물어봄
.\build\hw\eda\Release\hw_eda.exe 20261014                 # sweep 묶음 (1차): v4 발표자료 구성
.\build\hw\eda\Release\hw_eda.exe 20261020 --ref 20261014  # grid 묶음 (2차): 1차 결과로 평가
```

| 묶음 종류 | 판별 | 슬라이드 |
|---|---|---|
| sweep | 파일명 `Y+0.50.bmp`, `T-0.35.bmp` (한 축씩) | 표지, 기초 정보, 기준 위치(QR, 특징점, ECC), 0점, 이동 경로(특징점, ECC), 한 칸 이동량, 노이즈 표, 노이즈 그림 2장, 해석 |
| grid | 파일명 `Y+0.05 T+0.01 (1).bmp` (두 축 같이) | 표지, 기초 정보, 기준 위치, center 반복, 같은 조합 반복 흔들림, 1차 모델 예측 vs 측정, 세로 이동, 조합별 결과표, 해석 |

## 출력

| 위치 | 내용 | Git |
|---|---|---|
| `docs/validation-runs/hw/eda/<묶음>/SEMES_EDA_<묶음>.pptx` | 발표자료 | 제외 |
| `docs/validation-runs/hw/eda/<묶음>/csv/` | `shots.csv`(사진별 이동), `corners.csv`, `trials.csv`, `steps.csv`, `conditions.csv` | 제외 |
| `docs/validation-runs/hw/eda/<묶음>/fig/` | 사진 그림 | 제외 |
| `docs/validation-runs/hw/eda/<묶음>/log.txt` | 실행 기록 | 제외 |
| `docs/hw/eda/sets/<묶음>/summary.yml` | 핵심 숫자 (묶음끼리 비교, `--ref` 가 읽음) | 추적 |
| `docs/hw/eda/sets/<묶음>/notes.md` | **직접 쓰는 해석** → 마지막 슬라이드 (`## 제목`, `- 내용`) | 추적 |

## 측정 방법

| 방법 | 내용 |
|---|---|
| QR 라벨 | center 사진 가운데 띠를 오츠 이진화, 나란한 밝은 사각형 3개 |
| ECC | 라벨 + 바깥 8px 패치. 템플릿 매칭(가로 ±320px) → ECC 아핀(흐림 5px), 패치 가운데 점의 이동 |
| 특징점 | 라벨 안 코너 검출 → 주변 4칸으로 종류 판정 → "흐린 2x2 모듈" 밝기 모델을 9x9 창에 맞춤 (레벤버그-마쿼트, soft-L1). 사진마다 ECC 로 자리를 예측해 다시 맞춤. 대각선 모서리는 평균에서 빼고 흔들림 비교에만 씀 |

## 구조

| 파일 | 하는 일 |
|---|---|
| `src/main.cpp` | 실행 흐름, 인자 |
| `src/core.*` | 묶음 읽기, 파일명 해석, 이미지·CSV·summary 입출력, 통계 |
| `src/measure.*` | QR 라벨, ECC, 특징점 밝기 모델 맞춤 |
| `src/measurements.*` | 묶음 전체 측정 (사진은 한 번만 읽음) |
| `src/slides_common.*` | 슬라이드 틀, 표, 기준 위치 슬라이드 |
| `src/slides_sweep.cpp` | sweep 분석과 슬라이드 |
| `src/slides_grid.cpp` | grid 평가 슬라이드, notes.md → 슬라이드 |
| `src/pptx.*` | pptx 쓰기 (템플릿 풀기, 도형, PowerPoint 기본 차트, Windows tar 로 zip) |
| `report/template.pptx` | v4 발표자료에서 슬라이드를 뺀 디자인. `hw_eda --make-template <원본.pptx>` 로 다시 만듦 |

새 데이터 종류가 오면 `src/slides_<종류>.cpp` 를 추가하고 `main.cpp` 에서 종류별로 부른다.
