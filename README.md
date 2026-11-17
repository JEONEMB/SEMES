# S15P31A507

2026-11-09 평가 후보: joint 기본값은 [QR·바·나사 66특징 가중 선형 final](experiments/hye/00000_final_QR_bar_pin66/README.md)입니다. 실행·교차검증·5장 반복정밀도 명령과 환경별 검증 한계는 해당 README에 정리했습니다.

## 개발 규칙과 모듈 구성

작업 전 [AGENTS.md](AGENTS.md)와 [알고리즘 공통 규칙](ALGORITHM_RULES.md)을 확인합니다. 팀장이 전달한 헤더 선언·CPP 구현 분리 지침과 현재 입출력 계약은 [모듈 구성 지침](docs/team-module-layout.md)에 정리했습니다.

데이터 정제 도구의 빌드·실행은 [tool/README.txt](tool/README.txt), Git 작업은 [Git 컨벤션](docs/gitConvention.md)을 참고하세요.

## Quick Start (Windows x64)

VS Code, CMake, Visual Studio Build Tools 2026(MSVC v141), OpenCV 3.4.15(`C:\opencv`, vc15)가 설치된 환경이 필요합니다. 처음 설치하는 팀원은 [Windows 개발환경 설정](docs/windows-setup.md)의 설치 체크리스트부터 진행하세요.

### 1. 사진 데이터 준비

사진은 Git에 올리지 않습니다. 전달받은 `zip_20261014.zip`, `zip_20261020.zip`을 `data/raw/`에 넣고 정제 도구로 `data/clean/`을 만듭니다. 방법은 [tool/README.txt](tool/README.txt)를 따릅니다.

```
data/clean/clean_20261014/center.bmp, Y/, Theta/
data/clean/clean_20261020/Y+0.05 T+0.01/, ...
```

### 2. 빌드

모든 명령은 **저장소 루트**(`S15P31A507`)의 PowerShell에서 실행합니다. 다른 폴더에서 실행하면 `main.exe`와 사진 경로를 찾지 못합니다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build --config Release
```

`cmake`를 찾지 못하면 Build Tools에 포함된 cmake를 현재 창의 PATH에 추가하거나, 시작 메뉴의 Developer PowerShell for VS 2026을 사용합니다.

```powershell
$env:Path = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;" + $env:Path
```

빌드 로그의 `You might need to add C:\opencv\build\x64\vc15\bin to your PATH`는 안내 문구입니다. 빌드 후 OpenCV DLL을 실행 파일 옆에 복사하므로 따로 설정하지 않아도 됩니다.

### 3. 사진 한 장 실행

```powershell
.\build\Release\main.exe joint   "data/clean/clean_20261014/center.bmp"
.\build\Release\main.exe control "data/clean/clean_20261014/Y/Y+0.50.bmp"
```

사용법은 `main.exe joint|control IMAGE [ASSETS_DIRECTORY]`입니다. assets 경로를 생략하면 [selected_modules.cmake](experiments/selected_modules.cmake)에서 고른 모듈의 `assets/`를 사용합니다.

### 4. 출력 읽기

한 줄 JSON으로 출력합니다. 2026-10-28 Release 빌드의 실제 출력입니다.

```
{"y":0.027915035182525416,"t":-0.00049927191030941023,"measurement_valid":true,"outside_training_range":false}
{"y":0.49985708234730181,"t":0,"measurement_valid":true,"outside_training_range":false}
```

| 항목 | 의미 |
|---|---|
| `y`, `t` | 현재 사진의 Y·T 자세 추정값. 사진 파일명의 명령값과 같은 단위·부호. 보정 명령(부호 반전·장치 단위 변환)이 아님 |
| `measurement_valid` | 사진 측정과 계산이 정상 완료됨. 오차 0.09 이내를 보증하지 않음 |
| `outside_training_range` | 추정값이 학습 사진의 명령 범위 밖. 값을 잘라 바꾸지 않고 진단만 표시 |

| 명령 | 사진 (파일명 명령값) | 추정 y | 추정 t | 해석 |
|---|---|---:|---:|---|
| joint | center (Y 0, T 0) | 0.0279 | −0.0005 | Y 오차 0.028, 기준 0.09 이내 |
| control | Y+0.50 (Y 0.50, T 0) | 0.4999 | 0 | control은 Y와 T 중 한 축만 출력하므로 t=0이 정상 |

두 사진 모두 모델 학습에 쓴 사진입니다. 위 결과는 C++ 모듈이 정상 동작한다는 확인이며 새 사진 성능이 아닙니다.

측정에 실패하면 `(0,0)`을 출력하지 않고 오류 문장을 stderr에 쓰고 종료 코드 1을 반환합니다.

### 5. 검사와 전수 재실행 (선택)

```powershell
ctest --test-dir build -C Release --output-on-failure
.\build\control\Release\validate_control.exe `
  "experiments/hye/20261028_1327_final_Y_T control" data/clean/clean_20261014 `
  "experiments/hye/20261028_1327_final_Y_T control/results/NEW_VALIDATION_DIRECTORY"
```

사진 폴더 전체를 채점하려면 `benchmark.exe`를 씁니다. 결과는 웹 평가 탭([apps/web](apps/web/README.md))에서 엽니다.

```powershell
.\build\Release\benchmark.exe "data\clean" "evals\새_결과_폴더"
```

`ctest`는 공개 API를 검사합니다(초기화 누락, 잘못된 입력 거부, 원본 영상 불변 등). `validate_control`은 280장을 다시 측정해 원본 Python 결과와 비교합니다. 결과 폴더는 매번 새 이름으로 지정합니다.

빌드 없이 기존 결과만 보려면 아래 파일을 엽니다.

- joint 341장 재실행: [native_20261028_001/README.md](experiments/hye/20261028_1258_final_Y_T_joint/results/native_20261028_001/README.md)
- control 280장 재현: [native_qr536_20261028_001/README.md](experiments/hye/20261028_1327_final_Y_T%20control/results/native_qr536_20261028_001/README.md)
- QR536 상하선 비교 보고서: `experiments/hye/20261028_140538_control_QR536_상하선_비교/결과/report.html`을 브라우저로 엽니다.

## joint / control 모듈

공개 함수 선언은 [alignment.hpp](include/alignment.hpp)에 있습니다. `Alignment_t`의 y,t는 추정값이며 로봇 명령이 아닙니다.

- [1차 joint 구현·사용법](experiments/hye/20261028_1258_final_Y_T_joint/README.md)
- [2차 QR536 control 구현·검증 도구](experiments/hye/20261028_1327_final_Y_T%20control/README.md)
- [호출 예제](src/main.cpp): 사진 읽기, 초기화, 모드에 따른 함수 호출

빌드에 쓸 1차·2차 모듈은 [experiments/selected_modules.cmake](experiments/selected_modules.cmake)의 두 경로로 고릅니다. 이 파일만 고치면 다음 빌드에서 해당 모듈과 `assets/`가 사용됩니다.

원래 `alignment(image)`는 `joint(image)`로 연결했습니다. 오류는 예외로 전달하고 추가 진단은 `alignment_diagnostics()`로 읽습니다.

| | joint (1차) | control (2차) |
|---|---|---|
| 용도 | Y와 T가 섞인 사진에서 두 축을 함께 추정 | 한 축만 남은 사진에서 그 축을 정밀 추정 |
| 관측 | QR 3개의 변위 54개 + 위·아래 선 이동 48개 | QR 내부 고정 536점의 변위 |
| 출력 | `(y, t)` 동시 | Y 후보면 `(y, 0)`, T 후보면 `(0, t)` |
| 모델 | 204차원 특징의 선형 투영 | 12차원 특징의 6후보 곡선 역산 |
| 기존 결과 (학습 사진 재실행) | 341장 중 341장 두 축 0.09 이내 | 280장 중 280장 두 축 0.09 이내 |

두 모듈 모두 고정 기준 사진 `assets/center.bmp`와 현재 사진을 비교합니다. 사진 파일명, TXT, 정답값은 추론에 쓰지 않습니다. 실제 이동과 재촬영은 호출하는 쪽이 담당합니다.

### joint 알고리즘 단계

구현: [joint.cpp](experiments/hye/20261028_1258_final_Y_T_joint/src/joint.cpp), [observer.cpp](experiments/hye/20261028_1258_final_Y_T_joint/src/observer.cpp), [line_profile.cpp](experiments/hye/20261028_1258_final_Y_T_joint/src/line_profile.cpp)

1. **입력 검사**: 3840×2160, 8비트 영상인지 확인하고 컬러면 회색조로 바꿉니다.
2. **QR 대략 위치**: 기준 사진의 QR 3개 영역을 템플릿으로 NCC 탐색합니다. 탐색 반경을 130→260→512px로 넓히고, 3개 모두 상관 0.6 이상이면 멈춥니다.
3. **QR 정밀 정합**: QR 주변을 잘라 ECC 호모그래피(최대 200회 반복)로 기준 QR에 맞춥니다. ECC 상관 0.8 미만이거나 기준 점이 10px 넘게 움직이면 실패로 처리합니다.
4. **QR 관측 54개**: QR마다 가로·세로 20/50/80% 위치의 고정 점 9개를 호모그래피로 옮겨 dx, dy를 읽습니다(QR 3개 × 9점 × 2).
5. **선 관측 48개**: 위·아래 긴 가로 구조 경계에 놓인 64×49px 블록 48개에서 가로 평균 밝기 프로파일을 만듭니다. 기준 프로파일과 맞도록 세로 이동량을 찾습니다(±12px 정수 탐색 → spline 보간과 Brent 최적화로 소수 이동, 밝기 gain/offset 보정). 상관 0.9, 대비 5 미만이면 실패입니다.
6. **특징 204개**: 관측 102개에, 각 관측에 QR 평균 가로 이동을 곱한 상호작용 102개를 붙입니다.
7. **추정**: 저장된 선형 모델 `(특징 − origin) × projector × initial × scale`로 y, t를 계산합니다. 학습 범위 밖이면 `outside_training_range`를 켭니다.

### control 알고리즘 단계

구현: [control.cpp](experiments/hye/20261028_1327_final_Y_T%20control/src/control.cpp), [tracker.cpp](experiments/hye/20261028_1327_final_Y_T%20control/src/tracker.cpp), [spline.cpp](experiments/hye/20261028_1327_final_Y_T%20control/src/spline.cpp), [model.cpp](experiments/hye/20261028_1327_final_Y_T%20control/src/model.cpp). 함수별 흐름은 [FUNCTION_FLOW.md](experiments/hye/20261028_1327_final_Y_T%20control/docs/FUNCTION_FLOW.md)에 있습니다.

1. **입력 검사**: joint와 같습니다.
2. **QR 대략 위치**: QR별 NCC로 정수 픽셀 이동을 찾습니다.
3. **점 추적**: QR마다 잘라 sigma 0.8로 흐리게 한 뒤 기준 점 630개를 15×15 창 LK로 정방향·역방향 추적합니다.
4. **소수 정밀화**: 17×17 창을 cubic spline으로 샘플링하고 밝기 gain/offset을 함께 맞추는 Gauss-Newton을 최대 12회 반복합니다.
5. **점 선택과 기준 차감**: 고정된 내부 536점만 사용합니다. 기준 사진을 같은 방법으로 측정한 변위를 빼서 추적기 자체의 치우침을 없앱니다. 사진마다 점을 빼거나 바꾸지 않습니다.
6. **특징 12개 (P12)**: 점 위치 `(u, v)`의 2차 다항식 `[1, u, v, u², uv, v²]`으로 dx와 dy를 각각 최소제곱 적합한 계수 6+6개입니다.
7. **6후보 역산**: 원본 백색화 행렬 W로 특징을 정규화합니다. 후보 Y1+, Y1−, Y2+, Y2−, T+, T−마다 `특징 ≈ origin + 값×slope + 값²×curve` 곡선에서 값을 8회 반복으로 구합니다. 후보 부호와 맞지 않는 값은 버리고, 잔차가 가장 작은 후보를 고릅니다.
8. **출력**: Y 후보면 `(값, 0)`, T 후보면 `(0, 값)`을 반환합니다.

