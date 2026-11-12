# QR-only 536점 control C++ 모듈

사용자가 선택한 `release_qr536_20261021/pipeline.py`를 C++17/MSVC v141/OpenCV 3.4.15로 이식했다. 새로 작성했던 조건부 Y GLS 모델과 학습 도구는 활성 실행에서 제거하고 `results/replaced_conditional_20261028_001/`에 보존했다. 알고리즘의 합의 기준은 [ALGORITHM_RULES.md](../../../ALGORITHM_RULES.md)를 따른다.

## 팀에서 호출하는 방법

공개 선언은 [alignment.hpp](../../../include/alignment.hpp), 진입점 구현은 [src/control.cpp](src/control.cpp)다.

```cpp
#include "alignment.hpp"
// 작업 시작 시 한 번. joint 초기화는 필요 없다.
initialize_control("experiments/hye/00000_final_Y_T control/assets");
// 현재 사진을 IMREAD_GRAYSCALE로 읽은 cv::Mat
Alignment_t estimate = control(image);
// estimate.y, estimate.t는 추정값. 이동·단위 변환·재촬영은 호출 측에서 처리.
AlignmentDiagnostics diagnostic = alignment_diagnostics();
```

입력은 3840×2160 uint8 1/3/4채널 영상이며 수정하지 않는다. 원본 평가와 비교할 때는 동일 BMP 바이트를 grayscale로 디코딩한다. 함수는 원본과 동일한 **Y-or-T** 결과를 반환한다. Y 후보가 선택되면 `(추정 Y,0)`, T 후보면 `(0,추정 T)`다. Y와 T를 동시에 회귀하는 `joint()`와 출력의 성격이 다르다. 파일명·TXT·정답·시행 번호는 추론에 들어가지 않는다.

초기화·디코딩·입력·수치 계산 오류는 `std::exception`으로 전달한다. 실패를 `(0,0)`으로 반환하지 않는다. `maximum_control_t` 인자는 기존 호출 호환을 위해 남겼지만 사용하지 않는다. 원본 모델에 없던 joint 기반 T 게이트를 추가하지 않았다. 초기화는 worker 실행 전에 하고 실행 중 재초기화하지 않는다.

## 원본에서 보존한 계산

1. 고정 center와 630개 점·점 ID를 로드한다. QR별 NCC로 정수 이동을 찾는다.
2. 각 QR를 별도로 잘라 sigma 0.8 blur, LK15 정방향·역방향 추적을 수행한다.
3. 17×17 창의 cubic spline 영상 샘플링과 밝기 gain/offset을 포함한 GN을 최대 12회 계산한다.
4. 고정 내부 536점만 선택하고, center를 같은 추적기로 측정한 변위를 차감한다.
5. 위치 다항식 `[1,u,v,u²,uv,v²]`에 dx/dy를 최소제곱 적합해 P12 특징을 만든다.
6. 원본 whitening 행렬과 6개 학습 절편 후보에서 8회 곡선 역산을 수행한다. 부호가 맞는 최소 비용 후보를 선택한다.

6개 후보는 센터 사진 6장이나 정답으로 고른 후보가 아니다. 원본의 고정 536점을 유지하므로 validity 진단을 보고 사진마다 점을 제거하지 않는다. 재학습·새 파라미터 튜닝 없이 원본 `model.json`을 그대로 사용했다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `src/control.cpp` | 초기화·추론 공개 함수, 결과/진단 반환 |
| `src/tracker.cpp` | 영상 검사, NCC/LK/GN, center 차감, P12 |
| `src/spline.cpp` | SciPy cubic spline 대응 수치 구현 |
| `src/model.cpp` | 원본 파라미터 읽기, 6후보 비용과 값 계산 |
| `include/control_internal.hpp` | 내부 선언 |
| `assets/center.bmp` | 고정 기준 영상, SHA-256 검사 |
| `assets/points.json` | 630점 좌표·QR 번호·고정 536점 마스크 |
| `assets/model.json` | 원본 최종 모델 그대로 |
| `assets/provenance.json` | 원본 출처·해시 |
| `validation/` | Python 기준 특징·예측, 원본 fold 모델. 검증 전용 |
| `tool/validate_control.cpp` | 원본 사진부터 C++ 측정 재실행, 수치 대응/전수 평가 |
| `docs/FUNCTION_FLOW.md` | 함수별 입력·처리·출력 |

실행에는 `assets/`와 C++ 라이브러리 및 OpenCV DLL이 필요하다. `validation/`은 검증할 때만 사용한다. Python·NPY·저장소 밖 실험 폴더는 실행에 필요하지 않다.

## 빌드와 검증

저장소 루트 PowerShell에서 실행한다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build --config Release -- /m:1 /nr:false
ctest --test-dir build -C Release --output-on-failure
.\build\Release\main.exe control "data/clean/clean_20261014/Y/Y+0.50.bmp"
.\build\control\Release\validate_control.exe `
  "experiments/hye/00000_final_Y_T control" data/clean/clean_20261014 `
  "experiments/hye/00000_final_Y_T control/results/NEW_VALIDATION_DIRECTORY"
```

control만 쓸 때는 `semes_control`, joint만 쓸 때는 `semes_joint`, 두 모듈과 기존 `alignment()`를 함께 쓸 때는 `semes_alignment`를 링크한다. `.cpp`를 include하지 않는다. 초기화에는 모델 자료 디렉터리, 추론에는 현재 영상을 전달한다. 서비스 연결은 포함하지 않는다.

## 검증 결과의 의미

이번 결과는 [results/native_qr536_20261028_001](results/native_qr536_20261028_001/)에 저장했다. `summary.json`에 C++ 실측 수치, `predictions.csv`에280장×2방식의 전체 행을 남겼다.

전수 모델 재대입과 당시 고정4-fold 모델 평가를 구분한다. 검증 도구에서만 각 시험 사진에 대응하는 원래 fold 모델을 선택한다. 실제 `control()`은 최종 모델1개만 사용한다. 이번 평가는 언어 이식의 재현 검사이며 이미 개발에 사용한 사진의 성적이다.

`measurement_valid`는 유한한 측정·계산이 완료됐다는 뜻이다. `valid_points`는 고정536점 중 진단상 유효한 점 수다. `model_cost`와 `normalized_cost_margin`은 오차 확률이 아니다. `pose_stability_assessed=false`이며 실제T≈0이나 안정성을 이 모델만으로 보증하지 않는다. 2차 자세 조건 확인과 이동·재촬영은 호출 측의 책임이다.
