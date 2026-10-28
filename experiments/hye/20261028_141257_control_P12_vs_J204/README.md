# control의 P12 / J204 재학습 비교

2차에서 기존 12차원 표현 대신 joint의 204차원 표현을 쓰는 것이 도움이 되는지 확인하는 C++ 실험이다. 기존 final 모듈을 교체하지 않는다. 실행 전 [평가 계획](PROTOCOL.md)을 고정했으며 공통 기준은 [ALGORITHM_RULES.md](../../../ALGORITHM_RULES.md)를 따른다.

실측 수치와 해석은 [결과 보고서](REPORT.md)에 정리했다. 주 결과는 [results_141520/summary.csv](results_141520/summary.csv)다.

## 비교 조건

| 후보 | 특징 | 학습식 | 적용 전제 |
|---|---|---|---|
| `Y_GLS_P12` | QR536점의 P12 | Y 전용 1차 전방 모델 + GLS | 호출 측에서 T≈0 확보 |
| `Y_GLS_J204` | joint의 QR54 + 선48 관측 → 204차원 | 같은 Y 전용 GLS | 동일 |
| `YorT_P12` | QR536점의 P12 | 기존 6절편·2차식·부호 게이트 | Y 또는 T 단일축 |
| `YorT_J204` | joint의 204차원 | 같은 6절편·2차식·부호 게이트 | 동일 |

같은 학습식 안에서 P12/J204를 비교한다. GLS와 Y-or-T 사이에서는 학습 자료와 전제가 다르므로 특징만의 차이라고 설명하지 않는다. P12/J204도 단순히 숫자 개수만 늘린 비교는 아니다. QR 추적법과 선 포함 여부까지 다른 표현 전체의 비교다.

두 특징 모두 고정 센터 특징을 차감한다. J204는 기존 `Observer::measure()`와 `feature_map()`을 그대로 링크한다. 사진의 정답·세션·fold는 학습과 채점 도구에서만 읽고 예측 함수에는 전달하지 않는다.

## 학습과 평가

주 비교는 Y80장 OOF다. 기존 고정 4-fold의 Y20장씩을 시험으로 삼고, Y 전용 모델은 나머지 Y60장과 고정 센터1장으로 학습한다. 절편·응답·공분산은 이61장에서만 구한다. GLS 공분산 축소값은0.01로 고정한다.

Y-or-T 모델은 기존 전체280장의 같은 210/70 분할을 사용한다. 이 경우 센터는 영상 기준으로만 사용하고 6후보 학습에는 넣지 않는다. 공분산 축소값은 원본과 같은0.05다. 재학습한 P12의 최종·fold 예측이 원본과1e-6 이내로 일치하고 축도 같은지 자동 확인한다. 저장한20개 모델은 다시 읽어 예측 차1e-12 이하인지 검사한다.

재대입과 OOF는 따로 보고한다. 측정 실패는 통과율 분모에 포함하며, MAE/최대오차는 유효 측정 기준임을 컬럼 이름에 표시한다. 학습 측정이 실패한 사진은 임의로 제외하지 않는다.

20261020의60장은 날짜 간 참고 평가다. 혼합57장/센터3장과 혼합 중 abs(T)<=0.09/나머지를 구분한다. 이 구분은 채점용이며 모델 선택이나 입력에 사용하지 않는다. 실제로 T를 보정하고 새로 촬영한 사진은 아니다.

## 빌드와 실행

저장소 루트 PowerShell에서 실행한다.

```powershell
cmake -S "experiments/hye/20261028_141257_control_P12_vs_J204" -B build-feature-comparison-20261028 -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build-feature-comparison-20261028 --config Release -- /m:1 /nr:false
.\build-feature-comparison-20261028\Release\compare_control.exe `
  data/clean "experiments/hye/00000_final_Y_T control" experiments/hye/00000_final_Y_T_joint `
  "experiments/hye/20261028_141257_control_P12_vs_J204/results_HHMMSS"
```

출력은 존재하지 않는 새 경로여야 한다. 기존 결과를 덮어쓰지 않는다. 측정 캐시로 학습·평가만 재현하려면 마지막에 원래 결과의 `measurements.yml` 경로를 추가한다. 캐시의 영상·모델·기하·선 마스크 SHA-256을 검사한다.

## 구현 흐름

| 파일·함수 | 역할 |
|---|---|
| `include/experiment.hpp` | Sample/Prediction/Model 자료형과 함수 선언 |
| `src/main.cpp` | 자료·고정 fold 준비 → 영상 측정 → fold별 학습 호출 → 채점·저장·시간 측정 |
| `src/models.cpp`의 `train_y()` | 학습 Y와 센터에서 전방 1차식·잔차 공분산·GLS 역산 계수 계산 |
| `src/models.cpp`의 `train_yort()` | 학습 단일축 자료에서 6절편·공유 2차 응답·잔차 공분산 계산 |
| `Model::predict()` | 특징만 받아 추정값 반환. 정답·세션·fold 인자 없음 |
| `Model::save()/load()` | 실험 모델 저장과 다시 읽기 |

원시 J204를 모델에 바로 넣지 않고, 학습 때처럼 고정 센터 특징을 차감한다.

```cpp
// observer는 기존 joint/assets로 초기화한 alignment_detail::Observer
cv::Mat zero = alignment_detail::feature_map(observer.measure(center).values);
cv::Mat x = alignment_detail::feature_map(observer.measure(current).values) - zero;
study::Prediction estimate = loaded_model.predict(x);
```

Y/T는 파일명 명령 기준 추정량이며 T의 물리 단위를 새로 정하지 않는다. 모델은 이동·재촬영·안정성 판단·장치 전송을 수행하지 않는다. Python 런타임과 저장소 밖 실험 경로는 실행에 필요하지 않다.

## 결과 파일

- `summary.csv`: 조건·OOF/재대입·부분집합별 평균/최대오차와 통과 수.
- `predictions.csv`: 사진별 예측·오차·유효성·통과·선택 후보 전체.
- `models/`: 각 조건의 4개 fold 모델과 1개 전수 모델.
- `measurements.yml`: 센터 차감 전 P12/J204, 측정 유효성·시간·영상 해시.
- `timing.csv`: 같은 사진을 번갈아 측정한 P12/J204의 중앙값/p95와 모델 계산 평균 시간. 초기화·디코딩·로봇 시간은 구분한다. 캐시 모드의 초기화0은 재측정하지 않았다는 뜻이다.
- `audit.json`: 학습 수·축소값·원본 P12와의 대응·캐시 사용 여부.
- `final_preservation_before.json`: 실행 전 최종 모듈 해시.

현재 사진은 이미 개발에 사용한 자료다. 결과는 독립 신규 사진 성능이나 실제 센터 복귀 성공률을 뜻하지 않는다.
