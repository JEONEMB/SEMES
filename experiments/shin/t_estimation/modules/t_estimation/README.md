> 후속 원본575장 재계측과 추가 촬영234장 평가는 [RAW_EVALUATION.md](RAW_EVALUATION.md)에 기록한다. 아래는 최초57장 캐시 실험 이력이다. 현재 assets/model.json은337장 보정 모델이며, Y고정 T전용 모델은 assets/t_axis_only_model.json이다. 최초57장 모델은 results-final/deployment.json에 보존한다.

# T축 단독 추정 실험

## 결과와 범위

**최대 절대오차 1 tick 목표는 미달성이다.** T만 출력하는 별도 C++17 라이브러리와 실행 파일을 구현하고, T 전용 특징 선택·비선형 회귀·강건 손실·저차원 회귀를 비교했다. 기존 동시추정 결과보다 개선됐다고 주장하지 않는다.

기준 브랜치 `feature/shin/QRwithBar`의 `55c5ebe`에서 `feature/shin/T-estimation`을 생성했다. 기본 checkout은 다른 팀원의 `feature/hye/alignment-service`였으므로 기존 QRwithBar 작업 폴더에서 작업했다. 기본 checkout의 미추적 실험은 수정하지 않았다. Git 규칙의 기본 develop 분기보다 이번 사용자의 명시적 기준 브랜치 지시를 우선했다.

| 개발용 외부 좌표 보류 절차 | MAE | RMSE | 최대 절대오차 | MAE / RMSE / 최대 (tick) | 1 tick 이내 |
| --- | ---: | ---: | ---: | --- | ---: |
| phase 0: T 전용 선형 특징·정규화 선택 | 0.008000169 | 0.010513519 | 0.027060327 | 0.800017 / 1.051352 / 2.706033 | 38/57 |
| phase 1: 2차·강건 손실 추가 | 0.008196440 | 0.010812980 | 0.028191546 | 0.819644 / 1.081298 / 2.819155 | 37/57 |
| phase 2: 주성분 회귀 추가 | 0.008196440 | 0.010812980 | 0.028191546 | 0.819644 / 1.081298 / 2.819155 | 37/57 |

최종 `assets/model.json`은 가장 보수적인 phase 0 절차로 전체 57장을 보정한 모델이다. 추가 후보를 포함한 모델은 `results-final/deployment_phase1.json`, `deployment_phase2.json`에 비교용으로 보존했다. 외부 성적을 보고 phase 0을 선택했으므로 이 선택 자체는 개발 의사결정이며, 선택 후 성적을 독립 시험이라고 해석하면 안 된다. 전체 학습 모델의 재대입은 위 표의 OOF 성적과 별개이다.

## T tick과 정답

1 tick = **T 명령 라벨 0.01**이다. 근거는 `docs/offaxis-eda/README.md`의 T=0.01,0.02,0.03 표와 `docs/offaxis-eda/zero-position-v3/README.md`, 기존 `modules/qrwithbar_lanczos/tools/train.cpp`의 T/.01 정규화, manifest 및 실제 경로명이다. T는 카메라 영상의 회전각이 아니라 기존 장비 명령의 T축이다. 문서에서 degree는 가정이라고 명시했으므로 물리적 도 단위의 인증값으로 부르지 않는다. 출력 부호는 학습 명령 라벨의 부호를 따른다.

이번 학습 정답은 파일명/폴더명의 명목 T를 파싱한다. 기존 manifest의 sidecar 상대 Actual 정답은 명목값과 약 1e-6 차이가 있으므로 그 값을 묵시적으로 혼합하지 않았다. filename은 학습/평가 도구의 라벨 출처에만 쓰고 이미지 추론 API에는 전달하지 않는다. 반올림·최근접 명령 스냅·반복 번호별 보정은 사용하지 않는다.

## 기존 T가 상대적으로 정확했던 이유

기존 분석의 공통 가로 감도는 Y 약 14.375964 px/mm, T 약 80.539419 px/degree 가정이다. 명목 tick으로 환산하면 약 0.719 px/Y tick, 0.805 px/T tick이다. 물리 단위의 오차 숫자만 직접 비교하면 단위·tick 차이가 섞인다. 기존 성적은 Y MAE 0.044733mm=0.895tick, 최대 3.131tick, T MAE 0.800tick, 최대 2.706tick으로 T가 약간 낫다. 단순히 감도가 5배이므로 5배 정확하다는 해석은 틀리다.

QR의 공통 x 이동은 두 축 모두에 반응한다. Y를 출력하지 않아도 미지의 Y가 T 관측에 영향을 준다. Y=0이라는 추가 가정은 두지 않았다. QR 간 차이와 선의 위치/기울기를 포함해 Y로 인한 간섭을 줄일 후보를 비교했다. 기존 다출력 선형 최소제곱은 출력 열마다 독립적으로 풀리므로, Y 열을 삭제하는 것만으로 T 계수가 좋아지지는 않는다. 이번에는 T만으로 모델을 선택했으나 phase 0에서 모든 외부 fold가 같은 10특징/무정규화 선형식을 선택했고 기존 T OOF와 사실상 같은 결과를 얻었다.

## 처리와 추가 개선 시도

- 영상 측정: 고정 center, QR 내부630점과 허용된 상·하 선을 사용하는 기존 Lanczos-3 측정. 별도 T 추론은 `qrwithbar::Estimator::estimate()`나 Y 회귀를 호출하지 않는다.
- 특징 후보: 전체10개, QR6개, QR dx3개, QR 평균 dx/dy, 공통 이동+QR간 차이+선 위치, 공통 이동을 없앤 QR차이+선 기울기/간격.
- phase 0: 위 표현의 표준화 선형 ridge. alpha는 0,.001,.01,.1,1,10,100.
- phase 1: 모든 2차 항과 교차항 후보 추가. 선형 후보에는 잔차 역가중 IRLS와 큰 잔차 가중 IRLS도 추가한다. IRLS는 12회 절반 감쇠, 잔차 하한 .25tick, 큰 잔차 상한20tick이다. 강건 후보 alpha는 0,.1,1. 해당 고정 반복은 최적해 보증이 아니다.
- phase 2: 전체10특징의 표준화 SVD 상위2/3/4/6/8성분 회귀 후보 추가. 저분산 관측 잡음을 제거할 수 있는지 비교했으나 내부 검증에서 최종 선택되지 않았다.
- 선택 점수는 내부 보류의 `최대 절대오차(tick) + 0.1 * RMSE(tick)`이다. Y 정답이나 Y 예측은 적합·선택에 사용하지 않는다.

## 누출 방지와 평가 한계

외부는 19좌표 leave-one-coordinate-out이다. 같은 (Y,T)의 반복3장은 함께 보류한다. 54장으로 내부 좌표 보류를 다시 수행해 표현·차수·손실·rank·alpha를 선택한 뒤, 외부3장을 한 번 추론한다. 평균/표준편차, 2차항 중심, 회귀계수, IRLS 가중치와 SVD도 각 학습 분할에서만 계산한다. 외부 모델57개를 모두 저장했다. fold번호는 평가 모델 선택에만 쓰고 배포 API에는 존재하지 않는다.

이 57장은 과거 개발에 이미 노출되었고, 이번에도 성적을 보며 후보를 추가했다. 직접 시험 라벨로 적합하는 누출은 막았지만 적응적 평가 오염은 남는다. 고정 center/reference는 허용된 보정 입력이다. 새로운 세션·장치·QR·음의 T·미촬영 좌표에 대한 독립 검증이나 실제 로봇 이동 검증은 하지 않았다.

현재 작업 폴더와 기본 저장소의 data에는 원본57장과 T 단독200장이 없었다. 따라서 **이번 정량 평가는 보존된 Lanczos 측정 캐시를 입력으로 수행했다.** 원본의 해시를 이번에 다시 검사하거나 57장 영상 측정을 재실행한 것으로 표현하지 않는다. 영상 경로는 포함된 기준 PNG로 실행·입력 불변을 확인했다. 참고용 과거 분석의 280/341장 성적은 이번 모델의 성적으로 합산하지 않았다.

사용한 캐시 SHA256:
- `manifest.csv`: 04cd6a345a6c01031750e4aaef113b1b1cb69624c3d50bb17a3b6842f7254ab3
- `comparison_features.csv`: 796d34e784f162ea7a93252b7edff8d292f63a46b4c722398274c88dfba40f95

57개 고유 경로의 Lanczos 행을 일대일 확인하며 중복·누락·실패 계측이면 실행을 중단한다. 캐시에는 계측 실패0건이다. 실패나 보류를 분모에서 임의 삭제하지 않는다.

## 실패 분석

phase 0에서 19장이1 tick을 넘는다. 전수 목록은 `results-final/failures.csv`, 좌표별 편향과 반복 변동은 `group_diagnostics.csv`다.

| 실패 사진 | 오차(tick) |
| --- | ---: |
| Y+0.05 T+1.00 (3) | -2.706033 |
| Y+0.05 T+0.03 (2) | +2.693635 |
| Y+1.00 T+2.00 (3) | -2.058666 |
| Y+1.00 T+0.10 (1) | +1.951907 |
| Y+1.00 T+1.00 (2) | +1.908653 |

T=1/Y=.05 그룹의 평균 편향은 -1.665tick이며 반복 예측 범위는1.976tick이다. T=.03/Y=.05는 평균 편향+1.369tick, 반복 범위2.518tick이다. group7은 평균 편향이 -0.104tick으로 작아도 반복 범위2.625tick이다. 좌표별 모델 편향과 반복 관측 변동을 함께 개선해야 한다. 이 값은 모델 예측의 편차이며 실제 장비 오차의 물리적 하한은 아니다.

과거 분석은 반복 공통 dx residual RMS .135712px, 개별점 .072769px를 보고했다. 공통 성분은 QR 점 수를 늘려 단순 평균해도 제거되지 않는다. 촬영 순서·기구·조명·측정 중 어떤 원인인지는 인증할 수 없다. 잔차 가중 강화는 일부 편향에 과도하게 반응했고 외부 최대오차가 악화됐다. 2차항과 저차원 제한도 내부에서 채택되지 않았다. 실패 좌표를 직접 보정표로 기억시키는 대신 원본 영상에서 공간별 관측·선 잔차·세션 차이를 검증할 다음 단계가 필요하다.

## 사용 API와 실행

`include/t_estimator.hpp`의 `t_only::Estimator(calibration_directory, model_file)`를 한 번 만들고 `estimate(image)`를 호출한다. 입력은3840x2160,uint8,1/3/4채널이며 원본을 변경하지 않는다. 출력은 T 명령 라벨값 `t`, `ticks`,10특징,보정 T 범위 밖 진단이다. Y 필드는 없다. 예외로 측정/모델/입력 실패를 알리며0으로 숨기지 않는다. 출력은 로봇 명령이 아니다. T 범위 [.01,2] 진단은 Y/영상 특징의 범위나 정확도를 보증하지 않으며 예측을 clipping하지 않는다.

저장소 루트 PowerShell, MSVC v141/OpenCV3.4.15:

```powershell
cmake -S . -B build-t-only -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
cmake --build build-t-only --config Release --parallel 1
ctest --test-dir build-t-only -C Release --output-on-failure
.\build-t-only\modules\t_estimation\Release\t_train.exe modules/qrwithbar_lanczos modules/t_estimation/results-reproduced
.\build-t-only\modules\t_estimation\Release\t_estimate.exe "IMAGE.bmp" modules/qrwithbar_lanczos/assets modules/t_estimation/assets/model.json
```

기존 `main.exe`와 기존 alignment API는 동시추정 경로로 보존했다. **T 단독 실행은 `t_estimate.exe`를 사용한다.** 전달물은 exe/OpenCV DLL, 기존 QR calibration assets(reference와 template3개),새 T model.json이다. 기존 joint model.json은 T 추론에 쓰지 않는다. 추론은 저장소 밖 실험 코드/Python에 의존하지 않는다.

전체 Release 빌드 및 CTest 2/2 통과. T 검사에는 외부 라벨 변경에 대한 fit 격리,세 scalar loss,빈 학습 집합,빈/잘못된 크기/잘못된 채널 영상 거부가 포함된다. 학습 도구는 모델 저장/재로딩 후 예측차0을 확인했다. 기준 PNG CLI smoke는 t=.0066558675(.66558675tick),범위 밖 진단true였고 입력 픽셀 불변을 검사했다. 이는 알려진 기준 이미지 실행 확인이며 독립 정확도 성적이 아니다.

## 다음 검증과 Git

원본 데이터 확보 후 T단독200장·Y단독80장·혼합57장을 모두 새 측정해, 혼합과 단축 촬영에 대한 성적을 분리해야 한다. T 단독만 정밀 측정하는 경우에도 Y=0을 보장할지, 미지 Y를 허용할지를 운영 조건으로 확정해야 한다. 실제 별도 촬영은 모델과 선택 절차를 고정하고 중복 검사 후 정답을 분리해 한 번만 평가한다. 현재 목표를 보장하는 실행물로 배포하면 안 된다.

아직 push하지 않았다. 추후 push 전에는 사용자 프로젝트 지침대로 Atlassian Rovo에서 활성 스프린트에 `[신윤석]`으로 시작하는 Story/동급 Task를 만들고 Fibonacci story point를 지정해야 한다. 이번에는 push와 Jira 발행을 수행하지 않았다.
