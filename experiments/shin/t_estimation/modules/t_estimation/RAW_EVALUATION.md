# T 단축 추정: 원본 재계측 및 최종 평가

**Y가 기준 위치에 고정된 T 단독 촬영 조건에서는 개발 검증 최대오차 0.471 tick을 얻었다. Y가 변하는 조건을 포함하면 최대오차 1 tick 목표는 미달성이다.** 두 조건의 모델과 성적을 구분한다.

## 데이터 무결성과 분할

사용자가 전달한 기본 저장소의 `data/SSAFY Test Image Set`에서 BMP 575장을 전수 계측했다. 원본은 수정하지 않았다.

| 용도 | 구성 | 장수 |
| --- | --- | ---: |
| 기존 개발 자료 | T 단독 200 + Y 단독 80 + 혼합 19좌표×3회 | 337 |
| 기준 영상 | 기존 center 1 + 신규 세션 center 3 | 4 |
| 추가 촬영 평가 | Y=.05, T=.04~.82 중 기존 혼합 자료에 없던 78좌표×3회 | 234 |

추가 234장은 사용자가 실제 추가 촬영 원본이라고 확인했다. 복사 시각으로 촬영 날짜나 시간 순서를 추정하지 않았다. 고정 center1을 영상 참조로 사용하고 나머지 기준 사진을 회귀 학습에 추가하지 않았다.

계측 실패 0/575, 동일 파일 중복 0건이다. 계측 후 575장의 SHA256을 다시 계산했으며 변화 0건, 과거 inventory의 341장과 불일치 0건이었다. 근거는 `results-raw575/audit.csv`와 `integrity.json`이다. 기존 혼합 57장도 원본에서 별도 재실행했으며, 보존된 특징 및 보류 예측과 최대차이 0을 확인했다(`results-raw57/validation.json`).

T의 1 tick은 **명령 라벨 0.01**이다. 정답은 파일명의 명목 T를 사용한다. sidecar Actual과 묵시적으로 섞지 않는다. degree는 기존 문서의 가정이므로 물리 단위 인증값으로 표현하지 않는다. 추론 API는 정답·파일명·반복 번호·Y값을 받지 않는다.

## 공통 관측과 모델 선택

기존 Lanczos-3의 QR 내부 630점 및 허용된 상·하 선 10특징을 보존했다. 추가로 QR별 국소 affine 변위 기울기 4개씩, 총 12개를 계산했다. 기준 QR 좌표를 중심화하고 100px 단위로 정규화한 x,y에 현재-기준 dx,dy를 각각 적합한다. 평균 이동에서 사라지는 내부 모양 변화로 Y 간섭과 T 반응을 구분할 수 있는지 시험한 것이다. 모든 630점을 유지하며 T각을 영상 affine 회전각으로 대체하지 않는다.

`Features`는 22개로 확장했지만 기존 표현 0은 앞의 10개만 사용하므로 기존 모델도 읽는다. 표현 6은 전체 22개, 표현 7은 기울기 12개다. 실제 이미지 추론과 계측 도구가 같은 특징 함수를 호출하며 Y 회귀를 실행하지 않는다.

동일 (Y,T) 명령 좌표를 정수 tick 문자열로 만들고 FNV-1a 해시%5로 개발 fold를 고정한다. 같은 좌표의 반복 3장은 항상 같은 fold다. 외부 5-fold의 각 학습 집합에서 나머지 4-fold 내부 검증으로 표현·차수·alpha를 선택한다. 정규화, 다항식 중심, 회귀계수는 해당 학습 집합에서만 추정한다. Y는 동일 좌표 그룹을 만드는 평가 메타데이터로만 쓰며 추론 특징이 아니다.

- mode 0: 기존 10특징에서 파생한 6개 표현의 선형 회귀.
- mode 1: 전체 22특징 및 기울기 12특징의 선형 후보 추가.
- mode 2: 기존 6개 표현의 2차·교차항 후보 추가.
- 확장 실험의 mode 3: 평균 QR 이동 2특징 및 차이/선 기하 5특징의 3차항 후보 추가.

alpha는 0,.001,.01,.1,1,10,100이다. 내부 선택 점수는 최대 절대오차(tick)+.1×RMSE(tick)이다. 최종 절차 선택 자체도 개발 결과를 본 결정이므로 OOF를 새로운 독립 시험으로 부르지 않는다. 전체 학습 모델의 재대입과 OOF 성적을 구분한다.

## Y 변동을 포함한 T 추정

기존 337장 안에서 mode 0~2를 비교했다. 공간 기울기와 비선형 후보를 추가해도 외부 성적은 같았고 mode 0을 선택했다.

| 평가 자료 | MAE(tick) | RMSE(tick) | 최대(tick) | 1 tick 이내 |
| --- | ---: | ---: | ---: | ---: |
| 기존 337장 중첩 OOF | .719823 | .929140 | 3.786583 | 249/337 |
| └ T 단독 200장 | .649426 | .815440 | 2.217165 | 156/200 |
| └ Y 단독 80장(T 정답 0) | .765764 | .965778 | 2.292892 | 57/80 |
| └ 기존 혼합 57장 | .902349 | 1.209073 | 3.786583 | 36/57 |
| **고정 모델의 추가 234장 1회 평가** | **1.408944** | **1.639710** | **3.467762** | **81/234** |

추가 평가의 라벨 단위 MAE=.014089437, RMSE=.016397099, 최대=.034677619다. 추가 사진은 모델 선택에서 제외했고, 모델을 커밋 `57b22e9`에서 고정한 뒤 한 번 평가했다. 모델 SHA256은 평가 전후 `ada43c98528d7b33994ec0d45977105d0ca4f3d402392348aa62faabfe6625bd`로 같았다. 이 자료는 학습에 미사용인 촬영 자료지만 다른 장비·세션·전체 Y/T 범위의 인증 시험은 아니다.

실패 153장은 `results-dataset/evaluation_failures.csv`에 있다. 최악은 Y=.05/T=.27의 (1) 사진으로 -3.467762 tick이다. T=.34의 (1)은 -3.409376, T=.35의 (3)은 -3.291773 tick이다. 실제 CLI로 .27 사진을 입력한 결과 .23532238143795717은 배치 예측과 정확히 일치했다.

T=.30~.39의 평균 오차는 -2.472196 tick으로, 중간 구간을 낮게 추정하는 경향이 나타났다. 기존 Y=.05 혼합 보정점에는 T=.1과 1 사이가 비어 있었다. 이 구간의 모델 형태를 제약하기 어려웠다는 해석은 가능하지만 물리 원인을 기구 coupling·드리프트·측정 편향 중 하나로 확정할 수는 없다. 구간별 사후 진단은 `evaluation_bias_by_T.csv`에 보존했다.

## 실패 후 추가 개선

234장 평가 결과를 확인한 이후에는 이 자료도 개발 자료로 취급했다. 총 571장(기준 4장 제외)으로 같은 중첩 좌표 보류를 수행하고 3차 후보까지 추가했다. 이후 성적은 독립 평가가 아니다.

선택된 확장 절차의 MAE=.712829 tick, RMSE=.906292 tick, 최대=3.849279 tick, 통과 428/571이었다. 추가 234장 부분의 개발 OOF는 MAE=.650014 tick, 최대=2.256821 tick, 통과 186/234였으나 전체 목표는 여전히 실패했다. 앞선 234장 시험과 학습·평가 조건이 달라졌으므로 동일 조건의 개선율로 해석하지 않는다. 최악 표본은 Y=1/T=1의 (2), Y=1/T=2의 (3)이었다. 모델과 모든 외부 예측은 `results-expanded/`에 비교용으로 보존했다.

## Y 고정 조건을 활용한 T 전용 모델

**Y가 기준에 고정된 운영 조건**을 별도로 시험했다. T 촬영 200장만으로 보정하고, 각 학습 fold 내부에서 특징·차수·alpha를 다시 선택했다. Y 간섭을 상쇄하는 민감한 계수를 학습할 필요가 줄어드는 조건이다. 이는 Y가 미지인 상황에서 Y를 0으로 가정해도 된다는 뜻이 아니다.

| T200장 중첩 개발 검증 | MAE(tick) | RMSE(tick) | 최대(tick) | 통과 |
| --- | ---: | ---: | ---: | ---: |
| 선형 후보(mode 0) | .150258 | .183396 | .490375 | 200/200 |
| 공간 기울기 후보 추가(mode 1) | .148405 | .182782 | .490375 | 200/200 |
| **비선형 후보 추가(mode 2)** | **.120406** | **.153658** | **.471386** | **200/200** |

최종 T 전용 모델의 라벨 단위 MAE=.0012040645, RMSE=.0015365847, 최대=.0047138567이다. 전체 200장 보정에서는 QR6특징의 2차 회귀, alpha 0을 선택했다. OOF의 각 fold는 각자 내부에서 선택한 모델을 사용하므로 배포 모델 하나의 재대입 성적과 다르다.

이 200장은 과거 분석에 노출된 개발 자료다. **별도 신규 T 단독 촬영으로 조건부 목표를 인증한 것은 아니다.** 연속 범위나 다른 장비·QR·세션에 최대오차를 보장하지 않는다.

Y 고정 조건을 벗어난 적용도 확인했다. 이 모델의 최대 T오차는 Y 단독 80장에서 35.911 tick, 기존 혼합 57장에서 44.490 tick, 추가 234장에서 4.039 tick이었다(`results-t-axis/stress_metrics.csv`). 따라서 단축 조건의 통과를 혼합 조건의 통과로 확대하면 안 된다.

## 실행 모델과 검증

| 모델 파일 | 적용 조건 | 상태 |
| --- | --- | --- |
| `assets/t_axis_only_model.json` | Y가 기준에 고정, T=-1~1 명령 범위 | 개발 OOF 최대 .471 tick |
| `assets/model.json` | Y 변동을 포함한 기존 337장 보정 | 추가 234장 최대 3.468 tick, 목표 미달 |
| `results-expanded/frozen_model.json` | 571장으로 확장한 개발 비교 | 독립 후속 시험 없음, 목표 미달 |

T 전용 모델은 API·CLI에서 `requires_y_fixed_at_reference=true`를 반환한다. 입력 영상만으로 Y 고정을 자동 인증하지 않으므로 호출자가 운영 조건을 보장해야 한다. `outside_calibration_range`는 모델의 학습 T 범위에 대한 진단이며 정확도 보증이 아니다. 출력 clipping이나 명령 tick 반올림은 하지 않는다.

T+.50 원본의 T 전용 CLI 출력은 t=.50037466916630047이었다. 이는 학습 사진의 실행 확인이며 OOF와 별개다. 두 CLI 실행에서 입력 픽셀 불변을 검사했다. CTest 2/2가 통과했으며, 기존 입력 오류·학습 격리 검사에 공간 기울기 복원과 3차 기저 회귀 검사를 추가했다.

## 재현 명령

저장소 루트 PowerShell, C++17/MSVC v141/OpenCV3.4.15:

```powershell
cmake --build build-t-only --config Release --parallel 1
ctest --test-dir build-t-only -C Release --output-on-failure
.\build-t-only\modules\t_estimation\Release\t_extract.exe "IMAGE_ROOT" modules/qrwithbar_lanczos "NEW_OBSERVATION_DIRECTORY"
.\modules\t_estimation\tools\audit.ps1 -ImageRoot "IMAGE_ROOT" -ResultDirectory "NEW_OBSERVATION_DIRECTORY" -HistoricalInventory docs/offaxis-eda/sampling.csv
.\build-t-only\modules\t_estimation\Release\t_dataset.exe train modules/t_estimation/results-raw575/observations.csv "NEW_337_RESULT_DIRECTORY"
# 모델을 고정한 뒤 추가 자료를 한 번 평가한다.
.\build-t-only\modules\t_estimation\Release\t_dataset.exe evaluate modules/t_estimation/results-raw575/observations.csv "NEW_337_RESULT_DIRECTORY"
.\build-t-only\modules\t_estimation\Release\t_dataset.exe expand modules/t_estimation/results-raw575/observations.csv "NEW_571_RESULT_DIRECTORY"
.\build-t-only\modules\t_estimation\Release\t_dataset.exe axis modules/t_estimation/results-raw575/observations.csv "NEW_T200_RESULT_DIRECTORY"
.\build-t-only\modules\t_estimation\Release\t_dataset.exe stress modules/t_estimation/results-raw575/observations.csv "NEW_T200_RESULT_DIRECTORY"
.\modules\t_estimation\tools\summarize.ps1 -ResultDirectory "RESULT_DIRECTORY"
# Y가 실제로 기준에 고정된 T 단독 조건
.\build-t-only\modules\t_estimation\Release\t_estimate.exe "IMAGE.bmp" modules/qrwithbar_lanczos/assets modules/t_estimation/assets/t_axis_only_model.json
# Y 변동을 포함한 보정 모델
.\build-t-only\modules\t_estimation\Release\t_estimate.exe "IMAGE.bmp" modules/qrwithbar_lanczos/assets modules/t_estimation/assets/model.json
```

평가 도구는 기존 1회 평가 파일을 덮어쓰지 않는다. 원본 BMP 및 TXT는 코드 커밋에 포함하지 않는다. 필요한 전달물은 exe/OpenCV DLL, QR reference·template 자산, 선택한 T 모델이다. 추론에 Python이나 외부 개인 실험 경로가 필요하지 않다.

Push와 Atlassian 발행은 수행하지 않았다. 추후 push 전 활성 스프린트의 `[신윤석]` Story/동급 Task와 Fibonacci story point를 설정해야 한다.
