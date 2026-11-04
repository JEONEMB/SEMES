# 새 실험 코드 평가 감사

확인 파일: `zero_experiment.py`, `lattice_experiment.py`, `modeling/experiment.py`, `modeling/unified.py`, `validate_calibrated_runs.py`, `direct_alignment/direct_pair.py`.

## 확인한 정상 동작

- `zero_experiment.correct_labels`는 outer train에 있는 영상의 값·평균 QR 이동만으로 보정 모델을 맞춘다. 시험 라벨은 raw 명령값으로 유지한다. 기존 v7의 세션 홀드아웃 보정 누출을 반복하지 않는다.
- `zero_experiment.fit`의 column-vector whitening은 Cholesky(inv(Corr)).T를 표준편차로 나눈 형태다. `modeling.MotionModel`의 row-vector whitening은 `diag(1/sd) @ eigvectors @ diag(1/sqrt(eigenvalues))`다. 두 방향 모두 목적함수의 Mahalanobis norm에 맞는다.
- `modeling/experiment.py`의 후보 선택은 outer train 안의 inner folds에서 수행한다. 정규화, 공분산, 점 순위, quality 가중, SVM은 해당 fit의 학습 데이터에만 의존한다. center 영상에서 후보점을 선택하는 것은 이번 입력 조건에 맞는다.
- `lattice_experiment.py`의 nested 선택도 outer train 안에서 후보를 비교한다. `fixed_metrics.csv`의 최고 행을 다시 골라 독립 검증 성적이라고 해석하면 안 되지만, nested 결과 자체의 코드 흐름에서 직접 시험 라벨 누출은 발견하지 못했다.
- 0.09 strict 지표는 축 및 부호를 모두 맞히고 오차가 허용값 안이어야 통과한다. 2축 벡터 지표는 잘못 선택된 축에 생긴 오차까지 반영한다. 조건부 오차는 축을 맞힌 사진만이라는 의미를 유지해야 한다.

## 결과 해석에 반드시 남겨야 할 범위

`oracle_zero`는 추론 시 시험 사진의 실제 세션 ID를 사용하여 세션 절편을 선택한다. 세션 ID에는 실제 축/부호 정보가 포함된다. 따라서 이는 **정답 세션을 아는 진단 실험**이며 2장만 받는 배포 방식의 결과에 섞을 수 없다. 이 기능을 제외한 일반 추론 경로에서는 시험 세션을 사용하지 않는다.

`inferred_session`은 시험 세션을 정답으로 받지는 않는다. 학습 세션별 절편을 후보로 만들고 이미지 적합 비용으로 후보를 선택한다. 그러나 각 후보의 부호를 학습 세션의 부호에 제한한다. T+ 또는 T− 세션 전체를 뺀 검증에서는 빠진 T 부호를 낼 후보 자체가 없다. 그때의 실패는 이 코드의 학습 영역 제한이며, 알려진 세션 내 높은 정확도가 미관측 세션 일반화까지 입증하지 않는다. 세션별 절편으로 촬영조건을 암기할 가능성도 함께 보고해야 한다.

`grid='nonzero'`는 입력이 정지가 아닌 단일축 명령 격자(Y 0.05, T 0.01)에 놓인다는 추가 가정이다. 격자의 간격은 사용자가 준 이동 간격으로 설정되며 테스트 정답값을 직접 읽지는 않는다. 그러나 현재 물리 위치가 center 기준 같은 격자 위라는 보장은 백래시/0점 문제와 별개다. 임의 연속 이동, 혼합축, 정지 영상을 처리하는 모델로 확장해서 해석하면 안 된다. 격자 prior가 없는 연속 모델의 성적을 나란히 남기는 것이 적절하다.

연속/격자 모델은 모두 하나의 중심 예측값을 반환한다. Mahalanobis 비용, 비용비, 영상 MSE는 정확도 확률도 아니고 물리적 오차구간도 아니다. 추론 시 신뢰도/오차구간을 내보낼 경우 학습 내부 out-of-fold 예측으로 calibration/잔차분위수를 구하고, 실제 미관측 자료에서 coverage를 따로 평가해야 한다. 분류 오답을 제외한 오차분위수는 “축이 맞다는 조건의 오차”로 표시해야 한다.

원시 filename은 검증에 사용할 유일한 가용 정답이며 독립 외부 계측은 아니다. 0점 보정 가정에 맞춰 얻은 작은 오차를 장비의 절대 정확도로 바꿔 말할 수 없다. 여섯 구간은 축/부호마다 반복 세션이 골고루 있는 설계가 아니므로 일반화 실험이 제한적이다.

## 경미한 수치 주의

`zero_experiment.fit`의 residual covariance 분모는 모드와 관계없이 선형에서 n−6, 2차에서 n−8이다. 실제 회귀 자유도는 center/free/session 모드에 따라 다르다. 현재 공유 covariance의 전체 배율만 바뀌므로 분류 및 최적 이동량은 바뀌지 않는다. 향후 chi-square 해석, 확률, 신뢰구간 또는 covariance 크기를 쓰려면 각 축 설계행렬의 rank를 합산하여 residual dof를 계산해야 한다.

직접 정합의 최종 결과(`direct_p12_newinit.npz`)는 독립 초기화로 계산한 X를 사용한다. `X_oldinit_same_support`는 초기값에 따른 수렴 차이를 확인하는 진단 열이며 배포 추론의 입력이 아니다. 독립 `DirectPairExtractor(center).extract(current)`의 T+0.01 재실행은 배치 결과와 최대 계수 차이 2.47e−17이었다. 모든 281개 X가 유한값이고, center 자체 측정의 최대 절댓값은 2.77e−15, 반복 횟수 최대8이었다.

## 통합 nested 코드 추가 확인

`modeling/unified.py`는 측정법 선택까지 outer train 내부에서 수행한다. 데이터셋 간 사진 이름 정렬을 검증하며, direct 측정의 가상75좌표는 이미 얻은 P12 계수를 다시 P12로 표현하는 입력 경로다. 75개의 새로운 독립 관측을 확보했다고 주장하지 않는 한 적절하다. 최종 배포 모델 선택에 쓴 전체자료 CV 성적은 독립 시험 성적이 아니라는 metadata도 포함한다.

`independent_data()`는 현재 validity 마스크를 확인하지 않는다. 감사 시 lk31, lk51, gn17, gn31, gn17_spread, gn17_spread14의 모든 QR 추적이 valid이고 모든 좌표가 유한값임을 실제로 확인했다. 따라서 이번 성적에는 영향이 없지만 향후 추적 실패 데이터에는 assert 또는 영상별 masked fit이 필요하다.

통합 `MotionModel`의 `grid=True`는 0을 포함한 격자를 사용한다. 별도 `lattice_experiment`의 `grid='nonzero'`는 0을 제외한다. 같은 격자 모델로 이름을 뭉뚱그리지 않고 구분해야 한다.

`validate_calibrated_runs.py`의 20개 seed 재분할은 동일280장을 재사용한다. 5,600개의 독립 촬영을 검증한 것이 아니다. `blocked_within_session5`는 한 세션의 연속 값 블록을 빼지만 같은 세션의 다른 블록으로 절편을 학습하므로 “새로운 0점/세션을 검증했다”는 뜻이 아니다. 파일 머리말처럼 개발 후 선택한 설정의 스트레스 테스트로 해석해야 한다.

검토한 흐름에서 직접 시험 라벨 누출이나 whitening 방향 오류는 발견하지 못했다. 이 결론은 코드의 해당 평가 흐름에 관한 것이며 데이터 수집 설계의 독립성 또는 실제 장비 성능 보증을 뜻하지 않는다.
