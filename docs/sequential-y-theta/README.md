# 평균 제거 후 θ 선추정 → 잔여 평균 이동에서 Y 추정

## 초기화

사용자 요청에 따라 소스와 tracked tree를 shin 28f6759로 복원했다(9d09a7c). 이전 실험은 backup/ytheta-joint-e967026 로컬 브랜치와 기존 커밋 이력에 보존했다. force push하지 않았다. src/, include/, CMakeLists.txt는 shin과 동일하다. 이전 직접정합/카메라 물리 모델은 새 추정기에서 사용하지 않는다.

## 알고리즘

기준 center와 현재 영상의 동일 payload/module ID accepted 점만 대응한다. 평균 이동을 Y 이동으로 간주하지 않는다.

1. d_i = 관측점 - 기준점 - 학습된 세션 기준 차이.
2. e_i = d_i - mean(d). 관측 평균은 θ 추정 목적함수에 들어가지 않는다.
3. θ 단독 자료에서 학습한 b_i*theta + c_i*theta²의 평균 제거 패턴과 e_i를 비교하여 θ를 먼저 추정한다. 방향과 부호를 보존한다. 1차와 2차 모델을 비교했다.
4. 추정 θ의 전체 변위 b_i*theta+c_i*theta²를 계산한다. 이 단계에는 θ의 큰 공통 이동도 포함한다.
5. mean(d - theta변위)를 Y 단독 자료에서 학습한 평균 이동 감도에 투영하여 Y를 구한다. θ 추정 결과에 Y 평균을 다시 피드백하지 않는다.

스칼라 θ는 감쇠 Gauss–Newton의 여러 초기값을 비교한다. 동시에 목적함수의 4차 다항식 도함수의 모든 실근과 경계를 검사하여 이 스칼라 모델에서 국소 최솟값 때문에 잘못된 해가 선택되는 일을 줄였다. θ 범위 [-1,1]deg는 기존 데이터 범위다.

예측에서도 원본 영상 좌표를 사용한다. 비교하는 동일 점 부분집합으로 관측/모델 평균을 둘 다 다시 계산한다. 전체116개 점을 항상 관측하는 것으로 가정하지 않고 누락점은 버린다. 최소24점, QR세개 분포를 요구한다. 모듈/원본 좌표만 사용하므로 K,D는 요구하지 않는다.

학습은 특징점별로 세션 절편, y,t,t²를 동시에 회귀하되 y열은 Y 단독 영상에서만, t/t²는 θ 단독 영상에서만 활성화한다. 추론의 θ 목적함수에는 y열이나 관측 평균을 넣지 않는다. 새 영상 추정 입력에는 실제 y/theta 라벨을 주지 않는다.

## 재수집과 평가

shin의 기존 evaluate.cpp/QR 추출기를 4프로세스, 프로세스당 OpenCV1스레드로 재실행했다. 281장 수집36.9초. 현재 src/include는 shin과 동일하며 manifest.json에 소스 해시를 기록했다.

6개 세션 후보(Y±0.10계열/0.05시작계열, T±)를 사용했다. 세션 기준 차이는 학습 프레임에서만 적합하며 보류 프레임에서 재보정하지 않는다. 보류 시 세션을 알고 있으므로 미지 촬영 세션 영점 문제보다 유리한 조건이다. 파일명은 실측이 아닌 명목 구동값이다. 실제 혼합 운동 영상은 없다.

- interleaved: 위치순 매3번째 보류, 학습184/보류96.
- blocked: 위치순 중앙 연속1/3 보류, 학습186/보류94.
- 중심 기준 영상은 좌표 기준이며 명목 label 학습 프레임으로 넣지 않는다.
- 각 모델116개 특징점 적합, 모든 보류 영상이 대응점 요구조건을 통과했다.

| 분할 / θ모델 | 명목 구동축 | n | Y RMSE mm | θ RMSE deg | 두 축 ±0.09 동시통과 |
|---|---|---:|---:|---:|---:|
| interleaved / 1차 | Y | 28 | 0.9364 | 0.1673 | 1/28 |
| interleaved / 1차 | θ | 68 | 0.9792 | 0.1749 | 4/68 |
| interleaved / 2차 | Y | 28 | 0.9277 | 0.1657 | 1/28 |
| interleaved / 2차 | θ | 68 | 0.8474 | 0.1513 | 4/68 |
| blocked / 1차 | Y | 28 | 0.9208 | 0.1645 | 0/28 |
| blocked / 1차 | θ | 66 | 0.9718 | 0.1734 | 5/66 |
| blocked / 2차 | Y | 28 | 0.8705 | 0.1554 | 0/28 |
| blocked / 2차 | θ | 66 | 0.9814 | 0.1751 | 6/66 |

## 병목 확인: 명목 θ를 알고 있다고 가정한 대조

실제 추정 성능과 구분되는 oracle 진단이다. θ 추정값을 파일명의 명목 θ로 바꾸고 동일 Y 복원만 수행했다. 실제 입력에서는 이 값을 알 수 없으므로 운영 추정기에 사용하지 않는다.

| 분할 / θ모델 | Y구동 Y RMSE mm | θ구동 Y RMSE mm |
|---|---:|---:|
| interleaved / 1차 | 0.00255 | 0.00565 |
| interleaved / 2차 | 0.00255 | 0.00347 |
| blocked / 1차 | 0.00192 | 0.00272 |
| blocked / 2차 | 0.00192 | 0.00388 |

이 대조는 θ의 큰 공통 이동을 빼서 Y를 구하는 계산이 현재 명목 자료에 잘 맞음을 보여준다. 하지만 θ를 평균 제거 상대변형만으로 측정하는 단계의 오차가 Y로 크게 전파된다. 2차 항이나 스칼라 최적화로 이번 자료에서 이 병목은 해소되지 않았다. 이 결과가 모든 상대변형 측정 방법의 불가능성을 뜻하지는 않는다. 현재 shin 코너 특징점과 이 모델의 결과다.

## 검증과 사용 상태

5개 unittest: 혼합12조합 무잡음 복원, 임의의 큰 공통 이동 추가에 대한 θ 불변성, 누락점 부분집합 평균 일관성, 대응점 부족과 상대변형 감도 없는 θ 거부. 실제 혼합 운동 정확도 인증은 아니다. 기존 main의 center 특징점 추출 회귀와 Release 빌드도 확인한다.

현재 구현은 연구용이며 정확도 목표 미달이다. 운영 alignment()의 NaN 반환은 유지한다. 평균 제거 후 θ가 상대 변형으로 식별되는 정도와 세션 기준 의존성을 더 검증해야 한다. 카메라 전체 보정이나 실제 장비 영점 복원을 달성했다고 주장하지 않는다.

## 재현과 추정 입력

Python numpy,pandas가 필요하다. 기존 docs/qr-stability/scripts/README.md 방식으로 shin의 evaluate.cpp와 현재 src/*.cpp(main.cpp 제외)를 빌드하고 수집한다. parallel_evaluate.py는 4프로세스 실행을 지원한다.

```powershell
python tool/sequential/evaluate.py --data docs/validation-runs/sequential/data --output docs/validation-runs/sequential/evaluation --repo .
python -m unittest discover -s tool/sequential -p "test_*.py" -v
python tool/sequential/sequential.py docs/sequential-y-theta/experimental_model.json observation.json --session "Y+0"
```

observation.json은 accepted 점의 {"FP11L:31":[u,v], ...} 원본 픽셀 좌표다. session을 생략하면 center 기준을 사용하며 미지 세션 영점 성능은 검증되지 않았다. experimental_model.json은 보고서 재현용 1차 blocked 모델이다. 보류 결과와 요약은 이 디렉터리, 전체 학습 모델/원시 좌표/worker 로그는 Git 제외 validation-runs/sequential에 있다.

Jira: Epic S15P31A507-2, Story S15P31A507-9, 복원 Task23(2SP), 순차추정 Task24(5SP). 활성 sprint59016.
