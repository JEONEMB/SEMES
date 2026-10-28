# 2차 control C++ 모듈

> **현재 상태: 신규 조건부Y 실험 구현이며 미채택이다. 사용자가 지정한 초기100% 기존 모델은 아직 연결하지 않았다.**
> 당시 코드·모델·결과는 [20261028_02 실험 기록](../20261028_02_control_T0_조건부Y_신규모델_미채택/README.md)에 보존했다.

공통 기준은 [ALGORITHM_RULES.md](../../ALGORITHM_RULES.md)에 있다.
선언은 [alignment.hpp](../../include/alignment.hpp), 구현은 [src/control.cpp](src/control.cpp),
호출 예제는 [main.cpp](../../src/main.cpp)다. 빌드 진입점은 저장소 루트다.

## 입력과 출력

`Alignment_t control(cv::Mat& src)`는 T 정렬 후 새로 촬영한 현재 영상을 받는다.
먼저 joint와 같은 측정으로 현재T를 추정하고, `abs(T)<=maximum_control_t`일 때만 조건부Y를 계산한다.
전제가 충족되지 않으면 `std::domain_error`를 던지고 `control_precondition_met=false`를 남긴다.
반환 `y`는 조건부Y 추정, `t`는 같은 현재 영상에서 얻은 공동 추정T다. T를 강제로0으로 만들지 않는다.
서비스 연결·로봇 이동·재촬영·추정값 음수 변환은 수행하지 않는다.

기본T게이트0.09는 설정 가능한 **추정값 기반 운용 게이트**다. 실제 자세T가0임을 독립 계측한 보증이 아니다.
이 수치는 `initialize_alignment()`의 세 번째 인자로 변경할 수 있다.
불안정 구간 자동 이동·복귀는 이 계산 함수의 후속 제어 정책이다.

## 모델의 성격

이번에 추가했던 2차 신규 실험 모델이다. 현재 채택 모델이 아니며 기존337/341 공동 추정 결과나 초기100% 모델과 같은 것으로 취급하지 않는다.
`data/clean/clean_20261014/Y/*.bmp`80장과 기존센터1장, 총81장의 **T=0 통제 사진**으로 학습한다.
Y=0/T변화 사진은 조건부Y의 학습 정답으로 섞지 않는다. T 판정은 joint 모델이 담당한다.
관측은 joint와 동일한102개 값과204차원 특징이다.
도구가 `z≈origin+Y*response`를 맞춘 뒤 잔차 공분산을0.01로 축소한1차원GLS 역산 계수를 저장한다.
파일명 명령은 이 학습/평가 도구에서만 읽고 `control()`에는 영상만 전달한다. 동봉TXT를 대체 정답으로 사용하지 않는다.

## 모델 재생성

저장소 루트에서 빌드 후, **기존 파일과 다른 출력 경로**로 실행한다.

```powershell
.\build-joint-control-20261028\experiments\00000_final_Y_T_joint\Release\train_control.exe `
  data/clean/clean_20261014 `
  experiments/00000_final_Y_T_joint/assets `
  "experiments/00000_final_Y_T control/assets/control_model_v2.yml"
```

기본 `assets/control_model.yml`은 이번 C++ 도구가 생성한 파라미터다. 기존 파일은 덮어쓰지 않는다.
생성된 파라미터의81장 재대입 오차는 학습 적합도이며 새 사진 성능이 아니다.

```powershell
.\build-joint-control-20261028\Release\main.exe control "data/clean/clean_20261014/Y/Y+0.50.bmp"
```

최종 사용 전 T게이트와 불안정 판정, 실제 두 단계 이동 결과는 별도 실측 평가가 필요하다.
