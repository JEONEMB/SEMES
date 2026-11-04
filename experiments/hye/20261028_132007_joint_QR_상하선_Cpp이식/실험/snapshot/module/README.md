# 1차 joint C++ 모듈

공통 목표·관측 제약·평가는 [ALGORITHM_RULES.md](../../ALGORITHM_RULES.md)를 따른다.
이 폴더는 QR+위·아래 선의 기존 공동 추정 모델을 C++17/OpenCV3.4.15에서 실행한다.
서비스·실제 이동·재촬영은 호출 측이 담당한다.

## 먼저 읽을 파일

- 공개 선언: [include/alignment.hpp](../../include/alignment.hpp)
- 1차 함수 본문: [src/joint.cpp](src/joint.cpp)
- 사진 측정: [src/observer.cpp](src/observer.cpp)
- 선 프로파일 정합: [src/line_profile.cpp](src/line_profile.cpp)
- 공개 호출 예제: [src/main.cpp](../../src/main.cpp)
- 기존 `alignment()` 호환 연결: [src/alignment.cpp](../../src/alignment.cpp)
- 2차 구현: [control.cpp](../00000_final_Y_T%20control/src/control.cpp)

## 함수 계약

```cpp
void initialize_alignment(const std::string& joint_assets,
                          const std::string& control_model = "",
                          double maximum_control_t = 0.09);
Alignment_t joint(cv::Mat& src);
Alignment_t control(cv::Mat& src);
Alignment_t alignment(cv::Mat& src); // joint(src)로 위임하는 기존 API
AlignmentDiagnostics alignment_diagnostics();
```

초기화는 작업 스레드를 시작하기 전에 한 번 호출한다. 실행 중 다른 스레드에서 재초기화하지 않는다.
`joint_assets`와 `control_model`은 UTF-8 경로이며, 현재 작업 폴더 기준 상대 경로나 절대 경로다.
저장소 밖의 실험 폴더나 Python 런타임은 실행에 필요하지 않다.

입력은3840×2160,8비트,1/3/4채널 영상이다. 원본 `src`는 수정하지 않는다.
원본 Python의 BMP 바이트 관측과 비교할 때는 BMP를 **IMREAD_GRAYSCALE로 디코딩한 영상**을 권장한다.
BGR/BGRA 입력은 함수 안에서 회색조로 변환하며 디코더/색 변환 버전에 따른 수치 차이가 가능하다.
출력 `y,t`는 추정값이다. 음수 보정량 변환·장치 단위 변환·이동 명령 전송은 수행하지 않는다.
T의 단위와 축 부호는 원래 명령값 기준이며 물리 각도 단위를 새로 가정하지 않았다.

초기화 실패, 빈 영상, 크기/타입 불일치, QR/선 측정 실패는 예외로 전달한다.
호출 측은 `std::exception`을 잡아 실패 처리해야 한다. 실패를0,0으로 반환하지 않는다.
`alignment_diagnostics()`는 호출한 스레드의 최근 진단을 반환한다.
`measurement_valid`는 영상 정합 유효성이고 자세 오차0.09의 보증이 아니다.
`outside_training_range`는 추정값이 실제 보정 명령 범위를 벗어났는지 나타내는 관측용 진단이며 값을 잘라 바꾸지 않는다.
안정 구간 자동 탐색·이동 후 복귀 정책은 이 함수에 포함하지 않았다.

## 내부 처리

1. `joint()`가 입력 측정과 모델 계산을 호출한다.
2. `Observer::measure()`가 QR별 NCC 초기 탐색과 ECC 호모그래피 정합을 수행한다.
3. 고정 가상 위치27개의 dx/dy를 읽어 QR54개 값을 만든다. 실제 모듈 코너27개 직접 측정이 아니다.
4. 허용된 선48구간의 수직 프로파일 이동을 측정한다.
5. `feature_map()`이102개 관측과 평균 QR가로 이동과의 상호작용을 결합해204차원으로 만든다.
6. `JointModel::predict()`가 저장된 origin/projector/initial/scale로 y,t를 계산한다.

모델은 `final/models/observation_full341_frozen/both.json`에서 복사한 기존 전수 모델의 직렬화다.
이전 저장소 밖 자료는 이식 출처이고 런타임 의존성이 아니다. 출처·해시는 assets/provenance.json에 기록한다.
OpenCV5 ECC 수식을 호환 구현한 third_party/ecc_compat.cpp는 원 라이선스 헤더를 보존했다.
팀 실행은OpenCV3.4.15다. 같은 모델을 사용해도 영상 디코딩/연산 차이로 Python 결과와 완전히 같지는 않을 수 있다.

## 빌드 및 실행

저장소 루트 PowerShell에서:

```powershell
cmake -S . -B build-joint-control-20261028 -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build-joint-control-20261028 --config Release --parallel 4
.\build-joint-control-20261028\Release\main.exe joint "data/clean/clean_20261020/Y+0.05 T+0.01/Y+0.05 T+0.01 (1).bmp"
ctest --test-dir build-joint-control-20261028 -C Release --output-on-failure
```

동일 라이브러리에 `joint`와 `control`을 링크하므로 소스를 `.cpp` include로 합치지 않는다.
빌드 대상은 `semes_alignment`(라이브러리), `main`(호출), `train_control`(학습 도구),
`evaluate_alignment`(정지 사진 평가), `alignment_checks`(검사)로 나눴다.

## 검증 범위

기존337/341은 외부4fold의 개발 평가다. 이 폴더의 실행 기본 모델은 **전체341장 학습 모델**이다.
동일 clean341장을 다시 예측한 결과는 재대입/개발 재생 검사이며337/341의 OOF 재현이나 독립 평가로 부르지 않는다.
새 C++ 결과와 Python 전수 모델의 비교는 results/native_20261028_001에 별도로 저장한다.
실제 로봇 이동과 새로운 사진 촬영은 수행하지 않았다.

모듈을 공유할 때 이 폴더의 소스·assets와 control 폴더·공개 헤더·호출 코드를 함께 전달한다.
assets는 실행 필수 자료다. data/raw와data/clean 전체는 런타임 추론 한 장에 필수는 아니다.
대용량 영상·모델의 Git 추적 방식을 임의 변경하거나 `git add .`로 올리지 않았다.
