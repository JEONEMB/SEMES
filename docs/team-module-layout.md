# 팀장 화이트보드와 C++ 모듈 구성

출처: 2026-10-28 사용자가 전달한 팀장 화이트보드 사진. 아래는 이미지의 수기 내용을 직접 판독한 기록이다. 읽힌 글자와 그 취지에 대한 해석을 구분한다.

## 사진에서 확인한 내용

```text
왼쪽: main.cpp
  #include "alignment.hpp"
  main() { }

가운데: alignment.hpp
  [반환형 판독 불명확] control();
  joint();
  함수 본문 형태의 중괄호에 X 표시

오른쪽: alignment.cpp
  #include "align…"   [헤더명의 뒷부분은 생략되거나 불명확]
  control() { ?? }
  joint()   { ?? }

양쪽 .cpp에서 가운데 .hpp를 가리키는 연결 표시
```

아래쪽·오른쪽 한글 메모는 각각 “뭔가 있다”, “뭔가 해야 한다”와 유사하게 보이지만 정확한 문구는 판독이 불명확하다. 이 메모를 근거로 새로운 함수명, 반환형, 인자형을 확정하지 않는다. 사진에는 `control`·`joint`의 정확한 입력·출력 계약이 제시되어 있지 않다.

## 적용할 구조

사진은 호출 코드와 함수 선언, 실제 구현을 분리하라는 지침으로 해석한다.

| 위치 | 책임 |
|---|---|
| `include/alignment.hpp` | 공개 자료형과 외부에서 호출할 함수 선언 |
| `src/alignment.cpp` | 헤더를 include하고 알고리즘 함수 본문 구현 |
| `src/main.cpp` | 같은 헤더를 include하고 모듈 호출 |

호출부는 헤더를 통해 함수를 알고, 링커가 `.cpp`의 구현을 연결한다. `main.cpp`에서 `alignment.cpp`를 include하거나, 헤더에 일반 알고리즘 함수의 본문을 넣지 않는다.

단계별 모듈 내부에서 여러 소스/비공개 헤더로 나누는 것은 가능하다. 공개 선언과 실제 구현의 경계를 유지하고, 각 `.cpp`를 CMake 대상에 한 번씩만 포함한다. 독립 실행 도구에는 별도 진입점과 빌드 대상을 사용한다.

## 현재 dev에 실제로 있는 계약

다음은 사진을 바탕으로 만든 가상의 인터페이스가 아니라, 현재 [alignment.hpp](../include/alignment.hpp)에 있는 선언이다.

```cpp
typedef double error_t;

typedef struct _alignment_t {
    error_t y;
    error_t t; // theta
} Alignment_t;

Alignment_t alignment(cv::Mat &src);
Alignment_t joint(cv::Mat &src);
Alignment_t control(cv::Mat &src);
```

- 입력: `cv::Mat` 한 장이다. 이번 구현은3840×2160 uint8의1/3/4채널을 받고 원본을 수정하지 않는다. 기준·모델은 작업 전 `initialize_joint()` 또는 `initialize_control()`로 각자 로드한다. 기존 `initialize_alignment()`는 두 모듈 초기화의 호환 진입점이다.
- 출력: **추정한 `(y, t)`**를 `Alignment_t`의 두 필드에 담는다. 센터 목표값을 매번 고정 반환하는 계약이 아니다. 추정값과 이를 이용해 산출한 실행 명령은 구분하고, 단위·축 방향·부호·허용 범위를 구현 문서에서 명시해야 한다.
- 상태: `Alignment_t` 두 필드는 유지한다. 별도 `alignment_diagnostics()`로 스레드별 최근 진단을 확인하며 오류는 예외로 전달한다. 측정 유효성을 자세 안정성 보증으로 해석하지 않는다.
- 2026-10-28 구현: [alignment.cpp](../src/alignment.cpp)의 placeholder를 제거하고 기존 `alignment()`를 `joint()`에 위임했다. 함수 본문은 [joint.cpp](../experiments/hye/00000_final_Y_T_joint/src/joint.cpp)와 [control.cpp](../experiments/hye/00000_final_Y_T%20control/src/control.cpp)에 있다. 두 구현은 같은 공개 헤더를 include하며 CMake가 라이브러리로 연결한다.
- [main.cpp](../src/main.cpp)는 이미지 경로와 모드를 받아 사진을 읽고 해당 모듈을 호출한다. 흰 영상 데모와 GUI 대기는 제거했다. 학습·평가는 별도 도구다.

## `control()`·`joint()`를 구현할 때

1. 실제 단계와 알고리즘은 [공통 규칙](../ALGORITHM_RULES.md#현재-적용-기준-2026-10-28)을 따른다.
2. 함수 이름만 보고 입력·출력을 임의로 정하지 않는다. `alignment()`와의 연결, 결과의 의미, 모델/기준 영상 초기화 방법을 문서화한다.
3. 불안정 판정, 실패 전달, 이전 관측·실행한 상대 명령을 저장할 필요가 있으면 상태의 소유자와 수명을 정한다. 현재 단일 영상 API에 파일명이나 숨은 정답을 이용해 이력을 끼워 넣지 않는다.
4. 서비스 연결과 실제 로봇 이동은 호출 측이 담당한다. 계산 모듈은 합의된 데이터와 진단을 반환한다.
5. 공개 선언을 확장하면 구현, 호출 예제, 관련 테스트와 CMake도 함께 맞춘다. 기존 헤더를 실제 동작으로 확인하지 않은 채 완료했다고 보고하지 않는다.

이 문서는 사진 판독과 코드 구성 지침을 관리한다. 알고리즘의 허용 관측·목표·평가 수치는 `ALGORITHM_RULES.md`에서만 관리한다.
