# control 함수별 처리 흐름

호출 측의 `cv::imdecode(..., IMREAD_GRAYSCALE)`로 사진을 읽고 `control(cv::Mat&)`에 전달한다.
공개 선언은 `S15P31A507/include/alignment.hpp`에 있다. 아래 상대 경로는 이 control 폴더 기준이다.

| 순서·함수 | 경로 | 입력 → 처리 → 출력 |
|---|---|---|
| 준비 `initialize_control()` | `src/control.cpp` | assets 디렉터리 → Model/Tracker 준비, 성공한 후 교체 → 추론 상태 보관 |
| `Model::Model()` | `src/model.cpp` | model.json → 차원·숫자·6후보 검사 → W와 후보별 origin/slope/curve 보관 |
| `Tracker::Tracker()` | `src/tracker.cpp` | assets → center 해시 검사, 630점과 마스크 읽기, QR별 템플릿 구성, center 자체 변위 측정 → 고정 참조 보관 |
| `reference_models()` | `src/tracker.cpp` | QR 기준 영상·고정점 → 흐림/기울기, 17×17 템플릿, 의사역행렬 → 점별 GN 고정 계수 |
| 본 처리 `control()` | `src/control.cpp` | 현재 영상 → 입력 검사, measure/features/predict → `Alignment_t`와 스레드별 진단 |
| `checked_gray()` | `src/tracker.cpp` | cv::Mat → 타입·크기·채널 검사, 필요하면 회색조 변환 → 회색조 영상 |
| `Tracker::measure()` | `src/tracker.cpp` | 회색조 영상 → QR별 NCC 정수 이동, QR 자르기, LK15 왕복, GN17 → 630×2변위·품질·유효 플래그·NCC 값 |
| `spline_coefficients()` | `src/spline.cpp` | 흐림 처리한 double 영상 → cubic spline 사전 필터 → 보간 계수 영상 |
| `nearest_prefilter()` | `src/spline.cpp` | 참조 영상 또는 기울기 → 가장자리12px 복제 후 사전 필터 → SciPy nearest/prefilter 대응 계수 |
| `spline_sample()` | `src/spline.cpp` | 계수 영상·소수 좌표 → 4×4이웃의 3차 보간 → 밝기 값1개 |
| `Tracker::features()` | `src/tracker.cpp` | 630점 변위 → 고정536점 선택, center 변위 차감, 위치 다항식6항×dx/dy 최소제곱 → 1×12특징 |
| `Model::predict()` | `src/model.cpp` | 12특징 → W 백색화, 6후보의8회 역산, 부호 게이트, 최소 비용 선택 → 선택 축·추정량·후보명·비용 |
| `Tracker::valid_selected()` | `src/tracker.cpp` | 추적 결과 → 고정536점 안의 valid 합계 → 진단용 점 수. 특징점을 바꾸는 데 사용하지 않음 |
| 반환값 정리 | `src/control.cpp`의 `control()` 마지막 | Y선택이면(y,0), T선택이면(0,t) → 호출한 곳으로 return |
| `alignment_diagnostics()` | `S15P31A507/src/alignment_diagnostics.cpp` | 인자 없음 → 현재 스레드의 최종 진단 복사 → 호출한 곳으로 return |

`read_gray()`는 초기화와 검증 도구의 BMP 읽기, `sha256()`는 기준 영상 검사, `read_numbers()`는 모델 배열 검사를 담당한다.

로봇에 전송하는 함수는 없다. 호출 측이 `estimate.y / estimate.t`와 진단을 받아 장치의 축·단위·허용 범위에 맞춰 상대 지령을 만들고, 실제 이동 후 찍은 사진을 다시 모듈에 전달한다. 단순 부호 반전을 기계 연결 사양으로 내장하지 않았다.
