# S15P31A507 작업 지침

이 지침은 이 Git 저장소 전체에 적용한다. 팀원이 저장소를 clone한 상태에서도 읽고 빌드할 수 있도록 정리한다.

## 작업 전 확인

1. QR 관측, 모델, 보정, 평가, 설명 자료를 바꾸기 전에 [ALGORITHM_RULES.md](ALGORITHM_RULES.md)를 읽는다. 알고리즘의 목표·제약·평가 규칙은 이 파일 하나에서 관리하고 다른 문서에 복제하지 않는다.
2. 공개 입출력은 [include/alignment.hpp](include/alignment.hpp), 실제 구현은 [src/alignment.cpp](src/alignment.cpp), 호출 예제는 [src/main.cpp](src/main.cpp)를 먼저 확인한다.
3. 헤더/구현 분리와 팀장 화이트보드의 판독·해석은 [모듈 구성 지침](docs/team-module-layout.md)을 따른다.
4. 빌드는 [개발환경](docs/windows-setup.md), Git 작업은 [Git 컨벤션](docs/gitConvention.md)을 따른다. 문서 확인만을 이유로 브랜치 이동·커밋·push를 수행하지 않는다.

HTML 보고서를 작성·수정할 때는 [보고서 기준 형식](docs/report-format.md)을 읽고, 사용자가 지정한 hr-v2 디자인과 기준 보고서를 따른다.

## 언어와 실행 의존성

- 팀에 전달하는 알고리즘, 데이터 정제, 검증 도구는 **C++17**로 작성한다.
- 팀 기준은 **Windows x64, MSVC v141, OpenCV 3.4.15 vc15**이다. 버전을 임의로 올리거나 Python 도구로 대체하지 않는다.
- 최종 실행물은 저장소 밖의 `final`, `hw-analysis`, `codex_experiments_*`, 개발자 개인 경로, Python 런타임에 의존하지 않는다. 외부 실험 코드는 이식 및 수치 비교를 위한 참고로만 사용한다.
- 필요한 모델·파라미터·설정은 전달 대상에 포함하고 상대 경로 또는 명시적인 설정으로 찾는다. 형상 관리에 넣을 파일과 외부 제공 파일을 문서에 구분한다.

## 코드의 역할 분리

- `include/*.hpp`: 공개 자료형과 함수 **선언**을 둔다. 일반 알고리즘 함수 본문을 헤더에 구현하지 않는다.
- `src/*.cpp`: 대응하는 헤더를 include하고 함수 **구현**을 둔다. `.cpp`를 다른 소스에서 include하지 않는다.
- `src/main.cpp`: 입력 준비, 모듈 호출, 결과 확인을 담당한다. 알고리즘을 `main()` 안에 몰아넣지 않는다.
- 공통 추론 모듈에 서비스·네트워크·실제 로봇 명령 전송을 섞지 않는다. GUI 대기, 대화형 입력, 프로세스 종료도 호출부/도구의 책임으로 둔다.
- `tool/`: 정제 등 독립 실행 도구를 둔다. 도구의 `main`이 애플리케이션 `main`과 함께 링크되지 않게 빌드 대상을 분리한다.
- `examples/`, `tests/`: 사용 예제와 검증 코드를 둔다. 별도 실행 파일의 `main`을 공통 모듈 소스에 넣지 않는다.

## 기존 API와 출력 계약

- 현재 선언은 `Alignment_t alignment(cv::Mat &src)`이며, **출력은 추정한 `(y, t)`**다. `Alignment_t`의 `double` 기반 두 필드에 각각 Y·T 추정값을 담는다. 시그니처·필드명·호출부를 확인 없이 바꾸지 않는다. T의 물리 단위와 부호를 임의로 정하지 않는다.
- 센터 목표값과 함수 출력 계약을 구분한다. 초기 두 필드 대입은 미구현 placeholder였으며 고정 반환값 요구사항이 아니다. **2026-10-28 구현:** `joint()`·`control()`을 추가하고 `alignment()`는 `joint()`로 위임한다. 계산한 `y`, `t`를 반환하며 모델·측정 오류를0,0으로 숨기지 않는다.
- 현재 반환형에는 성공/실패·안정성 필드가 없다. 상태/진단이 필요하면 기존 호출과의 호환성을 검토해 별도 진단 API 또는 명시적인 확장 계약으로 설계하고 문서화한다. 측정 실패를 정상적인 `(0, 0)`으로 숨기지 않는다.
- 사진의 `control()`·`joint()`를 무인자 함수나 확정된 API라고 가정하지 않는다. 기존 인터페이스와 연결하는 실제 입력·출력·오류·상태 보관 방법을 먼저 정의한다.

## 최종 정리 작업 위치

- `experiments/hye/00000_final_Y_T_joint/`
- `experiments/hye/00000_final_Y_T control/`

2026-10-28 사용자 지시로 실험 폴더를 `experiments/hye/` 아래로 옮겼다. 빌드에 사용하는 모듈은 [experiments/selected_modules.cmake](experiments/selected_modules.cmake)의 `SEMES_JOINT_DIR`·`SEMES_CONTROL_DIR`로 고른다. 후보 모듈은 같은 대상 이름(`semes_joint`/`semes_control`)과 같은 공개 함수를 제공하고 실행 자료를 `assets/`에 둔다. 새 실험은 final의 `.cpp`를 경로로 직접 컴파일하지 않고 이 대상을 링크한다.

각 폴더의 단계별 역할과 기준 모델은 [알고리즘 규칙의 현재 적용 기준](ALGORITHM_RULES.md#현재-적용-기준-2026-10-28)을 따른다. 초기 검토 시 비어 있던 두 폴더에 2026-10-28 C++ 모듈과 실행 자료를 추가했다. 각 README와 새 results의 검증 범위를 확인하며 이름에 `final`이 있다는 이유로 독립 성능·실제 복귀 검증 완료로 취급하지 않는다.

공유 파일과 기존 `src`, `include`, 루트 `CMakeLists.txt`는 요청에 필요한 범위에서만 변경한다. 실험별 CMake/라이브러리 대상을 사용해 기존 main과 분리하고, 루트 CMake의 `src/*.cpp` 재귀 수집에 다른 `main`이 섞이지 않도록 한다.

## Git 작업 흐름 (hye, 강제)

**통합 브랜치에서 직접 작업하지 않는다.** `chore/hye/merge`, `dev`, `main`은 보호 브랜치이며 [tool/hye/githooks](tool/hye/githooks)의 `pre-commit`이 커밋을, `pre-push`가 push를 거부한다. clone 직후 한 번 설치한다. `cp tool/hye/githooks/pre-commit tool/hye/githooks/pre-push .git/hooks/ && chmod +x .git/hooks/pre-commit .git/hooks/pre-push` (저장소마다 1회, 모든 worktree와 모든 브랜치에서 켜진다. `core.hooksPath`를 저장소 안 경로로 두면 그 파일이 없는 브랜치에서 훅이 꺼지므로 쓰지 않는다.)

한 작업의 전체 수명은 다음 순서를 벗어나지 않는다.

1. `chore/hye/merge`를 최신으로 받는다. `git fetch origin && git switch chore/hye/merge && git pull --ff-only`
2. 거기서 작업 브랜치를 만든다. `git switch -c feature/hye/<영문-기능명>`
3. 작업하고 변경 경로만 골라 stage한다. `git add .`로 일괄 추가하지 않는다. 커밋 메시지는 [Git 컨벤션](docs/gitConvention.md)의 `<type>: <한국어 설명>`을 따른다.
4. push한다. `git push -u origin feature/hye/<영문-기능명>`
5. MR을 만든다. 대상 브랜치는 `chore/hye/merge`다. glab이 없으므로 아래 링크를 사용자에게 전달한다.
   `https://lab.ssafy.com/s15-final/S15P31A507/-/merge_requests/new?merge_request[source_branch]=<작업브랜치>&merge_request[target_branch]=chore/hye/merge`
6. **머지는 사용자가 GitLab에서 직접 승인한다.** 로컬에서 `chore/hye/merge`로 merge하거나 push하지 않는다.
7. 머지된 뒤 1번으로 돌아간다. 다음 작업은 반드시 **새 브랜치**에서 시작한다. 이미 머지된 작업 브랜치를 이어 쓰지 않는다.

지키지 않는 흔한 형태와 금지 사항:

- 통합 브랜치에 쌓인 미커밋 변경을 그대로 커밋하는 것. → 작업 브랜치를 만들어 옮긴다.
- 작업 브랜치를 머지한 뒤 같은 브랜치에서 다음 작업을 계속하는 것. → 새로 판다.
- MR 없이 로컬 merge로 통합하는 것. → MR 링크만 만들고 멈춘다.
- 여러 폴더(worktree)에 같은 저장소를 펼쳐 두고 어느 쪽이 최신인지 잃는 것. → 작업 폴더는 `S15P31A507` 하나만 쓴다. worktree는 끝나면 `git worktree remove`로 정리한다.

급히 우회해야 할 때만 `SEMES_ALLOW_PROTECTED=1 git commit ...`로 훅을 건너뛴다. 기본값은 우회하지 않는 것이다.

## 데이터와 기록

- 원본은 `data/raw/`, 정제 결과는 `data/clean/`이며 정제 도구는 [tool/01_prepare_clean.cpp](tool/01_prepare_clean.cpp)다. 정제·빌드·실행 방법은 [도구 사용 설명](tool/README.txt)을 따른다.
- 원본/기존 clean/모델/평가 결과를 임의로 덮어쓰거나 지우지 않는다. 새 결과는 구분되는 위치에 저장하고 출처를 남긴다.
- 데이터 해석·추론 입력 제한·성능 지표는 공통 알고리즘 규칙을 따른다. 파일명 명령과 동봉 TXT 원시값을 합치거나 임의로 재보정하지 않는다.
- 이미지·ZIP·대용량 생성물은 코드 변경과 구분한다. 현재 `.gitignore`가 raw/clean을 제외하지 않는다는 점을 확인하고 `git add .`로 일괄 stage하지 않는다. 데이터 공유/추적 방식을 요청 없이 임의 변경하지 않는다.

## 완료 시 전달할 내용

- 변경 목적과 처리 흐름을 팀원이 이해할 수 있는 한국어로 설명한다.
- 공개 입력·출력, 좌표/단위/부호, 전제 조건, 실패·불안정 처리, 필요한 파라미터를 명시한다. 미확정 사항은 추정해서 채우지 않는다.
- 저장소 기준 빌드·실행 명령과 실제 수행한 검증 결과를 남긴다. 실제 촬영, 정지 사진 평가, 시뮬레이션을 구분한다.
- 문서만 바꾼 작업에는 불필요한 전체 학습·대용량 데이터 재생성을 수행하지 않는다.
