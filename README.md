# QR 판 정렬 측정

사진 한 장에서 판이 기준 위치에서 얼마나 벗어났는지 **Y (mm), T (도)** 를 잰다.

- 입력: 3840×2160 사진 한 장
- 출력: `y_mm`, `t_deg`, 상태(정상 / 경고 / 실패), 알림 메시지
- 속도: 한 장 1스레드 17.6 ms, 4스레드 9.4 ms (노트북, SEMES 사진 702장 중앙값, 사진 읽기 제외)
- 언어와 환경: C++17, OpenCV 3.4.15

| 문서 | 읽는 때 |
|---|---|
| [docs/1_한눈에보기.md](docs/1_한눈에보기.md) | 처음 5분. 그림으로 보는 원리와 단계 |
| [docs/2_포팅안내서.md](docs/2_포팅안내서.md) | 코드를 옮길 때. 함수마다 순서와 확인 값 |
| [docs/3_근거보고서.md](docs/3_근거보고서.md) | 왜 이렇게 만들었는지, 문제가 생기면 어떻게 되는지 |

## 1. 바로 써 보기

```cpp
#include "alignment.hpp"

alignment::initialize("data");                                   // 한 번
alignment::Result r = alignment::measure_alignment(photo);       // 사진마다
// r.y_mm, r.t_deg  → 보정 이동은 (−r.y_mm, −r.t_deg)
// r.status, r.message → 경고나 실패가 있으면 여기에
```

아무것도 정하지 않으면 아래 기본값으로 동작한다.

| 항목 | 기본 | 바꾸는 곳 |
|---|---|---|
| QR 재는 법 | 흐림 NCC | (고정) |
| 나사 | 4개 모두 | `config/settings.json` 의 `screws` |
| 영점 | 기준 사진과 같은 화면이 (0, 0) | `set_zero_from_photos` |
| QR 탐색 | ±110 px → 못 찾으면 ±250 px → 사진 전체 폭 | (자동) |
| 값이 서로 안 맞을 때 | 경고만 | `fail_on_disagreement` |
| 알림 출력 | 콘솔(stderr) | `log_file` |

## 2. 빌드와 확인

```
cmake -S . -B build -DOpenCV_DIR=<OpenCV 폴더>
cmake --build build --config Release
```

예제 사진 5장으로 값이 맞는지 본다(폴더 맨 위에서 실행).

```
build/Release/check_photos data tests/photos/list.tsv out.tsv tests/photos/expected.tsv
→ 기대값과 비교 5장: Y 차이 최대 0.00000000 mm, T 차이 최대 0.00000000 도
```

사진 한 장의 단계별 값을 본다.

```
build/Release/measure_one data tests/photos/sample_1.png
```

## 3. 폴더

```
include/alignment.hpp        쓰는 사람이 부르는 함수 전부
src/
  alignment.cpp              아래 단계를 차례로 부르고 알림을 붙인다
  steps/
    01_find_qr               QR 이 얼마나 움직였나
    02_find_screws           나사가 얼마나 움직였나
    03_compute_yt            Y, T 계산
    04_check_result          결과 점검
  setup/
    settings                 설정 읽기
    find_new_reference       기준 사진 다시 잡기
  common/reference           기준 사진과 QR·나사 위치
config/settings.json         설정 (기본값과 설명)
data/                        기준 사진, 위치, 계수, 점검식
examples/measure_one.cpp     사진 한 장 재 보기
examples/show_steps.cpp      단계마다의 값을 모두 찍기 (옮긴 코드와 비교할 때)
tests/                       확인 프로그램과 예제 사진·기대값
tool/make_figures.cpp        문서의 그림 만들기
docs/                        문서 3개와 그림
```

**읽는 순서**: `include/alignment.hpp` → `src/alignment.cpp` 의 `measure_alignment` → `steps/01` → `02` → `03` → `04`. 각 단계의 `.hpp` 맨 위에 그 단계가 하는 일을 적어 두었다.

## 4. 사진 한 장이 거치는 단계

```
사진
 │
 ├─ 1. find_qr_shift       QR 3개의 가로 이동 (px)
 │     ±110 px 에서 찾음 ─ 못 찾으면 → ±250 px ─ 못 찾으면 → 사진 전체 폭
 │     하나도 못 찾으면 → 실패 E_QR_NOT_FOUND
 │
 ├─ 2. find_screw_shifts   나사 4개의 가로 이동 (px)
 │     나사마다: 대략 찾기(NCC) → 밝기로 다듬기 → 테두리로 정밀 맞춤(ECC)
 │     못 찾은 나사는 빼고 계속한다
 │
 ├─ 3. compute_yt          찾은 나사 조합의 계수로 Y, T 계산, 영점 빼기
 │
 └─ 4. check_…             값들이 서로 맞는지, 학습 범위 안인지 → 아니면 경고
```

### 원리

- 판이 **회전(T)** 하면 QR 과 나사가 화면에서 거의 같이 움직인다(1도에 약 80 px).
- 판이 **Y 로 이동**하면 QR 과 나사가 다르게 움직인다(1 mm 에 QR 약 14 px, 나사 약 11 px).
- 그래서 "전체가 얼마나 움직였나" 로 T 를, "QR 과 나사가 얼마나 다르게 움직였나" 로 Y 를 안다.
- 세로 이동은 쓰지 않는다. 카메라가 처져 화면이 위아래로 밀려도 값이 변하지 않게 하기 위해서다.

### NCC 와 ECC

| | NCC | ECC |
|---|---|---|
| 하는 일 | 조각을 한 칸씩 옮겨 가며 가장 닮은 자리를 고른다 | 대략 맞는 자리에서 조금씩 밀어 가며 가장 닮게 맞춘다 |
| 장점 | 멀리 움직여도 찾는다 | 소수 px 까지 정밀하다 |
| 쓰는 곳 | QR(포물선으로 소수 px 보완), 나사 대략 찾기 | 나사 정밀 맞춤 |

## 5. 알림 (경고와 실패)

어떤 함수도 예외를 던지지 않는다. 결과의 `status`, `code`, `message` 로 알리고, 같은 내용을 콘솔(stderr)에 한 줄 쓴다. `log_file` 을 정하면 파일에도 쓴다.

**경고일 때도 `y_mm`, `t_deg` 는 나온다.** 값이 없는 것은 실패(E_…)뿐이다.

| 코드 | 언제 | 값 | 해야 할 일 |
|---|---|---|---|
| `W_OUT_OF_RANGE` | Y, T 가 학습 범위(Y ±2 mm, T ±1도) 밖 | 나옴. 오차가 커질 수 있음 | 그 값으로 옮긴 뒤 다시 측정 |
| `W_FAR` | QR 이 250 px 넘게 떨어져 사진 전체 폭에서 찾음 | 나옴. 부정확할 수 있음 | 그 값으로 옮긴 뒤 다시 측정 |
| `W_SCREW_MISSING` | 나사 일부를 못 찾음 (좌우에 하나씩은 남음) | 나옴. 남은 나사로 계산 | 가림·이물 확인 |
| `W_VALUES_DISAGREE` | 값 하나가 다른 값들과 맞지 않음 | 나옴. 의심해야 함 | 가림·반사 확인. 모든 사진에서 나오면 영점 다시 잡기 |
| `W_QR_PARTIAL` | QR 3개 중 1~2개만 찾음 | 나옴. 부정확할 수 있음 | QR 이 카메라 안에 다 들어오게 |
| `W_COARSE` | 나사가 한쪽만 보이거나 안 보임 | **대략값**. 방향과 크기만 믿는다 | 그 값으로 센터 쪽으로 옮긴 뒤 다시 측정 |
| `W_ZERO_RESET` | 기준을 다시 잡을 때 이전 영점을 이을 수 없었음 | 그 사진이 (0, 0) | 영점이 바뀌었음을 기록 |
| `N_WIDE_SEARCH` (참고) | QR 이 110 px 넘게 움직여 넓은 창(±250 px)으로 다시 찾음. 상태는 정상 | 나옴. 정확도 같음 | 없음 (약 6 ms 더 걸림) |
| `E_QR_NOT_FOUND` | QR 이 하나도 안 보임 | 없음 (NaN) | QR 이 카메라 안에 보이는지, 가림·반사 확인 |
| `E_PHOTO` | 사진 크기·형식이 다름 | 없음 | 3840×2160 8비트 사진을 넣기 |
| `E_NOT_INITIALIZED` | `initialize` 를 부르지 않음 | 없음 | 먼저 `initialize` |
| `E_DATA`, `E_SETTINGS` | 초기화 때 data 폴더나 설정 파일을 읽지 못함 | 시작 못 함 | 받은 폴더를 그대로 다시 복사, 설정 값 확인 |
| `E_ZERO`, `E_REFERENCE` | 센터 사진에서 QR·나사를 못 찾아 영점·기준을 바꾸지 못함 | 이전 상태 그대로 | 센터 사진 확인. 화면이 많이 밀렸으면 기준부터 다시 잡기 |

알림이 여러 개면 `message` 에 모두 적히고, `code` 는 가장 중요한 것 하나다. 참고 알림(N_…)은 상태를 정상으로 둔 채 `code`, `message` 와 콘솔에 `[알림 …]` 으로만 알린다.

## 6. 날마다 하는 설정

| 함수 | 하는 일 | 언제 |
|---|---|---|
| `set_zero_from_photos(센터 사진들)` | 이 사진의 자세를 (0, 0) 으로 삼는다. 5~10장을 넘기면 평균을 쓴다 | 날마다. 카메라·조명·판을 건드린 뒤에는 꼭 |
| `set_zero_value(y, t)` / `clear_zero()` | 적어 둔 영점을 다시 넣기 / 처음 영점으로 되돌리기 | 프로그램을 다시 켰을 때 |
| `set_reference_from_photos(센터 사진들)` | QR·나사 위치를 이 사진에서 다시 찾는다. 영점은 이어진다 | 화면이 많이 밀려 QR·나사를 못 찾을 때만 |

영점을 날마다 잡아야 하는 이유: 카메라가 0.05도만 돌아도 Y 가 0.27 mm 틀어진다. 이런 화면 전체의 변화는 영점을 다시 잡으면 없어진다.

## 7. 확인한 것 (SEMES 사진 702장)

| 확인 | 결과 |
|---|---|
| 이 폴더의 코드와 정리 전 모듈의 값 | 702장 모두 같음 (차이 0) |
| 정확도 (학습에 안 쓴 사진으로 시험, 10회 평균) | 560장 중 0.2장이 허용오차(Y 0.045 mm, T 0.01도) 밖 |
| 처음 보는 촬영 회차 | 98.2% 통과 |
| 화면을 가로로 옮겨 큰 이동 흉내 | ±600 px 까지 값을 냄. ±1000~2000 px 는 대략값(T 가 3~8% 틀림). 2300 px 는 QR 이 사진 밖이라 실패 |
| 나사 하나를 가림 | 나머지 3개로 계산하고 경고. 가리기 전과 Y 0.007 mm, T 0.0006도 차이 |
| 화면이 위아래로 밀림 | 20 px 까지 값이 그대로. 그 이상은 QR 을 못 찾음 |
| 화면이 아래로 40 px 밀림 | QR 을 못 찾음 → 기준을 다시 잡으면 처음 값과 Y 0.004 mm, T 0.0004도 차이 |
| 영점 잡기 | 그 사진이 (0, 0), 다른 사진은 정확히 그만큼 빠짐 |

## 8. 알아 둘 것

- 값은 정지 사진으로 확인했다. 실제로 정렬을 반복해 센터로 돌아오는 시험은 아직 하지 않았다.
- 큰 이동 시험은 사진을 통째로 옮긴 것이다. 실제 큰 Y, 큰 회전에서의 정확도는 확인하지 못했다.
- 계수는 SEMES 사진(Y ±2 mm, T ±1도)으로 학습했다. 그 밖에서는 T 를 1~1.5% 크게 읽는 경향이 있었다.
- QR 판과 나사 사이가 통째로 밀리면(판 재조립) 사진만으로는 알 수 없다. 판을 손댄 뒤에는 영점을 다시 잡는다.
- 536점 추적(QR 을 더 정밀하게 재는 다른 방법)과 바 기울기는 이 폴더에 아직 넣지 않았다.
- data 폴더 경로에 한글이 있으면 Windows 에서 설정·위치 파일을 읽지 못할 수 있다. 영문 경로에 둔다.

## 9. 옮길 때 신경 쓸 점

항목마다 **지금 어떻게 되어 있는지**와 **바꾸려면 어디를 고치는지**를 적었다. 무엇을 고치든 마지막에는 `check_photos` 로 예제 사진 5장의 값이 그대로인지 확인한다(2절).

### 9-1. 컬러 순서 (가장 먼저 확인)

- **지금**: 3채널 사진은 **BGR** 로 보고 회색으로 바꾼다. 카메라가 RGB 로 주면 회색 밝기가 달라져 값이 조용히 틀린다. 경고도 나오지 않는다.
- **위치**: [src/common/reference.cpp](src/common/reference.cpp#L52) `to_gray` 52줄 `cv::COLOR_BGR2GRAY`, 53줄 `cv::COLOR_BGRA2GRAY`
- **바꾸려면** (셋 중 하나)
  - 넣기 전에 회색(1채널)으로 바꿔서 넣는다. 1채널은 그대로 쓰므로 순서 문제가 없다.
  - 넣기 전에 `cv::cvtColor(photo, photo, cv::COLOR_RGB2BGR)` 로 순서를 바꾼다.
  - 52줄을 `cv::COLOR_RGB2GRAY`, 53줄을 `cv::COLOR_RGBA2GRAY` 로 바꾼다.

### 9-2. 사진 크기와 카메라 조건

- **지금**: 사진은 3840×2160 8비트만 받는다. QR 상자와 나사 중심은 기준 사진의 **px 절대 좌표**로 박혀 있다.
- **위치**: 크기는 [src/common/reference.hpp](src/common/reference.hpp#L9) 9~10줄 `kPhotoWidth`, `kPhotoHeight`. 좌표는 `data/reference.json` 의 `qr`, `screws`
- **바꾸려면**
  - 화면이 조금 밀린 정도(카메라가 처지거나 살짝 움직임)면 코드는 그대로 두고 `set_reference_from_photos` 를 부른다(6절).
  - 해상도나 잘라내기(ROI)를 바꾸면 상수만 바꿔서는 안 된다. 기준 사진, 좌표, 계수를 모두 새 조건으로 다시 만들어야 한다(9-3).

### 9-3. 계수를 다시 만들어야 하는 경우

- **지금**: `data/coefficients/` 와 `data/checks/` 는 SEMES 사진 702장(Y ±2 mm, T ±1도)으로 학습한 결과다. **학습 코드는 이 패키지에 들어 있지 않다.**
- **다시 만들어야 하는 때**: 카메라, 렌즈, 카메라와 판 사이 거리, QR 판이나 나사 배치가 바뀌었을 때. 영점이나 기준을 다시 잡는 것으로는 해결되지 않는다.
- **바꾸려면**: 새 조건에서 Y, T 를 알고 있는 사진을 모아 계수를 다시 학습해야 한다. 학습 코드와 방법은 요청하면 따로 전달한다. 파일 형식은 지금 파일의 머리줄 설명을 따르면 코드는 고치지 않아도 된다.

### 9-4. 프로그램을 다시 켜면 사라지는 것

- **지금**: 아래 상태는 메모리에만 있다. 다시 켜면 `initialize` 직후 상태(처음 기준 사진, 영점 0)로 돌아간다.
  - 영점 (`zero`)
  - 기준을 다시 잡은 결과 (새 QR·나사 위치, 영점을 잇는 값 `carry_over`)
  - 값 일치 점검의 기준 어긋남 (`baselines`)
- **위치**: [src/alignment.cpp](src/alignment.cpp#L44) 44~56줄 `struct State`
- **다시 넣을 수 있는 것**: 영점 값만 `set_zero_value(y, t)` 로 넣을 수 있다. 이때 점검의 기준 어긋남은 0 이 된다.
- **바꾸려면**
  - 코드 수정 없이: 켤 때마다 센터 사진으로 `set_zero_from_photos` 를 다시 부른다. 기준을 다시 잡았던 장비라면 그 앞에 `set_reference_from_photos` 도 부른다. 가장 단순하고 안전하다.
  - 코드 수정: `alignment.cpp` 에 `state.reference`, `state.zero`, `state.carry_over`, `state.baselines` 를 파일로 쓰고 읽는 함수를 더하고 `include/alignment.hpp` 에 선언한다. 기준 사진은 `center.bmp`, 위치는 `reference.json` 형식으로 쓰면 그 폴더를 `initialize` 에 그대로 넘길 수 있다.

### 9-5. 여러 스레드, 여러 카메라

- **지금**: 상태가 프로그램 전체에 하나뿐이고(`state`), 모든 공개 함수가 같은 잠금(`state_mutex`)을 건다. 여러 스레드에서 불러도 안전하지만 **한 번에 한 장씩** 잰다. 카메라 여러 대를 각각 다른 기준으로 재는 구조는 지원하지 않는다.
- **위치**: [src/alignment.cpp](src/alignment.cpp#L58) 58~59줄
- **바꾸려면**
  - 카메라마다 프로그램(프로세스)을 따로 띄운다. 코드 수정이 없다.
  - 한 프로그램에서 하려면 `State` 와 잠금을 클래스 멤버로 옮기고, 공개 함수 6개를 그 클래스의 함수로 바꾼다. 단계 코드(`src/steps/`)는 이미 상태를 갖지 않아서 고칠 필요가 없다.

### 9-6. 안쪽 병렬 처리와 속도

- **지금**: QR 3개, 나사 4개를 `cv::parallel_for_` 로 동시에 찾는다. 하나씩 찾을 때와 결과가 같다. 속도는 OpenCV 스레드 수에 따라 달라진다(1스레드 17.6 ms, 4스레드 9.4 ms, 사진 읽기 제외).
- **위치**: [src/steps/01_find_qr.cpp](src/steps/01_find_qr.cpp#L72) 72줄, [src/steps/02_find_screws.cpp](src/steps/02_find_screws.cpp#L229) 229줄
- **바꾸려면**: 코드 대신 `cv::setNumThreads(n)` 으로 스레드 수를 정한다. 병렬을 아예 끄려면 `cv::setNumThreads(0)` 이다.

### 9-7. 실패를 알리는 방식

- **지금**: 어떤 함수도 예외를 던지지 않는다. 결과의 `status` 로 알리고, 실패면 `y_mm`, `t_deg` 가 NaN 이다. 안에서 생긴 예외도 잡아서 `E_INTERNAL` 로 돌려준다.
- **위치**: [src/alignment.cpp](src/alignment.cpp#L79) 79줄 `failed`
- **바꾸려면**: 라이브러리는 그대로 두고, 부르는 쪽에서 `status == Status::Failed` 일 때 예외를 던지는 감싸는 함수를 만드는 것을 권한다.

### 9-8. 숫자 형식

- **지금**: 계산은 모두 `double` 이다. OpenCV `matchTemplate` 결과만 `float` 이다.
- **바꾸려면**: `float` 으로 바꾸면 포팅 안내서의 확인 값이 소수 4자리에서 어긋날 수 있다. 바꾼다면 `show_steps` 로 단계마다 비교한다.

### 9-9. OpenCV 버전

- **지금**: OpenCV 3.4.15 에서 확인했다. 쓰는 함수는 `GaussianBlur`, `matchTemplate`, `minMaxLoc`, `Sobel`, `filter2D`, `magnitude`, `cvtColor`, `imdecode`, `FileStorage`, `parallel_for_` 이다. ECC 는 OpenCV 것을 쓰지 않고 직접 풀었다(1/32 px 끊김 문제, 포팅 안내서 2-3).
- **위치**: [CMakeLists.txt](CMakeLists.txt#L5) 5줄 `find_package(OpenCV REQUIRED)`
- **바꾸려면**: 다른 버전으로 빌드한 뒤 `check_photos` 로 예제 5장을 비교한다. 흐림과 NCC 의 끝자리 차이는 생길 수 있고, 소수 4자리까지 맞으면 된다.

### 9-10. 소스 파일 인코딩

- **지금**: 소스는 UTF-8 이고 주석과 알림 메시지가 한글이다. Visual Studio 는 `/utf-8` 옵션이 없으면 경고나 오류가 나고 메시지가 깨진다.
- **위치**: [CMakeLists.txt](CMakeLists.txt#L20) 20줄, 31줄 `/utf-8`
- **바꾸려면**: CMake 를 쓰지 않고 다른 프로젝트에 파일을 넣을 때도 같은 옵션을 켠다. GCC, Clang 은 기본이 UTF-8 이라 따로 할 일이 없다.

### 9-11. 파일 경로에 한글

- **지금**: 사진(`center.bmp`)은 바이트로 읽어서 한글 경로도 된다. `reference.json`, `settings.json` 은 OpenCV `FileStorage` 로 읽어서 Windows 에서 한글 경로를 못 읽을 수 있다.
- **위치**: [src/common/reference.cpp](src/common/reference.cpp#L29) 29줄, [src/setup/settings.cpp](src/setup/settings.cpp#L10) 10줄
- **바꾸려면**: 영문 경로에 두는 것이 가장 쉽다. 코드로 고치려면 `read_gray_image`(같은 파일 12줄)처럼 파일을 바이트로 읽은 뒤 `cv::FileStorage(내용, cv::FileStorage::READ | cv::FileStorage::MEMORY)` 로 연다.

### 9-12. 조정 값 (탐색 창, 기준값)

- **지금**: 탐색 창 크기, NCC 기준값, 반복 횟수 같은 숫자는 파일 맨 위 상수로 모아 두었다.
- **위치**: [src/steps/01_find_qr.cpp](src/steps/01_find_qr.cpp#L9) 9~14줄 (흐림 σ 2.0, 가로 ±110 → ±250 px, 세로 ±22 px, NCC 0.8), [src/steps/02_find_screws.cpp](src/steps/02_find_screws.cpp#L14) 14~34줄 (조각 반지름 45 px, ±60 → ±260 px, NCC 0.6 등), [src/setup/find_new_reference.cpp](src/setup/find_new_reference.cpp#L10) 10~15줄
- **바꾸려면**: 상수만 고치면 된다. 다만 흐림 σ, 조각 크기, 원 반지름은 계수와 점검식을 학습할 때의 값이라 바꾸면 계수도 다시 만들어야 한다. 탐색 창 크기와 반복 횟수는 바꿔도 계수와 상관없다. 바꾼 뒤에는 `check_photos` 로 예제 5장을 확인한다.

### 9-13. 옮긴 뒤 확인하는 법

1. 단계를 하나 옮길 때마다 `show_steps` 출력과 비교한다. 단계별 확인 값은 [docs/2_포팅안내서.md](docs/2_포팅안내서.md) 에 있다.
2. 다 옮긴 뒤 예제 5장의 최종 값을 `tests/photos/expected_steps.tsv` 와 비교한다. 소수 4자리까지 맞으면 된다.
3. 영점과 기준 다시 잡기는 `check_setup` 과 같은 순서로 확인한다.
