# QR 판 정렬 측정

사진 한 장에서 판이 기준 위치에서 얼마나 벗어났는지 **Y (mm), T (도)** 를 잰다.

- 입력: 3840×2160 사진 한 장
- 출력: `y_mm`, `t_deg`, 상태(정상 / 경고 / 실패), 알림 메시지
- 속도: 한 장 1스레드 19.2 ms, 4스레드 9.6 ms (노트북, SEMES 사진 715장 가운뎃값, 사진 읽기 제외. [docs/평가_방법/2_속도.md](docs/평가_방법/2_속도.md))
- 언어와 환경: C++17, OpenCV 3.4.15

| 문서 | 읽는 때 |
|---|---|
| [docs/0_문제정의와_목표.md](docs/0_문제정의와_목표.md) | 가장 먼저. 무슨 문제를 풀고 어디까지 풀었나, SEMES 와 정해야 할 것 |
| [docs/0_전체안내.md](docs/0_전체안내.md) | 처음 읽는 문서. 사진 넣기, 학습, 평가, 실사용을 한 번에 |
| [docs/0_코드구조.md](docs/0_코드구조.md) | 폴더, 파일, 함수가 하는 일과 읽는 순서 |
| [docs/1_한눈에보기.md](docs/1_한눈에보기.md) | 처음 5분. 그림으로 보는 원리와 단계 |
| [docs/2_포팅안내서.md](docs/2_포팅안내서.md) | 코드를 옮길 때. 함수마다 순서와 확인 값 |
| [docs/구현_방법/](docs/구현_방법/) | 어떻게 만들었고 왜 이 방법인가. 사진 한 장이 거치는 순서대로 step 0 ~ 4 |
| [docs/평가_방법/](docs/평가_방법/) | 어떻게 재고 얼마나 나왔나. 정확도, 속도, 강건성, 부품별 정밀도, 직접 평가하는 법 |
| [docs/추가_논의/](docs/추가_논의/) | 더 이야기할 주제. 영점 자동 보정, 먼 자세의 T 오차, 적용 환경 |

## 1. 바로 써 보기

```cpp
#include "alignment.hpp"

alignment::initialize("data/model");                                   // 한 번
alignment::Result r = alignment::measure_alignment(photo);       // 사진마다
// r.y_mm, r.t_deg  → 보정 이동은 (−r.y_mm, −r.t_deg)
// r.status, r.message → 경고나 실패가 있으면 여기에
```

아무것도 정하지 않으면 아래 기본값으로 동작한다.

| 항목 | 기본 | 바꾸는 곳 |
|---|---|---|
| QR 재는 법 | 흐림 NCC | (고정) |
| 나사 | 4개 모두 | `config/settings.json` 의 `screws` |
| 영점 | 절대 영점(학습한 모든 회차의 공통 0)이 (0, 0). 매일 명령 0 사진으로 다시 잡는다 ([docs/구현_방법/step0_준비와_영점.md](docs/구현_방법/step0_준비와_영점.md)) | `set_zero_from_photos` |
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
build/Release/check_photos data/model data/sample/list.tsv out.tsv data/sample/expected.tsv
→ 기대값과 비교 5장: Y 차이 최대 0.00000000 mm, T 차이 최대 0.00000000 도
```

사진 한 장의 단계별 값을 본다.

```
build/Release/measure_one data/model data/sample/sample_1.png
```

받은 사진으로 정확도를 평가한다. 인자 없이 실행하면 영점 사진, 평가할 폴더를 차례로 묻고, 끝나면 결과를 저장한 곳을 알려 준다 (`docs/0_전체안내.md` 3-3).

```
build/Release/score_photos
```

받은 사진으로 계수를 새로 만든다. 인자 없이 실행하면 사진 폴더를 묻고, 사진 재기부터 정확도 확인까지 차례로 한다. 지금 쓰는 `data/model` 은 바꾸지 않는다 (`docs/0_전체안내.md` 3-1).

```
build/Release/train
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
data/
  model/                     측정할 때 읽는 파일: 기준 사진, 위치, 계수, 점검식 (장비에는 이것만 보낸다)
  new_photos/                설비에서 받은 사진을 넣는 곳 (Git 에 올리지 않음)
  sample/                    확인용 예제 사진 5장과 기대값
  results/                   도구가 만든 결과: train/ (새로 학습한 계수), eval/ (평가 결과). Git 에 올리지 않음
tools/
  try/                       써 보기와 확인
    measure_one                사진 한 장 재 보기
    show_steps                 단계마다의 값을 모두 찍기 (옮긴 코드와 비교할 때)
    check_photos, check_setup  예제 사진으로 빌드가 맞는지 확인
    make_figures               문서의 그림 만들기
  train/                     계수를 다시 만들 때: train (물어보며 한 번에), prepare_dataset, fit_coefficients
  evaluate/                  성능을 잴 때: evaluate_accuracy, measure_speed, score_photos
  environment/               다른 설치 환경의 화면을 만들어 반응 보기
  common/                    도구가 같이 쓰는 코드
docs/                        문서와 그림
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

## 5. 상태와 알림 (정상, 경고, 실패)

어떤 함수도 예외를 던지지 않는다. 결과의 `status`, `code`, `message` 로 알리고, 같은 내용을 콘솔(stderr)에 한 줄 쓴다. `log_file` 을 정하면 파일에도 쓴다.

| 상태 | 뜻 | 값 | 로봇 |
|---|---|---|---|
| 정상 (`Ok`) | 그대로 믿어도 된다 | 나옴 | 그 값으로 보정한다 |
| 경고 (`Warning`) | 값은 나오지만 정확도가 떨어질 수 있다 | 나옴 | 표의 "다음 할 일"대로 옮기고 다시 잰다 |
| 실패 (`Failed`) | 잴 수 없다 | 없음 (NaN) | 멈춘다 |

- 보정 이동은 언제나 (−`y_mm`, −`t_deg`) 이다.
- 콘솔에는 `[경고 W_COARSE] 나사가 한쪽에만 …` 처럼 `[상태 코드] 메시지` 한 줄이 나온다. 정상이면 아무것도 쓰지 않는다(참고 알림 N_… 만 `[알림 …]` 으로 쓴다).
- 알림이 여러 개면 `code` 에는 표의 위쪽(더 중요한 것) 하나만, `message` 에는 모두 ` / ` 로 이어서 들어간다. 상태는 하나라도 실패면 실패다.
- 아래 "위치"는 이 폴더 기준이다. "예상되는 위험"의 숫자는 SEMES 정지 사진 702장으로 확인한 것이고, 확인하지 못한 것은 "검증 전"으로 적었다.

### 5-1. 정상

| 코드 | 조건 | 예상되는 위험 | 다음 할 일 | 위치 |
|---|---|---|---|---|
| (없음) | 아래 경고, 실패 조건이 하나도 없다 | 없음. 개발 사진 560장을 5묶음으로 나눠 학습에 안 쓴 묶음으로 돌아가며 시험하기를 10번 반복해, 560장 중 평균 0.2장이 허용오차(Y 0.045 mm, T 0.01도) 밖. 처음 보는 촬영 회차 98.2% 통과 | 1. 콘솔에 아무것도 쓰지 않는다<br>2. 로봇을 (−`y_mm`, −`t_deg`) 만큼 옮긴다 | - |
| `N_WIDE_SEARCH` | ±110 px 창에서는 QR 3개를 다 못 찾았고, ±250 px 창에서 3개를 모두 찾았다 | 없음. 넓은 창에서도 정확도는 같다. 약 6 ms 더 걸린다 | 1. 콘솔에 `[알림 N_WIDE_SEARCH] QR 이 110 px 넘게 움직여 넓은 창(±250 px)으로 찾았습니다`<br>2. 정상과 같이 보정한다 | `src/alignment.cpp` |

### 5-2. 경고 (값은 나온다)

| 코드 | 조건 | 예상되는 위험 | 다음 할 일 | 위치 |
|---|---|---|---|---|
| `W_COARSE` | 찾은 나사가 왼쪽(1, 3)이나 오른쪽(2, 4) 한쪽에만 있거나 하나도 없어, 정밀하지 않은 계수 조합을 썼다 | **대략값이다.** Y 와 T 를 제대로 가를 수 없다. 한쪽 나사만 쓴 식은 560장 중 128~170장이 허용오차 밖이었다. 방향과 크기만 믿는다. 더 확인하려면: 실제로 나사가 화면 밖으로 나가는 큰 자세의 사진이 필요하다 | 1. 콘솔에 `[경고 W_COARSE] 나사가 한쪽에만 보이거나 보이지 않아 대략값입니다 (못 찾은 나사: …). 이 값으로 센터 쪽으로 옮긴 뒤 다시 측정하세요`<br>2. 로봇을 (−`y_mm`, −`t_deg`) 만큼 옮긴다<br>3. `measure_alignment` 를 다시 부른다<br>4. 경고가 없어질 때까지 반복한다 (반복 횟수 상한은 SEMES 와 정할 것) | `src/alignment.cpp`, 정밀 판정 `src/steps/03_compute_yt.cpp` |
| `W_QR_PARTIAL` | 사진 전체 폭까지 넓혀도 QR 을 1~2개만 찾았다 (NCC 0.8 이상이고 봉우리가 창 가장자리가 아닌 것만 찾은 것으로 본다). 찾은 QR 만 평균한다 | **검증 전.** QR 이 화면 끝에 걸리거나 일부가 가려진 실제 사진으로는 확인하지 못했다. 정확도가 떨어질 수 있다. 더 확인하려면: QR 하나를 가린 합성 사진, QR 이 화면 끝에 걸린 실제 사진으로 오차를 잰다 | 1. 콘솔에 `[경고 W_QR_PARTIAL] QR 3개 중 N개만 찾았습니다. QR 이 카메라 안에 다 들어오게 해 주세요. 값이 부정확할 수 있습니다`<br>2. 로봇을 (−`y_mm`, −`t_deg`) 만큼 옮긴다<br>3. `measure_alignment` 를 다시 부른다<br>4. QR 3개가 다 보일 때까지 반복한다. 그대로이면 가림을 확인한다 | `src/alignment.cpp`, 탐색 `src/steps/01_find_qr.cpp` |
| `W_VALUES_DISAGREE` | 점검값(값마다 \|측정값 − 기대값 − 기준 어긋남\| ÷ scale 중 가장 큰 것)이 그 나사 조합의 `threshold` 를 넘었다 (4개 다 쓰면 5.195). 나사 2개 이상 조합에만 점검식이 있다. `fail_on_disagreement` 가 0(기본)일 때 | **틀렸을 수 있다.** 나사 하나가 가려져 엉뚱하게 재진 경우(점검이 없으면 Y 최대 3 mm 틀림), Y 가 0.1 mm 넘게 틀린 경우는 모두 이 경고가 났다(자체 장비 사진). 0.1 mm 아래의 오차는 못 잡을 수 있다. 정상 사진의 괜한 경고: SEMES 702장 0장, 영점을 날마다 잡으면 0.2% | 1. 콘솔에 `[경고 W_VALUES_DISAGREE] (QR 또는 나사 번호) 값이 다른 값들과 맞지 않습니다. 가림·반사·이물을 확인하세요. 모든 사진에서 나오면 영점을 다시 잡으세요`<br>2. 이 값으로 보정하지 않는다<br>3. 가림, 반사, 이물을 확인하고 `measure_alignment` 를 다시 부른다<br>4. 계속 나오면 센터에서 `set_zero_from_photos` 를 부른다 | `src/alignment.cpp`, 점검 `src/steps/04_check_result.cpp` |
| `W_SCREW_MISSING` | 좌우 나사가 남아 정밀 계수를 썼지만, 설정에서 쓰기로 한 나사 중 못 찾은 것이 있다 | 작다. 나사 하나를 가렸을 때 가리기 전과 Y 0.007 mm, T 0.0006도 차이 | 1. 콘솔에 `[경고 W_SCREW_MISSING] 나사 N 을 찾지 못해 나머지 나사로 계산했습니다. 가림·이물을 확인하세요`<br>2. 그 값으로 보정해도 된다<br>3. 다음 측정 전에 나사의 가림, 이물을 확인한다 | `src/alignment.cpp` |
| `W_FAR` | ±250 px 창에서도 다 못 찾아, 사진 전체 폭에서 QR 3개를 모두 찾았다 | **검증 전.** 사진을 통째로 옮긴 합성 시험에서는 ±600 px 까지 값이 나왔다. 실제로 멀리 간 자세(렌즈 가장자리 왜곡, 원근 변화)에서는 확인하지 못했다. 더 확인하려면: 실제 큰 자세 사진으로 구간별 오차를 잰다 | 1. 콘솔에 `[경고 W_FAR] QR 이 기준에서 250 px 넘게 떨어져 있습니다. 값이 부정확할 수 있으니 옮긴 뒤 다시 측정하세요`<br>2. 로봇을 (−`y_mm`, −`t_deg`) 만큼 옮긴다<br>3. `measure_alignment` 를 다시 부른다<br>4. 경고가 없어질 때까지 반복한다 | `src/alignment.cpp` |
| `W_OUT_OF_RANGE` | 영점을 빼기 전 값(계산값 + 기준을 다시 잡을 때 이어 붙인 값)이 `data/model/reference.json` 의 학습 범위(Y −2.0372 ~ 1.9628 mm, T −1.00023 ~ 1.01252도) 밖이다 | 떨어질 수 있다. 범위 밖 예측 시험에서 Y 는 비례 오차 0.3%(허용오차 밖 없음), T 는 가장자리에서 1~1.5% 크게 읽었다. 예: T 2도면 0.02~0.03도 틀려 허용오차를 넘는다. 실제 범위 밖 사진으로는 확인하지 못했다 | 1. 콘솔에 `[경고 W_OUT_OF_RANGE] 학습한 범위(Y -2.04 ~ 1.96 mm, T -1.00 ~ 1.01 도) 밖입니다`<br>2. 로봇을 (−`y_mm`, −`t_deg`) 만큼 옮긴다<br>3. `measure_alignment` 를 다시 부른다<br>4. 범위 안에 들어오면 그 값은 정밀값이다 | `src/alignment.cpp`, 판정 `src/steps/04_check_result.cpp` |
| `W_ZERO_RESET` | `set_reference_from_photos` 에서, 그 사진을 바꾸기 전 기준으로 QR 이나 좌우 나사를 재지 못했다. 그 사진을 새 (0, 0) 으로 삼고 영점도 0 으로 되돌린다. 이 함수의 결과에만 붙는다 | 영점이 바뀐다. 그 센터 사진이 실제 센터가 아니면 이후 모든 값이 그만큼 치우친다 | 1. 콘솔에 `[경고 W_ZERO_RESET] 이전 기준으로는 잴 수 없어 영점을 이을 수 없었습니다. 이 사진을 (0, 0) 으로 삼았습니다`<br>2. 영점이 바뀌었음을 기록한다<br>3. 로봇이 실제 센터에 있는지 확인하고, 필요하면 `set_zero_from_photos` 를 부른다 | `src/alignment.cpp` |

### 5-3. 실패 (값이 없다, 멈춘다)

| 코드 | 조건 | 예상되는 위험 | 다음 할 일 | 위치 |
|---|---|---|---|---|
| `E_SETTINGS` | (`initialize`) 설정 파일을 열지 못했다. `screws` 에 1~4 밖의 번호가 있다. 왼쪽(1, 3)이나 오른쪽(2, 4) 나사가 하나도 없다. 설정 파일 경로를 비우면 기본값을 쓰고 오류가 나지 않는다 | 시작할 수 없다. 그 전에 성공한 상태가 있으면 그대로 남는다 | 1. 콘솔에 `[실패 E_SETTINGS] 설정 파일을 읽지 못했습니다: (경로)` 또는 `설정의 screws 는 1~4 여야 합니다` 등<br>2. 로봇을 멈춘다<br>3. 설정 파일 경로와 값을 고친 뒤 `initialize` 를 다시 부른다 | `src/alignment.cpp`, `src/setup/settings.cpp` |
| `E_DATA` | (`initialize`) `center.bmp` 를 못 읽었거나 3840×2160 이 아니다. `reference.json` 을 못 열었거나, QR 3개와 나사 4개가 아니거나, QR 상자가 사진 밖이다. 계수 파일 16개 중 하나라도 없거나 잘못됐다. 점검식 파일이 잘못됐다(없는 것은 건너뛴다). 초기화 중 예외가 났다 | 시작할 수 없다 | 1. 콘솔에 `[실패 E_DATA] center.bmp 를 읽지 못했습니다: (경로). 받은 model 폴더를 그대로 다시 복사해 주세요` 등<br>2. 로봇을 멈춘다<br>3. 받은 `data` 폴더를 영문 경로에 그대로 다시 복사하고 `initialize` 를 다시 부른다 | `src/alignment.cpp` |
| `E_NOT_INITIALIZED` | `initialize` 가 한 번도 성공하지 않은 상태에서 `measure_alignment`, `set_zero_from_photos`, `set_reference_from_photos` 를 불렀다 | 잴 수 없다 | 1. 콘솔에 `[실패 E_NOT_INITIALIZED] 먼저 initialize 를 불러 주세요`<br>2. 로봇을 멈춘다<br>3. `initialize` 를 부른 뒤 다시 시작한다 | `src/alignment.cpp` |
| `E_PHOTO` | 사진이 비었다. 3840×2160 이 아니다. 8비트가 아니다. 채널이 1, 3, 4 가 아니다. (`set_zero_from_photos`, `set_reference_from_photos`) 사진을 한 장도 넘기지 않았거나 하나라도 위 형식이 아니다 | 잴 수 없다. 영점, 기준은 그대로다 | 1. 콘솔에 `[실패 E_PHOTO] 사진 형식이 다릅니다 (받은 크기 W×H). 3840×2160 8비트 사진을 넣어 주세요` 등<br>2. 로봇을 멈춘다<br>3. 카메라 설정(해상도, 비트 수)을 확인한다 | `src/alignment.cpp`, `src/common/reference.cpp` |
| `E_QR_NOT_FOUND` | ±110 px, ±250 px, 사진 전체 폭 어디에서도 QR 을 하나도 찾지 못했다 | 잴 수 없다. 판이 카메라 시야 밖이거나 QR 이 모두 가려졌다. 사진을 통째로 옮긴 시험에서는 2300 px 에서 이 실패가 났다 | 1. 콘솔에 `[실패 E_QR_NOT_FOUND] QR 을 찾지 못했습니다. QR 이 카메라 안에 보이는지, 가려지거나 반사되지 않았는지 확인해 주세요`<br>2. 로봇을 멈춘다<br>3. 판을 카메라 안으로 되돌리는 방법은 SEMES 와 정할 것 (수동 이동, 마지막 위치로 복귀 등) | `src/alignment.cpp` |
| `E_ZERO` | (`set_zero_from_photos`) 영점 사진(여러 장이면 평균)에서 QR 을 못 찾았거나, 좌우 나사를 못 찾아 정밀 계수를 쓸 수 없다 | 영점을 바꾸지 못한다. 이전 영점이 그대로다 | 1. 콘솔에 `[실패 E_ZERO] 영점 사진에서 QR 과 좌우 나사를 찾지 못해 영점을 바꾸지 않았습니다. 화면이 많이 밀렸으면 set_reference_from_photos 를 먼저 불러 주세요`<br>2. 로봇을 멈춘다<br>3. 영점 사진(가림, 반사)을 확인하고, 화면이 밀렸으면 `set_reference_from_photos` 를 먼저 부른 뒤 `set_zero_from_photos` 를 다시 부른다 | `src/alignment.cpp` |
| `E_REFERENCE` | (`set_reference_from_photos`) 새 센터 사진에서 QR 상자를 못 찾았다(±120 px, NCC 0.5 이하). 나사 중심을 못 찾았다(범위가 사진 밖이거나 테두리가 없다). 또는 새 기준으로 QR 이나 좌우 나사를 재지 못해 원래 기준으로 되돌렸다 | 기준을 바꾸지 못한다. 이전 기준이 그대로다 | 1. 콘솔에 `[실패 E_REFERENCE] 새 센터 사진에서 QR N 을 찾지 못했습니다. 기준은 바꾸지 않았습니다` 또는 `새 기준으로 QR 과 좌우 나사를 재지 못했습니다. 기준은 바꾸지 않았습니다`<br>2. 로봇을 멈춘다<br>3. 판이 센터에 있고 QR, 나사가 모두 보이는지 확인한 뒤 다시 부른다 | `src/alignment.cpp` |
| `E_INTERNAL` | 측정, 영점, 기준 함수 안에서 예상하지 못한 예외가 났다 | 잴 수 없다 | 1. 콘솔에 `[실패 E_INTERNAL] 측정 중 오류: (내용)` 등<br>2. 로봇을 멈춘다<br>3. 로그와 그 사진을 보내 준다 | `src/alignment.cpp` |
| `W_VALUES_DISAGREE` (실패로 설정) | 위 경고와 조건이 같고 `config/settings.json` 의 `fail_on_disagreement` 가 1 이다 | 경고와 같다. 지금 코드는 실패여도 `y_mm`, `t_deg` 를 채워서 돌려준다 | 1. 콘솔에 `[실패 W_VALUES_DISAGREE] …`<br>2. 로봇을 멈춘다 (값이 있어도 쓰지 않는다)<br>3. 가림, 반사를 확인한 뒤 다시 잰다 | `src/alignment.cpp` |

`set_zero_value`, `clear_zero` 는 상태를 돌려주지 않아 어떤 경우에도 실패하지 않는다.

## 6. 날마다 하는 설정

| 함수 | 하는 일 | 언제 |
|---|---|---|
| `set_zero_from_photos(명령 0 사진들)` | 로봇을 명령 0 으로 보내 찍은 사진의 자세를 (0, 0) 으로 삼는다. 5~10장을 넘기면 평균을 쓴다. 명령값을 모르는 "센터 사진"은 쓰지 않는다 ([docs/구현_방법/step0_준비와_영점.md](docs/구현_방법/step0_준비와_영점.md)) | 날마다. 카메라·조명·판을 건드린 뒤에는 꼭 |
| `set_zero_value(y, t)` / `clear_zero()` | 적어 둔 영점을 다시 넣기 / 처음 영점으로 되돌리기 | 프로그램을 다시 켰을 때 |
| `read_photo(path)` | 사진 파일을 회색으로 읽기 (한글이 든 경로도 읽힘) | 파일에서 사진을 읽어 `measure_alignment` 에 넣을 때 |
| `set_reference_from_photos(센터 사진들)` | QR·나사 위치를 이 사진에서 다시 찾는다. 영점은 이어진다 | 화면이 많이 밀려 QR·나사를 못 찾을 때만 |

영점을 날마다 잡아야 하는 이유: 카메라가 0.05도만 돌아도 Y 가 0.27 mm 틀어진다. 이런 화면 전체의 변화는 영점을 다시 잡으면 없어진다.

## 7. 확인한 것 (SEMES 사진 702장)

| 확인 | 결과 |
|---|---|
| 이 폴더의 코드와 정리 전 모듈의 값 | 702장 모두 같음 (차이 0) |
| 같은 촬영들 안의 처음 보는 사진 (572장을 5묶음으로, 10번) | 100% (허용오차 Y 0.045 mm, T 0.01도 밖 0장). 떼어 둔 마지막 확인 130장은 99.2% |
| 처음 보는 날 (받은 명령값으로 채점) | 그날 영점을 잡으면 95.5%, 잡지 않으면 31.5%. 자세한 것은 [docs/평가_방법/1_정확도.md](docs/평가_방법/1_정확도.md) |
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
- `data/model` 폴더 경로에 한글이 있으면 Windows 에서 설정·위치 파일을 읽지 못할 수 있다. 영문 경로에 둔다.

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
- **위치**: 크기는 [src/common/reference.hpp](src/common/reference.hpp#L9) 9~10줄 `kPhotoWidth`, `kPhotoHeight`. 좌표는 `data/model/reference.json` 의 `qr`, `screws`
- **바꾸려면**
  - 화면이 조금 밀린 정도(카메라가 처지거나 살짝 움직임)면 코드는 그대로 두고 `set_reference_from_photos` 를 부른다(6절).
  - 해상도나 잘라내기(ROI)를 바꾸면 상수만 바꿔서는 안 된다. 기준 사진, 좌표, 계수를 모두 새 조건으로 다시 만들어야 한다(9-3).

### 9-3. 계수를 다시 만들어야 하는 경우

- **지금**: `data/model/coefficients/` 와 `data/model/checks/` 는 SEMES 사진 702장(Y ±2 mm, T ±1도)으로 학습한 결과다. **학습 코드는 이 패키지에 들어 있지 않다.**
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
  - 코드 수정 없이: 켤 때마다 명령 0 사진으로 `set_zero_from_photos` 를 다시 부른다. 기준을 다시 잡았던 장비라면 그 앞에 `set_reference_from_photos` 도 부른다. 가장 단순하고 안전하다.
  - 코드 수정: `alignment.cpp` 에 `state.reference`, `state.zero`, `state.carry_over`, `state.baselines` 를 파일로 쓰고 읽는 함수를 더하고 `include/alignment.hpp` 에 선언한다. 기준 사진은 `center.bmp`, 위치는 `reference.json` 형식으로 쓰면 그 폴더를 `initialize` 에 그대로 넘길 수 있다.

### 9-5. 여러 스레드, 여러 카메라

- **지금**: 상태가 프로그램 전체에 하나뿐이고(`state`), 모든 공개 함수가 같은 잠금(`state_mutex`)을 건다. 여러 스레드에서 불러도 안전하지만 **한 번에 한 장씩** 잰다. 카메라 여러 대를 각각 다른 기준으로 재는 구조는 지원하지 않는다.
- **위치**: [src/alignment.cpp](src/alignment.cpp#L58) 58~59줄
- **바꾸려면**
  - 카메라마다 프로그램(프로세스)을 따로 띄운다. 코드 수정이 없다.
  - 한 프로그램에서 하려면 `State` 와 잠금을 클래스 멤버로 옮기고, 공개 함수 6개를 그 클래스의 함수로 바꾼다. 단계 코드(`src/steps/`)는 이미 상태를 갖지 않아서 고칠 필요가 없다.

### 9-6. 안쪽 병렬 처리와 속도

- **지금**: QR 3개, 나사 4개를 `cv::parallel_for_` 로 동시에 찾는다. 하나씩 찾을 때와 결과가 같다. 속도는 OpenCV 스레드 수에 따라 달라진다(1스레드 19.2 ms, 4스레드 9.6 ms, 사진 읽기 제외).
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
2. 다 옮긴 뒤 예제 5장의 최종 값을 `data/sample/expected_steps.tsv` 와 비교한다. 소수 4자리까지 맞으면 된다.
3. 영점과 기준 다시 잡기는 `check_setup` 과 같은 순서로 확인한다.
