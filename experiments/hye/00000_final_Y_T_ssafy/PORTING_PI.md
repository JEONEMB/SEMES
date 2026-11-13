# 00000_final_Y_T_ssafy — 라즈베리파이 포팅 안내 (2026-11-13)

혜원의 우리 장비용 Y·T 추정 모듈을 Pi(라즈베리파이 5, Linux, OpenCV 3.4.15)와 확인 앱(`verify_web`)에 붙이기 위한 안내다.
**Windows(MSVC v141)에서만 빌드·검증했다. Linux 빌드와 Pi 실측은 아직 없다.** 아래 "확인할 것" 의 숫자와 맞는지 봐 주세요.

## 1. 대상

| 항목 | 값 |
|---|---|
| 브랜치 | `feature/hye/ssafy-reference` (MR 대상 `chore/hye/merge`). 이 안내문이 들어간 커밋 이후 |
| 모듈 폴더 | `experiments/hye/00000_final_Y_T_ssafy` |
| 모델 | `final00_huber7_qrnccb2v5_rig20261112_1959_irls6_r1e-7` (11-12 장비 사진 1959장, 값 5개, QR 흐림 NCC) |
| 다른 모델 | `00000_final_Y_T_ssafy_2` 는 코드가 같고 `assets/` 의 `model7.tsv`, `provenance.json` 만 다르다(두 날 3638장). `00000_final_Y_T_semes` 도 코드가 같고 SEMES 사진용 assets 다(QR 536점 추적, 바 사용). 바꿀 때는 assets 폴더만 바꾼다 |

## 2. 무엇을 하나 (요약)

사진 한 장에서 값 5개를 재고 직선식으로 (Y mm, T 도)를 낸다. 코드는 값 7개(바 기울기 2개 포함)를 잴 수 있지만, 지금 모델은 바 가중치가 0 이라 바를 재지 않는다.

| 값 | 재는 법 |
|---|---|
| QR 값 1개 | QR 3개를 각각 센터 사진의 QR 조각과 NCC(센터 조각과 사진을 가우시안 σ2 로 흐린 뒤, 정수 봉우리 + 3점 포물선)로 맞춘 가로 이동의 평균. `assets/site.json` 의 `"qr_method": "ncc"` 로 고른다(없으면 536점 추적) |
| 나사 4개 | 나사 주변 기울기 영상을 센터 사진과 ECC 평행이동으로 맞춘 가로 이동 |

- 측정 기준은 `assets/center.bmp`(센터 사진)이다. 카메라가 처져 화면이 날마다 세로로 밀려도 값 5개는 가로 이동만 보므로 측정된다(11-13 역순 사진: 세로 14 px 밀림, 등록 없이 42/42 측정·정답 일치, 평균 Y −0.3 µm, T +0.6 mdeg). 기본 영점은 절대 영점이고, 그날 센터를 (0, 0) 으로 쓰려면 영점을 업데이트한다(6절).
- 처리 시간(Windows 노트북): 보통 9~12 ms(4스레드), 18~21 ms(1스레드). 특정 자세에서 나사 맞춤 반복이 길어 최대 56~65 ms. Pi 는 약 3배로 예상(보통 30~36 ms).

## 3. 필요한 파일

```
experiments/hye/00000_final_Y_T_ssafy/
  CMakeLists.txt          semes_joint 라이브러리
  include/ src/           모듈 소스 (src/tracker.cpp, src/spline.cpp 는 빌드하지 않는 기록용)
  third_party/            ecc_compat (기준 다시 잡기의 QR 호모그래피)
  assets/                 center.bmp, points.json, geometry.json, line_protocol.json, site.json, model7.tsv, health.tsv, provenance.json
                          (+ 다른 QR 방식용 model7_track536.tsv, health_track536.tsv)
  tool/standalone/        모듈 + final00_tool 만 빌드하는 CMake
include/alignment.hpp, include/detail/alignment_diagnostics.hpp, src/alignment_diagnostics.cpp   (저장소 공용)
src/alignment.cpp         alignment() → joint() 연결 (확인 앱 기본 빌드에서 씀)
```

- `assets/` 의 파일은 바이트가 바뀌면 안 된다(`provenance.json` 의 SHA-256 검사). `.gitattributes` 가 `experiments/**` 의 줄바꿈 변환을 막는다. 복사할 때 텍스트 변환을 하지 않는다.

## 4. Pi 빌드

```bash
# 저장소 루트에서. 모듈 + 도구만 (OpenCV 3.4.15 가 아니면 CMake 가 멈춘다)
cmake -S experiments/hye/00000_final_Y_T_ssafy/tool/standalone -B build_ssafy -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-include $(pwd)/verify_web/native/linux_prelude.hpp"
cmake --build build_ssafy -j4
```

- **`error_t` 충돌**: 저장소 `include/alignment.hpp` 의 `typedef double error_t;` 가 glibc 의 `error_t` 와 겹친다. 확인 앱처럼 `linux_prelude.hpp` 를 강제 include 하거나, 팀 헤더의 이름을 바꾼다(팀 공용 헤더라 이 브랜치에서는 고치지 않았다).
- SHA-256 은 순수 C++(`src/sha256.cpp`)로 바꿔 두어 Windows/macOS 전용 코드가 없다.
- `final00_tool` 의 사진 병렬 처리는 MSVC `/openmp` 로만 켜져 있다. Linux 에서는 `#pragma omp` 가 무시되어 한 장씩 처리한다(결과는 같다). 모듈 안의 QR·나사 병렬은 `cv::parallel_for_` 라 그대로 동작한다.
- `linux_prelude.hpp` 는 `origin/feature/lye/verify-demo-ui` 의 `verify_web/native/` 에 있다.

## 5. API

```cpp
#include "alignment.hpp"          // 저장소 공용: Alignment_t, initialize_joint, joint, alignment
#include "joint_reference.hpp"    // 모듈 include 폴더: 영점, 측정 기준

initialize_joint("/path/to/00000_final_Y_T_ssafy/assets");   // 한 번. 해시가 다르면 예외. 기본 영점 = 절대 영점, QR 방식 = site.json(ncc)
// QR 방식을 바꾸려면 위 대신: initialize_joint_method(assets, "track536");  // 536점 추적. 느리지만(사진마다 약 +25 ms) 정밀. README "QR 방식 고르기"
Alignment_t z = set_joint_zero(center_frames);   // (선택) 영점 업데이트. 1장(cv::Mat) 또는 여러 장(std::vector<cv::Mat>, 화소 평균)
Alignment_t p = joint(image);                    // 또는 alignment(image). p.y [mm], p.t [도]
```

| 항목 | 내용 |
|---|---|
| 입력 | 3840×2160 8비트, 1·3·4채널(내부에서 회색으로 바꾼다). **데이터셋과 같은 왜곡 보정·SEMES 화각 크롭**, 게인 3.0, 노출 8333 µs. 회색 1채널로 주면 변환 시간(1스레드 약 3 ms)이 빠진다 |
| 출력 | 영점 기준 현재 자세. Y mm, T 도. 기본 영점은 절대 영점(11-12 센터 사진과 같은 화면). **부호는 우리 스테이지 명령 방향**이라 보정 명령 = −출력(확인 앱 `--algo-output offset`) |
| 실패 | **예외를 던진다**(NaN 을 돌려주지 않는다). 확인 앱 `align_worker` 는 예외를 `ERR <메시지>` 로 바꾼다. 진단은 `alignment_diagnostics()` |
| 대표 실패 메시지 | `QR n NCC 가 낮다`, `QR n 이 탐색 창 끝에 걸렸다`(움직임 > 가로 ±125 px 또는 세로 ±22 px), `Pin NCC below 0.6`(나사), `나사 기울기 영상 맞춤에 실패했다`. 바를 재지 않으므로 `Line normal-profile match failed` 는 나오지 않는다 |
| 경고 (측정값 일치 점검) | 값은 그대로 내고 `alignment_diagnostics()` 에 남긴다: `normalized_cost_margin > 1` 이면 경고, `message` 가 `WARNING 측정값이 서로 맞지 않는다(…)` 로 시작. **확인 앱은 `joint()` 뒤에 이 값을 보고 화면에 경고를 띄운다.** 나사 하나가 가려지거나 반사되면 Y 가 크게(최대 3 mm) 틀린 값이 경고와 함께 나온다. `set_joint_health_check(true)` 를 부르면 그때 예외를 던진다 |
| 스레드 | `joint`, 영점·측정 기준 함수는 내부 잠금으로 한 번에 하나씩 처리한다 |
| `initialize_joint` 를 다시 부르면 | 영점과 측정 기준 등록이 지워지고 절대 영점으로 돌아간다 |

## 6. 영점

| 함수 | 하는 일 |
|---|---|
| (기본) | 절대 영점. 값 5개 모델은 날이 바뀌어 화면이 세로로 밀려도 등록 없이 측정되고 정답과 맞았다(11-13 역순 42/42, 평균 Y +1.2 µm, T +0.4 mdeg) |
| `set_joint_zero(사진들)` | 그 사진의 지금 출력을 영점으로 기억하고 이후 출력에서 뺀다(측정 기준은 그대로). 돌려주는 값 = 그 사진의 절대 영점 기준 자세. 성현 `alignment_set_reference` 와 같은 뜻 |
| `set_joint_zero_value(y, t)` / `clear_joint_zero()` / `joint_zero()` | 저장해 둔 영점 넣기(재시작 뒤 복원) / 절대 영점으로 / 지금 영점 읽기 |
| `set_joint_reference(사진들)` | **측정이 실패할 만큼 화면이 크게 밀렸을 때만.** 측정 기준(센터 사진, QR 상자, 630점, 나사 중심)을 그 사진으로 다시 만든다. 같은 사진을 바꾸기 전 기준으로도 재서 절대 영점과 영점을 잇는다(`absolute_kept`). 바꾸기 전 기준으로 잴 수 없으면 절대 영점을 잃고 그 사진이 (0, 0). 새 자료는 `/tmp/final00_reference_*`(약 8 MB, 재부팅 때 지워짐) |

영점 업데이트 절차(쓸 때):

1. 센터로 이동할 때 정렬과 같은 방향(−쪽 접근점을 지나 +방향)으로 들어간다(Θ 백래시 약 0.023°).
2. 흔들림이 멎은 뒤 **5~10장**을 찍어 한 번에 넘긴다(화소 평균). 그 순간의 스테이지 오차가 영점에 들어간다(11-13 센터 명령 사진 18장이 33분 동안 T 24 mdeg 범위).
3. 돌려받은 값(그날 센터의 절대 자세)을 기록해 두면 스테이지·카메라 상태를 날마다 볼 수 있고, `set_joint_zero_value` 로 복원할 수 있다.
4. 실패하면 예외가 나고 이전 영점이 그대로 남는다.

**영점 업데이트를 할지**: 업데이트하면 목표가 "그날 센터 명령 자리" 가 되고 그 순간의 스테이지 오차가 들어간다. 업데이트하지 않으면 목표는 절대 영점(11-12 센터 화면)이다. Pi 에서 두 방식으로 복귀 시험을 해 보고 정한다.

**영점 업데이트를 권하는 이유 (영상 교란 시험, 2026-11-13)**: 카메라가 0.05° 만 돌아도 Y 가 0.09 mm(SEMES 사진 0.27 mm), 화면이 가로로 2 px 밀리면 T 가 0.023 틀린다. 이런 화면 전체 변화는 그날 영점을 다시 잡으면 거의 모두 없어진다. 또 측정값 일치 점검도 그날 센터를 기준으로 삼아, 날이 바뀐 날의 괜한 경고가 15% → 0.2% 로 준다. **카메라·조명·판을 건드린 뒤에는 꼭 다시 잡는다.**

## 7. 확인 앱(verify_web) 에 붙이기

`align_worker` 는 두 방식으로 빌드된다.

- **기본 빌드**(`ALGO_SOURCES`, `ALGO_INCLUDES`): `alignment()` 만 부른다. 절대 영점으로 쓰면 이것으로 충분하다. 영점 등록(`ALGR`)은 "지원하지 않음" 이 된다.
- **`ALGO_DIR` 빌드**(`ALGO_HAS_REFERENCE`): 성현 API 를 부른다. 영점 등록 버튼을 쓰려면 아래 함수를 이 모듈 위에 얇게 구현한 어댑터가 필요하다(**작성·검증하지 않은 스케치**).

```cpp
// ksh_api_adapter.cpp — 성현 API 이름을 이 모듈 호출로 옮긴다
static std::string g_assets = "/path/to/00000_final_Y_T_ssafy/assets";
static AlignmentInfo g_last; static ReferenceInfo g_ref;
bool alignment_init(const std::string& /*config*/) { initialize_joint(g_assets); return true; }
bool alignment_set_reference(const cv::Mat& ref, bool /*save*/) {
    g_ref = {};
    try {
        Alignment_t z;
        try { z = set_joint_zero(ref); }                              // 보통은 영점만 바꾼다
        catch (const std::exception&) { set_joint_reference(ref); z = set_joint_zero(ref); }   // 측정이 안 될 만큼 밀렸으면 측정 기준부터
        g_ref.ok = true; g_ref.zero_y_mm = z.y; g_ref.zero_t_deg = z.t; return true;   // 새 영점(절대 영점 기준)
    } catch (const std::exception& e) { g_ref.message = e.what(); return false; }
}
void alignment_clear_reference() { clear_joint_zero(); }
const AlignmentInfo& alignment_last_info() { g_last.message = alignment_diagnostics().message; return g_last; }
const ReferenceInfo& alignment_reference_info() { return g_ref; }
// alignment(cv::Mat&) 는 src/alignment.cpp 의 것(joint 호출)을 쓴다. 실패는 예외 → align_worker 가 ERR 로 바꾼다.
// save=true(설정 파일에 영점 기록)를 쓰려면 z 를 파일에 적어 두고 시작할 때 set_joint_zero_value 로 넣는다.
```

- 성현 헤더(`OutPut_A507/Code/include/alignment.hpp`)와 저장소 헤더는 둘 다 `Alignment_t`, `alignment` 를 선언하므로 한 빌드에 같이 넣지 않는다. 어댑터용 헤더에는 성현 쪽 추가 구조체(`AlignmentRoi`, `AlignmentInfo`, `ReferenceInfo`)와 함수 선언만 옮기고, `Alignment_t`, `alignment` 는 저장소 헤더 것을 쓴다.
- ROI 그리기(`rois`)는 비워 둬도 동작한다(화면에 상자가 안 그려질 뿐).

## 8. 확인할 것 (Windows 결과와 비교)

| 확인 | 방법 | Windows 결과 |
|---|---|---|
| 빌드·해시 | `initialize_joint(assets)` 가 예외 없이 끝남 | 통과 |
| 같은 값 | `final00_tool predict <assets> <photos.tsv> <labels.tsv> <out>` 로 11-12 장비 사진 1959장(라벨 `experiments/hye/20261113_0950_…/결과/labels_L2.tsv`) | 1959장 측정, **1950장 통과**, 최대 \|eY\| 0.05478, \|eT\| 0.01092 |
| 영점 업데이트 | `final00_tool predict <assets> <photos> <labels> <out> zero=<rig_grid_rev/center.bmp>` (predict.txt 끝줄) | `zero 0.281930496 -0.04422359` (11-13 역순 센터의 절대 자세), 그 사진의 출력 (0, 0) |
| 측정 기준 다시 잡기 | 같은 명령에 `ref=<rig_grid_rev/center.bmp>` | `QR1 (0,14) … screw1 (1,7) screw2 (1,8) screw3 (0,8) screw4 (1,8) bar_upper 0~0 bar_lower 0~0 absolute_kept 1 bridge 0.276335 -0.0432488`, 센터 출력은 그대로 Y +0.2819, T −0.0442 |
| 측정값 일치 점검 | 위 1959장 `predict` 의 `health_warning`, 11-13 역순 1680장을 `zero=<rig_grid_rev/center.bmp>` 로 `predict` | 1959장 경고 0. 역순: 영점 업데이트 없이 경고 251, 영점 업데이트 뒤 경고 3 |
| 등록 없이 다른 날 | 오늘 사진을 등록 없이 `joint` | 측정 실패 없음. 센터 명령 사진의 출력 = 그날 센터가 절대 영점에서 떨어진 양 |
| 오늘 사진 | 확인 앱이 찍은 센터 사진 여러 장으로 `set_joint_zero` 뒤, 같은 날 사진을 `joint` | (Pi 에서 처음) 등록 사진은 (0, 0), 측정 실패 없음, 돌려받은 값 = 오늘 센터의 절대 자세 |
| 속도 | 한 장 `joint` 시간 | 노트북 보통 9~12 ms(4스레드) → Pi 예상 30~36 ms |

`photos.tsv` 는 `id<TAB>사진 경로`(머리줄 없음), 도구 인자 경로는 ASCII 로 준다.

## 9. 알려진 한계

- 정확도 숫자는 정지 사진, 영상 값에서 만든 라벨 기준이다. 실제 정렬 복귀 검증은 아직 없다(확인 앱으로 처음 하게 된다).
- 영상 교란(실험 `20261113_2130`, 인위 교란): 흐림 σ3·노이즈·밝기 ×0.6~±40 까지는 영향이 작다. 나사 하나 가림·반사가 가장 위험하고(경고로 대부분 알림), 카메라 회전·배율·가로 밀림은 영점 업데이트로 없어진다.
- 화면이 가로로 ±125 px 넘게 움직이면(Y·T 범위 밖) QR 이 탐색 창 끝에 걸려 실패로 나온다. 학습 범위는 Y −2.02~+2.04 mm, T −1.02~+0.98° 다.
- 측정 기준 다시 잡기가 잡는 것은 화면 밀림(카메라 처짐)까지다. QR 판과 나사(뒤판) 사이가 물리적으로 밀리면 Y 가 그만큼 틀어진다(나사 − QR 시차 0.1 px ≈ Y 28 µm).

문의: 혜원. 근거 실험은 `experiments/hye/20261113_1345_…`(기준 다시 잡기), `20261113_1700_…`(QR NCC), 이 폴더 `README.md` 의 "기준 다시 잡기", "속도".
