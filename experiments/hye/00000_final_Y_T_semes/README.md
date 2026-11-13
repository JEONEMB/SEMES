# 00000_final_Y_T_semes — 값 7개 Y·T 추정 모듈 (1차 joint, SEMES 사진 학습)

2026-11-12 작성. `experiments/selected_modules.cmake` 의 `SEMES_JOINT_DIR` 이 이 폴더를 가리킨다. `alignment()` 와 `joint()` 가 이 모듈로 계산한다.

알고리즘의 합의 기준은 [ALGORITHM_RULES.md](../../../ALGORITHM_RULES.md) 를 따른다. 이 문서는 이 모듈의 입출력과 검증 범위만 적는다.

## 무엇을 하나

사진 한 장에서 값 7개를 재고 직선식으로 (Y, T) 를 낸다.

| 값 | 개수 | 재는 방법 | 소스 |
|---|---:|---|---|
| QR 536점 이동의 가로 평균 | 1 | QR 구조 모서리 536점을 추적해 가로 이동을 평균 | `src/tracker.cpp`, `src/spline.cpp` |
| 위·아래 바 기울기 | 2 | 바마다 24칸의 세로 이동을 직선에 맞춘 기울기 | `src/observer.cpp`, `src/line_profile.cpp`, `src/model.cpp` |
| 나사 4개의 가로 이동 | 4 | 나사 주변 141×141 조각의 기울기 영상을 센터 사진과 맞춤 | `src/screw_gradient.cpp` |

세로 방향 값(바 세로 이동, QR·나사의 세로 이동)은 쓰지 않는다. 702장에서 세로 값은 변화의 75~90% 가 촬영 날 차이였고, 가로 값과 바 기울기는 99.9~100% 가 Y, T 움직임이었다.

## 2026-11-13 갱신 (우리 장비 모듈과 같은 코드)

코드(`src/`, `include/`, `tool/final00_tool.cpp`, `CMakeLists.txt`)를 [00000_final_Y_T_ssafy](../00000_final_Y_T_ssafy/README.md) 와 같게 바꿨다. **계수와 자료(`assets/`)는 그대로이고 값 7개도 그대로라 출력이 갱신 전과 같다**(702장 공개 API 출력 차이 최대 1.1e-8, 702장 통과). 근거: 실험 [20261113_1930](../20261113_1930_SEMES모듈갱신_QR방식과바비교/결과/결과.md).

| 바뀐 것 | 내용 |
|---|---|
| 속도 | 쓰지 않던 QR ECC 호모그래피 제거, 나사 시작 위치 탐색 축소, 나사 ECC 계산 범위 축소, QR·나사 병렬, 536점 추적에서 쓰지 않는 94점 제외. 측정 1스레드 51~75 ms, 4스레드 44~57 ms(노트북) |
| QR 값 | 기본은 그대로 **536점 추적**(`site.json` 이 없으면 `qr_method = track536`). 흐림 σ2 NCC 계수도 함께 있어 `initialize_joint_method(assets, "ncc")` 로 고를 수 있다(주 검증 0.2, 마지막 확인 0, 약 25 ms 빠름). 흐림 없는 NCC 는 SEMES 에서 주 검증 탈락 0.1 → 2.6 장이라 쓰지 않는다 |
| 바 기울기 | 그대로 **쓴다(값 7개)**. 빼면 주 검증 0.1 → 0.0, 처음 보는 날 98.8 → 98.2% 로 차이가 작고, SEMES 에서는 바 측정이 실패한 일이 없다 |
| 영점 | 기본은 절대 영점. `set_joint_zero`, `set_joint_zero_value`, `clear_joint_zero`, `joint_zero` 로 영점 업데이트(`include/joint_reference.hpp`) |
| 측정 기준 다시 잡기 | `set_joint_reference`(화면이 크게 밀려 측정이 실패할 때만). 절대 영점과 영점을 잇는다 |
| 리눅스 | SHA-256 을 순수 C++ 로. `error_t` 충돌은 팀 공용 헤더 문제라 그대로 |
| 모델 식별자 검사 | `final00_huber7_` 로 시작하면 받는다 |

사용법과 세부는 ssafy README 의 "영점과 측정 기준", "속도" 와 포팅 안내문 [PORTING_PI.md](../00000_final_Y_T_ssafy/PORTING_PI.md) 를 본다(SEMES 는 assets 경로만 이 폴더로).

## QR 방식 고르기 (2026-11-13)

QR 값을 재는 방법은 두 가지이고, assets 마다 두 방법의 계수를 함께 둔다. 기본은 `site.json` 의 `qr_method`(없으면 `track536`).

```cpp
initialize_joint(assets);                          // 기본 방식 (SEMES = track536, 우리 장비 = ncc)
initialize_joint_method(assets, "track536");       // 또는 "ncc" 로 바꿔 쓴다. 그 방식의 계수(provenance.json 의 alt_models)를 읽는다
joint_qr_method(); joint_model_id();               // 지금 방식과 계수 확인
```

도구: `final00_tool measure|predict ... method=track536` 또는 `method=ncc`.

| | track536 (QR 536점 추적) | ncc (흐림 σ2 + QR NCC + 포물선) |
|---|---|---|
| 재는 법 | QR 구조 모서리 536점을 LK → 점마다 17×17 스플라인 정밀 맞춤, 가로 이동 평균 | 센터 조각과 사진을 가우시안 σ2 로 흐린 뒤 QR 3개를 통째로 NCC, 봉우리 3점 포물선 소수 이동, 3개 평균 |
| 장점 | 가장 오래 검증한 방식. SEMES 에서 처음 보는 날 성적이 조금 낫다(98.8 대 98.4%). 점이 많아 QR 일부 가림에 덜 민감할 것으로 본다(시험 안 함) | 빠르다(사진마다 약 20~25 ms 짧고 초기화가 반). 코드가 단순하다. 정확도는 같은 수준 |
| 단점 | 느리다. 536점이 모두 유효해야 측정된다 | QR 을 한 덩어리로 보므로 일부 가림에 약할 수 있다(시험 안 함). 흐림 없이 쓰면 선명한 사진(SEMES)에서 소수 위치 치우침이 ±0.03 px 생긴다(그래서 흐림 σ2 를 넣었다, 실험 20261113_2000) |

정확도 (C++ 고정 검증, 같은 사진·라벨·분할, 통과 \|eY\| ≤ 0.045 mm 그리고 \|eT\| ≤ 0.01)

| 모듈 (값) | 방식 | 주 검증 탈락 | rms Y / T | 마지막 확인 탈락 | 처음 보는 날 |
|---|---|---:|---|---:|---:|
| semes (값 7개, 702장) | **track536 (기본)** | 0.1 / 560 | 0.0132 / 0.0029 | 0 / 142 | 98.8% |
| semes | ncc | 0.2 / 560 | 0.0132 / 0.0029 | 0 / 142 | 98.4% |
| ssafy (값 5개, 1959장) | **ncc (기본)** | **5.1 / 1567** | 0.0110 / 0.0021 | 5 / 392 | – |
| ssafy | track536 | 8.6 / 1567 | 0.0113 / 0.0021 | 6 / 392 | – |
| ssafy_2 (값 5개, 3638장) | **ncc (기본)** | 10.5 / 2910 | 0.0119 / 0.0040 | 2 / 728 | – |
| ssafy_2 | track536 | 9.5 / 2910 | 0.0117 / 0.0040 | 2 / 728 | – |

측정 시간 (Windows 노트북, 사진이 메모리에 있을 때, 보통 자세 / 느린 자세. 값 7개 측정 기준이라 바 1.5 ms 포함)

| 사진 | 방식 | 1스레드 | 4스레드 | 초기화 |
|---|---|---:|---:|---:|
| 우리 장비 | ncc | 26~27 / 82 ms | 14~18 / 62 ms | 0.2 초 |
| 우리 장비 | track536 | 46~48 / 95 ms | 42~44 / 91 ms | 0.4 초 |
| SEMES | ncc | 23~28 / 50 ms | 15~16 / 31 ms | 0.6 초 |
| SEMES | track536 | 53~55 / 79 ms | 45~48 / 61 ms | 1.1 초 |

Pi 5 는 이 노트북의 약 3배로 본다(재지 않음). 예: ncc 4스레드 약 45~55 ms, track536 약 130 ms.

고르는 법: 정확도는 두 방식이 같은 수준이다. **속도가 중요하면 ncc**, **SEMES 에서 가장 오래 검증한 방식을 쓰려면 track536**(SEMES 기본). 우리 장비 기본은 ncc 다.

## 공개 입출력

- `void initialize_joint(const std::string& assets)`: `assets/` 폴더 경로(UTF-8). 자료 해시가 `provenance.json` 과 다르면 예외를 던진다.
- `Alignment_t joint(cv::Mat& src)`: 입력은 3840×2160 8비트 영상(1·3·4채널). 입력을 바꾸지 않는다. 반환은 추정값 `(y, t)` 이며 이동 명령이 아니다.
- **기준이 절대 영점이다.** `(0, 0)` 은 "센터 사진(`assets/center.bmp`)과 화면이 같은 자세"다. 촬영 날의 로봇 명령 0 이 아니다. Y 는 mm, T 의 단위와 부호는 학습 라벨(파일명 명령)을 그대로 따른다.
- 실패는 예외로 알린다. `(0, 0)` 으로 숨기지 않는다.
  - QR 고정 536점 중 하나라도 유효하지 않으면 예외. 범위 밖(예: T=+2.00) 사진은 여기서 거부된다.
  - 나사 맞춤이 수렴하지 않으면 예외.
- 진단(`alignment_diagnostics()`): `selected_branch = "huber_linear7"`, `tracked_points = 536`, `valid_points`, QR·바 상관, `outside_training_range`(추정값이 학습 라벨 범위 Y −2.04~+1.96, T −1.00~+1.01 밖).
- 세로 값을 쓰지 않으므로 판이 세로로 밀리는 이상은 이 모듈이 알아채지 못한다. 품질 판정은 호출 측의 책임이다.
- **측정값 일치 점검 (2026-11-14, 실험 `20261113_2230`)**: 값 7개가 한 자세로 함께 설명되지 않으면 값은 그대로 내고 경고를 남긴다.
  - 경고 표시: `normalized_cost_margin > 1`, `message` 가 `WARNING 측정값이 서로 맞지 않는다(…)` 로 시작한다.
  - 멈추게 하려면 `set_joint_health_check(true)` 를 부른다(예외).
  - 점검식은 `assets/health.tsv`(track536, 기준값 8.29), `health_ncc.tsv`(ncc, 8.33)다.
  - 702장 공개 API 경고 0. 회차 하나를 빼고 만든 점검식으로 그 회차를 재면 702장 중 4장(20261020, 위 바 기울기)이 경고했다.
  - SEMES 사진에서 나사가 가려지거나 반사되면 원래 나사 NCC 0.6 검사에서 예외가 나므로, 이 점검은 주로 우리 장비 사진에서 의미가 있다.
  - 설명은 [ssafy README](../00000_final_Y_T_ssafy/README.md) "측정값 일치 점검".

## 자료 (`assets/`)

| 파일 | 내용 |
|---|---|
| `center.bmp` | 센터 사진(1차). 다른 final 모듈과 같은 파일이다 |
| `points.json` | QR 구조 모서리 630점과 그중 쓰는 536점 |
| `geometry.json`, `line_protocol.json` | QR 위치와 바 측정 칸 48개 |
| `model7.tsv` | 702장 전체로 학습한 계수(평균, 표준편차, Y·T 가중치) |
| `model7_ncc.tsv` | QR 흐림 NCC 방식의 계수(`initialize_joint_method(assets, "ncc")`) |
| `health.tsv`, `health_ncc.tsv` | 측정값 일치 점검식(`final00_tool health` 로 만듦) |
| `provenance.json` | 모델 식별자, 학습 범위, 자료 해시 |

`.gitattributes` 가 텍스트 자료의 줄바꿈을 LF 로 고정한다(해시 검사 때문).

## 빌드와 실행

저장소 루트 빌드는 [개발환경](../../../docs/windows-setup.md) 을 따른다. 이 모듈과 도구만 빌드하려면 저장소 루트에서:

```
cmake -S experiments/hye/00000_final_Y_T_semes/tool/standalone -B .batch_build/final00 -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR="C:/opencv/build"
cmake --build .batch_build/final00 --config Release
```

도구 `final00_tool` (인자 경로는 ASCII 로 준다):

```
final00_tool measure <assets> <photos.tsv> <출력 폴더>                      값 7개를 잰다 → features7.tsv
final00_tool eval <features7.tsv> <labels.tsv> <splits.tsv> <출력 폴더>      학습·평가 → eval.txt, model7.tsv
final00_tool predict <assets> <photos.tsv> <labels.tsv> <출력 폴더>          공개 API 로 예측 → predictions.tsv, predict.txt
```

입력 표는 `python tool/prepare_inputs.py <폴더>` 가 만든다(자료 준비만 한다). 이 스크립트와 `tool/compare_with_python.py` 는 대조용이고 추론에는 필요 없다. `prepare_inputs.py` 는 `apps/distribution/reoof.py` 와 `experiments/hye/20261112_1835_…/실험/combo.py` 를 쓴다.

## 검증 결과 (702장, 정지 사진 평가)

사진 702장(ID 702 제외, T=+2.00 12장 제외), 절대 영점 라벨, 자세 묶음 단위 분할. 통과는 |eY| ≤ 0.045 이고 |eT| ≤ 0.01.

| 항목 | 결과 |
|---|---|
| 주 검증: 개발 560장 섞은 80/20, 10회 탈락 | 0 0 0 0 0 0 1 0 0 0 (평균 0.10) |
| 보조: 세션별 절반 50/50, 10회 탈락 | 평균 0.20 |
| 참고: 세션 통째로 빼기 | 98.8% (1차 100, 2차 100, 3차 100, 4차 93.9) |
| 마지막 확인 142장 | 탈락 0 (최대 \|eY\| 0.0389, \|eT\| 0.0086) |
| 702장 전체로 학습해 다시 넣기 | 탈락 0/702 |
| Python 분석값과의 차이 | QR·바 0, 나사 최대 0.0002 px |

근거 파일은 `evidence/` 에 있다. `evidence/이식검증_결과.md` 가 이식 과정과 대조를 설명한다. 구성을 고른 과정은 `experiments/hye/20261112_1835_조합마다검증방식다름_방법론별증강과고정검증/결과/결과.md` 에 있다.

## 한계

- 702장은 개발에 쓴 자료다. 구성도 같은 자료에서 골랐다. 독립 성능이 아니다.
- 정지 사진 평가다. 실제 로봇으로 센터에 복귀하는 검증은 하지 않았다.
- 촬영 날이 4개이고 날 차이의 종류는 사실상 2가지다. 가로 값에 영향을 주는 새로운 날 차이가 오면 다시 봐야 한다.
- `cv::warpAffine` 은 OpenCV 3.4.15 에서 좌표를 1/32 px 로 끊는다. 나사 맞춤은 이를 피해 직접 보간한다. QR·바 관측(`src/observer.cpp`, `third_party/ecc_compat.cpp`)은 `warpAffine`/`warpPerspective` 를 그대로 쓴다. 바 기울기는 여기에 영향받지 않는 프로파일 맞춤이다.
- 값 7개에 필요 없는 QR 54값 계산이 관측기 안에서 함께 돈다. 속도는 재지 않았다(702장 측정이 병렬로 약 2분, 공개 API 로 한 장씩 돌리면 더 오래 걸린다).
- 값 교란 강건성 시험은 Python 분석으로만 했다(가장 약한 교란에서 통과 88%).

## 소스의 출처

| 파일 | 출처 |
|---|---|
| `src/joint.cpp`, `src/estimator.cpp`, `src/screw_gradient.cpp`, `src/huber_fit.cpp`, `include/final00.hpp`, `include/screw_gradient.hpp`, `include/huber_fit.hpp`, `tool/final00_tool.cpp` | 이 모듈에서 새로 작성 |
| `src/tracker.cpp`, `src/spline.cpp`, `include/control_internal.hpp`, `assets/points.json` | `20261028_1327_final_Y_T control` 에서 복사(내용 그대로) |
| `src/observer.cpp`, `src/line_profile.cpp`, `src/pin_observer.cpp`, `src/model.cpp`, `third_party/ecc_compat.cpp`, `include/observer_internal.hpp`, `include/pin_observer.hpp`, `include/ecc_compat.hpp`, `include/final04.hpp`, `assets/geometry.json`, `assets/line_protocol.json` | `20261111_0944_final_Y_T` 에서 복사(내용 그대로) |

2차 모듈(`semes_control`)도 같은 tracker 소스를 쓴다. 한 실행 파일에 함께 링크될 때 기호가 겹치지 않도록, 이 모듈을 컴파일할 때만 `legacy_control` 이름공간을 `final00_tracking` 으로 바꾼다(`CMakeLists.txt`).
