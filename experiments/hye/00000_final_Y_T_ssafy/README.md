# 00000_final_Y_T_ssafy — 값 7개 Y·T 추정 모듈 (우리 장비 사진 학습)

2026-11-13 사용자 지시로 만들었다. [00000_final_Y_T_semes](../00000_final_Y_T_semes/README.md) 와 계산 방식(값 7개, Huber 직선식, 절대 영점)이 같다. 다른 점은 화면 위치를 코드에 고정하지 않고 `assets/site.json` 에서 읽는다는 것뿐이다.

- 출처: `experiments/hye/20261113_0950_우리장비사진거부_장비기준자료와라벨보정/실험/module` 을 옮겼다. 자료(`assets/`)는 같은 실험의 `실험/assets_rig` 를 복사했다. 만든 과정과 한계는 그 실험의 [결과.md](../20261113_0950_우리장비사진거부_장비기준자료와라벨보정/결과/결과.md) 에 있다.
- `experiments/selected_modules.cmake` 는 지금 `00000_final_Y_T_semes` 를 가리킨다. 이 모듈을 쓰려면 `SEMES_JOINT_DIR` 을 이 폴더로 바꾼다.

## SEMES 모듈과 다른 점

| 항목 | 00000_final_Y_T_semes | 00000_final_Y_T_ssafy |
|---|---|---|
| QR 상자 3개, 나사 4개 중심, 잘라낼 영역, 센터 사진 해시 | 코드에 고정 | `assets/site.json` (`src/site.cpp`, `include/site.hpp`) |
| QR 가로 탐색 반폭 | 110 px | `site.json` 의 `search_x` (장비는 125 px. Y+2, T+1 에서 QR 이 114 px 움직인다) |
| 받는 모델 식별자 | `final00_huber7_702_irls6_r1e-7` 만 | `final00_huber7_` 로 시작하는 이름 |
| 계수 | SEMES 702장 | 장비 1959장(rig_1st 280, rig_grid 1679), 보정 라벨 L2 |

- 같은 코드인지 확인: 이 모듈 도구에 SEMES 자료를 주면 SEMES 사진 8장의 출력이 원본과 바이트까지 같았다(위 실험 결과 2).
- 장비 사진은 부호가 SEMES 와 반대이고 T 는 도(deg) 단위다. QR 은 SEMES 화면보다 세로로 62 px 아래에 있다.
- 각 모델은 자기 장비 사진에만 쓴다. SEMES 자료로는 장비 사진이 거부된다.

## 성적 (장비 1959장, 정지 사진, 고정 검증)

통과는 |eY| ≤ 0.045 mm 이고 |eT| ≤ 0.01 도. `evidence/eval_L2/eval.txt`.

| 주 검증 탈락 (1567장, 10회 평균) | rms Y / T | 마지막 확인 392장 | 세트 통째로 빼기 |
|---:|---:|---:|---:|
| 9.9 | 0.0112 / 0.0021 | 탈락 6 | 99.6% 통과 |

- L2 라벨은 같은 영상 값으로 명령을 보정한 것이다. 독립된 정답과의 비교가 아니라 반복 정밀도에 가깝다.
- 한 날, 두 세트다. 날 차이와 실제 복귀 검증은 보지 않았다.
- `evidence/predict_L2` 는 공개 API(`initialize_joint`, `joint`)로 1959장을 다시 넣은 확인이다(학습 사진이라 성능 지표가 아니다).

## 빌드

저장소 루트에서:

```
cmake -S experiments/hye/00000_final_Y_T_ssafy/tool/standalone -B .batch_build/rig00 -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR="C:/opencv/build"
cmake --build .batch_build/rig00 --config Release
```

도구 사용법은 SEMES 모듈 README 와 같다(`final00_tool measure | eval | predict`). 입력 표는 위 실험의 `실험/prepare_rig_inputs.py` 가 만든다.
