# analysis_v3 실험 출처

## 문제

과거 조합 결과 재현과 QR만 사용하는 추정의 오차

## 조사

특징점 계산용 GN과 픽셀 직접 추정 GN, 호모그래피 표현 안정성

## 해결 방법

조합별 회귀와 C6, 픽셀 직접 G1을 별도 비교

## 유지한 부분

v1_data 측정·분할을 재현한 비교는 해당 스냅샷을 유지

## 바꾼 부분

GN 적용 위치와 변환값 표현

## 다른 실험과의 구분

QR 밖 점을 포함한 비교는 QR 전용 결과와 구분

## 시간과 작성 흔적

관측된 최초 파일 생성: 2026-10-13T13:05:20.490+09:00. 관측된 최종 파일 수정: 2026-10-13T16:55:32.105+09:00. 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.

### build_deck.py

- 원본: `analysis_v3/build_deck.py`
- 생성 KST: 2026-10-13T15:18:16.632+09:00
- 수정 KST: 2026-10-13T15:19:31.183+09:00
- SHA256: `afcd36a87e0db8a7c35206aa32096ca14ab674ea6c4b453983f37b8d7668c9c9`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-13T15:17:57+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1601행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### build_ppt.py

- 원본: `analysis_v3/build_ppt.py`
- 생성 KST: 2026-10-13T13:42:37.964+09:00
- 수정 KST: 2026-10-13T14:58:00.847+09:00
- SHA256: `136b8df75991d832a38f715f79cf0ba38be00d95031ebb34a6e3ea16decf6623`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-13T13:40:48+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1182행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### common.py

- 원본: `analysis_v3/common.py`
- 생성 KST: 2026-10-13T13:05:59.044+09:00
- 수정 KST: 2026-10-13T13:05:59.085+09:00
- SHA256: `6f293b857df56b671410640b67523195a750d27ae17bc437c60b9e3a344d65dc`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_shared_historical_data_and_C1_C5_v3_callers_pending`
- C++ 위치: `experiments/analysis_v3_v1_data/src/common.hpp`
- 검증·차이: 원본의 grid_repro 앞부분 실행형 import를 C++ 데이터·C1~C5 함수로 대체. v3 후속 개별 실험은 대기.
- 이식 기록: `experiments/analysis_v3_v1_data/README.md`
- 새 실행 근거: `results/historical_v1_grid_parity_003/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### deck_figs.py

- 원본: `analysis_v3/deck_figs.py`
- 생성 KST: 2026-10-13T15:13:38.123+09:00
- 수정 KST: 2026-10-13T15:13:56.732+09:00
- SHA256: `0b9bf1de40815bcb27d9a4ffd3ca34869301fd9915ec7c44485f8a7fdaa77522`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-13T15:13:35+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1566행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### draw_twin_lines.py

- 원본: `analysis_v3/draw_twin_lines.py`
- 생성 KST: 2026-10-13T13:30:36.477+09:00
- 수정 KST: 2026-10-13T13:30:36.563+09:00
- SHA256: `d33c9c689160675cf65782b3fcc7296ff9e1a03f5a3605868a7db91a63df1bea`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### draw_twin_photos.py

- 원본: `analysis_v3/draw_twin_photos.py`
- 생성 KST: 2026-10-13T13:34:13.079+09:00
- 수정 KST: 2026-10-13T13:34:13.083+09:00
- SHA256: `170b6bb2f8340c2ce166b2b07905c06130021adf61ecd47fea0e7eef53bd3db8`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### fig_pixel_explain.py

- 원본: `analysis_v3/fig_pixel_explain.py`
- 생성 KST: 2026-10-13T15:25:33.049+09:00
- 수정 KST: 2026-10-13T15:25:33.049+09:00
- SHA256: `b8e0954ae6439fd866ac6eef66e569bff7d13418ae633c4ec4c989ed1691e1b6`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-13T15:24:07+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1739행.
- 2026-10-13T15:25:10+09:00 Claude Edit 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1780행.
- 2026-10-13T15:25:13+09:00 Claude Edit 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1781행.
- 2026-10-13T15:25:15+09:00 Claude Edit 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1782행.
- 2026-10-13T15:25:18+09:00 Claude Edit 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1783행.
- 2026-10-13T15:25:19+09:00 Claude Edit 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1784행.
- 2026-10-13T15:25:29+09:00 Claude Edit 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1807행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### gn_features.py

- 원본: `analysis_v3/gn_features.py`
- 생성 KST: 2026-10-13T13:05:59.108+09:00
- 수정 KST: 2026-10-13T13:06:17.544+09:00
- SHA256: `150647e2fdd3ff4ba8cc1af6eeff40362548cb21434cd7c1b21d272e4c8da020`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_quadratic_GN_12_feature_sets_metrics_parity`
- C++ 위치: `experiments/analysis_v3/src/gn_features.hpp`
- 검증·차이: 원본 이차 기저·가중 GN·감쇠 조건으로 C6 구현. 기존 v3 지표 대조에서 구분 수·오판율 동일, 오차 지표 차이 별도 기록.
- 이식 기록: `experiments/analysis_v3/README.md`
- 새 실행 근거: `results/historical_v3_parity_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### gn_pixel.py

- 원본: `analysis_v3/gn_pixel.py`
- 생성 KST: 2026-10-13T13:14:05.721+09:00
- 수정 KST: 2026-10-13T13:14:48.265+09:00
- SHA256: `b7a80a8bb206251a70ad09eb7c3528fa83bb0647410c07e3789dcc69c113ce38`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-13T13:12:33+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 885행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### gn_pixel_slow.py

- 원본: `analysis_v3/gn_pixel_slow.py`
- 생성 KST: 2026-10-13T13:12:05.214+09:00
- 수정 KST: 2026-10-13T13:12:05.214+09:00
- SHA256: `6bb3e78a070a8808fc782bea79e91a12dab9388604012a24a31a54dd26f86350`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### make_v3_outputs.py

- 원본: `analysis_v3/make_v3_outputs.py`
- 생성 KST: 2026-10-13T13:36:07.487+09:00
- 수정 KST: 2026-10-13T13:36:07.489+09:00
- SHA256: `6fe8664aabb5d58e4e4331e4694b141cdf28b418394422b65d341b09867719f9`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-13T13:11:37+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 861행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### run_g1.py

- 원본: `analysis_v3/run_g1.py`
- 생성 KST: 2026-10-13T13:08:51.427+09:00
- 수정 KST: 2026-10-13T13:15:25.844+09:00
- SHA256: `a969131b68860efffbffe0f8b965f9f35cbaf22300e02e242325953e5bf07377`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### run_v3.py

- 원본: `analysis_v3/run_v3.py`
- 생성 KST: 2026-10-13T13:09:23.896+09:00
- 수정 KST: 2026-10-13T13:09:23.980+09:00
- SHA256: `463028552f2ad276ce4a034cdeffc0a17e3b84ce6b65214dd4ac35f84fc68601`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_65_feature_fits_plus_explicit_G1_saved_prediction_replay`
- C++ 위치: `experiments/analysis_v3/src/run_v3.cpp`
- 검증·차이: C++ 특징 모델65행과 원본 실행처럼 저장 G1예측1행으로66행 재현. G1 픽셀 직접 추론기는 별도 미이식. 구분불가 수·오판율 모두 동일.
- 이식 기록: `experiments/analysis_v3/README.md`
- 새 실행 근거: `results/historical_v3_evaluate_001/audit.json`
- 새 실행 근거: `results/historical_v3_parity_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### slide_data.py

- 원본: `analysis_v3/slide_data.py`
- 생성 KST: 2026-10-13T13:37:34.773+09:00
- 수정 KST: 2026-10-13T13:37:59.831+09:00
- SHA256: `454a44817eb2fba3f1a37bd039b031f7f58abbe58837b2f1c0b228632c4e94d7`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-13T13:37:32+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1136행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### slide_panels.py

- 원본: `analysis_v3/slide_panels.py`
- 생성 KST: 2026-10-13T13:38:31.579+09:00
- 수정 KST: 2026-10-13T13:38:31.581+09:00
- SHA256: `6c89f4ff5557317679fc992cc487aefc5ecee318e666dfb23e0af42d0f2f618f`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-13T13:38:29+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1155행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### combos.py

- 원본: `analysis_v3/v1_data/combos.py`
- 생성 KST: 2026-10-13T13:05:20.534+09:00
- 수정 KST: 2026-10-13T13:05:20.534+09:00
- SHA256: `e6111fedd2ebf59352e993b1f3775afbf4703bd1e02c0891f6a933032e5a01fc`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_2701_pairs_and_greedy_ids_match`
- C++ 위치: `experiments/analysis_v3_v1_data/src/combos.cpp`
- 검증·차이: 84점 중74점·2701쌍·greedy 계산. good/best_pair/greedy/bolt ID 동일, 특징 통계 최대차8.53e-14.
- 이식 기록: `experiments/analysis_v3_v1_data/README.md`
- 새 실행 근거: `results/historical_v1_combos_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### ecc_tune.py

- 원본: `analysis_v3/v1_data/ecc_tune.py`
- 생성 KST: 2026-10-13T13:05:20.524+09:00
- 수정 KST: 2026-10-13T13:05:20.524+09:00
- SHA256: `ac8ebc8d6ffc559433ab7727085f67a38eb60a48ac8e9604bc5268b589a9ce91`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### grid.py

- 원본: `analysis_v3/v1_data/grid.py`
- 생성 KST: 2026-10-13T13:05:20.506+09:00
- 수정 KST: 2026-10-13T13:05:20.506+09:00
- SHA256: `e858a91d07f854dfbabf47a6c1e6cde4b493300eaaccacce891c05e1aa342852`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_49_configurations_440_of_441_rounded_metrics_equal`
- C++ 위치: `experiments/analysis_v3_v1_data/src/grid.cpp`
- 검증·차이: 281행·49조합·441개 집계 중 440개 표시값 동일. O5/C2 Y최대 한 칸은 0.0001 차이로 보존. 과거 전체 자료 특징 선택·교대2fold 조건.
- 이식 기록: `experiments/analysis_v3_v1_data/README.md`
- 새 실행 근거: `results/historical_v1_grid_002/audit.json`
- 새 실행 근거: `results/historical_v1_grid_parity_003/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### grid_repro.py

- 원본: `analysis_v3/v1_data/grid_repro.py`
- 생성 KST: 2026-10-13T13:05:20.636+09:00
- 수정 KST: 2026-10-13T13:05:20.657+09:00
- SHA256: `2ea8be9be8b4df7b73aa1d985c517fda9da5dff87a312b211e7760df49ab0655`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_49_configurations_440_of_441_rounded_metrics_equal`
- C++ 위치: `experiments/analysis_v3_v1_data/src/grid.cpp`
- 검증·차이: 281행·49조합·441개 집계 중 440개 표시값 동일. O5/C2 Y최대 한 칸은 0.0001 차이로 보존. 과거 전체 자료 특징 선택·교대2fold 조건.
- 이식 기록: `experiments/analysis_v3_v1_data/README.md`
- 새 실행 근거: `results/historical_v1_grid_002/audit.json`
- 새 실행 근거: `results/historical_v1_grid_parity_003/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### qr_only.py

- 원본: `analysis_v3/v1_data/qr_only.py`
- 생성 KST: 2026-10-13T13:05:20.524+09:00
- 수정 KST: 2026-10-13T13:05:20.524+09:00
- SHA256: `35d759ef5ea952e36c31d740ada3d926a9b5a642131899cdb28fe7ca2739f7c8`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### stable.py

- 원본: `analysis_v3/v1_data/stable.py`
- 생성 KST: 2026-10-13T13:05:20.528+09:00
- 수정 KST: 2026-10-13T13:05:20.528+09:00
- SHA256: `ece9011b5d5fbd8af2a71d6457be03118b5e3ce59b6ad5a1f0524e934675057b`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_51_kept_and_best_pair_ids_match_maps_rendered`
- C++ 위치: `experiments/analysis_v3_v1_data/src/stable.cpp`
- 검증·차이: 23점 제외·51점 유지·선택쌍[11,46] 원본과 동일. modulo4 fit/selection/final, greedy10단계와 지도 C++ 생성. 그림 픽셀은 다름.
- 이식 기록: `experiments/analysis_v3_v1_data/README.md`
- 새 실행 근거: `results/historical_v1_stable_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### track_all.py

- 원본: `analysis_v3/v1_data/track_all.py`
- 생성 KST: 2026-10-13T13:05:20.518+09:00
- 수정 KST: 2026-10-13T13:05:20.518+09:00
- SHA256: `f5952c3a6a50df945e09839d30bf74062e75d50ec38b04864e42f67210c92c82`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_frozen_84_points_282_rows_detector_mode_unvalidated`
- C++ 위치: `experiments/analysis_v3_v1_data/src/track_all.cpp`
- 검증·차이: 원본 고정84점·중복center 포함282행 측정 실패0. 원본 대비 dx0.0038682px/dy0.0064014px 차이. center 검출 모드는 구현 후 검증 대기.
- 이식 기록: `experiments/analysis_v3_v1_data/README.md`
- 새 실행 근거: `results/historical_v1_track_frozen_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### zero_offset_photos.py

- 원본: `analysis_v3/zero_offset_photos.py`
- 생성 KST: 2026-10-13T14:50:22.268+09:00
- 수정 KST: 2026-10-13T14:50:22.289+09:00
- SHA256: `9b0559b0458cd034e3e8ead9783d99bfb26b016cc20bc42d2fbbe5ba686c9cf0`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-13T14:50:06+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\0478c349-0ea5-4fa5-b878-4da27eb190c2.jsonl`의 1386행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

## 보존된 결과와 문서

- `analysis_v3/v1_data/combos.json` (2026-10-13T13:05:20.506+09:00)
- `analysis_v3/v1_data/repro_17.csv` (2026-10-13T13:05:21.394+09:00)
- `analysis_v3/v1_data/stable.json` (2026-10-13T13:05:20.506+09:00)
- `analysis_v3/v3_results.csv` (2026-10-13T13:35:30.933+09:00)
- `analysis_v3/zero_offset_photos.csv` (2026-10-13T14:50:28.649+09:00)
