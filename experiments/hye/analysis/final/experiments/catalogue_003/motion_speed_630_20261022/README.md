# motion_speed_630_20261022 실험 출처

## 문제

630점 경계 측정의 실행시간

## 조사

원본 TRF, 해석 미분 TRF, 컴파일된 LM의 단계별 시간과 수치 차이

## 해결 방법

compiled.py LM 목적함수·미분·경계·두 창 진단을 C++로 이식

## 유지한 부분

630 ID, 초기점과 기준 좌표의 구분, 목적함수·모델·전체점 회귀

## 바꾼 부분

Numba/Python 연산을 C++와 OpenCV 선형대수로 대체; 측정 스레드 기본 1

## 다른 실험과의 구분

Python LM 표본 좌표 최대차 1.44e-7px, 진단 차이 0점; TRF와는 차이 존재

## 시간과 작성 흔적

관측된 최초 파일 생성: 2026-10-22T15:11:05.369+09:00. 관측된 최종 파일 수정: 2026-10-22T15:27:58.299+09:00. 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.

### baseline.py

- 원본: `motion_speed_630_20261022/baseline.py`
- 생성 KST: 2026-10-22T15:11:05.369+09:00
- 수정 KST: 2026-10-22T15:11:05.666+09:00
- SHA256: `91b93e305276055665de2428b7d5088e9cbb91fccca6a9ae373a45081e331ecf`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### benchmark.py

- 원본: `motion_speed_630_20261022/benchmark.py`
- 생성 KST: 2026-10-22T15:20:34.224+09:00
- 수정 KST: 2026-10-22T15:20:34.227+09:00
- SHA256: `20e5f02da5144c1cec7e75b6273a4237dc1c0f0959c1f348463ad91427359045`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### bench_one.py

- 원본: `motion_speed_630_20261022/bench_one.py`
- 생성 KST: 2026-10-22T15:12:11.667+09:00
- 수정 KST: 2026-10-22T15:13:00.143+09:00
- SHA256: `46d2d0ad6d7cf43722db7c751357e08a09a78e9dd4c56fbfd6d51acb3218b86d`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### client_example.py

- 원본: `motion_speed_630_20261022/client_example.py`
- 생성 KST: 2026-10-22T15:27:06.275+09:00
- 수정 KST: 2026-10-22T15:27:06.277+09:00
- SHA256: `ba7eda19c24d8db9ba9dfea688eeda609a2fee52776a1edaf3efb6c9a73a5a40`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### compiled.py

- 원본: `motion_speed_630_20261022/compiled.py`
- 생성 KST: 2026-10-22T15:15:41.394+09:00
- 수정 KST: 2026-10-22T15:17:16.406+09:00
- SHA256: `0da38151bd17206c0599563c40b2e55aa7721d87558ffa5f457d207fe4c766fd`
- 관측된 작업자: unknown
- 이식 상태: `ported_compiled_lm_validated_sample_and_280_image_predictions`
- C++ 위치: `src/alignment.cpp`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### engine.py

- 원본: `motion_speed_630_20261022/engine.py`
- 생성 KST: 2026-10-22T15:12:11.661+09:00
- 수정 KST: 2026-10-22T15:12:11.661+09:00
- SHA256: `cec6cff8a768487a1751b607e8082fbe51df6d9bb8e3260e84c1d60093f0d272`
- 관측된 작업자: unknown
- 이식 상태: `ported_feature_and_inference_engine`
- C++ 위치: `src/alignment.cpp`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### fast_fit.py

- 원본: `motion_speed_630_20261022/fast_fit.py`
- 생성 KST: 2026-10-22T15:11:34.510+09:00
- 수정 KST: 2026-10-22T15:11:34.510+09:00
- SHA256: `9d91cfa6d91e02929f25b60ff1f859a4037e1f2202c1102ba6e5eb3aa69c2da1`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### make_report.py

- 원본: `motion_speed_630_20261022/make_report.py`
- 생성 KST: 2026-10-22T15:25:39.446+09:00
- 수정 KST: 2026-10-22T15:27:54.320+09:00
- SHA256: `21bcdb410217d30c6f32c50d4533ce14a814e51bd3cdc1249a2eed94a2e001e5`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### optimized.py

- 원본: `motion_speed_630_20261022/optimized.py`
- 생성 KST: 2026-10-22T15:11:34.777+09:00
- 수정 KST: 2026-10-22T15:11:34.779+09:00
- SHA256: `9ebea5d0ede38e2d32c37d25cbc725d095ef4d6e8290fb693ebc0f332650d073`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### predict.py

- 원본: `motion_speed_630_20261022/predict.py`
- 생성 KST: 2026-10-22T15:21:34.611+09:00
- 수정 KST: 2026-10-22T15:21:34.611+09:00
- SHA256: `6876a96fb8963ca986388801da02e3ad51bc88709aa8e9115cb1927ae1ca7eaa`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### serve.py

- 원본: `motion_speed_630_20261022/serve.py`
- 생성 KST: 2026-10-22T15:21:34.603+09:00
- 수정 KST: 2026-10-22T15:21:34.611+09:00
- SHA256: `62c9d47a10f56310dd0580f05cd5153dcf19adbdce491c25ec4d3ed9eadfa4ce`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### start.ps1

- 원본: `motion_speed_630_20261022/start.ps1`
- 생성 KST: 2026-10-22T15:27:06.279+09:00
- 수정 KST: 2026-10-22T15:27:06.279+09:00
- SHA256: `08a7394e09a85c3da999456ce0c271804aa8e8dd63ed035186db2fa2385af1f8`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### test_math.py

- 원본: `motion_speed_630_20261022/test_math.py`
- 생성 KST: 2026-10-22T15:27:06.282+09:00
- 수정 KST: 2026-10-22T15:27:06.282+09:00
- SHA256: `8a6a1f9f9aeb150413e9ae3feeef6c135d92263d37fd58f049163d68921b13ad`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### verify280.py

- 원본: `motion_speed_630_20261022/verify280.py`
- 생성 KST: 2026-10-22T15:18:24.446+09:00
- 수정 KST: 2026-10-22T15:21:35.094+09:00
- SHA256: `dd53b0c45bbac0b2d54c87e5ab196e55dd843bd011fa959e14077682c0ea792b`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

## 보존된 결과와 문서

- `motion_speed_630_20261022/assets/corners.json` (2026-10-22T10:25:42.267+09:00)
- `motion_speed_630_20261022/assets/model.json` (2026-10-21T17:23:50.354+09:00)
- `motion_speed_630_20261022/assets/provenance.json` (2026-10-22T10:25:42.290+09:00)
- `motion_speed_630_20261022/report.html` (2026-10-22T15:27:58.297+09:00)
- `motion_speed_630_20261022/results/baseline_Y+0.50.json` (2026-10-22T15:14:46.091+09:00)
- `motion_speed_630_20261022/results/compiled280.csv` (2026-10-22T15:19:23.344+09:00)
- `motion_speed_630_20261022/results/compiled280_summary.json` (2026-10-22T15:21:35.101+09:00)
- `motion_speed_630_20261022/results/compiled_failures.csv` (2026-10-22T15:19:23.344+09:00)
- `motion_speed_630_20261022/results/compiled_Y+0.50.json` (2026-10-22T15:17:45.665+09:00)
- `motion_speed_630_20261022/results/headline.json` (2026-10-22T15:27:58.299+09:00)
- `motion_speed_630_20261022/results/math_checks.json` (2026-10-22T15:27:09.003+09:00)
- `motion_speed_630_20261022/results/optimized_Y+0.50.json` (2026-10-22T15:13:17.664+09:00)
- `motion_speed_630_20261022/results/paired_timings.json` (2026-10-22T15:22:48.848+09:00)
- `motion_speed_630_20261022/results/pilot_contended_baseline.json` (2026-10-22T15:12:44.775+09:00)
- `motion_speed_630_20261022/results/service_checks.json` (2026-10-22T15:23:38.332+09:00)
