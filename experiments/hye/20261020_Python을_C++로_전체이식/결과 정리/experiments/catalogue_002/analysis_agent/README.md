# analysis_agent 실험 출처

## 문제

QR 3개만으로 Y와 T를 구분하고 이동량을 추정할 수 있는지 확인

## 조사

QR 측정과 회귀, Y가 고정됐다는 조건의 T 홀짝 검증

## 해결 방법

measure_qr.py, evaluate.py, evaluate_theta_only.py의 독립 경로

## 유지한 부분

원본 이미지와 기존 측정·결과를 보존

## 바꾼 부분

T만 움직인다는 조건부 평가와 Y/T 추정을 구분

## 다른 실험과의 구분

현재 630점 경계 모델과 다른 초기 실험

## 시간과 작성 흔적

관측된 최초 파일 생성: 2026-10-12T15:47:42.930+09:00. 관측된 최종 파일 수정: 2026-10-12T16:07:38.975+09:00. 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.

### evaluate.py

- 원본: `analysis_agent/evaluate.py`
- 생성 KST: 2026-10-12T15:50:59.176+09:00
- 수정 KST: 2026-10-12T15:52:01.392+09:00
- SHA256: `5d666675d0fed31bd1aecd3b27d4324b8e68914b1140f26e7d6c10eed1521988`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_frozen_measurements_regression_parity_passed`
- C++ 위치: `experiments/analysis_agent/src/evaluate.cpp`
- 검증·차이: 140개 시험 행, 원본 저장 측정에서 T/Y 예측 최대차 각각 1.16e-13/6.47e-13. 그림 렌더러는 OpenCV로 변경.
- 이식 기록: `experiments/analysis_agent/README.md`
- 새 실행 근거: `results/historical_agent_parity_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### evaluate_theta_only.py

- 원본: `analysis_agent/evaluate_theta_only.py`
- 생성 KST: 2026-10-12T16:07:29.531+09:00
- 수정 KST: 2026-10-12T16:07:29.531+09:00
- SHA256: `7bc6ff98a57f13bad2947cbb1a9e442f2617e66e13622836201983b41958601c`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_frozen_measurements_theta_parity_passed`
- C++ 위치: `experiments/analysis_agent/src/evaluate_theta_only.cpp`
- 검증·차이: 100개 시험 행, theta_pred 최대차 2.22e-16. 원본의 T-only·원점 통과·홀짝 분할 조건 보존.
- 이식 기록: `experiments/analysis_agent/README.md`
- 새 실행 근거: `results/historical_theta_parity_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### measure_qr.py

- 원본: `analysis_agent/measure_qr.py`
- 생성 KST: 2026-10-12T15:48:32.540+09:00
- 수정 KST: 2026-10-12T15:49:46.524+09:00
- SHA256: `729c86a406cf9ef558c95882f985d7a298aef444e6f05cad5b233824d1eb6170`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_281_images_ecc_numeric_difference_unresolved`
- C++ 위치: `experiments/analysis_agent/src/measure_qr.cpp`
- 검증·차이: 281장 실패 0. NCC 점수 대조 통과. ECC translation은 기존 CSV와 최대 약 0.0363px 차이; 정확한 실행환경 차이 원인 미확정.
- 이식 기록: `experiments/analysis_agent/README.md`
- 새 실행 근거: `results/historical_ecc_parity_001/checks.json`
- 새 실행 근거: `results/historical_ecc_ncc_parity_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

## 보존된 결과와 문서

- `analysis_agent/heldout_predictions.csv` (2026-10-12T15:52:06.873+09:00)
- `analysis_agent/qr_measurements.csv` (2026-10-12T15:50:11.915+09:00)
- `analysis_agent/README.md` (2026-10-12T15:53:03.088+09:00)
- `analysis_agent/theta_only_holdout.csv` (2026-10-12T16:07:38.975+09:00)
