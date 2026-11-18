# analysis_v2 실험 출처

## 문제

QR 평균 이동이 비슷한 Y/T 사진의 구분

## 조사

판 호모그래피와 ECC, 특징별 회귀와 마할라노비스 판정

## 해결 방법

extract_homography, estimate, judge 경로

## 유지한 부분

기존 QR 측정값과 결과표

## 바꾼 부분

내부 픽셀 정합과 공분산을 반영한 판정

## 다른 실험과의 구분

최신 고정 630점 측정 이전의 방법

## 시간과 작성 흔적

관측된 최초 파일 생성: 2026-10-13T11:41:09.023+09:00. 관측된 최종 파일 수정: 2026-10-13T11:43:39.135+09:00. 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.

### estimate.py

- 원본: `analysis_v2/estimate.py`
- 생성 KST: 2026-10-13T11:43:38.174+09:00
- 수정 KST: 2026-10-13T11:43:38.175+09:00
- SHA256: `22ac1197e26844357787eda034d3a51b099a314d901c8f2637ebe050e74f8db6`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_six_feature_groups_estimation_parity_passed`
- C++ 위치: `experiments/analysis_v2/src/estimate.cpp`
- 검증·차이: 여섯 조합 지표 대조 통과. 280개 결합 행·전체 표준화 후 교대2fold·원본 ridge조건 유지; 현재 OOF와 구분.
- 이식 기록: `experiments/analysis_v2/README.md`
- 새 실행 근거: `results/historical_v2_estimate_parity_002/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### extract_homography.py

- 원본: `analysis_v2/extract_homography.py`
- 생성 KST: 2026-10-13T11:41:09.023+09:00
- 수정 KST: 2026-10-13T11:41:09.050+09:00
- SHA256: `00f369ec05cdb89152bc8cb35ebfd1a29226f7ba5bfff56fabfef2978c3f0340`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_281_images_homography_numeric_difference_unresolved`
- C++ 위치: `experiments/analysis_v2/src/extract_homography.cpp`
- 검증·차이: 281장 측정 실패 0. OpenCV 3.4.15 재측정 좌표와 원본 CSV의 최대차 약 0.0349px. 1e-4px 동등성 검사는 실패로 보존.
- 이식 기록: `experiments/analysis_v2/README.md`
- 새 실행 근거: `results/historical_v2_homography_001/audit.json`
- 새 실행 근거: `results/historical_v2_homography_parity_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### judge.py

- 원본: `analysis_v2/judge.py`
- 생성 KST: 2026-10-13T11:42:22.152+09:00
- 수정 KST: 2026-10-13T11:42:22.156+09:00
- SHA256: `11ada2705c785f25040447534c1e54135e47f6c23152bf95bda5196f5055eac2`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_frozen_features_1960_decisions_parity_passed`
- C++ 위치: `experiments/analysis_v2/src/judge.cpp`
- 검증·차이: 기존 Excel의 판정 1960개 불일치 0. 151개 통계 거리 최대차 1.93e-8. 원본 전체 자료 보정 조건을 역사적 분석으로 구별.
- 이식 기록: `experiments/analysis_v2/README.md`
- 새 실행 근거: `results/historical_v2_workbook_parity_001/checks.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### make_outputs.py

- 원본: `analysis_v2/make_outputs.py`
- 생성 KST: 2026-10-13T11:43:39.088+09:00
- 수정 KST: 2026-10-13T11:43:39.135+09:00
- SHA256: `325ce0365572f95cb596cc3307ccf297f232066a2882b91f9feca6b62bdb3c55`
- 관측된 작업자: unknown
- 이식 상태: `cpp_ported_xlsx_heatmap_independent_formula_and_render_checks`
- C++ 위치: `experiments/analysis_v2/src/make_outputs.cpp`
- 검증·차이: 요약/Y/Theta 세 시트·1120개 평균식·히트맵 C++ 생성. 독립 도구에서 수식 재계산·입력 변경 반영·세 시트 렌더링 확인. 실제 Excel 앱 재계산은 미확인.
- 이식 기록: `experiments/analysis_v2/README.md`
- 새 실행 근거: `results/historical_v2_outputs_001/audit.json`
- 새 실행 근거: `results/historical_v2_outputs_001/independent_qa.json`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

## 보존된 결과와 문서

- `analysis_v2/estimate_result.csv` (2026-10-13T11:43:39.004+09:00)
- `analysis_v2/homography_features.csv` (2026-10-13T11:41:36.970+09:00)
- `analysis_v2/homography_test.csv` (2026-10-13T11:41:10.195+09:00)
