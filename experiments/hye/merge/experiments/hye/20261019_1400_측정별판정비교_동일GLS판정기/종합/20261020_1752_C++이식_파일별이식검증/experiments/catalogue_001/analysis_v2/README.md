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
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### extract_homography.py

- 원본: `analysis_v2/extract_homography.py`
- 생성 KST: 2026-10-13T11:41:09.023+09:00
- 수정 KST: 2026-10-13T11:41:09.050+09:00
- SHA256: `00f369ec05cdb89152bc8cb35ebfd1a29226f7ba5bfff56fabfef2978c3f0340`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### judge.py

- 원본: `analysis_v2/judge.py`
- 생성 KST: 2026-10-13T11:42:22.152+09:00
- 수정 KST: 2026-10-13T11:42:22.156+09:00
- SHA256: `11ada2705c785f25040447534c1e54135e47f6c23152bf95bda5196f5055eac2`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### make_outputs.py

- 원본: `analysis_v2/make_outputs.py`
- 생성 KST: 2026-10-13T11:43:39.088+09:00
- 수정 KST: 2026-10-13T11:43:39.135+09:00
- SHA256: `325ce0365572f95cb596cc3307ccf297f232066a2882b91f9feca6b62bdb3c55`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

## 보존된 결과와 문서

- `analysis_v2/estimate_result.csv` (2026-10-13T11:43:39.004+09:00)
- `analysis_v2/homography_features.csv` (2026-10-13T11:41:36.970+09:00)
- `analysis_v2/homography_test.csv` (2026-10-13T11:41:10.195+09:00)
