# eda_all_points_20261022 실험 출처

## 문제

대표점 그림 대신 모든 점의 이동 확인

## 조사

기존 C++ EDA 전체 점 내보내기와 그림

## 해결 방법

export.cpp와 plot_all.py의 별도 경로

## 유지한 부분

원본 CSV와 그림

## 바꾼 부분

전체점 출력 C++는 재사용 가능, Python 그림 생성은 개별 이식 대상

## 다른 실험과의 구분

전체점 진단과 모델 평가를 구분

## 시간과 작성 흔적

관측된 최초 파일 생성: 2026-10-22T15:41:09.817+09:00. 관측된 최종 파일 수정: 2026-10-22T15:46:00.001+09:00. 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.

### plot_all.py

- 원본: `eda_all_points_20261022/plot_all.py`
- 생성 KST: 2026-10-22T15:43:54.293+09:00
- 수정 KST: 2026-10-22T15:46:00.001+09:00
- SHA256: `57d9007ec67d0424e84cfe81c11d18ae85266c25a1805223536c5cc28cec3007`
- 관측된 작업자: Codex
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-22T15:43:54+09:00 Codex Add 호출. 로그 `C:/Users/SSAFY/.codex/sessions/2026/10\22\rollout-2026-10-22T15-24-53-01a16248-afec-7ce0-b2eb-799209a4eca9.jsonl`의 231행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### export.cpp

- 원본: `eda_all_points_20261022/src/export.cpp`
- 생성 KST: 2026-10-22T15:41:09.817+09:00
- 수정 KST: 2026-10-22T15:41:09.819+09:00
- SHA256: `191f9e993538c1912aa856b42bc5673b5da105780fc3a9e938531010e9518c08`
- 관측된 작업자: Codex
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-22T15:41:07+09:00 Codex Add 호출. 로그 `C:/Users/SSAFY/.codex/sessions/2026/10\22\rollout-2026-10-22T15-24-53-01a16248-afec-7ce0-b2eb-799209a4eca9.jsonl`의 192행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

## 보존된 결과와 문서

- `eda_all_points_20261022/audit.json` (2026-10-22T15:45:24.572+09:00)
- `eda_all_points_20261022/new_points.csv` (2026-10-22T15:43:57.899+09:00)
- `eda_all_points_20261022/point_linearity_metrics.csv` (2026-10-22T15:45:24.526+09:00)
- `eda_all_points_20261022/summary_metrics.csv` (2026-10-22T15:45:24.568+09:00)
- `eda_all_points_20261022/결과.html` (2026-10-22T15:46:00.000+09:00)
