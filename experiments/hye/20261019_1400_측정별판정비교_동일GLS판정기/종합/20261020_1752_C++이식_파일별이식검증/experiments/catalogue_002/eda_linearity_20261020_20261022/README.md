# eda_linearity_20261020_20261022 실험 출처

## 문제

복합 촬영에서 축별 선형성과 반복 산포

## 조사

조건별 평균·표준편차, lack-of-fit와 반복 오차

## 해결 방법

기존 shots.csv의 조건별 회귀

## 유지한 부분

두 날짜의 원본 EDA 측정

## 바꾼 부분

조건 평균의 직선 이탈과 반복 잡음을 분리

## 다른 실험과의 구분

조건 평균 곡선과 개별 촬영 궤적을 구분

## 시간과 작성 흔적

관측된 최초 파일 생성: 2026-10-22T15:28:16.030+09:00. 관측된 최종 파일 수정: 2026-10-22T15:46:00.002+09:00. 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.

### analyze.py

- 원본: `eda_linearity_20261020_20261022/analyze.py`
- 생성 KST: 2026-10-22T15:28:16.030+09:00
- 수정 KST: 2026-10-22T15:37:21.072+09:00
- SHA256: `7947d204324e74f7bd81f2d9eb39317874373347d9e6bbbf9482f47b706a6b35`
- 관측된 작업자: Codex
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-22T15:28:14+09:00 Codex Add 호출. 로그 `C:/Users/SSAFY/.codex/sessions/2026/10\22\rollout-2026-10-22T15-24-53-01a16248-afec-7ce0-b2eb-799209a4eca9.jsonl`의 62행.
- 2026-10-22T15:28:48+09:00 Codex Update 호출. 로그 `C:/Users/SSAFY/.codex/sessions/2026/10\22\rollout-2026-10-22T15-24-53-01a16248-afec-7ce0-b2eb-799209a4eca9.jsonl`의 82행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

## 보존된 결과와 문서

- `eda_linearity_20261020_20261022/condition_statistics.csv` (2026-10-22T15:37:25.708+09:00)
- `eda_linearity_20261020_20261022/linearity_metrics.csv` (2026-10-22T15:37:26.767+09:00)
- `eda_linearity_20261020_20261022/old_new_comparison.csv` (2026-10-22T15:37:29.391+09:00)
- `eda_linearity_20261020_20261022/old_zero_axis_metrics.csv` (2026-10-22T15:37:28.383+09:00)
- `eda_linearity_20261020_20261022/shot_residuals.csv` (2026-10-22T15:37:27.742+09:00)
- `eda_linearity_20261020_20261022/source_shots.csv` (2026-10-22T15:37:25.693+09:00)
- `eda_linearity_20261020_20261022/summary.json` (2026-10-22T15:37:29.732+09:00)
- `eda_linearity_20261020_20261022/결과.html` (2026-10-22T15:46:00.002+09:00)
