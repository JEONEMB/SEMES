# 리눅스 검증 기록 (2026-10-28)

컨테이너(2코어, OpenCV 4.6 · 3.4 호환 API, g++ -O2)에서 이 코드로 실행한 결과. 경로만 바꾼 같은 설정 파일.

| 파일 | 내용 |
|---|---|
| field_report.md | 가상 장면 요약 (`--build`) |
| validation.md / .csv | 합성 vs 실제 사진 모델 추정 비교 (`--validate`, 티칭 = center 사진) |
| monte_기본_100회.* | 기본 로봇 오차, 100회 |
| monte_가혹_50회.* | 배율 σ 5 %, 반복 15 µm / 0.003° (stress_config.yml), 50회 |
| *_teach_rendered_origin* | 티칭 사진 = 자세 (0,0) 합성 사진일 때 (상수항 치우침 확인용) |
