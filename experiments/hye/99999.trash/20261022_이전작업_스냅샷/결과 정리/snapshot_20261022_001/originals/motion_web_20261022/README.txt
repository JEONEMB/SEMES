Motion Studio · 630 structural boundary model

실행: python server.py (이 폴더)
접속: http://127.0.0.1:8765
필수: numpy, scipy, opencv-python

기본 추론: v1_edge630_boundary_fixed
고정 center SHA-256 확인 → 디코딩 격자 755점 중 대각선 접촉 125개 제외
→ 고정 630개 구조 ID → NCC ±130 px → 9x9 erf 경계 모델 soft_l1 맞춤
→ 창 이동 최대3회 → 11x11 재맞춤 진단 → 현재−center 변위
→ 진단 실패점 포함 전체630점 동일 가중치 P12 최소제곱
→ 학습된 원점통과 Y/T 이차곡선 GLS 비용 비교(8회 갱신).
LK/GN17 패치 추적 및 6개 원점 후보는 사용하지 않습니다.

모델: ../codex_experiments_20261014/center_patch_v2_20261021/v1_edge630_model.json
측정 원본: ../analysis_grid_corners_20261019/track_corners.py
구조/기준 좌표 스냅샷과 출처: structural_assets/
630개 ID를 원래 번호로 표시하며 불확실성 이유도 공개합니다.
파일명은 서버에 전송하지 않으며 브라우저에서 예측 후 정답 평가에만 사용합니다.
정답 없는 이름은 평가 불가로 표시합니다. T 물리 단위는 미확정입니다.

기존 205점 웹 소스 보존: archive_205/
검증: verification_630/verify.py → checks.json, recomputed280.csv, failures.csv
보존된 280장 측정값 기반 모델 재계산과 원본 이미지 일부 직접 측정 대조를 구분합니다.
전체 학습 재대입 결과이며 새 OOF 또는 현장 독립 성적으로 해석하지 않습니다.
규칙 단일 원본: ../ALGORITHM_RULES.md


폴더 / ZIP 일괄 평가
페이지: http://127.0.0.1:8765/batch
기본 버튼: 지정 ZIP 폴더 불러와 평가
대상: ../hw-analysis/data/hw/raw/zip/*.zip
실제 자료 2개 압축 안에 분할ZIP을 복원해 총341장(단일축280, 복합57, center4)을 찾습니다.
다른 폴더 선택 / 폴더 드래그 / 이미지·ZIP 여러 파일 선택도 가능합니다.
각 실행은 runs/<새 UUID>/에 별도로 보존합니다. 원본 ZIP은 변경하지 않습니다.
동시1~4장 처리, 진행률, 중단 후 남은 사진 재개, 저장 실행 조회를 지원합니다.
사진별 표, Y/T 산점도(오분류 포함 벡터 성분), 축 혼동행렬, 개별630점 상세,
CSV/JSON 내보내기. 복합입력은 단일축 모델 범위 밖으로 별도 집계합니다.
이름/라벨은 바이트 추론 함수에 전달하지 않습니다. 추론 결과 뒤에서만 평가합니다.
단일축 처리 실패도 단일축 성적 분모에 포함합니다. center/미라벨/복합은 별도입니다.
전체341장 실추론을 사전 실행한 것은 아닙니다.
검증: verification_batch/checks.json, verify.py (실제ZIP 목록, 업로드·취소·재개·상세·분모 검증)
