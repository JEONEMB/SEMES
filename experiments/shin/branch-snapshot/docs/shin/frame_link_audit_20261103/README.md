# 표준 원점 v3: 좌표 연결 감사와 개발 재검증

2026-11-03. 원본 575장 보존. 통계적 기준이며 물리적 참원점이나 T 1 tick 정확도의 인증이 아니다.

## 결과

- `standard_origin_v3.json`: 원점 정의, 날짜 이동, 반응식, 구간 보정, 최적화 상태.
- `coordinates_standard_v3.csv`: 원본/정정 라벨, Y/T 표준 좌표, 학습 가중치. 행별 원본 경로와 row_id 보존.
- `standard_origin_v3_QR630.csv`, `standard_origin_v3_natural.csv`: 영상 좌표계의 가상 표준 특징 위치.
- `refinement_CV.csv`: 6개 전략 × 2개 분할, 각 555장 전방 예측 점수.
- `refinement_zero_holdouts.csv`: 원점 사진을 한 장씩 제외한 재적합 평가.
- `export_checks.json`: 데이터/분할/출력 수치 일치 검사. 단위 px는 좌표 성분 RMS이며 T 오차가 아니다.
- 최종 PDF: `../../output/pdf/T_axis_common_origin_v3_audit_20261103.pdf`.

## 반영 사항

파생 라벨 .33→.32 한 건, .57 이미지 세 장 복원(Actual 로그 충돌은 유지), 원점 노출과 움직임 구간 분리, 반복 변위 후보 네 장 soft weighting(삭제 없음), Y4/Y5 반응 항 추가.

center3 보류 QR .46970→.01519px, 오른쪽 특징 .66716→.19756px. 일반 보류 555장에서도 QR와 오른쪽 잔차 개선. 오른쪽 B는 원점 적합에 미사용이지만 학습 추정 좌표로 반응식을 보정하므로 독립적인 위치 참값은 아니다. 후보 선택에 같은 개발 자료를 재사용했다. 원시 구간 변수 12개가 경계에 도달해 보정량은 제약과 사전척도에 민감할 수 있다.

## 재현

기존 `analysis/session_eda_20261102`, `analysis/standard_origin_20261102`, `analysis/full_scene_origin_20261102`의 관측·캐시·코드가 필요하다. 원본 이미지 경로는 forensics/export/report 코드에서 확인한다. `sources/`와 원본 영상은 수정하지 않는다.

프로젝트 루트에서 번들 Python으로 forensics.py → refine_origin.py --fresh → export_v3.py → report_v3.py → verify_exports.py 순서. `--fresh`가 없으면 완성된 전략의 체크포인트를 재사용한다. 새 PDF 저작 시에는 PDF 스킬의 marker 요구를 먼저 따른다. 이번 PDF의 marker는 이미 실행했다.

필요 패키지: numpy, pandas, scipy, opencv-python, matplotlib, pillow, reportlab, pypdf. 본 환경은 `.analysis_packages`를 PYTHONPATH에 추가하고 OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1을 사용했다. 글꼴은 Windows Malgun Gothic이다.

선택 후보 두 개의 per-feature 상세 CSV는 체크포인트 재사용 때문에 현 파일에 포함되지 않는다. 전체 이미지 평가 CSV에는 여섯 후보가 모두 있다. 전체 상세까지 재생성하려면 --fresh를 쓴다.
