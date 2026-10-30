# 기존 ECC 수치 차이에 대한 OpenCV4 분리 대조

초기 `analysis_agent/measure_qr.py`의 affine 및 `analysis_v2/extract_homography.py`의 homography를 이식한 C++ 알고리즘을 공유하고, OpenCV4.11만 연결한 실행 파일을 따로 만들었다. 원본 시각·해시·관측된 작업자 쓰기 호출은 [전체 파일 목록](../../../실험%20방법/전체_파일_목록.md)에서 확인한다. 이번 작업자는 Codex다.

문제는 OpenCV3.4.15 C++ 재측정과 기존 저장 CSV의 translation 값이 약0.03px 달랐다는 것이다. 영역·초기화·반복수·epsilon·관측 필드를 유지한 채 라이브러리만4.11로 바꾸어 원인을 조사했다. 실행 파일은 기존 OpenCV3 alignment 라이브러리와 링크하지 않는다.

2026-10-23 10:30:46 KST에 `../../results/historical_affine_opencv4_20261023_001`에서281장을 실패 없이 재측정했다. 완료 후 대조인 `historical_affine_opencv4_parity_20261023_002`에서 기존 CSV와 최대 translation 차이는0.036315918px로 남았다. `historical_affine_opencv3_vs4_20261023_001`의 ECC 변환 필드는 동일하고 NCC 점수만 최대3.16e-6 차이다. 라이브러리 버전 변경으로 기존 CSV 차이가 해결되었다고 주장할 수 없다. affine 원본의 완료 체크는 보류한다.

`historical_affine_opencv4_parity_20261023_001`은 생성 중 CSV를 먼저 대조한 잘못된 실행이다. `invalid_comparison.json`을 붙이고 보존했으며 유효한 근거로 사용하지 않는다.

homography 실행 파일은 빌드했으나 아직 이 환경의 전체 실행·대조 전이다. 저장된 이전 결과와 새 계산을 섞지 않는다. 원본과 DATA는 현재 위치를 유지하고 legacy·Git 작업은 하지 않는다.
