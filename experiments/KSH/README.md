# experiments/KSH — 김성현 실험 기록

실제 알고리즘 코드는 저장소 루트의 `include/`, `src/`, `tool/ksh/`, `config/` 에 있다 (2026-11-04 구조 변경, 이전 위치 `src/KSH/`).
이 폴더는 그 알고리즘을 고르기까지의 실험 기록이며 루트 빌드에 포함하지 않는다.

| 폴더 | 내용 |
|---|---|
| `EDA/` | 데이터 탐색 (촬영 순서, 화질, Y·Θ 감도) |
| `Motion_Model/` | 운동 모델 (핀홀 카메라, 깊이별 이동) |
| `Hypothesis_1` ~ `Hypothesis_5/` | 가설 검증 기록 (H1 Gabor, H2 ECC, H3·H6·H7, H4 서브픽셀·시차, H5 최적 ROI) |
| `DOE/` | **실험계획** — 정렬 순서 · 보간 방법 · s·d 식 차수를 같은 기준(허용 ±45 µm · ±0.005°, 블록 교차검증 J)으로 비교, 최종 채택 근거와 결과 CSV |
| `analyze/` | 분석 스크립트 |
| `MFC/` | QR Y-Θ Viewer (초기 MFC 측정 앱) |
| `Simulator/` | 영점 맞춤 시뮬레이터 (콘솔 + MFC, 2026-10-28 판, 기록용) — 최신 판은 `tool/simulator/` |
| `Integrated/` | 통합 정렬기 (ksh 추정 + 팀 판 검출·QR 검증) — 루트에서 `-DKSH_BUILD_INTEGRATED=ON` 으로 빌드 |
| `Retrain/` | **1~4차 전체 재학습 (2026-11-10)** — 차수 1~3 · 기준 센터 · 사분면 균형 · 반복 학습 비교, 4차 데이터 진단, 폐루프 영향 분석, 후보 수식 9개 (배포 수식은 그대로) |
| `TextAlign/` | **center1 기준 재학습 · 텍스트 설정(절대 ROI) · 엔코더 영점 (2026-11-16)** — 1차식(s1d1) 채택, 배포 코드는 `OutPut_Text/Code/` |
| `Alignment_vs/` | 이전 `src/KSH/Alignment` 의 Visual Studio 프로젝트·CMake 사본 (로컬 `QR_Image.slnx` 용, 파일 경로는 이전 배치 기준) |

사진(`img/`)은 저장소에 올리지 않는다. 실험 폴더 안의 상대 경로(`../Alignment` 등)는 이전 배치 기준이라 그대로는 빌드되지 않을 수 있다.
