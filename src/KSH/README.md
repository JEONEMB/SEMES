# src/KSH — 김성현 작업 폴더 안내

| 폴더 | 내용 |
|---|---|
| `EDA/` | 데이터 탐색 (촬영 순서, 화질, Y·Θ 감도) |
| `Motion_Model/` | 운동 모델 (핀홀 카메라, 깊이별 이동) |
| `Hypothesis_1` ~ `Hypothesis_5/` | 가설 검증 기록 (H1 Gabor, H2 ECC, H3·H6·H7, H4 서브픽셀·시차, H5 최적 ROI) |
| `MFC/` | QR Y-Θ Viewer (초기 MFC 측정 앱) |
| `analyze/` | 분석 스크립트 |
| `Alignment/` | **제출 형식 함수** `alignment()` — 사진 한 장 → Y·Θ 어긋남 |
| `Simulator/` | **영점 맞춤 시뮬레이터** (콘솔 + MFC) — 파노라마 가상 장면에서 `alignment()` 로 20회 이내 영점 맞춤 시험 |

Visual Studio 에서는 `QR_Image.slnx` (로컬 `source/repos/QR_Image`) 에 연구 · 모델 · 시뮬레이터가 폴더별로 묶여 있다.
사진(`img/`)은 저장소에 올리지 않는다.
