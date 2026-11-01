# DOE — 실험계획으로 최종 방법 고르기 (2026-10-29)

`QR_Image` 연구 프로젝트에 추가한 `src/doe` 모듈과 결과. 실행 : `QR_Image.exe verify doe --img <사진 폴더> --out <출력 폴더>`
(처음 실행은 특징 측정에 수 분, 이후는 `doe_features.csv` 캐시 사용)

## 공통 평가

- 허용 범위 : Y ±45 µm, Θ ±0.005° (과제의 절반 — 프로젝트 기본)
- 학습·선택 : 1차 첫 번째 측정 240장, 연속 14 블록 교차검증
- J = √mean((eY/45 µm)² + (eΘ/0.005°)²), 항마다 100 에서 자름. 작을수록 좋음
- 2차 57장은 선택에 쓰지 않고 결과 확인만

## 실험과 최종 채택

| 실험 | 인자 · 수준 | 최종 채택 | 근거 |
|---|---|---|---|
| 1 정렬 순서 | Coarse only / Fine only / Coarse→Fine / Fine→Coarse (보간 = OpenCV ECC, 식 = 1차) | **Coarse → Fine** | 합성 성공 45/45 · 오차 0.013 px, J 0.721 (나머지 0.874 / 136 / 102) |
| 2 보간 | 선형, Keys −0.5, OpenCV 3차 −0.75, Lanczos3, B-스플라인, O-MOMS × 가우시안 σ 0 / 0.5 / 1.0 / 1.5 | **O-MOMS, σ 0 (IC-GN 직접 구현)** | J 0.682 (최소), 합성 치우침 0.0036 px |
| 3 식 차수 | s, d 각각 Θ 1·2·3차 + Y² | **s Θ3, d Θ2** | J 0.682 → 0.604, 허용 안 213 → 229/240 |

## 폴더

| 경로 | 내용 |
|---|---|
| `code/doe/doe.hpp, doe.cpp` | DOE 모듈 (커널, IC-GN, 합성 시험, 블록 교차검증, 차수 비교) |
| `code/yt/yt_verify.cpp, .hpp` | `verify doe` 단계 등록 |
| `results/` | `RESULT.md`, 실험별 CSV, 채택 모델 `doe_best_model.yml` |

C++17, OpenCV 3.4.15 API. 결과 CSV 는 보고서 그림(`중간보고서_그림/2차보강_20261029/figs.py`)의 입력이다.
