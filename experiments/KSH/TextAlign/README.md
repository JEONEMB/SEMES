# TextAlign — center1 기준 재학습 · 텍스트 설정(절대 ROI) · 엔코더 영점 (2026-11-16 ~ 11-17)

기준 이미지 없이 **텍스트 설정 파일(`alignment.yml`)만으로** 정렬하는 방식을 고르기까지의 실험 기록이다.
배포용 C++ 코드는 `OutPut_Text/Code/` 에 있고, 이 폴더는 루트 빌드에 포함하지 않는다.

## 결과 요약

| 항목 | 결과 |
|---|---|
| y −0.1 치우침 원인 | 1차 세트의 0.1 mm 배수 음(−)쪽 백래시 부분집합 → 배율 오차 약 1.1 % (+21 µm vs −20 µm), 센터 사진이 엔코더 영점에서 +35~44 µm 떨어진 상수 오프셋, 백래시 약 ±20 µm |
| center1 기준 재학습 | 학습 1차+4차 / 평가 2차+3차. 세트별 상수항(고정효과) + 세트 균형 가중(1/n). 과적합을 피하려고 **1차식(s1d1)** 채택 |
| 텍스트 절대 ROI | QR 3개 중심(행·열 서브픽셀 에지 + 강건 직선 맞춤) + 배경 구멍 4개(방사 대칭 중심). Python 과 C++ 특징 차이 2.3×10⁻¹¹ px (718장) |
| 엔코더 영점 | `zero` = 센터 측정값 + `center_to_encoder` (Y −0.0353 mm, Θ −0.0030°) |
| 판정(모터 엔코더 기준) | 2차 48/57, 3차 234/234, 합계 282/291. σ Y 13.7 µm · Θ 0.0035° |
| 알려진 한계 | 구멍 근처 먼지(24~30 px 거리에서 무경고 오차 가능), Θ 영점 ±0.003°, Θ > 1° 미학습, 대략 위치 탐색 범위 ±350 px |

## 폴더

| 경로 | 내용 |
|---|---|
| `code/retrain_center1/` | center1 기준 재학습 실험 (`exp1.py` 차수·기준 비교, `exp2.py` 후보 정리, `semes.py` 세메스 표 대조, `lib.py` 공통) |
| `code/text_roi/` | 텍스트 절대 ROI 측정 Python 원형 (`absmeas.py` 측정, `roi_setup.py` 세팅, `text_align.py` 정렬, `text_config.py` 설정 읽기·쓰기, `exp3.py` 평가, `batch.py`·`absjoin.py`·`bgtest.py` 보조) |
| `code/setup_package/` | 보정식 학습·평가·영점 도구 `pose_model_tool.py` 와 절차 문서 |
| `data/` | 719장 특징 (`features_719x4ref_labeled.csv`, `abs_features_719.csv`, `abs_features.jsonl`), 배경 구멍 시험 `bg_s3.json` |
| `models/` | 후보 수식 (`model_center1_s1d1*.yml`, `model_center1_s1d2*.yml`, `model_text_s1d2.yml`) 과 텍스트 설정 원형 `alignment_text*.yml` |
| `reports/` | 검토 결과 보고서(HTML), 실험 결과 JSON, 2·3차 사진별 예측 CSV, 세메스 표 대조 |
| `figures/` | 항목 4 그림, center1 ROI 확인 그림, ROI 미리보기 |

사진(`img/`)과 마스킹 검증 PNG(약 1 GB)는 저장소에 올리지 않는다.
스크립트 안의 절대·상대 경로(`/mnt/user-data/...`, `data/features_719x4ref.csv` 등)는 작업 당시 배치 기준이라 그대로는 실행되지 않을 수 있다.
