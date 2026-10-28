# Hypothesis 5 — 최적 ROI 학습 (OpenCV, C++)

플레이트 QR 3개에 어떤 배경 영역(ROI)을 더해야 Y·Θ 오차가 가장 작은지를 데이터로 정했습니다.
- ROI 선택과 학습 : 1차 **첫 번째 측정**(center, Y 0.1 mm 배수, Θ 스윕)만 사용
- 시험 : **2차 57장**만 사용

## 결론
- **최적 ROI = 하단 문 좌우 홀 W2 (2867, 1357) + W3 (1010, 1677), 각 120×120 px.** 후보 58곳(하단 문·문 아래 바닥·중간 돌출부·벽) 중에서 골랐습니다.
- 돌출부는 플레이트와 깊이가 같아 시차가 없습니다. 좌우 홀 짝은 원근 휨이 서로 지워져 잡음이 1/2.5 로 줄어듭니다.
- **2차 목표 근처(Θ ≤ 0.2°) 33/33 허용 안** (Θ RMS 0.0024°, Y RMS 11.2 µm).
- Θ 1° 는 9/12, Θ 2° 는 0/12 입니다. 오차가 Θ 에 비례(약 1 %)하는 날짜 간 배율 차이며, 반복 보정으로 확인할 과제입니다.
- **1차 0.05 mm 묶음의 중심 어긋남은 실재합니다** (0.1 묶음 대비 +0.20 px). 상대값으로 1/3 로 줄지만, 사전 기준(Θ ≤ 0.0025°)을 근소하게 넘어(0.00253°) 학습에서 뺐습니다.
- **서브픽셀이 필요한 근거** : 측정을 정수 화소로 반올림하면 1차 240장 중 78장만 허용 안. 0.1 px 단위까지는 유지됩니다.
- 기존 4곳 평균(H4 트랙 B)과의 차이는 작습니다(J 2.5 %, 2차 동일). 운영 코드는 오히려 단순해졌습니다(벽 4 → 2, Θ² 제거).

## 폴더

| 폴더 | 내용 |
|---|---|
| `docs/H5_ROI_optimization.md` | 수행 방법, 결과 표, 시행착오, 재검증, 오버엔지니어링 검토 |
| `report/` | 쉬운 설명 HTML 보고서 (실제 사진으로 만든 영상 4편, Y–Θ 좌표 그래프, 과녁 그래프). `index.html` 을 웹 서버로 열기 (`data.json` 을 읽음) |
| `results/verify/` | 통합 실행기 `verify all` 결과 (H4 M1~트랙 A, H5_roi) 와 콘솔 기록. 영상 프레임 PNG 는 빼고 mp4 만 `report/media/` 에. H4 결과는 단계별 실행과 완전히 같고, H5 는 1e-5 이내 (측정 캐시의 소수 자릿수 차이) |
| `results/windows_debug_20261028/` | Visual Studio Debug 빌드로 `verify all` 을 실행한 결과 CSV·RESULT.md, 콘솔 원문(`run_all.log`), 정리와 리눅스 비교(`RUN_SUMMARY.md`) |
| `config/roi_search.yml` | ROI 탐색 구역 설정 (VS 프로젝트가 처음 실행 때 만든 기본값) |
| `figs/` | 한국어 ROI 그림과 그리는 스크립트 (C++ 결과 CSV 를 읽어 그리기만 함) |
| `code/` | VS 프로젝트 `src` 와 같은 내용 (`roi/`, `report/` 추가, `yt/yt_model` 에 Θ² 선택, `yt_verify` 에 roi·media 단계) |
| `audit/` | 독립 재검증 스크립트와 결과 (`audit_result.json`, `remeasure_result.txt`), 촬영 기록 사본 |
| `tools/encode_videos.sh` | 영상 프레임 → mp4 (ffmpeg) |
| `backup/` | 이전 판 코드(v1~v6)와 결과 폴더, 실행 기록. 결과 폴더의 큰 그림은 JPG 로 바꿔 보관 |

## 실행 (VS 프로젝트 QR_Image, 작업 폴더 = 프로젝트 폴더)
```
QR_Image.exe verify roi      H5 최적 ROI (처음 약 5분, 이후 측정 캐시로 수 초)
QR_Image.exe verify media    보고서 영상 프레임 (PNG). mp4 변환은 tools/encode_videos.sh
QR_Image.exe verify all      H4 전체 + H5
```
설정 : `config/roi_search.yml` (탐색 구역·후보 수·허용 오차). 없으면 기본값으로 생성합니다.
결과 : `results/verify/H5_roi/RESULT.md`, `roi_selected.yml` (선택 ROI + 모델 계수), 예측 CSV.
