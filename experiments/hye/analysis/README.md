# 저장소 밖 분석·실험 코드 통합 (2026-10-30)

10-06 ~ 10-29에 저장소 밖 `projects/2026-10_SEMES/`에서 한 분석·실험의 코드와 문서를 원래 폴더 구조 그대로 옮겼다. 폴더 이름은 원래 이름이다. 루트에 있던 낱개 문서는 `_root/`에 있다.

## 넣은 것과 뺀 것

- 넣음: `.py .cpp .hpp .h .c .cmake .md .txt .ps1 .bat .toml .yml .yaml`, `CMakeLists.txt`, `CMakePresets.json` 중 1MB 이하
- 뺌: 영상(BMP·PNG·MP4), 측정·모델 파일(npz·joblib·json·csv), 발표자료(pptx), 1MB 넘는 파일, `data/`, `build*/`, `deps/`(파이썬 패키지 사본), `third_party/`의 OpenCV·OpenBLAS 사본
- `final/third_party/ecc5`(ECC 호환 소스)는 넣었다
- `hw-analysis`는 같은 저장소의 옛 clone이라 커밋된 적 없는 `tool/hw`, `docs/hw`, `docs/validation-runs`만 넣었다
- 원래 폴더에 있던 하위 `.gitignore`는 원래 배치 기준이라 넣지 않았다

뺀 결과·데이터·모델은 원래 폴더(`projects/2026-10_SEMES/<폴더명>/`)에 그대로 있다. 그래서 여기 있는 스크립트는 대부분 이 폴더에서 바로 실행되지 않는다. 문서 안의 절대 경로도 원래 위치를 가리킨다.

## C++ 이식이 끝난 원본

`final/config/port_progress.json`에서 `cpp_complete`인 원본 55개는 여기에 넣지 않았다. 원본은 저장소 밖 `projects/2026-10_SEMES/00000.legacies/`로 옮겼고, 대응하는 C++ 구현은 `final/` 아래(`experiments/analysis_v*`, `src`, `include`, `tool`)에 있다. 원본이 빠져서 같은 폴더의 다른 Python 스크립트가 import에 실패할 수 있다.

| 원래 폴더 | 옮긴 원본 수 |
|---|---|
| `make_robot_motion.py` (루트) | 1 |
| `video_output` | 1 |
| `analysis_agent` | 3 |
| `analysis_v2` | 4 (폴더 전체) |
| `analysis_v3` | 17 |
| `analysis_v4` | 1 (폴더 전체) |
| `hw-analysis/tool/hw/eda/src` | 15 |
| `codex_experiments_20261014` | 13 |

## 주요 폴더

| 폴더 | 내용 |
|---|---|
| `analysis_v2` ~ `analysis_v8`, `analysis_*_20261019` | 10-12 ~ 10-19 초기 분석 (QR 측정, Y/Θ 구분, 특징점·패치 이동장) |
| `eda_*_20261022`, `experiment_documentation_20261021` | 10-20 ~ 10-22 EDA와 실험 문서 |
| `codex_experiments_20261014` | 10-14 ~ 10-29 joint/control 실험 (구조점630, ECC, 위·아래 선, 337/341, 구조점 vs ECC 비교) |
| `joint_center_630_20261022`, `motion_*_20261022` | 구조점630 기준과 동작·속도 실험 |
| `final` | Python → C++ 전체 이식 작업 공간 (체크리스트 `final/전체_파일_목록.md`) |
| `legacy` | 10-22 이전 스냅샷 |
