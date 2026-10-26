# QR Y·Θ EDA 툴

QR 3개 플레이트 영상 세트를 단계별로 분석하고 보고서까지 만드는 Python 툴입니다.
계획은 [`../EDA_PLAN.md`](../EDA_PLAN.md), 1·2차 세트 분석 결과는 [`../EDA_보고서.md`](../EDA_보고서.md)에 있습니다.

## 1. 준비 (처음 한 번)

```powershell
cd src\KSH\EDA
python -m venv .venv
.venv\Scripts\python -m pip install numpy opencv-python pandas matplotlib pyyaml pillow
```

- MP4 영상은 시스템 `ffmpeg`(PATH)를 사용합니다. 없으면 GIF 만 만듭니다.
- 그림 글꼴은 Malgun Gothic (Windows 기본)입니다.

## 2. 실행

`src\KSH\EDA` 에서 실행합니다. 영상은 `data\img\<세트>` 에서 읽기만 합니다.

| 명령 | 동작 |
|---|---|
| `.venv\Scripts\python eda_tool\qr_eda.py --all` | set.yaml(또는 초안)이 있는 모든 세트, Stage 0~5 |
| `... --all --stages 2-4` | 일부 단계만 (앞 단계 결과 표를 읽어 이어서 실행) |
| `... --set ..\..\..\data\img\1st --set ..\..\..\data\img\2nd` | 세트 지정 |
| `... --all --no-video` | 영상 생략 (빠른 확인) |
| `... --all --force` | 캐시 무시하고 다시 계산 (파일 메타 캐시는 유지) |
| `... --results <폴더>` | 다른 결과 폴더에 쓰기 (시험 실행용) |

- 계산값은 파일 해시 기준으로 `eda_results\cache\` 에 저장됩니다. 새 사진만 계산합니다.
- 실행 시간 (1·2차 341장 기준): 처음 약 10분, 캐시가 있으면 전체 약 2분, 새 사진 몇 장 추가 시 30초 안팎.

## 3. 새 세트 추가

1. `data\img\<새 폴더>\` 에 영상을 넣습니다.
2. 같은 폴더에 `set.yaml` 을 만듭니다 (아래 예, 기존 초안은 `eda_results\set_yaml_draft\`).
3. `qr_eda.py --all` 을 실행하면 표·그림·영상·`summary_all_sets.csv`·PDF 가 새 세트를 포함해 다시 만들어집니다.

```yaml
set_id: set3_1030               # 표·그림에 쓰이는 이름 (세트마다 달라야 함)
date: 2026-10-30
type: [mixed]                   # y_sweep | theta_sweep | mixed
reference_images: ["기준 위치/center1.bmp"]
time_source: mtime              # 촬영 시각 = 파일 수정일시 (복사하면 생성일시가 바뀜)
filename_rule:                  # 위에서부터 차례로 적용
  - {glob: "기준 위치/center*.bmp", kind: reference, name_regex: '^center(?P<shot>\d+)$', y: 0.0, t: 0.0}
  - {glob: "Y*T*/*.bmp", kind: mixed, dir_regex: '^Y(?P<y>[+-]?\d+(?:\.\d+)?) T(?P<t>[+-]?\d+(?:\.\d+)?)$', name_regex: '\((?P<shot>\d+)\)$'}
motor_log: {type: txt_absolute, origin_from: reference}   # 같은 이름 .txt 의 절대 좌표 − 기준 위치 좌표 = 지령값
label_reference_csv: "src/KSH/.../parallax_set3.csv"       # (선택) 배경 시차 정답, 없으면 Stage 4 일부 생략
```

- 정규식 그룹 `y`, `t` 는 지령값(mm, °), `shot` 은 같은 지령값의 촬영 번호입니다.
- 해석되지 않은 영상은 `inventory_summary.csv` 의 `unparsed_files` 에 나옵니다.

## 4. 단계와 산출물 (`eda_results\`)

| 단계 | 모듈 | 주요 산출물 |
|---|---|---|
| 0 목록 점검 | `stage0_catalog.py`, `media.py` | `tables\catalog.csv`, `inventory_summary.csv`, `figures\s0_*`, `videos\<세트>_shooting.*` |
| 1 화질 | `stage1_quality.py` | `tables\quality.csv`, `noise_repeats.csv`, `reflection_summary.csv`, `figures\s1_*`, `videos\*_shooting_quality.*` |
| 2 Y vs Θ | `stage2_features.py`, `qr_geom.py`, `center_def.py` | `tables\features.csv`, `linearity.csv`, `separability.csv`, `patch_flow.csv`, `center\center_definition.json`, `center_by_image.csv`, `figures\s2_*`, `qr_info_*`, `videos\*_sync_*` |
| 3 혼합 이동 | `stage3_mixed.py` | `tables\superposition.csv`, `drift.csv`, `center\center_by_set.csv`, `figures\s3_*`, `videos\<2차>_sync.*` |
| 4 라벨 | `stage4_labels.py` | `tables\label_check.csv`, `label_uncertainty.csv`, `figures\s4_*` |
| 5 보고서 | `report.py` | `summary_all_sets.csv`, `EDA_보고서.pdf`, `EDA_보고서_요약.png` |

영상은 GIF(가로 960 px, 20 MB 넘으면 720 px), MP4(균일 간격), `_realtime.mp4`(촬영 간격 비율) 세 가지입니다.

## 5. 알아 둘 점 (개발 중 확인한 것)

- **OpenCV 5.0 `phaseCorrelate` 의 창 함수 인자**를 넘기면 결과가 0 이 되는 경우가 있어 쓰지 않습니다. 또 서브픽셀 값이 치우칩니다 (0.30 px → 0.254 px). 변위장(Stage 2)은 정성 비교용입니다.
- **numpy 2.x** 는 2차원 `np.cross` 를 지원하지 않아 직접 계산합니다.
- **pandas 3** 은 datetime 내부 단위가 초일 수 있어 `astype("int64")` 대신 `total_seconds()` 를 씁니다.
- QR 후보 검출의 닫기 연산은 5×5 입니다. 9×9 는 Y −1.15 mm 이하에서 왼쪽 QR 과 나사 반사가 붙어 검출에 실패합니다.
- 화질 경고는 세트 내 중앙값 ± 3 MAD 와 실무 최소 편차(밝기·대비 2 %, 초점·잡음 10 %)를 **둘 다** 넘을 때만 냅니다.
