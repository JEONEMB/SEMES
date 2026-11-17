# 세메스 설비 적용 — 초기 세팅 · 보정식 학습 절차 (2026-11-16)

알고리즘은 **사진 한 장 → 특징 두 개(s, d) → 보정식 역산 → (Y, Θ)** 로 동작한다.
설비에 넣기 전에 한 번 해야 하는 일은 ① ROI 박스 확인, ② 영점(센터) 기록, ③ 학습 사진 촬영, ④ 보정식 학습·평가, ⑤ (선택) 영점 치우침 보정이다.

## 0. 구성 요소

| 구분 | 파일 | 역할 | 언어 |
|---|---|---|---|
| 실행 (제출) | `QR_Image/OutPut/` (= `PJT/output/Code`) | `alignment(cv::Mat&)` 사진 한 장 추정. 설정 `config/alignment.yml`, 수식 `config/model_*.yml` | C++ / OpenCV 3.4.15 |
| 특징 측정 | `PJT/tool/ksh/align_train.cpp` (`--features` 로 CSV 저장) · `PJT/experiments/KSH/Retrain/code/featdump.cpp` | 라벨 사진마다 기준 사진 대비 s, d 측정 → CSV | C++ |
| 학습·평가·영점 | `setup_package/pose_model_tool.py` | `train` / `eval` / `zero` (C++ `pose_model.cpp` 와 같은 식·역산, 모델 파일 형식 동일) | Python 3 + numpy |
| 텍스트 방식(검토) | `text_roi/` (`roi_setup.py`, `absmeas.py`, `text_align.py`, `alignment_text.yml`) | 기준 사진 없이 절대 좌표 ROI 로 측정 | Python 3 + OpenCV (C++ 이식 전 단계) |

## 1. ROI 박스 확인 (설치 직후 1회)

- **기준 사진 방식(현재 제출본)** : `alignment.yml` 의 ROI 좌표는 `img/roi_template.bmp` 위의 좌표. 기준 사진을 바꾸면
  `relocation`(위상 상관)이 같은 물체 위치로 ROI 를 자동 이동 (허용 300 px, QR 3개 이동 차이 2 px 넘으면 실패 → 카메라 장착 확인).
- **텍스트 방식** : `python text_roi/roi_setup.py <센터 사진> --repeat <센터2> <센터3>`
  → 대략 좌표에서 QR 3개 · 뒷벽 홀 4개 중심을 찾아 박스 좌표를 다시 적고, 확인 그림(`roi_check.png`)을 저장. 경고가 없으면 끝.

## 2. 영점 (센터) 기록 — 정렬할 때와 **같은 방향으로 접근**해서 찍는다

- 백래시 때문에 반대 방향에서 들어오면 같은 지령값에서도 위치가 ±20 µm 정도 달라진다 (1차 사진에서 확인, 보고서 4번).
- 3장 연속 촬영 → 기준 사진 방식은 그중 1장을 `reference_path` 로, 텍스트 방식은 3장 평균 s·d 를 `zero_s_px`, `zero_d_px` 로.
- 기준 사진(또는 영점)은 **측정하는 날 다시 잡는다**. 다른 날 영점을 쓰면 Y 60~95 µm 치우침 (보고서 1·2번 표 b 행).

## 3. 학습 사진 촬영 계획 (1차 + 4차 구성을 그대로 권장)

| 묶음 | 내용 | 장수 | 목적 |
|---|---|---|---|
| 축 위 (1차형) | Y 단독 −2.00 ~ +2.00 mm (0.05 간격), Θ 단독 −1.00 ~ +1.00° (0.01 간격) | 280 | 기울기 a, b (Y·Θ 감도) |
| 사분면 격자 (4차형) | Y ±{0.05, 0.1, 0.2, 0.4, 0.8, 1.6} × Θ ±{0.03, 0.06, 0.12, 0.24, 0.48, 0.96} | 144 | 동시 이동 · 2차 항 · 4개 사분면 균형 |
| 같은 날 센터 | 묶음 시작·끝에 센터 3장씩 | 6 | 상수항(영점 오프셋) |

- 한 방향으로만 접근 (예 : 매 촬영 전 −쪽으로 0.5 mm 지나쳤다가 목표로). 왕복 순서가 섞이면 백래시가 라벨 잡음이 된다.
- 파일 이름 = 라벨 (`Y+0.10 T-0.24.bmp`), 가능하면 모터 실제값 `.txt` 도 함께.

## 4. 특징 측정 → 학습 → 평가

```bash
# (1) 특징 CSV : set, Y, T, s<k>, d<k> 열 (featdump 출력 + 라벨). 학습 제외 행은 exclude=1
# (2) 학습 : 기울기는 학습 세트 전체(세트별 상수항 + 세트 균형 가중), 상수항은 같은 날 센터가 있는 세트에서
python pose_model_tool.py train --features feats.csv --train S1,S4 --s-col s1 --d-col d1 \
       --zero-set S1 --zero-s-col s0 --zero-d-col d0 --order-s 1 --order-d 2 --out model_new.yml
# (3) 평가 : 학습에 쓰지 않은 세트(다른 날 · 다른 패턴)로만
python pose_model_tool.py eval --model model_new.yml --features feats.csv --sets S2,S3 --s-col s1 --d-col d1 --out-csv pred.csv
```

- 차수는 s1d2 (s 는 Θ 1차, d 는 Θ 2차) 권장. s1d3 · s2d2 는 범위 밖(Θ 2°)에서 크게 틀림 (보고서 1번).
- 판정 : 명세 Y ±90 µm · Θ ±0.01°, 권장 멈춤 기준은 절반 (Y ±45 µm · Θ ±0.005°).

## 5. (선택) 영점 치우침 보정 — 설비에서 알려진 위치 10~30점으로

```bash
# zero.csv : true_y_mm,true_t_deg,est_y_mm,est_t_deg  (설비에서 지령 이동 후 alignment() 결과)
python pose_model_tool.py zero --model model_new.yml --table zero.csv --out model_new_zero.yml
```
- 수식의 상수항만 옮겨, 같은 특징에서 평균 치우침(Y, Θ)만큼 작게 읽게 한다 (기울기는 그대로). 큰 이상값(부호 반전 등)은 자동 제외.
- 세메스 결과 35건 예 : 치우침 Y +44.1 µm, Θ −0.0110° → 2겹 검증 Y rms 19.6 µm, 최대 45 µm (보고서 4번).

## 6. 배포

- `config/` 에 새 모델 파일을 넣고 `alignment.yml` 의 `model_file`, `reference_path` 만 바꾼다 (코드 수정 없음).
- 예 : `_work_20261116_개선/deploy_center1/config/alignment.yml` (center1 + `model_center1_s1d2.yml`, 원본 대비 3줄만 다름).
