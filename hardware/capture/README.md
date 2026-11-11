# capture — 학습·평가 사진 자동 촬영

우리 장비에서 스테이지를 정해진 간격으로 옮기며 사진을 찍는 도구다. 김성현·이혜원 알고리즘의 ROI 재설정과 재학습에 쓴다.
폴더와 파일 이름은 SEMES가 준 사진과 똑같이 만든다. 그래서 두 사람의 도구(`align_train`, `dataset.cpp`의 라벨 규칙 등)를 고치지 않고 바로 읽을 수 있다.

| 계획 | 찍는 사진 | 장수 | 용도 |
|---|---|---|---|
| `set1` | `center.bmp`, `Y/Y-2.00 … Y+2.00` (0.05mm 간격), `Theta/T-1.00 … T+1.00` (0.01° 간격) | 1 + 80 + 200 | 학습 (SEMES 1차와 같은 구성) |
| `y`, `t` | `set1` 중 한 축만 | 1 + 80 / 1 + 200 | Θ 펌웨어 전 Y만 먼저 찍을 때 |
| `set2` | `center1.bmp`, `Y+0.50 T+0.10/Y+0.50 T+0.10 (1…3).bmp` | 1 + 20자세 × 3 | 평가 (학습에 쓰지 않음) |

모든 값은 **센터 기준 상대값**이다. 파일 이름의 부호는 우리 스테이지 명령 방향과 같다(Y+ = `MOVE` +방향).

## 빌드 (라즈베리파이)

`hardware/stage`와 `hardware/camera`(feat/lye/camera-calib)를 함께 빌드한다.

```bash
cd hardware/capture
cmake -S . -B build -G Ninja
cmake --build build
./build/plan_test            # 파일 이름 규칙 시험 (장치 불필요)
```

PC나 WSL에서는 `plan_test`만 빌드된다.

## 실행

준비: 드라이버를 올리고(`hardware/stage/README.md`), 스테이지 원점 복귀, 초점·노출·조명 고정.

```bash
# 1) stagectl 로 플레이트를 센터(배경 띠와 평행, 화면 가운데)에 맞춘다
sudo ../stage/build/stagectl home y
sudo ../stage/build/stagectl move 25.0          # 예시. 화면을 보며 조정
sudo ../stage/build/stagectl status             # 이 좌표를 적어 둔다 → 다음부터 --center 로

# 2) 계획 확인 (장치 없이)
./build/capture_dataset --plan set1 -o ~/data/rig_1st --dry-run

# 3) 촬영
sudo ./build/capture_dataset --plan y    -o ~/data/rig_1st --calib ../camera_calib/camera_calib.yml --center-here
sudo ./build/capture_dataset --plan set1 -o ~/data/rig_1st --calib ../camera_calib/camera_calib.yml --center 25.000 0.000
sudo ./build/capture_dataset --plan set2 -o ~/data/rig_2nd --calib ../camera_calib/camera_calib.yml --center 25.000 0.000
```

- Ctrl-C를 누르면 지금 찍던 사진까지 저장하고 멈춘다. 같은 명령에 `--resume`을 붙이면 없는 사진만 이어서 찍는다.
- Θ 펌웨어가 준비되면 `set1`을 **한 번에 처음부터** 찍는다. 기준 사진은 찍고 나서 시간이 지나면 정확도가 떨어진다(KSH 3rd 평가에서 228/234 → 179/234). 그래서 Y와 Θ를 다른 날 나눠 찍지 않는다.

## 한 장 찍는 순서

1. **같은 방향에서 접근.** 목표보다 `--approach-y` 0.2mm, `--approach-t` 0.1°만큼 −쪽에 먼저 갔다가 목표로 간다. 백래시 영향을 모든 사진에서 같게 하기 위해서다.
2. **흔들림 대기 후 촬영.** `Camera::captureSettled`를 쓴다. 안 멎으면 `--settle-retries`번 다시 시도한다.
3. **STATUS로 실제 위치를 읽는다.** 목표와 다르면 경고한다.
4. **저장.**
   - 3840×2160 BMP: 알고리즘 입력
   - 원본 PNG: `<출력>_raw/`
   - 같은 이름의 TXT: 명령값, 절대 좌표, STATUS, 안정 대기, 노출
   - `manifest.csv`에 한 줄 추가

## 센터 확인 (`_check/`)

센터 사진은 처음, `--check-every`(100)장마다, 마지막에 다시 찍어 처음 센터와 비교한다. 결과는 `_check/drift.txt`에 남는다.

| 값 | 뜻 |
|---|---|
| 전체 이동 | 배경까지 다 움직였다면 카메라나 배경판이 움직였다는 뜻 |
| 플레이트 창 이동 (`--plate-roi`, 기본 SEMES 플레이트 위치) | 스테이지가 같은 좌표로 돌아왔을 때의 반복 오차 |

0.3px(Y 허용 45µm ≈ 0.64px의 절반)를 넘으면 `← 큼`으로 표시한다. 장비를 건드렸다면 다시 찍는다.

## 출력

```
rig_1st/
  center.bmp  center.txt  manifest.csv
  Y/Y-2.00.bmp  Y/Y-2.00.txt  …
  Theta/T-1.00.bmp  …
  _check/session.txt  drift.txt  center_after_0100.bmp  center_end.bmp
rig_1st_raw/            원본 4056×3040 PNG (같은 하위 경로)
```

용량: BMP 8.3MB × 281장 ≈ 2.3GB, 원본 PNG까지 합치면 약 4GB. `--no-raw`를 주면 원본을 저장하지 않는다.

## 주요 옵션

| 옵션 | 기본 | |
|---|---|---|
| `--y-range` / `--y-step` | 2.0 / 0.05 mm | |
| `--t-range` / `--t-step` | 1.0 / 0.01 ° | |
| `--poses FILE.csv` | 20자세 | `set2` 자세 목록, 한 줄에 `y_mm,t_deg` |
| `--repeat` | 3 | `set2` 자세마다 찍는 장수 |
| `--average` / `--center-average` | 1 / 4 | 몇 장을 평균할지 |
| `--home` | none | `y`, `t`, `both`: 시작 전 원점 복귀 |
| `--exposure`, `--gain`, `--zoom`, `--settle-*` | camera 기본값 | `cam_capture`와 같음 |
