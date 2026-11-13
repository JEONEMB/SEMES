# QR 순차 보정 · verify_web 이식

로컬에서 확인한 기준은 S15P31A507의 `origin/feature/lye/verify-demo-ui`
`6ab989eb`입니다.
이 폴더에 QR 순차 보정 구현을 두고 확인 앱에서 불러옵니다.

## 폴더 구조

QR 관측, 제어 계산, 확인 앱 어댑터는 이 실험의 `src/`에 있습니다.
Lanczos QR+Bar 실험은 [별도 폴더](../qrwithbar_lanczos/README.md)에 있습니다.

## 연결부

- `src/qr_observer.py`는 보정된 3840×2160 회색조 영상에서
  QR 3개의 가로 이동 중앙값을 읽습니다.
- `src/verify_adapter.py`는 확인 앱의 `set_reference(image)`,
  `infer(image, truth)`, `status()`, `close()` 계약을 따릅니다.
  결과의 Y는 STATUS, Θ는 QR 기준입니다. 각 출처와 QR 변위를 회차 로그에 남깁니다.
- [`verify_web/runner.py`](../../../verify_web/runner.py)는 카메라, 정착 대기, 상태 갱신, 이동, 중단을 기존 방식으로 관리합니다.
  Y가 센터에서 0.005 mm 넘게 떨어져 있으면 Y부터 보정하고, 그 뒤에는 Θ만 보정합니다.
  Θ는 계산량의 100%, 한 명령에 최대 0.30°를 사용합니다.
  QR 가로 편차 0.30 px 이내를 두 번 연속 확인하고, 설정한 STATUS 허용오차도 확인합니다.

QR 모드에서는 절대 목표를 직접 보내고 백래시 보정은 Pi에 맡깁니다.
보정 단계에는 최대 20개의 목표 명령을 보냅니다. 기준 사진을 등록하는 이동과
시작점으로 보내는 이동은 보정 단계 이전에 수행합니다. 마지막 명령 뒤에는 촬영으로
결과를 확인합니다. 기준 사진은 현재 장비의 설정된 센터에서 등록합니다.

## 이식 및 실행

기존 Pi 실행 옵션에 `--algo qr-sequential`을 추가합니다.

```bash
python -m verify_web serve --mode hardware --algo qr-sequential --calib /path/to/camera_calib.yml
```

Θ를 활성화하고 현재 장비의 센터 좌표를 설정한 뒤,
확인 앱에서 기준을 등록하고 시작 위치를 잡아 보정을 실행합니다.
기존 worker 및 mock 선택지는 같은 명령으로 사용할 수 있습니다.

## 검증 범위

로컬에서 확인 앱 Runner와 실제 QR 관측 코드를 모의 장치에 연결했습니다.
양·음 Θ 시작점에서 Y 선행 이동, Θ 명령 상한, 이동 예산과 최종 두 번 확인을
검증했습니다. 실제 Pi 카메라와 스테이지 연결은 시험하지 않았습니다.
