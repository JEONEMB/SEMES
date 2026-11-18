# SSAFY QR 자세 추정기

SSAFY Test Image Set과 같은 카메라 시점, QR 배치, 정류 상태에서 촬영한 3840×2160 이미지의 QR 세 개로 Y(mm)와 θ(°)를 추정한다. 모델에 QR 기준 패치가 포함되어 있어 실행 시 원점 사진을 읽을 필요가 없다. 사진 입력과 JSON 출력으로 연결하며, 프레임은 순차 처리한다.

## 실행

Python 3.12 환경에서 검증했다. 이 저장소의 루트에서 실행한다.

```sh
python -m pip install -r experiments/shin/qr_only_multiple_exe/requirements.txt
python experiments/shin/qr_only_multiple_exe/src/ssafy_qr_runtime.py image.bmp
```

여러 파일을 전달하면 사진마다 JSON 한 줄을 출력한다. `inference_ms`는 로딩과 이미지 디코딩을 제외한 프레임당 추론 시간이다. 입력 오류나 QR 추적 실패는 stderr에 원인을 남기고 종료 코드 1을 반환한다. 프로그램에서 직접 호출할 때는 `QRPose(model_path, target=(y_mm, theta_deg)).predict(frame)`을 사용한다. `frame`은 정류된 `uint8` 그레이스케일 배열이다.

## 보정 목표 바꾸기

`experiments/shin/qr_only_multiple_exe/config/ssafy.json`의 `y_mm`, `theta_deg`에 학습 좌표계 기준 목표 위치를 지정한다. `model`의 상대 경로는 설정 파일 디렉터리를 기준으로 해석한다.

```json
{"model": "../models/ssafy_qr_pose.npz", "y_mm": 0.05, "theta_deg": 0.01}
```

출력 `y_mm`, `theta_deg`는 **추정 위치 − 설정 목표**다. 예를 들어 학습 기준점 사진을 위 설정으로 처리하면 약 `(-0.05 mm, -0.01°)`가 나온다. 보정기의 좌표계와 일치하는 경우 목표로 향하는 상대 이동량은 출력의 부호를 뒤집어서 얻는다. 한 번 이동한 뒤 새 프레임으로 잔차를 다시 추정할 수 있다. 카메라와 QR의 장착 위치가 바뀌는 경우에는 캘리브레이션을 다시 확인해야 한다.

## 확인한 범위

- SSAFY 혼합 자세 287장에 대한 그룹 분할 검증에서 `|Y 오차| ≤ 0.045 mm` 및 `|θ 오차| ≤ 0.01°` 동시 통과는 216장(75.3%)이었다. 부호 있는 오차의 평균 ± 표준편차는 Y `+0.0012 ± 0.0434 mm`, θ `−0.0002 ± 0.0078°`였다.
- Y 좌표 값을 통째로 분리한 평가에서는 같은 기준으로 52/287장(18.1%)이 통과했다. 평균 ± 표준편차는 Y `−0.0116 ± 0.1396 mm`, θ `+0.0010 ± 0.0250°`였다.
- SSAFY 원본 575장에서 패키지 추정값이 개발판과 일치했다. Windows에서 혼합 자세 287장을 단일 스레드로 실행한 추론 시간은 중앙값 6.09 ms, 95백분위 11.51 ms였다.
- 위 수치는 오프라인 한 프레임 평가 결과다. 새로운 목표 근처에서의 반복 보정 결과는 수신측 장비에서 새 사진과 독립 위치 기준으로 확인해야 한다.
