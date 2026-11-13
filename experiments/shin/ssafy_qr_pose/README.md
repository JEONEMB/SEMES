# QR 단일 프레임 Y·Θ 추정

보정된 3840×2160 흑백 또는 BGR 영상에서 QR 세 개의 특징점 42개를
추적합니다. 공통 이동과 상대 변형을 사용해 센터 대비 Y(mm)와 Θ(°)를
동시에 반환합니다. 이 실험은 별도 실행 파일로 빌드됩니다.
실장비 순차 보정은 [ssafy_qr_estimation](../ssafy_qr_estimation/README.md)에 있습니다.

- `src/`와 `include/`에는 추론 구현과 공개 API가 있습니다.
- `config/`에는 동일한 학습 계수와 세 촬영별 센터 템플릿이 있습니다.
- `tools/`에는 QR 추적 및 통합 학습·평가 코드가 있습니다.
- `results/pooled-qr-metrics.json`에 보류 평가 결과가 있습니다.

## 실행

Windows에서 OpenCV 3.4.15 및 MSVC v141을 사용한 예입니다.

```powershell
cmake -S experiments/shin/ssafy_qr_pose -B build-qr-pose -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
cmake --build build-qr-pose --config Release
.\build-qr-pose\Release\ssafy_qr_pose.exe "path\to\image.bmp"
```

기본 설정은 역순 격자의 센터(Y 19.816 mm, Θ 0°)입니다.
`QR_POSE_MODEL`에 다른 센터 보정 파일의 경로를 지정할 수 있습니다.
선택한 센터와 촬영 구도가 일치해야 합니다. QR 정합 실패 시 오류를 반환합니다.

## 학습 및 검증

`rig_1st` 280장, 정방향·역방향 격자 각각 1,680장으로 학습했습니다.
NumPy와 OpenCV가 설치된 환경에서 `tools/track_sessions.py`와
`tools/train_pooled.py`를 순서대로 실행하면 재현됩니다.
추적 도구는 홈 디렉터리의 `Desktop`에서 세 원본 ZIP을 읽습니다.

| 보류 방식 | Y RMSE | Θ RMSE | 동시 통과 |
|---|---:|---:|---:|
| Θ 연속 구간 5-fold | 0.0926 mm | 0.0240° | 683 / 3,640 |
| Y 연속 구간 5-fold | 0.0862 mm | 0.0180° | 1,109 / 3,640 |
| 촬영 세션 전체 | 0.1531 mm | 0.0381° | 296 / 3,640 |
| 촬영 날짜 전체 | 0.1642 mm | 0.0411° | 223 / 3,640 |

동시 기준은 Y ±0.045 mm와 Θ ±0.01°입니다. 보류 세션도 기준 센터
사진을 사용합니다. 위 수치는 명령 라벨 기준이며 물리적 자세의 절대
정확도는 검증하지 않았습니다.
