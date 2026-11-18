# QR 전용 자세 추정과 반복 보정

| 사용 환경 | 실행 파일 | 설정과 모델 |
| --- | --- | --- |
| SSAFY 납품 | `src/ssafy_qr_runtime.py` | `config/ssafy.json`, `models/ssafy_qr_pose.npz` |
| Rig 로봇 실증 | `src/run_sequential.py` | 기존 `verify_web` 캘리브레이션과 Rig 영점 사진. 혼합 자세 비교 시 `models/rig_grid*-qr-pose.npz` 사용 |

SSAFY 실행기는 사진의 QR 세 개로 Y·θ를 추정하고 외부 파일에 지정된 목표까지의 차이를 출력한다. Rig 실행기는 기존 스테이지에 Y 이동을 명령한 뒤 QR 수평 변위를 이용해 θ를 반복 보정한다. 모터 명령은 최대 20회다.

[납품·실증 사용법과 결과](../../../docs/shin/qr_only_multiple_exe/README.md)
