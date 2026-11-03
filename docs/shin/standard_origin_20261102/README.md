# 연속 스캔 기반 표준 원점 v1

## 결과

- `output/pdf/T_axis_standard_origin_v1.pdf`: 분석 결과와 시각화, 정의·검증·한계를 담은 13쪽 PDF.
- `standard_origin.json`: 단일 통계적 원점, 공통 영상 반응, 15개 스캔 그룹의 잠정 Y/T 바이어스.
- `standard_origin_630points.csv`: QR 630점의 가상 원점 좌표.
- `standard_origin_120edges.csv`: 상·하 외부 선 120표본의 가상 원점 좌표.
- `coordinates_in_standard_frame.csv`: 575장의 원래 라벨과 잠정 표준 좌표. 이상 라벨 4장은 적합에 쓰지 않았으며 표에 명시.
- `conditional_origin_intervals.csv`: 기준 사진·모델·영상 등록을 고정한 60회 자세 묶음 bootstrap 구간.
- `prior_and_shape_sensitivity.csv`, `anchor_frame_sensitivity.csv`: 모델 가정과 기준 영상 등록의 민감도.
- `cv_summary.csv`, `cv_predictions.csv`: 567장 비기준 사진의 5-fold 동일 자세 묶음 전방 검증. T 참값 정확도 평가가 아님.

기존 시간 세션 3의 양의 T와 음의 T 스캔을 서로 다른 바이어스 그룹으로 분리했습니다. 그룹 3=양의 T, 그룹 15=음의 T, 나머지는 기존 세션 후보 번호입니다.

## 채택 규약

원점 기준 4장의 추정 Y/T 좌표 평균을 (0,0)으로 고정합니다. 10/15 영상은 기준 사진에서 측정한 공통 이동으로 10/4 영상 좌표계에 등록합니다. 이 이동의 물리적 원인을 특정하지 않습니다.

공통 반응의 2·3차 후보 및 바이어스 사전 폭을 비교했고, 3차 반응·Y 사전 SD 0.05mm·T 사전 SD 0.01°를 개발 CV로 선택했습니다. 좌표 변환은 `추정 표준 좌표 = 원래 명령 라벨 + 그룹별 바이어스`입니다.

원점은 재현 가능한 가상 기준입니다. 특히 그룹별 Y/T 분해는 사전 폭과 최적화 경계에 민감합니다. 이를 확정된 실제 위치 또는 라벨 수정 정답으로 취급하지 마세요. 모든 모델 차이는 보고서에 공개했습니다. 원본 파일과 기존 모델은 바꾸지 않았습니다.

## 재현

이전 `analysis/session_eda_20261102/remeasured/`의 575장 Lanczos-3 전수 재측정 캐시와 이 워크스페이스의 Python 패키지를 사용합니다. 원본 픽셀을 재보간하지 않습니다.

```powershell
& .\analysis\standard_origin_20261102\reproduce.ps1
```

전방 모델·기울기 검증, 동일 자세 보류 평가, 원점 좌표 일치 검사, 60회 bootstrap, 사전·모델 차수·영상 등록 민감도, 5개 초기값 안정성 검사를 포함합니다. 참고: bootstrap은 모델 가정에 따른 오차와 기준 사진 자체의 오차를 포함하지 않으므로 물리적 원점 신뢰구간이 아닙니다.
