# control 측정기 교체 검토: GN 정밀화의 효과

[검출기 비교](../../20261029_112722_QR특징점검출기_비교/결과/README.md)의 단순 판정기에서는 같은 구조점에 OpenCV LK만 쓴 측정(276/280)이 채택 측정기(273/280)보다 나았다. 이 실험은 측정만 바꾸고 채택 control의 실제 학습식으로 다시 확인한다. 채택 모듈(`00000_final_Y_T control`)은 바꾸지 않는다.

## 조건

처음 실행(`results/`)은 A와 B만 비교했는데, B는 GN만 뺀 것이 아니라 흐림·창 크기·피라미드·점 선택까지 달라 원인을 가를 수 없었다. 그래서 채택 측정기에서 GN만 뺀 C를 더해 다시 실행했다(`results_v2/`, 주 결과).

| | A (채택) | B | C (GN만 뺌) |
|---|---|---|---|
| 초기값 | QR별 NCC (탐색 ±110/±22 px) | 같음 | 같음 |
| LK 입력 | QR 영역 σ 0.8 흐림 | 흐림 없음 | σ 0.8 흐림 |
| LK 창 / 피라미드 | 15×15 / 없음 | 21×21 / 2단 | 15×15 / 없음 |
| 반복 기준 | 60회, 1e-4 | 40회, 1e-3 | 60회, 1e-4 |
| 정·역방향 검사 | 함 | 함 (0.5 px 이하만 사용) | 안 함 (A도 P12에는 반영 안 함) |
| GN 정밀화 | 17×17 스플라인 GN 12회 | 없음 | 없음 |
| P12에 쓰는 점 | 536점 전부 | 역방향 검사 통과 점만 | 536점 전부 |

공통: QR 내부 536점, 같은 P12 정규화, center 측정 차감, 학습식 `train_yort`(6절편·2차 응답·축소 0.05, [P12/J204 비교](../../20261028_141257_control_P12_vs_J204/결과/README.md)의 `models.cpp`), `validation/python_predictions.csv`의 원래 4-fold와 세션. A가 기존 fold 모델과 전수 모델을 3.3e-13 이내로 재현해야 실행이 끝난다(`audit.json`).

## 결과 (`results_v2/summary.csv`, `results_v2/timing.csv`)

| | A: 채택 (LK + GN) | B: LK21 피라미드, 유효점 | C: 채택 LK 단계, GN 없음 |
|---|---|---|---|
| 고정 4-fold 두 축 통과 | 280/280 | 280/280 | 280/280 |
| 축 판정 정답 | **280/280** | 279/280 | 279/280 |
| 틀린 축 | - | T+0.01 → Y | T+0.01 → Y |
| Y 평균 오차 (Y 80장) | **0.00183** | 0.00178 | 0.00229 |
| Y 최대 오차 (280장) | **0.0366** | 0.0429 | 0.0423 |
| T 최대 오차 | **0.0030** | 0.0100 | 0.0100 |
| 10/29 혼합 참고 | **130/234** | 127/234 | 122/234 |
| 측정 시간 중앙값 | 48 ms | 14 ms | 5 ms |

## 해석

- GN만 뺀 C가 A보다 나빠졌다: 축 한 장 오판, Y 평균 오차 25 % 증가, 최대 오차 증가. 차이는 GN 정밀화에서 온다.
- 검출기 비교 표에서 B가 앞섰던 것(276 vs 273)은 영점 후보를 뺀 단순 판정기의 결과이고, 실제 학습식에서는 A가 가장 좋다.
- B·C는 3~9배 빠르지만, 2단계 흐름의 시간은 joint(약 1초)가 대부분이라 전체 시간 차이는 작다.
- 결론: 채택 측정기(A)를 유지한다.

## 빌드와 실행

저장소 루트 PowerShell에서 실행한다. 지금 코드는 `results_v2/`에 쓰며, 이미 있으면 덮어쓰지 않고 멈춘다. `results/`는 A·B만 비교한 첫 실행 기록이다.

```powershell
cmake -S "experiments/hye/20261029_132047_control_LK측정_재학습" -B build-control-lk-20261029 -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build-control-lk-20261029 --config Release
.\build-control-lk-20261029\Release\compare_lk.exe
```

OpenCV `FileStorage`가 한글 경로에 쓰지 못해, 모델은 임시 폴더(`%TEMP%/control_lk_models_20261029_v2`)에 저장한 뒤 `results_v2/models/`로 복사한다.
