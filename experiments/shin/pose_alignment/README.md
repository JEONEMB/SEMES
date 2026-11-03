# Lanczos 전체 보유 데이터 학습·평가

사용자 요청: 보유 원본 전체의 Y/yaw 예측·오차 그래프, 전수 학습, 지정한 「촬영 세션별 오차 EDA 시각화」의 감사 판정 반영. 기존 배포 알고리즘은 보존하고 새 전수 모델을 별도 출력한다.

- 입력: `data/SSAFY Test Image Set` 전체 575장(BMP), Y 단일축80 / yaw 단일축200 / 혼합291 / 원점4. 다른 실험·모듈의 복사본과 합성 증강은 중복 집계하지 않는다. 현재 `data/clean`은 존재하지 않는다.
- 영상 측정: `feature/shin/QRwithBar`의 `55c5ebeda5647e3ee864758be41ba456ded0eb32`에서 `git archive`로 가져온 `modules/qrwithbar_lanczos`. 기존630 QR점과120 경계 표본, Lanczos-3 정합과10특징을 그대로 사용한다. 측정 결과를 한 번 저장해 세 평가에서 공유한다.
- 기준 자산: 기존 center1 영상 및 QR 템플릿을 고정한다. 모든 추론은 영상만 입력받으며 파일명·세션·정답으로 예측값을 교정하지 않는다. 파일명은 회귀 학습 및 평가기에만 제공한다.
- 기존57장 모델 고정 출력, 새575장 모델 재대입, 동일좌표 전체 보류 교차검증을 구분한다. 각 사진은 세 방식 모두 결과를 가진다. 교차검증은 일부 사진을 샘플링해 점수 내는 방식이 아니다.

## EDA 감사 반영

지정 대화 `01a19c7a-f632-7e72-aa5e-55ec255877e5`와 최신 `analysis/frame_link_audit_20261103/HANDOFF_model_v3.md`, `result_summary_v3.json`을 확인했다. 출처의 감사 기록은 `audit/`에 읽기용 복사했다.

1. 최신 판정의 사진 hard removal은0장. 최초에 의심했던 .57 사진3장은 영상 궤적 근거로 복원한 상태다.
2. `Y+0.05 T+0.32/Y+0.05 T+0.33 (3).bmp`의 파생 라벨만 T=.32로 정정한다. 원본 이미지·파일명·TXT를 변경하지 않는다.
3. 반복 편차4장은 삭제하지 않는다. 감사에서 QR 및 독립 자연 특징 부위의 동반 움직임으로 계산한 사진별 `common_norm`을 품질 측정치로 가져온다. 이 값은 같은 좌표의 다른 반복에만 의존한다. 교차검증에서 같은 좌표가 전부 빠지므로 학습 사진의 반복 값 계산에 시험 사진이 섞이지 않는다. 각 학습 fold에 존재하는 반복 진단값으로 기준을 다시 계산한다.
4. 기준 = median(norm)+4×max(1.4826×MAD,0.02px). 넘은 사진의 가중치=max(0.2,(기준/norm)^2), 나머지1. 전수 모델의 실제 가중치는 `results/training_weights.csv`에 저장한다. 시험 오차 계산은 항상 무가중이다.
5. 기존 EDA의 잠재 `Y_standard/T_standard`나 세션 바이어스를 새 정답으로 사용하지 않는다. 이번 목표는 감사 후 명령 라벨에 대한 기존10특징 모델의 전수 학습·예측이다. EDA의 자연 특징·아래 선 탈락 판정은 해당 원점 모델의 관측 품질 판단이며, 사진 삭제 목록과 구분한다. 현재 QR+양쪽 경계 알고리즘 자체를 그 원점 모델로 바꾸지 않는다.

## 학습과 검증

기존 선형10특징 회귀 구조와 α=0을 사전 고정하고, 가중 표준화·절편·SVD 회귀 계수를 학습한다. 특이값 cutoff는 최대 특이값의1e-10, scale 하한1e-8이다. 가중 평균·분산과 sqrt(weight)를 곱한 중심화 설계행렬을 사용한다.

- 전수 학습: 성공575장의 특징 및 감사 후 명령을 전부 사용한다. 모델·기준 영상·템플릿을 `results/full_assets/`에 별도 저장한다. 기존 배포 모델은 변경하지 않는다.
- 전체 교차검증: 감사 후378개 좌표를 하나씩 통째로 제외한다. 반복3장과 원점4장도 각각 같은 그룹이다. 매 fold에 학습용 가중치 기준·표준화·절편·계수를 다시 계산하고 제외된 사진을 예측한다. 모든575장의 보류 예측을 합친다. fold별 모델은 `held_coordinate_models.json`에 남긴다.
- 고정 영상 보정 자산(center1, QR템플릿)은 CV에서도 유지한다. 따라서 원점4장을 회귀 학습에서 제외한 것과 기준 영상 자체를 보류한 검증은 다르다.
- 새 모델의 내보내기 후 실제 Estimator로 서로 다른 부분집합3장을 다시 예측해 특징 기반 회귀와 대조한다. `full_training.json`에 차이와 모델/특징 SHA-256을 기록한다.
- 이 사진들은 과거 분석과 EDA에서 이미 본 개발 자료다. 새 촬영 독립 성능·물리 참원점·20턴 실제 복귀 검증으로 주장하지 않는다. 최초57장 Actual 상대 라벨 교차검증 성적과 직접 개선율을 계산하지 않는다.

주 산출물: [대화형 보고서](results/report.html), [전체 예측 CSV](results/full_predictions.csv), [조건별 지표](results/all_metrics.csv), [전수 학습 그래프](results/full575_overview.png), [전체 보류 그래프](results/OOF575_overview.png).

## 재현

이 실험 폴더에서 PowerShell, 팀 기준 Windows x64 / MSVC v141 / OpenCV3.4.15:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
$env:MSBUILDDISABLENODEREUSE='1'
cmake --build build --config Release --target full_evaluation full_train --parallel 1 -- /nr:false
./build/Release/full_evaluation.exe '../../../data/SSAFY Test Image Set' './modules/qrwithbar_lanczos' './results'
./build/Release/full_train.exe './modules/qrwithbar_lanczos' './results' '../../../data/SSAFY Test Image Set'
```

`full_evaluation`이 원본 해시·명령·기존 모델 예측·10특징을 저장한다. `full_train`은 저장된 전수 특징으로 전수/보류 모델을 적합하고, 원본3장으로 내보낸 모델을 확인한다. 이미지측정 전체 재실행과 캐시 회귀 학습을 구분한다. 원본과 다른 실험 결과를 덮어쓰지 않는다. C++17만 사용하며 Python 런타임은 필요 없다.

`predictions.csv/jsonl`, `summary.json`, SVG3개는 최초 고정57장 모델 및 정정 전 파일명 라벨 기록이다. 최종 비교와 보고서는 `full_predictions.csv`, `all_metrics.csv`, `full_data.js`의 감사 후 라벨을 사용한다. 비교를 위해 원 기록도 보존했다.

## 실제 실행 결과

575장 모두 측정 성공, 고유 SHA-256 575개. 교차검증378개 그룹·378개 모델, 학습/보류 이미지 수 대조 불일치0. EDA 감사 가중 감소4장, 삭제0장, 라벨 정정1장. 내보낸 전수 모델의 실제 영상3장 재실행과 특징 회귀 최대차이0.

| 평가 | Y 평균/최대 절대오차(mm) | yaw 평균/최대 절대오차(°) | 두 축≤0.09 |
|---|---:|---:|---:|
| fixed57 | 0.073846 / 0.360042 | 0.025742 / 0.105577 | 381/575 |
| full575 | 0.034767 / 0.191548 | 0.006906 / 0.034311 | 548/575 |
| OOF575 | 0.035675 / 0.208000 | 0.007095 / 0.035229 | 543/575 |

전수 학습의 적합 오차와 전체 교차검증 오차가 비슷하지만, 교차검증에서32장이 Y 허용오차0.09를 넘는다. yaw는575장 모두0.09 이내이다. 모든 사진의 두 축 원점 도달을 입증한 결과는 아니다. 전체 데이터 사용과 EDA 가중 반영이 함께 바뀌었으므로 개선을 데이터 양 증가 하나의 효과로 단정하지 않는다.

