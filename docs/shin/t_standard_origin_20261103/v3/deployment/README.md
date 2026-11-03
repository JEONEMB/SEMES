# v3 표준 좌표 기반 T 추정 모델 평가

575장 유지, 기준 4장 제외 일반 예측 571장. 가정 v3 좌표로 재학습했고 원점까지 외부/내부 fold마다 재적합한 개발 검증도 수행했다. 원래/정정 명목 라벨과 표준 좌표에 대한 점수는 `three_target_metrics.csv`에서 분리한다.

동일 고정 v3 목표에서 통제 명목 모델 RMSE 2.713 / 최대 5.309 tick → v3 재학습 RMSE 1.004 / 최대 4.792 tick, 451/571 통과. 원점까지 재적합하면 표준 좌표가 식별되는 568장 RMSE 1.005 / 최대 4.802 tick, 450/568 통과. 미식별 3장도 추론은 수행했으며 명목 라벨 기준 평가는 전체 571장이다. 나사·지그 A 패치 확장은 개선되지 않았다.

## 평가 조건

- 기존 구성원 배포 하이퍼파라미터를 고정하고 학습 좌표·가중치를 변경했다. 이번 결합 가중치는 학습 내부에서 선택했다. 이전의 모든 하이퍼파라미터 검색을 반복한 실험과 구분한다.
- 정정 명목 자세를 한 fold에 묶어 .32 오기 이미지도 반복과 함께 보류한다. 예전 저장 OOF와 한 장의 fold가 달라질 수 있다.
- 전체자료 v3 좌표를 고정한 결과는 조건부 적합 평가다. 원점 재적합도 네 앵커·기존 감사/모델 선택을 조건으로 한 개발 검증이다. 물리적 정답이나 신규 세션의 1 tick 성능을 인증하지 않는다.
- `crossfit`는 25개 적합(외부5 + 내부20). 시험 구간이 학습에서 사라져 바이어스를 식별할 수 없는 경우 `target_available=0`으로 기록한다.
- 기존 v2 실험, 원본 영상, 이전 모델, sources/를 보존했다. 개선 모델이 기존 권장 모델을 대체하지 않는다.

## 실행

프로젝트 루트에서 `analysis/t_standard_origin_20261103/v3/reproduce.ps1`로 원점·빌드·학습·평가·PDF를 재현한다. 이미 만들어진 배포 폴더만으로 사진 추론:

```powershell
deployment/runtime/v3_estimate.exe IMAGE.bmp deployment/qr_assets deployment/qr_model.json
python deployment/scene_estimator.py --image IMAGE.bmp --model deployment/scene_model.json
```

C++ 실행기는 Windows x64/MSVC 런타임을 사용한다. 확장 모델 Python은 numpy, scipy, pandas, opencv-python이 필요하다. 동봉 DLL은 MSVC v141/OpenCV3.4.15로 빌드됐다. 추론에는 명목 라벨·세션 ID·시각·개발 CSV가 필요 없다. 기준 사진은 동봉한 고정 reference_scene.png와 qr_assets를 쓴다.

`verification.json`: 14개 실행의 571장 예측 일관성, 25개 원점 학습/시험 격리, 저장·재로딩, 원본 사진 3장 직접 추론·측정 캐시 동일성, 입력 해시 보존 검증 결과.
