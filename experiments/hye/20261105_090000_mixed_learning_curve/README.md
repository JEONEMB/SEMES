# 촬영 1·2·3차 혼합, 소량 학습과 전체 모델 5-fold 검증

## 결과 확인 전에 고정한 조건

- 원본 571장, center 제외. 기존에 저장한 QR102, 나사8, 위·아래 바 관측을 재사용한다. 1·2·3차(20261014/1020/1029)를 날짜 층화나 날짜 holdout 없이 무작위로 섞는다.
- 비교 8방법: v1 joint GLS, v2 joint, 나사만 선형/2차식, QR+나사 선형/2차식, QR+나사+위·아래 바 선형/2차식. 모두 Y/T 동시 출력이다. 이전 축 선택 control은 이 8방법의 일부가 아니다.
- 외부 seed 20261105/20261106/20261107, 각 5-fold. 매번 이미지가 test에 1회 배정된다. 같은 자세의 반복 사진은 서로 다른 fold에 들어갈 수 있다.
- 각 train fold에서 라벨을 보지 않고 2개 seed로 섞은 중첩 부분집합을 만든다. 학습량 10/20/40/80/160/320장. 전체 456~457장 조건은 부분집합 seed로 중복 반복하지 않는다.
- 방법·예산마다 소량 조건은 30개 학습/시험 조합, 전체 조건은 15개 조합이다. 같은 분할과 학습 사진으로 8방법을 비교한다.
- 허용오차는 Y ±0.045, T ±0.01. 측정 실패와 학습 실패를 분모에서 제외하지 않는다. 계수, 표준화, 전방 시뮬레이터 모두 해당 train 부분집합에서만 학습한다.
- 사진별 예측, 부분집합 회원 목록과 사진 해시, 날짜 구성, fold별 점수, 오차와 20턴 결과를 저장한다. 반복 평가 사진을 새로운 독립 관측으로 세지 않는다.
- 실제 영상 측정시간은 날짜와 정답으로 고르지 않은 고정 30장에 대해 1개 이미지 작업/1개 OpenCV 스레드로 다시 잰다. 회귀 추론시간은 저장 특징 입력 기준으로 따로 잰다.
- 20턴은 이전과 동일한 train-only 선형/2차 전방 특징 모델과 고정 test 잔차를 쓰는 이상적 이동 시뮬레이션이다. 실제 로봇 이동·재촬영 성공이 아니다.
- 571장은 이미 개발에 노출된 자료다. 이 검증으로 아무 자료, 새 날짜, 미촬영 범위를 항상 예측할 수 있다고 주장하지 않는다. 표본 수뿐 아니라 임의로 뽑힌 데이터 구성에 대한 민감도를 본다.

실행 코드와 완료된 결과·해석은 이 폴더에 남긴다. 기존 11-04 결과와 운영 모듈은 보존한다.

## 완료 결과와 실행

[결과 보고](REPORT.md), [학습량별 8방법 전체 표](RESULTS_TABLE.md)를 본다. `results/aggregate.tsv`에는 정확도, 최악 조합, 최대오차, 시뮬레이션과 추론시간을 함께 남겼다.

저장소 최상위, Windows 팀 환경에서 실행한다.

```powershell
$cmake = 'C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
& $cmake -S experiments/hye/20261105_090000_mixed_learning_curve -B .batch_build/mixed_learning_curve -G 'Visual Studio 18 2026' -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
& $cmake --build .batch_build/mixed_learning_curve --config Release --target learning_curve --parallel 4
$env:PATH = 'C:/opencv/build/x64/vc15/bin;' + $env:PATH
$exe = './.batch_build/mixed_learning_curve/Release/learning_curve.exe'
$out = '.batch_build/new_mixed_learning_curve_results'
& $exe evaluate . $out
& $exe audit . $out
& $exe benchmark . $out
```

기존 결과를 보존하기 위해 새 출력 경로를 사용한다. `evaluate`와 `audit`는 커밋한 이전 실험의 `results_with_bars/features.tsv`, 원본 메타데이터와 line_protocol.json을 사용하며 BMP는 필요 없다. `benchmark`에는 고정 30장의 원본 BMP가 필요하다. 이전 측정 파일의 SHA-256은 이전 실험의 artifact_hashes_with_bars.json과 대조한다.

## 保存した記録

- `dataset.tsv`: 사진 571장의 경로, 날짜, 원본 해시, 라벨과 반복 그룹.
- `test_folds.tsv`: 분할 seed 3개 각각의 사진별 test fold.
- `train_membership.tsv`: 195개 학습 집합의 모든 인덱스와 추출 순서. 작은 예산 집합은 큰 예산 집합의 부분집합이다.
- `runs.tsv`: 1,560개 학습/평가 조건의 점수, 날짜 구성, 학습 Y/T 범위, 실패와 시간.
- `predictions/`: 8방법×3seed, 총 24파일. 178,152회 평가의 사진별 예측과 2가지 20턴 결과.
- `aggregate.tsv`: 8방법×7학습량. 가중 통과율, 최악/P05 조건 통과율, 최대오차, 시뮬레이션과 시간.
- `full_models/`: 전체 train 조건의 fold 모델 120개. 소량 모델은 학습 집합과 코드로 재생성한다.
- `observation_timing*.tsv`: 같은 30장에 대한 새로운 단일 작업 속도 측정.
- `integrity.txt`, `audit.txt`, `reproducibility.json`: 실행 조건, 저장 기록의 독립 재집계, 재실행 일치.

`runs.tsv`와 `aggregate.tsv`의 시간 열은 환경 부하에 따라 달라진다. 재현성 확인에서는 그 열만 제외했으며, 예측과 모델 등 148개 파일의 SHA-256은 완전히 일치했다. OpenCV 3.4.15/MSVC v141의 Windows에서 수행했고 macOS의 수치 일치와 속도는 미확인이다.
