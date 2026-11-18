# 나사 관측, 무작위 이미지 5-fold 동시 추정

사용 한도로 중단됐던 나사 탐색을 C++17로 재개한다. 기존 Python scratchpad는 참고 이력이며 실행에 사용하지 않는다. 원본과 기존 v1/v2 모듈, 이전 평가 결과는 보존한다.

## 고정한 평가 조건

위·아래 바까지 결합한 후속 비교는 [추가 결과 보고](REPORT_WITH_BARS.md)와 `results_with_bars/`에 있다. 기존 `results/`는 최초 6방법 비교 이력으로 보존한다.

- `data/fivefold_seed20261104/_metadata/images.csv`의 center 제외 571장. 실제 원본의 SHA-256과 메타데이터를 비교한다. center 4장은 test와 train에 넣지 않고, 고정 기준 사진 1장을 관측 기준으로 사용한다.
- 경로를 UTF-8 바이트 순으로 정렬하고 seed 20261104, mt19937, rejection sampling Fisher–Yates로 섞은 뒤 순서 modulo 5로 test fold를 배정한다. 115장 test 한 fold, 114장 test 네 fold다. 각 이미지는 test 1회, train 4회다.
- 사용자 지정 무작위 **이미지 단위** 분할이다. 같은 자세의 반복 사진이 train과 test에 함께 있을 수 있다. 날짜 외삽 평가나 독립 성능 검증이 아니다.
- 모든 방법은 Y/T 동시 출력. Y ±0.045, T ±0.01, 두 축 동시 통과율을 보고한다. 측정 실패와 미예측도 전체 분모에 남긴다.
- v1 joint: 같은 관측의 204차원 확장과 forward GLS 재학습. v2 joint: QR 평균 가로 이동, 위/아래 선 기울기 차이를 쓰는 2특징 선형 재학습. 이전 모델을 고정한 재대입 성적이 아니다. Y-or-T control은 동시 추정 대조군으로 바꾸지 않았다.
- 후보 4개: 나사 8특징만 선형/2차식, QR 대비 나사 상대 특징 14개 선형/2차식. 모델과 ridge는 평가 전에 고정했다.
- 학습, 표준화와 전방 모델 적합은 train fold 안에서만 수행한다. 지금 자료는 이미 개발에 노출돼 있으므로 직접 누출 방지와 새 촬영 독립 검증을 구분한다.

## Windows 실행

저장소 최상위에서 실행한다. CMake와 OpenCV 경로는 해당 컴퓨터 설치 위치에 맞춘다.

```powershell
$cmake = 'C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
& $cmake -S experiments/hye/20261104_173000_pin_random5fold -B .batch_build/pin_random5fold -G 'Visual Studio 18 2026' -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
& $cmake --build .batch_build/pin_random5fold --config Release --target pin_eval --parallel 4
$env:PATH = 'C:/opencv/build/x64/vc15/bin;' + $env:PATH
$exe = './.batch_build/pin_random5fold/Release/pin_eval.exe'
$out = 'experiments/hye/20261104_173000_pin_random5fold/results'
& $exe selftest . $out
& $exe measure . $out
& $exe evaluate . $out
& $exe summarize . $out
```

위 순서대로 실행한다. 새 결과에는 별도 출력 경로를 지정해 기존 결과를 보존한다. macOS에서는 일반 CMake 컴파일러와 설치된 OpenCV를 명시하고, 실행 파일은 빌드 폴더 바로 아래에 생긴다. 이번 실측은 Windows에서만 수행했으며 macOS 빌드를 확인했다고 주장하지 않는다.

`measure`에는 원본 이미지가 필요하다. `evaluate`와 `summarize`는 커밋한 features.tsv와 메타데이터로 재실행할 수 있고, BMP나 임시 pickle이 필요하지 않다. measure_cache는 중단 재개용 로컬 캐시로 커밋하지 않는다.

## 20턴 평가의 의미

### 바 결합 비교 재실행

위에서 빌드한 실행 파일로 저장 특징을 사용한다. 원본 BMP를 다시 측정하지 않는다. 같은 6개 대조 조건에 QR+나사+위·아래 바 선형/2차 후보 2개를 더해 8개 방법, 40개 fold 모델을 평가한다.

```powershell
$out = 'experiments/hye/20261104_173000_pin_random5fold/results_with_bars'
New-Item -ItemType Directory -Path $out -Force | Out-Null
Copy-Item -LiteralPath experiments/hye/20261104_173000_pin_random5fold/results/features.tsv -Destination "$out/features.tsv"
& $exe evaluate_bars . $out
& $exe summarize . $out
& $exe selftest . $out
```

`evaluate_bars`와 `summarize`에는 커밋한 특징과 line_protocol.json만 필요하다. 이미지 입력을 검증하는 `selftest`에는 원본 BMP가 필요하다. 새 비교 결과를 보존하려면 다른 출력 폴더를 사용한다.

### 시뮬레이션 가정

실제 로봇에는 연결하지 않았다. 각 fold의 train 사진만으로 자세→관측 특징의 선형/2차 전방 모델을 적합한다. test 사진을 초기 관측 기준점으로 삼고, 그 잔차를 고정한 채 이동 후 특징을 생성한다. 제어기에는 생성된 특징만 주고 추정 Y/T를 동시에 보정한다. 1턴 이동 상한은 Y 0.5, T 0.25다. 추정값이 허용범위면 멈추고 최대 20턴에서 종료한다. 실제 잔차는 시뮬레이터만 보관해 마지막 채점에 쓴다.

이상적 이동, 고정 잔차, 무잡음 대리 모델이며 백래시, 이동 후 실제 사진, 매번 달라지는 조명, 실제 소요 시간을 재현하지 않는다. 추정상 멈췄어도 실제 잔차가 범위 밖이면 false_stop으로 기록한다. 실제 20턴 복귀 성공률이 아니다.

## 결과 파일

- `features.tsv`: 분할, 원본 해시, QR/나사 변위, 품질, 측정시간, 실패 사유.
- `predictions.tsv`: 사진별 OOF 예측과 실패. 6방법×571행.
- `summary.tsv`: fold별 분모, 학습 수, 축별/동시 통과 수, 오차.
- `aggregate.tsv`, `measured_results.md`: 전체와 fold 간 통과율 범위/표준편차. 표준편차는 5-fold의 모집단 표준편차다.
- `repeat_spread.tsv`: 같은 날짜와 자세의 OOF 예측에서 반복 수, 전폭, 평균 중심 최대편차. fold와 학습 모델도 달라 단일 고정 모델의 순수 촬영 반복정밀도는 아니다.
- `simulation.tsv`, `simulation_summary.tsv`: 대리 모델의 최대 20턴 평가.
- `*_foldN.yml`: train-only 모델. `development_all_*.yml`: 전체 학습의 개발용 모델.
- `selftest.txt`, `integrity.txt`, `environment.txt`: 측정 방향, 합성 이동 정확도, 모델 저장·재읽기, 수량, 환경.

최종 해석은 결과 확정 후 [결과 보고](REPORT.md)에 적는다. 문헌 정리의 나머지는 별도 작업으로 남기며, 이번 실험을 v3 실행이나 실제 로봇 검증 완료와 혼동하지 않는다.
