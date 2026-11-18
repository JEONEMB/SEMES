# v1·v2·LSJ 정지 사진 평가와 실기 복귀 검증 상태

2026-11-04. 목적: T ±0.01, Y ±0.045 기준으로 고정한 그룹 5-fold를 평가하고 실제 복귀 실험과 구분한다. 활성 모듈과 기존 assets는 변경하지 않는다. LSJ 원본 폴더는 읽기만 사용하며 커밋·push하지 않는다.

**실행 상태:** 571장 영상 재측정과 v1/v2 구조의 5-fold 학습·시험, 기존 배포 모델 재생, LSJ 최신·루트 각각 571장 실행을 완료했다. 원본 공개 API는 날짜별 4장씩 동일 12장으로 별도 시간 측정했고 재생 출력과의 차이가 0임을 확인했다. 실제 장비 이동·다중 step 복귀는 미실행이다. 그룹 누출·예측 중복·수량 검사는 [evaluation_integrity.json](results/evaluation_integrity.json), API 대응 검사는 [public_api_parity.json](results/public_api_parity.json)에 기록했다.

## 보고서

- [성능 비교 요약](성능비교_요약.md)
- [final v1 보고서](보고서_final_v1.md)
- [final v2 보고서](보고서_final_v2.md)
- [LSJ 보고서](보고서_LSJ.md)

## 평가 범위

| 구분 | 실제 실행할 계산 | 해석 |
|---|---|---|
| joint v1 구조 재학습 | 공통 관측 102값+상호작용 204차원, 전방 1차 GLS, 축소 .01 | v1 구조의 C++ 학습 재구현; 원 학습기와 완전 수치 동등성 검증은 별개 |
| joint v2 재학습 | s,d 두 특징 최소제곱 | 기존 v2 구조, 전체 T 범위 사용 |
| control v1 T=0 변형 | Y1±/Y2± 네 가지 2차 곡선, 잔차 공분산 축소 .05 | 원본 여섯 가지에서 T 가지 제거; 원본 그대로의 v1은 아님 |
| control v2 T=0 변형 | QR 평균 변위→Y 직선, 잔여 T 보정 비활성 | T≠0 보정 학습을 금지한 조건; 원본 v2 보정 효과와 구분 |
| 원본 배포 모델 | 기존 모델 계수로 새 측정값 예측 | 개발 재생; 재학습 OOF나 독립 시험 아님 |
| LSJ 최신 | 상·하단 Huber 각도→내부 Y/T 동시 해석 | 내부 Y는 보정 명령 아님; 고정된 계수 사용 |
| LSJ 루트 | θ→Y 순차 다음 명령/명령 가능 여부 | 실제 이동 없이 한 장 입력의 진단만 수행 |
| 실제 센터 복귀 | 로봇 이동·정지·재촬영·독립 실측 필요 | 연결된 장비 세션이 없어 미실행 |

## 입력과 분할

[데이터셋 명세](../../../data/fivefold_seed20261104/README.md), [배정 근거](../../../data/fivefold_seed20261104/_metadata/assignments.tsv)를 따른다. `input.tsv`는 원본 경로·명령 라벨·외부 test fold·그룹 ID를 포함한다. 이미지 자체는 정제 원본과 해시가 일치하는 사진을 사용한다. 동일 사진을 다섯 번 측정하지 않고 결정론적 영상 측정값을 캐시한 후 각 fold train에서만 회귀·공분산을 학습한다. 촬영 라벨은 측정기 입력에 제공하지 않는다.

center 제외 571장이다. control 기본 조건의 T=0은 80장으로 fold당 64장 학습/16장 시험이다. T≠0 491장은 고정된 control fold 모델의 민감도 평가에만 사용한다. 1029 파일명과 원본 상위 폴더의 라벨 불일치가 있는 사진은 임의 정정하지 않았으며 보고서에 제한을 기재한다.

## 소스와 산출물

- `src/eval.cpp`: 영상 재측정, 모델별 외부 5-fold 재학습, 예측·시간·실패 저장.
- `src/frozen.cpp`: 기존 배포 모델 재생. 이미 학습에 노출된 사진이 포함됨.
- `src/lsj_new.cpp`: 최신 LSJ 내부 Y/T 추출. 설정·계수 변경 없음.
- `src/lsj.cpp`: LSJ 루트 순차 코드 진단. 이동 명령을 실제 수행하지 않음.
- `make_reports.py`: 이미 생성된 C++ 출력에서 표·그림·Markdown 생성. Python에서 영상 추론·학습하지 않음.
- `results/fold_*.yml`: fold별 새 모델과 학습 수량·일부 학습 시간.
- `results/measurements.tsv`: 읽기/디코딩, 영상 측정 시간, 실패.
- `results/predictions.tsv`: fold별 새 모델의 사진별 외부 시험 예측.
- `results/frozen_predictions.tsv`: 원본 모델 개발 재생.
- `results/lsj_new.tsv`, `lsj_root.tsv`: LSJ 원본 출력.
- `return_trials_template.csv`: 실제 장비 시험에서 채울 빈 양식. 결과 아님.

## 재현 명령

저장소 루트에서 Visual Studio CMake 실행 경로를 PATH에 둔 뒤 아래처럼 구성한다. 입력 TSV와 출력 경로 인자는 Windows 다중 바이트 argv 혼동을 피하도록 ASCII 임시 경로를 사용했다. TSV 내용의 이미지 경로는 UTF-8로 읽는다. 출력은 새 위치에 두고 이전 실행을 덮어쓰지 않는다.

```powershell
cmake -S experiments/hye/20261104_123000_v1v2_LSJ_5fold평가 -B C:/Temp/hye_eval_20261104 -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
cmake --build C:/Temp/hye_eval_20261104 --config Release --target eval eval_frozen eval_lsj eval_lsj_new -- /m:1 /nr:false
# opencv_world3415.dll을 실행 파일 옆에 배치한다.
# input.tsv를 C:/Temp/hye_eval_input.tsv에 복사한다.
C:/Temp/hye_eval_20261104/Release/eval.exe <저장소루트> C:/Temp/hye_eval_input.tsv <새결과폴더>
C:/Temp/hye_eval_20261104/Release/eval_frozen.exe <저장소루트> C:/Temp/hye_eval_input.tsv <새결과폴더> <새결과폴더>/frozen_predictions.tsv
C:/Temp/hye_eval_20261104/Release/eval_lsj_new.exe <LSJ루트>/experiments/LSJ/algorithm/theta_huber_20261101/config/calibration.json C:/Temp/hye_eval_input.tsv <새결과폴더>/lsj_new.tsv
C:/Temp/hye_eval_20261104/Release/eval_lsj.exe <LSJ루트> <고정center.bmp> C:/Temp/hye_eval_input.tsv <새결과폴더>/lsj_root.tsv
# 결과 파일을 이 실험의 results에 보관한 후:
python experiments/hye/20261104_123000_v1v2_LSJ_5fold평가/make_reports.py
```

## 실제 복귀에서 추가로 필요한 것

가정에 따른 복귀 루프도 별도로 실행했다: [v1 센터 복귀 시뮬레이션 보고서](../20261104_140000_v1_센터복귀_시뮬레이션/보고서.md). 관측 생성기의 control Y 오차가 목표 수준보다 커 실제 복귀율 예측으로는 채택하지 않았다. fold별 가상 step·시간·5회 반복 표와 실제 실기 미실행 상태를 구분한다.

장비 제어·카메라 취득 연결, 이동 부호/단위·속도·안전 제한 확인, 촬영 대기시간, 독립 위치 측정, 시작 자세와 반복 시행 계획이 필요하다. 이 코드에는 장비 구동 명령을 보내는 기능이 없다. 정지 사진에 같은 모델을 반복 적용하거나 정답에서 보정 명령을 수식으로 빼는 것으로 실제 step·이동 시간을 만들어내지 않는다.
