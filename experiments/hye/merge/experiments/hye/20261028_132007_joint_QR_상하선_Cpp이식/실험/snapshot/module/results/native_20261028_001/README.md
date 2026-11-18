# C++ 모듈 재생 검사 (2026-10-28)

환경: Windows x64, C++17, MSVC v141(19.16), OpenCV3.4.15.
실행 입력은 저장소의 clean_20261014/clean_20261020 전체341장이다.
원본 및 기존 결과는 변경하지 않았다. 새 촬영·실제 로봇 이동은0회다.

## joint 전수 모델의 정지 사진 재생

| 항목 | 결과 |
|---|---:|
| 입력/측정 성공 | 341/341 |
| Y 평균 / 최대 절대오차 | 0.0169834 / 0.0743885 |
| T 평균 / 최대 절대오차 | 0.0030081 / 0.0140121 |
| 두 축 동시0.09 통과 | 341/341 |

이 모델은 전체341장으로 학습한 저장 모델이다. 위 성적은 **재대입/개발 재생 검사**다.
기존337/341 외부4fold 성적을 새로 재현한 결과 또는 새 사진 성능으로 보고하지 않는다.
동일한 Python 전수 모델 예측과 비교한 최대 차이는 Y0.00453868, T0.000656108이다.
원래 Python 수치와 완전히 동일한 포트는 아니며, 관측 구현/라이브러리 차이를 포함한 실제 차이를 보존했다.

## control 조건부Y 개발 검사

실제 파일명 명령에서T=0인84장(20261014의센터1+Y80,20261020의센터3)을 따로 집계했다.
84장 모두 게이트를 통과했으며 Y 평균/최대 절대오차는0.00287883/0.0599753이다.
Y0.09 통과는84/84다. 그중81장은학습사진이고나머지3장도이미개발에노출된반복센터다.
따라서 독립 성능으로 해석하지 않는다. 혼합사진을 실제로 T정렬한 뒤 얻은 결과도 아니다.

## API 검사

`ctest --test-dir build-joint-control-20261028 -C Release --output-on-failure`:1/1통과.
초기화 누락, 기존alignment위임 일치, 입력불변, 빈영상/잘못된크기/잘못된깊이 거부,
T+0.50사진의control전제거부를 확인했다.

## 파일과 재생 명령

- predictions.csv: 사진별 정답/두 모듈 예측/호출 성공 여부. T!=0사진의control은평가하지않아빈값이다.
- python_full_model_comparison.csv: 기존 Python 전수 예측과의 차이 및 절대오차.
- verification.json: 분모/오차/검사범위 기록.

저장소 루트에서 아래 명령을 실행한다. 결과는 **새 경로**를 지정한다.

```powershell
.\build-joint-control-20261028\experiments\00000_final_Y_T_joint\Release\evaluate_alignment.exe `
  data/clean experiments/00000_final_Y_T_joint/assets `
  "experiments/00000_final_Y_T control/assets/control_model.yml" `
  experiments/00000_final_Y_T_joint/results/native_next/predictions.csv
```

이 기록은 코드/정지사진 검사다. 자동 안정 구간 탐색·이동 후 복귀·실제20턴 도달은 검증하지 않았다.
