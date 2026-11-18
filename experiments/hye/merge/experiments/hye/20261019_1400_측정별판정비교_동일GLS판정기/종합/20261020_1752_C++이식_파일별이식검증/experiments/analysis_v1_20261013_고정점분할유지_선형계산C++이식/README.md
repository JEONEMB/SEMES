# v1_data 초기 특징점·알고리즘 비교

이 폴더는 원본 `analysis_v3/v1_data`의 별도 C++ 이식이다. 일괄 파일 생성 시각 2026-10-13 13:05:20 KST는 복사·보관 시점의 관측값으로 취급하며 실제 초기 실험 실행 시각이나 작성자로 단정하지 않는다. 원본별 수정 시각과 도구 호출 흔적은 실험 원장의 해당 경로에서 확인한다.

## 문제·조사·해결

QR 전용 특징과 QR 밖의 볼트·브래킷·라벨 특징이 Y/T 구분에 주는 차이를 비교한 초기 실험이다. 13개 특징 묶음과 C1~C5 계산을 조합한다. 1개 특징에만 정의된 C1, 다변량에서 정의된 C4/C5, 고차원에서 제외한 C2 조건을 유지하면 실행 가능한 조합은49개다.

## 현재 이식 범위

| 원본 | C++ | 상태 |
|---|---|---|
| `v1_data/grid.py` | `src/grid.cpp`, `src/common.hpp` | 저장된 관측에서49개 계산 경로 재구현 |
| `v1_data/grid_repro.py` | 위와 같은 공통 구현 | 경로·출력 위치만 달랐던 중복 계산을 하나로 유지 |
| `v1_data/track_all.py` | `src/track_all.cpp` | 84개 고정점 측정 실행. center 재검출 모드도 구현, 미실행 |
| `v1_data/combos.py` | `src/combos.cpp` | 2701쌍·greedy·좋은점·볼트 선택 ID 동일 |
| `v1_data/stable.py` | `src/stable.cpp` | 제외23점·유지51점·선택쌍[11,46] 동일, 지도 생성 |
| `v1_data/ecc_tune.py`, `qr_only.py` | [별도 OpenCV4 이식](../analysis_v1_20261013_QR관측유지_ArUcoC++이식/README.md) | QR 측정 대조 통과. ECC 수치 차이 보류. 기존 OpenCV3 설치 유지 |
| `analysis_v3/common.py` | 일부 기능은 `src/common.hpp` | 원본이 불러오던 데이터와 C1~C5 구현. 후속 v3 전체는 아직 미이식 |

## 유지한 부분

원본 `tracks.npz`의 순서, 중복 center 하나 제거 후281행, 원래 명령값, center 대비 크기·간격 변화를 유지한다. 특징 조합·선택된 ID·홀짝2fold·학습 자료로 정규화·C3 내부 alpha 선택·C4 잔차 표준편차·C5 1% 공분산 보정을 그대로 계산한다. C1은 Y=0을 가정한 과거 대조군이다.

점 추적은 48×48 템플릿, ±120/12 탐색, NCC, 포물선 보간을 사용한다. 기본 `frozen` 모드는 원본 center에서 정해진84개 좌표·ID로 측정을 비교한다. `detect`는 원본의 center 한 번 goodFeaturesToTrack 설정을 실행하는 별도 모드다. 현재 사진마다 새 점을 만드는 방식이 아니다.

## 바꾼 부분과 한계

Python의 실행형 `exec`와 pickle 로딩을 제거했다. 원본 NPZ의 숫자 배열은 C++로 읽고 파일명·경로 메타데이터는 명령 실행 없는 기호 파서로 해석한다. 원본 경로 문자열을 실제 데이터 경로로 실행하지 않는다. NumPy 선형대수는 OpenCV SVD/LU로 바꿨으며 새 결과는 CSV/JSON으로 저장한다.

원본은 특징 선택에 전체 자료를 사용했고 일부 방법 선택에도 개발 자료를 반복 사용했다. 이 조건을 역사적 계산 재현으로 유지하며 현재의 독립 검증이나 고정4fold OOF로 표시하지 않는다. 라벨·QR 밖 구조가 포함된 조합을 현재 QR 전용630점 추론의 성능으로 제시하지 않는다.

## 검증·새 실행 시각

2026-10-22 17:54:10 KST의 `../../results/historical_v1_grid_002`에서49개 조합을 재계산했다. 원본 `repro_17.csv`의 집계값441개 중440개가 소수 넷째 자리까지 같다. O5/C2의 Y 최대오차는 원본0.0634, C++0.0635로 마지막 표시 단위0.0001 차이다. 비반올림 C++ 수치와 사진별 예측도 보존했다. 이 한 칸을 정확히 같다고 표시하지 않는다.

`historical_v1_grid_parity_003`은 0.0001 허용값 대조다. 비교 도구에는 십진 소수의 double 표현 때문에 경계값이 아주 조금 넘는 경우에만 적용하는16ε 여유를 추가했다. 기존 엄격 검사 결과 `_001`, `_002`는 그대로 두었다.

2026-10-22 17:58:30 KST의 `historical_v1_track_frozen_001`은 중복center를 포함한282행·84점 전체 측정, 실패0이다. 원본 M배열과 최대차는 dx0.0038682px, dy0.0064014px, NCC점수0.0004789다. OpenCV 버전·float32 연산 등 차이의 원인 분리는 남아 있다. 새 관측을 기존 관측 기반 회귀 재현에 섞지 않았다.

2026-10-22 18:07:05 KST의 `historical_v1_combos_001`은74개 좋은점·2701쌍·greedy를 다시 계산했다. good/best_pair/greedy/bolt ID가 원본과 같고 특징별 통계의 최대차는8.53e-14다.

18:07:15 KST의 `historical_v1_stable_001`은 같은 제외 상자와 modulo4 분할로23점 제외·51점 유지·최종쌍[11,46]을 재현했다. greedy10단계와 전체 지도1920/960픽셀 축소본도 기록했다. 지도의 전체 이미지·제외 영역·점·ID를 확인했으며 Pillow와 GDI/OpenCV의 글꼴·외곽선·보간 픽셀은 다르다.

```powershell
# final 기준. 새 출력 폴더 사용
.\build\experiments\analysis_v1_20261013_고정점분할유지_선형계산C++이식\Release\historical_v1_grid.exe .. results/v1_grid_새이름
.\build\experiments\analysis_v1_20261013_고정점분할유지_선형계산C++이식\Release\historical_v1_track.exe .. ../hw-analysis/data/hw/bmp/20261014 results/v1_track_새이름 frozen
.\build\experiments\analysis_v1_20261013_고정점분할유지_선형계산C++이식\Release\historical_v1_combos.exe .. results/v1_combos_새이름
.\build\experiments\analysis_v1_20261013_고정점분할유지_선형계산C++이식\Release\historical_v1_stable.exe .. ../hw-analysis/data/hw/bmp/20261014 results/v1_combos_새이름/combos.json results/v1_stable_새이름
```

원본·DATA를 제자리에 유지한다. `legacy`에는 추가하거나 이동하지 않는다. 커밋·푸시는 사용자 지시에 따라 미룬다.
