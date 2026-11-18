# 20261013 특징 조합·판 원근 변환 실험

원본 네 파일을 각각 C++로 옮겼다. 아래 시각은 파일 생성 관측값이며 실제 실행 시각이나 작성자를 단정하는 근거가 아니다. 개별 수정 시각과 쓰기 호출 흔적은 `../catalogue_003/analysis_v2/README.md`에서 확인한다. 현재 C++ 이식 작업자는 이 채팅의 Codex다.

| 원본 | 생성 관측 KST | C++ |
|---|---|---|
| `extract_homography.py` | 2026-10-13 11:41:09.023 | `src/extract_homography.cpp` |
| `judge.py` | 2026-10-13 11:42:22.152 | `src/judge.cpp` |
| `estimate.py` | 2026-10-13 11:43:38.174 | `src/estimate.cpp` |
| `make_outputs.py` | 2026-10-13 11:43:39.088 | `src/make_outputs.cpp` |

## 문제와 방법

QR 위치가 비슷한 Y/T 사진을 구별하기 위해 QR 위치·크기, QR별 affine ECC, 판 전체 homography, 참고용 볼트를 여섯 조합으로 비교한 실험이다. 각 축의 같은 축 보간 잔차로 잡음을 구하고 3σ 기준과 공분산을 반영한 통계 거리로 상대 축과의 구분 가능 여부를 분석한다. 같은 특징으로 선형 회귀도 비교한다.

판 추출 창, float32 homography ECC 300회·epsilon 1e-7·필터5, 일곱 위치의 14개 변위와 특징 순서를 유지했다. 통계 판정은 끝점을 제외한 LOO 잔차, 두 축 중 큰 표준편차, 표본 공분산과 원본의 작은 대각 보정, chi-square 99.73% 기준을 유지했다.

## 유지한 부분과 한계

원본은 모든 결합 행으로 표준화한 뒤 두 fold로 나눈다. 이 과거 조건을 수치 재현용으로만 보존했으며, 현재의 학습 분할 내부 표준화나 고정 4-fold 평가로 표시하지 않는다. 볼트가 포함된 조합은 QR 전용 조합과 구분한다.

원본 `finder` CSV에는 center 행이 없다. 따라서 원본 코드의 내부 결합 결과도 280장이고 center는 빠진다. 소스에 center 처리 코드가 있다는 이유로 새 center 행을 넣지 않았다.

통계 판정 보정에도 원본과 같이 전체 자료를 사용한다. 판정 결과를 현재 고정 4-fold OOF 분류 정확도로 표시하지 않는다. QR 전체 결합은 실제 코드·표에서 44개 특징이다. 과거 문서의 ‘49개’ 설명을 새 코드의 차원으로 사용하지 않는다.

## 바꾼 부분

Python pickle 대신 원본 CSV 네 개로 같은 특징표를 C++에서 재구성한다. NumPy/SciPy 대신 C++ 선형대수와 incomplete-gamma 기반 chi-square 역함수를 사용한다. Excel은 C++ OOXML, 히트맵은 C++ OpenCV/GDI로 기록한다. 세 시트 역할과 사진별 평균 수식은 유지하고 글꼴·레이아웃·이미지·ZIP 바이트는 달라진다.

실제 사진 측정은 OpenCV 3.4.15로 수행했다. 사진별 실패를 기록하고 계속 진행하는 방식으로 바꾸었으며 원본의 중간 중단과 구분한다. 새 ECC 관측값을 저장 관측 재현 결과에 섞지 않는다.

## 검증

2026-10-22 17:11:50 KST 비교에서 여섯 조합의 모든 집계 지표가 대조를 통과했다. Y RMS 최대차는 약 2.67e-7µm, Y 최대오차 수치의 차이는 약 1.59e-5µm, T RMS 차이는 약 3.65e-11이다. 90µm 초과 사진 수는 여섯 조합 모두 같다. 이는 계산 재현 오차이며 실제 측정 정확도의 의미가 아니다.

근거는 `../../results/historical_v2_estimate_001`과 `historical_v2_estimate_parity_002`다.

2026-10-22 17:38 KST의 실제 281장 판 ECC 재측정은 실패0이지만 원본 CSV와 좌표 최대차 0.0349126px다. 1e-4 동등성 대조는 실패로 보존했다. 근거는 `historical_v2_homography_parity_001`이다. 정확한 라이브러리·실행환경 원인 분리는 남아 있다.

2026-10-22 17:41:20 KST에는 기존 Excel과 판정1960개를 대조해 불일치0을 확인했다. 거리151개 최대차는 1.93e-8이다. 근거는 `historical_v2_workbook_parity_001`이다.

2026-10-22 17:41:31 KST에 `historical_v2_outputs_001`의 요약/Y/Θ 3시트와 히트맵을 C++로 생성했다. 별도 스프레드시트 도구에서 오류0, 평균 수식 입력+2 변경 시 결과+1 반영, 세 시트 렌더링을 확인했다. 평균 수식은1120개다. 실제 Excel 앱의 재계산은 확인하지 않았다.

```powershell
# final 기준
.\build\experiments\analysis_v2_20261013_특징분할유지_통계C++이식\Release\historical_v2_estimate.exe .. results/v2_새이름
.\build\experiments\analysis_v2_20261013_특징분할유지_통계C++이식\Release\historical_v2_homography.exe ../14_데이터_QR마커위치.csv ../hw-analysis/data/hw/bmp/20261014 results/v2_hom_새이름
.\build\experiments\analysis_v2_20261013_특징분할유지_통계C++이식\Release\historical_v2_judge.exe .. results/v2_judge_새이름
.\build\experiments\analysis_v2_20261013_특징분할유지_통계C++이식\Release\historical_v2_outputs.exe .. results/v2_judge_새이름 results/v2_새이름 results/v2_output_새이름
.\build\experiments\analysis_v2_20261013_특징분할유지_통계C++이식\Release\historical_v2_workbook_check.exe .. results/v2_judge_새이름 ../23_YΘ구분표_v2.xlsx results/v2_check_새이름
```

`legacy`에 추가하거나 원본을 이동하지 않는다.

원본 CSV를 이용한 계산 재현과 새 ECC 측정으로 끝까지 다시 평가한 성적을 구분한다.
