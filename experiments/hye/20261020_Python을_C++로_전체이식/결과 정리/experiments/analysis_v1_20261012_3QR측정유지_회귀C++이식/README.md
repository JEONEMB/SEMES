# 20261012 QR 세 영역 초기 실험

원본은 `analysis_agent`의 `measure_qr.py`, `evaluate.py`, `evaluate_theta_only.py`다. 관측된 생성 시각은 각각 2026-10-12 15:48:32.540, 15:50:59.176, 16:07:29.531 KST다. 원본 작성자는 현재 확보한 직접 쓰기 기록만으로 확정하지 않는다. C++ 이식 작성자는 이 채팅의 Codex다.

## 문제와 조사

QR 세 영역의 평균 수평 이동과 가운데 QR의 상대 굽음으로 Y/T를 구분하는 초기 실험이다. Y가 고정됐다는 조건에서 T만 추정하는 실험도 따로 있다. 최신 630점 경계 맞춤과 다른 방법으로 보존한다.

## 유지한 부분

- QR 세 개의 180×192 영역, NCC ±130 초기화, affine ECC 150회·epsilon 1e-7·필터 3 설정.
- affine 변환에서 영역 중앙의 이동을 계산하는 식과 회귀 특징 두 개.
- 명령 크기의 홀수 단계 학습, 짝수 단계 평가. Y/T 140장 평가와 T 전용 100장 평가를 구분.
- T 전용 실험의 `80.06/14.25`는 QR 영상 이동의 Y 등가 환산이며 T의 물리 단위로 해석하지 않음.

## 바꾼 부분

세 실행 경로를 각각 C++ 실행 파일로 만들었다. 회귀는 C++ 선형대수로 계산하고, 산점도는 OpenCV로 그린다. 축·색·전체 점·특징값은 유지하지만 matplotlib의 글꼴과 레이아웃은 재현하지 않는다. 데이터 경로와 새 출력 폴더를 명시적으로 받으며 원본을 덮어쓰지 않는다.

## 수치 대조

| 경로 | 실행과 결과 | 상태 |
|---|---|---|
| `evaluate.cpp` | 원본 측정값으로 140장 재평가. T 예측 최대차 1.17e-13, Y 최대차 6.47e-13 | 수치 재현 통과 |
| `evaluate_theta_only.cpp` | 원본 측정값으로 T 100장 재평가. T 예측 최대차 2.23e-16 | 수치 재현 통과 |
| `measure_qr.cpp` | 실제 281장 측정, ECC 실패 0. NCC 점수 차이 최대 약 1.32e-6 | ECC 결과 차이는 별도 보존 |

새 ECC의 변환 이동 성분은 원본 CSV와 최대 약 0.0364px 차이가 있다. OpenCV 3.4.15 C++ 경로와 원본 Python 실행의 라이브러리·실행 버전 차이를 포함한 원인 분리는 완료하지 않았다. 따라서 새 ECC를 원본 측정과 수치적으로 같다고 표시하지 않는다. 원본 CSV를 이용한 회귀 재현과 새 ECC 측정을 섞지 않는다.

근거는 `../../results/historical_agent_parity_001`, `historical_theta_parity_001`, `historical_ecc_parity_001`, `historical_ecc_ncc_parity_001`에 있다. 2026-10-22 17:00 KST의 비교 기록이다.

```powershell
# final 기준. 매번 새 출력 폴더 사용
.\build\experiments\analysis_v1_20261012_3QR측정유지_회귀C++이식\Release\historical_agent_evaluate.exe ../analysis_agent/qr_measurements.csv results/agent_새이름
.\build\experiments\analysis_v1_20261012_3QR측정유지_회귀C++이식\Release\historical_theta_only.exe ../analysis_agent/qr_measurements.csv results/theta_새이름
.\build\experiments\analysis_v1_20261012_3QR측정유지_회귀C++이식\Release\historical_measure_qr.exe ../hw-analysis/data/hw/bmp/20261014 results/ecc_새이름
```

원본과 DATA는 현재 위치에 유지하며 `legacy`에 추가하거나 이동하지 않는다.
