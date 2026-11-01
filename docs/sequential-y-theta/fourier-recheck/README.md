# Fourier 교점 개선 후 순차 Y·θ 재평가

2026-10-19. 시작 커밋9609e45. 설정/추정기/분할을 튜닝하지 않고 corner와 Fourier를 각각 새로 수집했다.

## 결과

| 같은 보류94장 | Y RMSE mm | θ RMSE deg | 두 축 ±0.09 동시통과 |
|---|---:|---:|---:|
| corner | 0.956875 | 0.170794 | 5/94 |
| fourier | 0.229458 | 0.040893 | 26/94 |

기존 순차 추정기의 blocked/1차 θ 모델을 사전에 비교 기준으로 고정했다. 관측 평균은 θ 목적함수에 사용하지 않고, 상대 변형으로 θ를 추정한 뒤 θ 전체 이동을 제거하여 Y를 구한다. 새로운 K/D나 물리 모델을 추가하지 않았다. 2차와 interleaved 결과도 comparison.csv에 보존했으나 최선 결과를 고르기 위해 설정을 변경하지 않았다.

## 같은 ID 통제 비교

프레임마다 두 방법 모두 accepted인 (frame,payload,id)의 교집합만 사용했다. 학습186/보류94 및 기준 center는 동일하다. 두 방법 모두115점 모델을 적합했다.

| 명목 축 | corner Y/θ RMSE | Fourier Y/θ RMSE | corner 통과 | Fourier 통과 |
|---|---:|---:|---:|---:|
| Y | 0.9380 mm / 0.1675° | 0.2717 mm / 0.0485° | 2/28 | 7/28 |
| T | 1.0120 mm / 0.1806° | 0.2403 mm / 0.0428° | 8/66 | 21/66 |

공통 ID 조건에서도 개선됐으므로 채택점 증가만으로 설명되지는 않는다.

## 실행과 범위

- corner/Fourier 각각281장, 세 QR 특징점 검증281/281. 4프로세스, 프로세스당 OpenCV1스레드.
- Fourier 기본11×11, alpha2.4, 평균밝기제거, 홀수 각주파수1/3/5. 사용자 개선 구현을 수정하지 않았다.
- 새로 빌드한 test_refiners 통과: model/Fourier 미분 최대차이 약3.31e-10/7.08e-11, convolution identity 약3.33e-16, 제약검사 통과. 기존 순차 추정5시험 통과.
- 기존 교점 개선 보고서의0.230780mm/0.041132deg,27/94와 이번 값은 평가기의 점 선택 조건이 다르다. 기존 qr-subpixel/evaluate.py는 학습관측70% 이상 조건, 이번 기존 tool/sequential/evaluate.py는48개 이상/설계행렬 rank 조건을 사용한다. 서로 다른 평가 수치를 혼합하지 않는다.
- 파일명은 명목 구동값, 6세션 보정값은 학습에서만 적합하고 보류에서 고정. 세션을 알고 있는 유리한 평가조건이다.
- 이전 교점 설정 선택에 활용된 데이터/보류구간을 재사용했으므로 독립적인 새 시험 정확도나 통계적 유의성을 주장하지 않는다.
- 결과는 크게 개선됐지만 Y±0.09mm와 실제 혼합 운동 정확도는 달성/검증되지 않았다. 운영 alignment()를 자동 연결하지 않는다. 네이티브 교점 기본값은 이미 Fourier다.
- 새 모델은 fourier_experimental_model.json. 입력 형식과 절차는 상위 README와 동일하며 학습세션 영점에 의존하는 연구 모델이다.

## 재현

```powershell
cmake -S tool/qr-subpixel -B tool/qr-subpixel/build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
cmake --build tool/qr-subpixel/build --config Release --target compare validate test_refiners
python tool/sequential/recheck_fourier.py --repo . --output docs/validation-runs/fourier-recheck
```

Jira Task S15P31A507-26,3SP / Epic2 / Story9 / 활성 sprint59016. 원본 영상 변경 없음.
