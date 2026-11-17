# 20261013 픽셀 직접 정합 G1의 C++ 이식

원본 `analysis_v3/gn_pixel_slow.py`, `gn_pixel.py`, `run_g1.py`를 분리해 이식했다. 원본별 생성·수정 시각, SHA256, 확인된 Claude/Codex 쓰기 호출은 [전체 파일 목록](../../전체_파일_목록.md)에 연결한다. 이번 이식 작업자는 현재 채팅의 Codex이며, 아래 실행 시각은 원본 작성 시각이 아니다.

## 문제·조사·해결

판의 모서리 변위 모델을 픽셀 영상의 정합 비용과 연결해 T/Y를 추정하는 초기 G1 실험이다. 느린 원본은 정확한 homography와 OpenCV linear remap을 사용하고, 빠른 원본은 좌표 변화를 선형화하고 cubic spline으로 영상을 연속 샘플링한다. 두 구현의 차이를 유지했다.

`gn_pixel_slow.hpp`는 정확한 homography, 좌표 유한차분 1e-3, brightness 두 변수, 2/5개 자세·nuisance 변수, 감쇠 정규방정식과 40회 한도를 유지한다. `gn_pixel.hpp`는 원본 ±0.5px 좌표 선형화, float32 샘플, brightness, 세 nuisance 변수, 60회 한도와 두 초기값 중 낮은 비용 선택을 유지한다. `run_g1.cpp`는 홀짝 두 분할에서 학습한 모서리 모델과 각 학습 사진 전체의 평균 템플릿을 사용한다. `fast_demo.cpp`의 평균 템플릿은 원본 데모처럼 학습 사진 매4번째만 사용한다.

이 역사적 경로는 초기값으로 저장 QR 가로 이동 관측을 읽는다. 현재의 이미지 두 장만 받는 제품 API와 구분한다. 원본 홀짝 분할을 현재 고정4-fold 성적으로 바꾸지 않으며, 현재 규칙은 [루트 문서](../../../../../99999.trash/20261022_이전작업_스냅샷/실험%20방법/snapshot_20261022_001/originals/ALGORITHM_RULES.md)가 기준이다.

## 바꾼 부분

Python/NumPy/SciPy 실행을 native C++17, OpenCV4.11 선형대수와 C++ cubic B-spline 전처리·샘플러로 대체했다. spline의 mirror 전처리 및 nearest 계수 확장을 구현했으며 수학식 확인에 [SciPy 1.16 spline 소스](https://github.com/scipy/scipy/blob/v1.16.0/scipy/ndimage/src/ni_splines.c)를 참고했다. 원본 코드나 Python 프로세스를 실행하지 않는다.

경로·출력 디렉터리를 인자로 받으며 결과·반복 비용·실패를 별도 저장한다. 전체 실행의 일부 사진이 실패하면 NaN과 실패 CSV로 남긴다. 원본·DATA·기존 결과·legacy는 수정하지 않는다. OpenCV3 라이브러리와 이 실행 파일을 함께 링크하지 않는다.

## 검증과 시각

| KST | 결과 폴더 | 확인 범위 |
|---|---|---|
| 2026-10-23 10:28:46 | `historical_spline_20261023_001` | 내부 정수점 복원 최대차5.69e-14, 0.0001px 이동에도 샘플값 변화 확인 |
| 2026-10-23 10:28:48 | `historical_pixel_slow_20261023_001` | 원본4사진×2초기값, 유한값·비용 감소 확인. 저장된 원본 데모 숫자는 없어 수치 동등 미확인 |
| 2026-10-23 10:29:11 | `historical_pixel_fast_20261023_001` | 원본5사진 데모, 유한값·비용 감소 확인 |
| 2026-10-23 10:33:31 | `historical_run_g1_20261023_001` | 원본 순서281행 실제 영상 계산, 실패0. 저장 G1 예측 재생 아님 |
| 2026-10-23 10:39:30 | `historical_g1_parity_20261023_001` | T/Y562값·nuisance843값·비용281값 전수 대조 통과 |

원본 대비 최대차는 T2.56030e-5, Y1.44162e-4, nuisance7.88020e-6, 비용8.94713e-7이다. 이식 대조 허용차는 T1e-4/Y1e-3/nuisance1e-4/비용1e-5로 명시했다. 이 범위는 첫 전체 실행 후 정한 수치 이식 기준이며 사전 등록된 과학적 검증 기준으로 주장하지 않는다. 과거 모델의 정확도를 개선했다는 의미도 아니다. slow는 독립 원본 숫자 대조가 없어 전체 목록에서 완료 체크를 보류한다.

각 결과는 `../../results/<폴더>/checks.json`에 보존한다. 앞선 특징모델 평가의 G1 저장 예측 재생은 그대로 남기며 이번 실제 계산과 섞지 않는다.

## 실행

`final` 기준이며 출력은 새 이름을 사용한다. OpenCV4 선택 빌드는 `config/opencv411_build.json`을 따른다.

```powershell
$bin = './build/experiments/analysis_v3_20261013_정합모델유지_픽셀GNC++이식/Release'
& "$bin/historical_pixel_slow.exe" .. ../hw-analysis/data/hw/bmp/20261014 results/slow_새이름
& "$bin/historical_pixel_fast.exe" .. ../hw-analysis/data/hw/bmp/20261014 results/fast_새이름
& "$bin/historical_run_g1.exe" .. ../hw-analysis/data/hw/bmp/20261014 results/g1_새이름
& "$bin/historical_g1_compare.exe" ../analysis_v3 results/g1_새이름 results/g1대조_새이름
```
