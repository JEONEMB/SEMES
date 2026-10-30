# QR 3개 영상 직접 정합

최종 측정 캐시는 `direct_p12_newinit.npz`이다. center 포함 281장, X는 float64 `(281,12)`이며 사진 이름은 `names`에 있다. `direct_p12.npz`는 기존 점 추적을 초기값으로 쓴 inset3 예비 결과이므로 최종 strict QR 비교에는 사용하지 않는다.

`direct_pair.py`의 `DirectPairExtractor`가 배포 가능한 독립 측정 인터페이스다. 센터/현재 영상 경로만 받아 특징 벡터를 계산하며 파일명의 Y/T, 명령값, 촬영 시간, 세션, 과거 이동 캐시를 읽지 않는다. QR 위치는 현재 3840×2160 카메라 배치를 위한 고정 설정이다. 카메라 배치가 달라지면 QR 영역 설정과 학습 모델을 다시 보정해야 한다.

```python
from direct_pair import DirectPairExtractor
extractor = DirectPairExtractor(center_path)
measurement = extractor.extract(current_path)
X = measurement["X"]  # shape (12,)
```

```powershell
python direct_pair.py --center center.bmp --current current.bmp --out measurement.npz
```

1. 새 `measurement/extract_pair.py`의 QR-only GN17 측정으로 점의 대응을 구한다.
2. 유효점에 전체 2차 변형을 맞추어 직접 정합의 초기값을 만든다.
3. QR 코드 검출 경계에서 10px 안쪽 63,247개 픽셀만 목적함수에 사용한다. 거친 QR 정합은 대응 영역을 찾기 위해 이동 가능 범위에서 검색하며 나사를 별도 특징으로 쓰지 않는다.
4. sigma1 float64 영상에서 cubic spline 보간으로 위치를 읽는다. QR별 gain/bias 6개를 제거한 뒤 P12 변형 12개를 Gauss–Newton으로 최적화한다. 최종 단계는 모든 내부 픽셀을 쓴다.

정규화 좌표는 `x=(pixel_x−1906)/486`, `y=(pixel_y−1070)/122`이다. X의 순서는 `[1,x,y,x²,xy,y²]`마다 `[dx,dy]`를 번갈아 배치한다. 따라서 좌표 p에서 이동은 `B(p) @ X.reshape(6,2)`이다. X 자체는 픽셀 변형이며 Y/T 값과 확률은 별도의 학습 모델이 추론한다.

`mse_final`은 밝기 정합 잔차이며 물리 이동량 오차나 분류 확률이 아니다. `valid_points`, `condition`, `iterations`도 측정 진단값이다.

전281장을 기존 점 추적 초기값과 새 독립 초기값으로 각각 정합한 결과, QR 경계/중심에서 최종 변형 차이는 최대 0.000202px, 중앙값 0.00000187px였다. 두 최적화가 같은 해로 수렴하는지 확인한 수치다. 새 초기값 결과는 기존 캐시 없이 재현 가능하다. 비교 데이터는 `X_oldinit_same_support`와 JSON에 있다.

합성 16조건은 알려진 cubic 이동(최대 약80px), gain1.025/offset2.3, 밝기잡음0 또는0.6과 8bit 양자화를 사용했다. 좌표 RMSE 최대0.000440px, 위치오차95분위 최대0.001179px였다. 이 검증은 참 변형에 약0.18px 가까운 초기값으로 시작한 **직접 정합 수치 검증**이다. 이 숫자를 실제 장비 오차나 Y/T 정확도로 해석하지 않는다. 전체 초기 정합의 합성 검증은 `measurement` 결과와 함께 본다.

`extract_newinit.py`는 원본 전체의 독립 측정과 기존 초기값 비교를 재실행한다. `validate_synthetic.py`는 합성 검증을 재실행한다. 환경 의존 패키지는 numpy, scipy, OpenCV, pandas(점 측정 모듈의 주변 코드가 필요로 하는 경우)다.
