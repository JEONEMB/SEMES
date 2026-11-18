# 0점 위치 그래프 수정 (v3)

사용자가 지정한 원본 PPT 6쪽을 기준으로 했다. 가로축은 명령 Y(mm) 또는 θ(°), 세로축은 측정 가로변위 dx에서 공통 축 기울기 s×명령값을 뺀 값(px)이다. 그룹 절편은 남긴다. 다이아몬드는 명령 0에서의 그룹 직선 절편, 별은 기준 영상이다. 이는 dx–dy 이동 경로도, 절편까지 제거한 OLS 잔차도 아니다.

원본의 공통 제거 기울기 수치는 미기재다. 각 세션·축의 그룹별 직선 기울기를 산술 평균하여 사용했다. 기존 자료에서 Y=14.375964px/mm, θ=80.539419px/°이며 원본의 그래프 추세를 복원한다. 앞선 v2 0점 그래프는 그룹마다 그 그룹의 기울기를 제거하여 추세 차이를 평평하게 만든 한계가 있었다. 이번에는 하나의 공통 기울기를 제거한다. 그룹 절편 자체는 바뀌지 않는다.

신규 혼합 자료는 다른 축을 고정한 그룹이다. 절편에는 고정된 축의 실제 이동량이 포함되므로 순수한 기구 영점 오차로 읽을 수 없다. 원본 단독축 trial의 절편과 의미가 동일하지 않다. 고정축 기여를 임의로 빼지 않고 원시 수준을 보존했다. 그룹별 확대 그림도 제공한다.

zero_position_points.csv: 측정값 dx, 공통 기울기 common_slope_dx, 표시값 slope_removed_dx_common, 표시 직선 display_line_dx, 별도 순수 잔차 residual_dx. zero_position_fits.csv: 그룹별 기울기·절편. common_axis_slopes.csv: 제거한 공통 기울기.

수치 검증: 표시값 − 표시 직선 = 원래 OLS 잔차, 오차 허용 1e-10px. 원본 fitting이나 변위는 다시 변경하지 않았다.

재현: python tool/offaxis/zero_position_revision.py --previous docs/offaxis-eda/ppt-method-v2 --output docs/offaxis-eda/zero-position-v3

## 혼합축 주 그래프

주 그래프 zero_position_new_additive.png는 `dx−sY·Y−sTheta·theta`이다. 원본은 한 축만 움직였지만 신규 자료는 두 축이 동시에 움직여서 다른 고정축의 큰 이동량까지 제거해야 0점 차이를 볼 수 있다. sY·sTheta는 위에서 정의한 신규 그룹 기울기의 평균이다. 절편은 제거하지 않는다. 본 값에는 0점 offset, coupling과 비선형 모델 오차, 관측 흔들림이 남는다. 실제 원점을 두 축에서 새로 촬영한 것과 구분한다. fixed-axis 그룹의 보정 절편은 mixed_zero_intercepts.csv에 제공한다. 다른 축의 기여를 남긴 기본 그래프 zero_position_new.png와 원시 수준 CSV도 함께 보존한다.
