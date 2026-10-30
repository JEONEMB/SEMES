프로젝트 기하 생성 모델

목적: 기존 특징->Y/T 회귀 대신 Y/T->예상 영상 기하를 먼저 학습하고,
관측 영상 기하에 가장 잘 맞는 연속 Y/T 두 좌표를 최적화로 찾는다.
카메라 K/렌즈 보정/QR 실물 크기/T 물리 단위가 없으므로 경험적 영상
변환 모델이다. 실제 3D 자세 복원이나 기계 회전각 복원으로 부르지 않는다.

generative_model.py API
  model = GenerativeProjective(config, points, regions)
  model.fit(D_train, y_train, metadata_train)
  prediction = model.predict(D_test)

D는 (N,630,2) center 대비 변위이고 y는 (N,2) 명령값이다.
metadata에는 index,dataset만 필요하다. 명령/날짜는 fit에서만 사용한다.
predict는 metadata, 파일명, 정답을 받지 않는다.

직접 영상 homography API
  model.fit_homographies(H_train, y_train, metadata_train)
  prediction = model.predict_homographies(H_test)

H는 (N,3,3,3), QR별 고정센터 전역 픽셀->현재 전역 픽셀의 변환이다.
direct/features.npz의 H_homography_g3 또는 H_homography_g5와 호환된다.
모델 내부에서 각 QR 중심 기준 좌표로 정규화한다. 3x3 anchor grid는
homography를 표현하기 위한 가상 투영점이며 새로 측정한 실제 코너가 아니다.
좌표 관측의 homography에는 해당 QR의 모든 기존630 대응점이 기여한다.
새 직접영상 관측은 고정630 경계 측정을 사용하지 않는다.

네 구조
0. rational: H의 8개 자유계수를 1,Y,T로 생성, 동차 좌표 나눗셈 포함.
1. rational+YT: 위 식에 Y*T 상호작용 추가.
2. lie_YT: H0 exp(Y*A) exp(T*B).
3. lie_TY: H0 exp(T*B) exp(Y*A).

최초 ECC g3/전체 날짜 균형 조건 평균 네 구조를 먼저 고정·평가했다.
g5와 train new조건 평균 적합은 그 결과를 본 뒤 추가한 보조 실험이다.
모든20개 관측/적합/구조 조합을 summary_all.csv에 보존한다.
이20개 중 성적을 보고 고른 방법의 OOF를 독립 검증으로 부르지 않는다.

평가: 기존 동일4fold, 같은 혼합 명령 반복3장 묶음, 새center3장 fold0.
341장 모두 포함. 추론 실패는 NaN으로 보존하고 전체 분모에 포함한다.
현재20조합에서는 각341장 모두 수치 추론에 성공했지만 오차 목표는 실패.

사후 최선 개발 조합: ECC g5 / train new조건 / Lie TY.
혼합/새날짜60장: Y 평균.075099 최대.274970,
T 평균.013438 최대.048631, 동시0.09 통과41/60.
기존281장: 125/281, 전체341장:166/341.
다른 더 강한 dense 관측 decoder보다 성적이 낮으므로 최종 대체로 채택하지 않는다.

verification.json은 합성된 알려진 순방향 맵의 역산 및 실제 저장예측80개
fold모델의 재현, 전체 지표 재계산, 학습인덱스/조건 그룹 분리를 검증한다.
합성 검사 결과는 로봇 성적이 아니다. identifiability_diagnostic.json의
Y/T 거의 평행한 영상 반응도 경험식이 학습한 원점 부근 국소 진단이다.
두 축을 한 영상에서 추정하는 것이 원리적으로 불가능하다는 증명은 아니다.

기존 실험/알고리즘 파일은 수정하지 않았다. 실제 새 독립 촬영 또는
로봇 복귀 동작을 실행하지 않았다. 문헌 링크와 적용 범위는 research.json.
