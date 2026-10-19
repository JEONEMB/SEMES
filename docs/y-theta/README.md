# Y·θ 공동 추정 개발 기준

기존 영상만 사용한다. 벽/QR 고정, 카메라가 벽과 평행한 Y 이동과 수직축 yaw 회전한다고 가정한다. Y는 몸체 이동 명령이 아니라 벽 좌표계의 최종 위치다. 회전축과 카메라 중심의 차이는 미확정이다.

## 최신 제원

- 플레이트 66×11 mm. QR 간격 5 mm는 측정 기준 미확정이며 QR 자체 치수도 미확정.
- 픽셀 2.8 μm, 명목 렌즈 C-Mount 12 mm, optical format 2/3 inch, WD 307.80 mm.
- 수신 영상 3840×2160. 4096→3840은 resize/crop 미확정.
- 명목 K 초기 후보: fx=4017.857 또는 4285.714, fy=4285.714, 주점은 영상 중심 초기값. WD는 광학 중심 깊이와 같다고 확정하지 않는다.
- 허용 오차 Y 0.09 mm, theta 0.09 deg. 최소 구동 Y 0.05 mm, theta 0.01 deg는 별도 항목.

## 구현 순서와 push 단위

1. 데이터/세션/분할 및 검증 기준 문서화 (Task S15P31A507-20, 3 SP).
2. 원본 QR 픽셀 직접 정합, 기하 모델, 감쇠 Gauss–Newton 추정 및 의미 있는 합성 검증 구현.
3. 전수 실행, 보류 평가, 식별 가능성/모델 민감도 분석 및 보고서.

Epic S15P31A507-2, Story S15P31A507-9. Jira에서 Task와 Story가 같은 계층이므로 Task의 Epic parent와 Story Relates 링크를 사용한다. 활성 sprint 59016. 기존 feature/shin/y-thetaEstimation에서 작업하고 기본/통합 브랜치로 직접 push하지 않는다. docs/gitConvention.md는 develop을 명시하나 실제 원격 기본 브랜치는 dev다. 기존 브랜치를 재사용하며 자동 병합하지 않는다.

## 데이터와 평가 규칙

파일명은 명목 명령값이며 실측값이 아니다. Y의 ±0.10 계열과 ±0.05 시작 계열, theta ±구간을 6개 세션 후보로 다루되 수정 시각은 촬영 로그의 대용 단서다. 학습에서만 세션 절편을 적합하고 보류 영상에서 재적합하지 않는다. 위치순 매 3번째 보류와 연속 구간 보류를 분리해 기록한다. 공간별 QR 보류는 모델 일관성 검증이며 정지 반복성 검증이 아니다.

직접 정합은 기준 영상의 원본 좌표계에서 수행한다. 프레임별 정면 보정 영상을 최종 측정값으로 쓰지 않는다. QR 초기 대응은 기존 accepted 특징점과 payload/id를 사용한다. 잘못된 대응/정합 실패를 조용히 성공으로 취급하지 않는다.

물리 모델: 기준 3D 평면과 카메라 위치 C(y,t), 방향 R(t)를 투영. 명목 K 고정 후보를 먼저 비교하고 주점/왜곡 등은 잔차와 식별 검증이 뒷받침할 때만 확장. 직접 모델: 기구 관계를 보존한 homography family. 자유로운 두 축 회귀의 y*t 항은 단독 축 데이터로 식별할 수 없다.

두 축은 항상 공동 추정한다. 원본/정규화 Jacobian 특잇값, 조건수, 목표 단위 0.09mm/0.09deg에 대한 감도와 추정 불안정성을 기록한다. 수렴과 정확도를 구분하고, 두 모델의 합성 예측이 다르면 단독 축 적합만으로 하나를 확정하지 않는다. 기존 alignment()가 검증되지 않은 수치를 무조건 반환하도록 바꾸지 않는다.

합성 시험은 알고리즘/합성 모델의 일관성 검증이며 실제 합성 운동 정확도 인증이 아니다. 같은 보간으로 생성하고 추정하는 편향을 피하고, 별도 forward model/유한차분 Jacobian 검사를 포함한다. 원본 영상/자격 증명/바이너리는 커밋하지 않는다.

## 근거 논문

- Baker & Matthews (2004), Lucas–Kanade 20 Years On: https://doi.org/10.1023/B:VISI.0000011205.11775.fd
- Hild & Roux (2006), Integrated DIC: https://doi.org/10.1016/j.crme.2005.11.002
- Evangelidis & Psarakis (2008), ECC: https://doi.org/10.1109/TPAMI.2008.113
- Schreier et al. (2000), interpolation bias: https://doi.org/10.1117/1.1314593
- Bornert et al. (2017), synthetic evaluation bias: https://doi.org/10.1016/j.optlaseng.2016.11.014

## 구현 및 실행

오프라인 연구 도구로 구현했다. 운영 alignment()의 NaN 반환은 검증된 보정/영점 연결 전까지 유지한다.

```powershell
cmake --build build --config Release
python -m unittest discover -s tool/ytheta -p test_ytheta_model.py -v
python tool/ytheta/run_ytheta.py --repo . --output docs/validation-runs/ytheta --collect --workers 4
```

Python 의존성: numpy, pandas. 영상 수집은 C++/OpenCV 3.4.15이며 프로세스당 OpenCV 스레드 1개, 기본 4프로세스. affine/homography 두 방법으로 281장씩 처리한다. 실패 상태는 CSV에 보존하고 실패 프레임은 성공 평가의 분모에서 숨기지 않는다.

- include/direct_registration.hpp, src/direct_registration.cpp: accepted 대응점 RANSAC 초기화 → 원본 grayscale ECC. 현재 QR 영역 mask를 사용하며 Gaussian은 OpenCV ECC 기본 평활화다. ECC 상관 .95 미만, 좌표 불일치 5px 이상, QR 세 개 대응 부족은 실패한다. 이 임계값은 메트롤로지 합격 기준이 아니다.
- tool/ytheta_collect.cpp: 기준 QR 내부 고정 가상 격자 45점에 측정 warp 적용. 격자는 하나의 warp에서 파생된 상관 측정이며 45개 독립 관측으로 불확실성을 줄여 계산하지 않는다.
- tool/ytheta/ytheta_model.py: 카메라 중심 yaw, 고정 벽, D 생략 모델과 직접 다항 모델. 열 정규화·감쇠·backtracking Gauss–Newton, 원본 Jacobian을 SVD로 풀고 경계/수렴/조건수를 출력한다. y=[-2,2]mm, theta=[-1,1]deg는 데이터 범위에 따른 연구용 경계다.
- tool/ytheta/run_ytheta.py: 두 nominal K 후보에서 깊이/벽 기울기 및 세션의 2D 이미지 절편을 학습. 세션별 실제 물리 영점을 복원했다고 주장하지 않는다. 직접 모델은 점별 세션 절편과 y,t,y²,t² 계수를 사용하므로 물리 모델보다 자유도가 크고 y*t는 학습되지 않는다. 매3번째 보류와 중앙 연속1/3 보류를 각각 독립 적합한다.

저장 모델과 45×2 원본 영상 좌표 JSON을 사용한 추정:

```powershell
python tool/ytheta/ytheta_model.py model.json observation.json --session "Y+0"
```

session을 생략하면 모델 기준 영점을 사용하며 절대 기계 영점이 확인된 것은 아니다. 실사용 입력은 y/theta 라벨을 요구하지 않는다. session 옵션은 보정 시점 세션의 기준 차이를 적용하기 위한 것으로, 평가에서 실제 축을 추정기에 알려주는 제약으로 사용하지 않는다. 학습에 사용하지 않은 세션 보정은 지원하지 않는다.

단위시험은 독립 3D 회전행렬 투영, 해석/중앙차분 Jacobian, 혼합 y/theta 16개 조합의 무잡음 복원, 잘못된 관측/세션 거부를 포함한다. 무잡음 모델 시험은 실제 영상 정확도 인증이 아니다. 보고서에는 정합 누락, 보류 RMSE, 두 축 동시 허용 오차 통과 수, 수렴·경계 도달·조건수, K 후보 민감도를 함께 기록한다.
