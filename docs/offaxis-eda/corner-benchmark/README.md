# 모서리 추출 방법 비교: 투영 정답과 실제 사진

2026-10-26. 실제 341장, 가상 카메라 24자세 × 3조건. 원본 이미지는 변경하지 않았다.

## 주요 결과

질감 조건의 합성 L형에서 quadrant_model RMS=0.120438px였다. 경계 후퇴를 학습점에서만 보정한 실제 L형의 투영 잔차 중앙값은 quadrant_model에서 0.153408px였다. 실제 잔차는 추출점의 절대 정확도가 아니다. X형의 투영/반복 잔차는 작았지만 기존 밝기 적합 진단은 전부 탈락했다. 따라서 작은 기하 잔차만으로 X형이 더 정확하다고 주장하지 않는다.

## 평가에서 구분한 것

사용자 제안대로 이상적인 평면 위에 가상 특징점을 두고 핀홀 카메라 K[R|t]로 투영했다. 이 투영 위치와 검출 위치의 유클리드 거리를 합성 영상의 정확도로 사용했다. 실제 사진은 K·왜곡·실제 자세·인쇄물 모서리의 실측 좌표가 없으므로 같은 수준의 정답을 만들 수 없다. 실제 사진은 공간 블록 5-fold 보류 투영 잔차와 반복 일관성으로 평가했다. 평가할 점은 그 점의 homography 학습에서 제외했다.

각 QR에 개별 homography를 사용했으며 세 QR 사이 실제 간격이나 단일 보드 물리 좌표를 새로 알아냈다는 뜻이 아니다. 왜곡·판의 비평면성·인쇄 편차·코팅 경계 편향도 실제 잔차에 포함된다. 이 자료만으로 카메라 보정을 새로 추정해 정답으로 쓰지 않았다.

## 후보와 추출

QR 모듈 배치를 복원해 고정 ID를 붙였다. L형 모서리 630개, X형 가상 교점 125개다. L형에는 볼록 모서리, 오목 모서리, finder 경계가 포함된다. 기존 offaxis 분석도 이미 이 L630을 썼으므로, 새로 630개를 처음 발견했다고 주장하지 않는다. 운영 브랜치의 교점 중심 방식과 기존 분석을 구분한다.

자유 Shi–Tomasi 검출도 ROI 전체에서 수행했다. 고정 모듈 후보와 3px 이내 일대일 대응된 L 후보 비율은 100.00%였다. 대응이 안 된 후보에는 배경/코팅 질감/라벨 외곽의 유효한 모서리가 함께 있을 수 있어 모두 오검출이라고 부르지 않는다. 다만 알려진 물리 ID가 없으므로 위치 정답 평가에는 포함하지 않았다.

비교 방법은 3×3 반창 cornerSubPix, 두 경계 프로파일에서 구한 직선 교점, 기존 9×9 Gaussian 사분면 밝기 적합이다. 기존 모델의 실제 사진 추출값은 저장된 v2 값을 재사용했고 원본 해시를 341장 확인했다. 나머지 실제 추출은 새로 실행했다. L형은 동일 ID의 공통 유효점에서도 비교하며 X형은 위치 집합이 달라 직접적인 개선 배수로 해석하지 않는다. 전처리 평활화는 하지 않았다. 경계 알고리즘의 미분/프로파일 보간은 추정 과정이다.

## 합성 영상의 정답 정의

카메라 초점거리 450px, 주점 (128,128), QR 한 모듈=가상 길이 1, 거리 48~54, 회전벡터 각 성분의 난수 범위를 코드에 고정했다. 실제 장비 보정값이 아니다. 입력 도형은 모듈당 64샘플, 센서 픽셀은 6×6 면적 적분으로 생성했다. 비등방 Gaussian PSF, 1.5 gray-level 센서 잡음, 인쇄 경계 후퇴 2/64 module, 상관된 질감 조건을 포함한다. 추출기의 해석적 Gaussian 식 자체로 영상을 만들지 않았다.

인쇄 경계 후퇴가 있으면 L형의 실제 재료 모서리와 이상적인 격자점은 다른 위치다. `error_px`는 후퇴한 실제 재료 모서리 투영과의 거리, `ideal_grid_error_px`는 원래 격자 투영과의 거리로 둘 다 저장했다. X형은 검정이 분리되면 실제로 존재하지 않는 가상 격자 교점이므로 X 결과를 실제 재료 모서리의 오차와 동일한 의미로 읽지 않는다.

테플론 코팅은 사용자 제공 정보다. 합성 질감은 영향 확인용이며 표면 반사모델을 실측하거나 테플론 물성을 맞춘 것은 아니다. 24가상 자세는 3조건에서 재사용하고 초기값 난수도 같게 유지했다. 각 자세의 L48/X24 위치는 성능을 보기 전 격자 순서에서 균등 선택했다. 검출 위치의 초기 추측은 정답 격자에 ±0.45px를 더해 제공했으므로 합성 정확도는 정밀화 단계 평가다. 자유 검출 전체 성능과 혼동하지 않는다.

## 결과

상세 수치는 `synthetic_summary.csv`, `real_summary.csv`, `repeatability.csv` 및 [시각 보고서](report.html)에 있다. 점별 CSV에는 실패와 모든 보류 잔차를 남겼다. 실제 사진 수십만 점을 독립 실험 수로 보지 않도록 프레임/QR 집계도 제공한다. 합성 자세와 원본 프레임은 범위가 제한돼 있으므로 전체 장비 조건의 일반 성능으로 일반화하지 않는다.

**유효 판정 주의:** 본 위치 비교의 valid는 유한 좌표·초기값 대비 3.6px 범위 등의 기하 조건이다. 기존 광학 적합의 원래 accepted와 다르다. 기존 L형 accepted는 96.24%, X형은 0%이며 X형의 픽셀 밝기 RMSE는 약 25로 원래 기준 15를 모두 초과했다. 유효한 좌표와 정상적인 광학 형상 적합을 혼동하지 않는다. X형의 작은 기하 잔차는 같은 편향이 반복되는 상황에서도 나올 수 있다. 원래 진단은 original_photometric_diagnostics.csv에 보존했다.

### 인쇄 경계 후퇴를 고려한 보류 투영

각 QR/방법/프레임의 학습점만으로 H의 8변수와 가로/세로 유효 후퇴량 2변수를 robust 적합하고, 공간 블록을 보류해 예측했다. L의 볼록/오목 모양에 따라 후퇴 부호를 정했다. 원래 격자 투영 잔차와 별도로 material_projection.csv.gz에 남겼다. 이 모델도 실제 K·렌즈 왜곡·광학/코팅 물성을 알아낸 것이 아니며, 추출기의 모양별 편향을 흡수할 수 있는 기술적 모델이다. 실제 모서리 정답을 제공하지는 않는다.

밝기 형상 적합의 유효 후퇴량 중앙값은 (0.074089, 0.077651) module이다. 무잡음 투영 복원과 해석적 Jacobian 수치 검산을 수행했다. 두 모델의 잔차가 달라진 것을 추출 정확도가 그만큼 향상됐다고 해석하지 않는다.

## 문헌과 적용 범위

- [AprilTag 2, Wang & Olson (2016)](https://april.eecs.umich.edu/media/pdfs/wang2016iros.pdf): 경계 법선 방향의 gradient 가중 위치를 구하고 직선 적합 후 교점으로 모서리를 정밀화하는 접근을 참고했다. 이번 짧은 QR 내부 L변 프로파일 알고리즘은 논문 전체 구현이나 AprilTag 검출기가 아니다.
- [Canny/Devernay subpixel edges, Grompone von Gioi & Randall (2017)](https://www.ipol.im/pub/art/2017/216/): 부화소 경계 위치 추정의 근거로 참고했다. 제공 소스의 복제나 그 정확한 알고리즘 재현은 아니다.
- [Zhang (2000), A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/): 평면 패턴과 투영, 렌즈 왜곡/비선형 보정을 구분하는 근거다. 본 실험에서 실제 K, 왜곡을 보정했다는 뜻이 아니다.
- [OpenCV feature detection documentation](https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html): 자유 검출과 cornerSubPix 비교군의 공식 정의.

## 재현과 파일

numpy,pandas,scipy,opencv-python-headless,matplotlib,qrcode,Pillow 필요. `tool/offaxis/corner_benchmark.py --views 24`, 이어서 `tool/offaxis/corner_benchmark_report.py` 실행. 추출 원시값을 재사용할 때는 `--reuse-real --reuse-synthetic`. Python 경로와 라이브러리 버전/소스 해시는 manifest에 기록했다.

- report.html: 오프라인 시각 보고서, 원본 사진 위 점/보류 투영 위치 선택·확대·오차 확인.
- real_points.csv.gz: 실제 사진 모든 후보/방법의 추출값·실패·보류 투영 잔차.
- synthetic_points.csv: 합성 투영 정답·격자 정답·추출값·오차·실패.
- synthetic_cameras.json: 알려진 가상 K,R,t,H,PSF와 인쇄 후퇴량.
- free_detection.csv: 자유 모서리 검출과 고정 ID 대응 범위.
- real_frame_summary.csv, repeatability.csv: 프레임/QR 집계 및 동일 명령 좌표의 상대변형 반복 잔차. 반복은 정지한 장비의 센서 잡음만이 아니라 기구 재현오차도 포함한다.
- manifest.json: 입력/소스 해시, 가정, 수치 검산, 라이브러리 버전.
- material_projection_summary.csv, material_projection.csv.gz: 후퇴 보정 투영의 보류점 결과. corner_material_projection.py 실행 후 report 생성.

커밋·push·운영 알고리즘 교체는 하지 않았다.
