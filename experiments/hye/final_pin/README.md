# 나사 관측 Y/T 동시 추정 후보

운영 모듈을 교체하지 않는 별도 C++17 라이브러리다. 기준 영상에서 지정한 뒷벽 나사 4곳을 고정 추적한다. 이름의 final은 실기 검증 완료나 채택을 뜻하지 않는다.

## 입력과 출력

- 입력: 기준과 같은 3840×2160 `CV_8UC1` 영상. 원본을 수정하지 않는다.
- 출력: `cv::Vec2d(y, t)`. Y는 mm, T는 기존 회전 명령 단위다. 이동 명령이 아니라 현재 편차 추정이다.
- 모델에 파일명, 정답, 날짜, 반복 번호, fold 번호를 주지 않는다.
- 측정 실패는 예외로 알린다. 0,0으로 대체하지 않는다.

```cpp
#include "final_pin.hpp"
final_pin::Estimator model(qr_assets_directory, trained_model_file);
cv::Vec2d estimate = model.predict(grayscale_image);
```

`qr_assets_directory`는 저장소의 `experiments/hye/v2_Y_T_joint/assets/`다. center, geometry, line_protocol 파일이 필요하며 Python이나 임시 디렉터리에 의존하지 않는다. 이 실험 빌드가 정의하는 `semes_joint`와 `semes_final_pin`을 링크한다. 기존 공개 alignment API와 기존 모델은 수정하지 않는다.

## 측정과 특징

센터 좌표: (974,1352), (2864,1358), (1010,1677), (2817,1681). 각 91×91 패치에 ±260픽셀 NCC 탐색을 하고, 밝기 gain/offset을 포함한 이동 Gauss–Newton으로 정밀화한다. 연속 3차 보간을 직접 계산해 OpenCV 보간표의 1/32픽셀 양자화를 피한다. NCC 0.6 미만, 특이 Hessian, 발산, 미수렴을 실패로 보고한다. 수치 수렴이 올바른 물체 대응의 보증은 아니다.

기존 joint 관측기로 QR 3개를 정합한다. QR별 9개 확인 좌표의 평균 변위에서 나사 4개의 평균 변위를 뺀 6개 특징과, 나사별 x/y 변위 8개를 합친 14차원이다. 나사 평균을 빼는 것은 공통 평행이동 상쇄 모델이다. 회전, 원근, 깊이 차이까지 완전히 상쇄한다는 가정은 하지 않는다. 원 나사 변위도 포함해 잔여 반응을 학습한다.

QR 관측기는 현재 가로선도 함께 계산한다. 나사 후보의 회귀 입력에는 가로선 값을 넣지 않지만 이 재사용 경로에서는 가로선 측정 오류도 예외로 전달된다. QR 단독 관측기 분리는 후속 검토 사항이다.

회귀는 학습 fold에서만 평균과 표준편차를 추정하고 선형 또는 전체 2차 항을 만든다. 절편을 제외한 ridge는 1e-4로 고정한다. 평가 결과를 보고 계수를 탐색하지 않았다. 모델 파일은 평균, 척도, 차수와 Y/T 계수를 함께 보관한다.

## 평가와 모델 위치

[실험 폴더](../20261104_173000_pin_random5fold/README.md)의 동일 분할 비교를 본다. `results/*_foldN.yml`은 각 test fold를 뺀 모델이다. `development_all_qr_pin_linear.yml`, `development_all_qr_pin_quadratic.yml`은 571장 전체 학습 모델로, 독립 성능이나 OOF 성적을 주장하는 데 사용하지 않는다. `Estimator`에는 qr_pin 모델을 사용한다. pin_only 모델은 8차원 입력이라 이 영상 파이프라인용 모델이 아니다.

실제 로봇 재촬영과 센터 복귀 검증은 수행하지 않았다. 새 날짜, 나사 외형 변경, 조명 변경, 탐색 범위 밖, 카메라 재설치에 대한 보장은 없다.
