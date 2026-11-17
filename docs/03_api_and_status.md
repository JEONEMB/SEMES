# API, 영점과 보정 허용

## 호출 순서

공개 헤더는 include/alignment.hpp, 라이브러리 대상은 alignment다. API는 사진에서 위치를 추정한다. 카메라 촬영, 네트워크, 모터 명령 전송은 포함하지 않는다.

1. initialize로 해당 장비 모델을 읽는다.
2. set_zero_from_photos로 같은 설치 조건의 영점 사진을 설정한다.
3. measure_alignment에 현재 사진을 전달한다.
4. correction_allowed가 true일 때만 호출자가 장치용 보정 명령을 계산한다.

```cpp
#include "alignment.hpp"

bool measure_for_controller(const cv::Mat& image, double& y_mm, double& t_deg) {
    const auto r = alignment::measure_alignment(image);
    // notices는 호출자가 기록하거나 화면에 표시한다.
    if (!r.correction_allowed) return false;
    y_mm = r.y_mm;
    t_deg = r.t_deg;
    return true;
}
```

이 예제는 초기화와 영점 설정 이후에 호출한다. Warning을 모두 실패로 취급하는 status != Ok 조건을 사용하면 범위 초과 경고를 다시 차단하게 된다. 보정 여부는 correction_allowed로 판단한다.

## 상태와 알림

| 조건 | 상태 또는 코드 | 보정 허용 |
|---|---|---|
| 정상 검출과 유효한 영점 | Ok | 허용 |
| Y 또는 T 학습 범위 초과만 있음 | Warning, W_OUT_OF_RANGE, Correct | 허용 |
| 일부 나사 미검출이지만 좌우 정보 확보 | Warning, W_SCREW_MISSING, Correct | 다른 차단이 없으면 허용 |
| 좌우 나사 정보 미확보 | Warning, W_COARSE, Review | 금지 |
| 관측값 일치 점검 초과 | W_VALUES_DISAGREE, Recapture 또는 Stop | 금지 |
| QR 미검출 또는 개수와 간격 불일치 | Failed, E_QR_NOT_FOUND | 금지 |
| 나사 위치 계산 후 품질 기준 초과 | Warning, W_SCREW_UNSTABLE, 측정값 유지 | 다른 차단 조건이 없으면 허용 |
| 영점 미설정 또는 무효화 | Failed, E_ZERO_REQUIRED | 금지 |
| 모델, 입력 형식 또는 계산 오류 | Failed | 금지 |
| measure_diagnostic 호출 | 상태와 무관 | 금지 |

여러 알림은 notices에 모두 남는다. 가장 제한적인 조치가 대표 code와 action을 결정한다. 범위 경고가 다른 검출 실패나 정보 부족을 덮어써서 허용하지 않는다. Failed 결과의 Y와 T는 NaN이다.

## 영점과 기준 사진

set_zero_from_photos는 여러 사진의 같은 위치 밝기를 평균한 뒤 측정해 영점으로 저장한다. 실패하면 이전 영점을 계속 쓰지 않고 무효화한다. 학습 범위 초과만 있는 사진은 Warning으로 설정에 성공한다.

set_zero_value는 같은 좌표 기준에서 저장한 유한한 영점값을 복원한다. NaN이나 무한대는 거부하고 영점을 무효화한다. clear_zero도 영점을 무효화한다.

set_reference_from_photos는 QR와 나사 기준 위치를 다시 등록하는 기능이다. 성공 여부와 무관하게 운영 영점은 다시 설정해야 한다. 기준 재등록과 영점 설정은 서로 다른 작업이다.

## 설정

config/settings.json을 initialize의 두 번째 인자로 지정할 수 있다. screws는 사용할 나사 번호이며 왼쪽과 오른쪽을 하나씩 포함해야 한다. fail_on_disagreement=0은 일치 점검 초과를 Warning과 Recapture로 반환한다. 1이면 Failed와 Stop으로 반환하며 보정을 허용하지 않는다. log_file은 알림 기록 경로다.

설정을 생략하면 코드의 기본값을 사용한다. 모델 폴더 경로나 파일명 정답으로 조건을 몰래 바꾸지 않는다.
