# 상단 구조물만 사용하는 θ 비교 버전

하단 직선 추정을 제외한 C++ 작성본이다. 기존 `../theta_huber_20261101`은 수정하지 않는다.

## 변경 내용

상단 ROI·가림 범위·세로 Sobel·일반 Huber는 기존과 같다. 기존 `edge_points.cpp`, `huber_line.cpp`, `utf8_io.cpp`를 읽기 전용으로 재사용해 빌드한다. 하단 경계점을 추출하거나 하단 직선을 피팅하지 않으며, 기존 두 각도의 Y·θ 연립방정식도 사용하지 않는다.

```text
이미지 한 장
 → 세로 Sobel
 → 상단 경계점 2,712개
 → 일반 Huber 직선 피팅
 → 상단 영상 각도 φ
 → θ 추정 = (φ − b) / a
 → θ 보정치 = 0 − θ 추정
```

상단 각도 하나로 미지수 Y와 θ를 동시에 구할 수 없으므로, 이 비교 버전에서는 **Y 영향을 따로 보정하지 않는다.** 테스트 사진의 Y 라벨을 계산에 입력하지 않는다.

## 계수 계산

기존과 동일한 2차 (1)·(2)번 40장의 상단 각도를 사용해 `φ = b + a×θ`를 일반 최소제곱으로 피팅했다. 평균·가중치·추가 필터·정규화 항을 넣지 않았다.

- `b = 0.27101545388280657°`
- `a = -0.08829275104680276 영상 ° / 장비 °`
- 목표 θ: `0°`

계수는 테스트 실행 전에 고정했다. 실행 시 입력은 이미지 한 장과 설정 파일이다. 설정에 하단 ROI나 하단 각도 계수는 없다.

## 비교 범위

- 계수 계산: 기존과 동일한 40장.
- 테스트: 기존과 동일한 2차 (3)번 20장 + 3차 전체 234장.
- 판정: `|추정 θ − θ 라벨| ≤ 0.005°`.
- 변경하지 않은 항목: 상단 측정 과정, Huber 설정, θ 목표, 테스트 라벨, 성공 기준.
- 함께 변경된 항목: 하단 특징 제거, Y·θ 동시 환산 제거, 상단 전용 계수 재계산.

따라서 이 결과는 **상단 전용 모델과 상·하단 결합 모델의 비교**다. 개선되더라도 하단 경계점 추출 자체가 잘못됐다는 원인 확정은 아니다. 기존 실패를 본 뒤 동일한 테스트셋을 재사용한 비교 실험이며, 새로운 독립 데이터의 검증으로 표현하지 않는다.

## 코드

| 파일 | 역할 |
|---|---|
| `src/upper_theta.cpp` | 상단 측정과 상단 각도→θ 환산 |
| `src/alignment.cpp` | 기존 `Alignment_t alignment(cv::Mat &src)` 연결 |
| `src/main.cpp` | 이미지 한 장 입력, JSON 출력 |
| `config/calibration.json` | 상단 ROI·Huber·계수 |
| `tools/prepare_calibration.py` | 동일 학습 40장으로 계수 계산 |
| `tests/upper_tests.cpp` | 역산·오류 처리·하단 픽셀 비사용 검사 |

공통 선언을 위해 기존 `theta_pipeline.hpp`를 포함하지만 기존 상·하단 Detector와 `theta_decoder.cpp`는 링크하지 않는다. 직선 추정은 상단에 대해서만 한 번 호출한다.

## 실행

프로젝트 루트 PowerShell:

```powershell
& '.\algorithm\theta_huber_upper_only_20261101\tools\build.cmd'

& '.\.tmp\theta_huber_upper_only_20261101_build\Release\theta_upper_alignment.exe' `
  '.\algorithm\theta_huber_upper_only_20261101\config\calibration.json' `
  'C:\실제\이미지경로.bmp'
```

기존 스켈레톤 호출 방식은 유지한다.

```cpp
configureThetaAlignment("algorithm/theta_huber_upper_only_20261101/config/calibration.json");
Alignment_t command = alignment(image);
```

`command.t`는 θ 보정 각도이며 `command.y=0`은 Y 명령 없음이다. JSON에는 `lower_used=false`, `y_effect_corrected=false`를 명시한다. 실제 장비를 구동하거나 정렬 완료를 선언하지 않는다.

## 확인 및 결과

빌드와 기본 계산 검사를 통과했다. 학습 이미지 한 장에서 상단 ROI의 Sobel 지지 영역 아래를 전부 검정/흰색으로 바꾸어도 상단 기울기와 θ 보정치가 동일함을 확인했다. 원본 파일은 변경하지 않고 메모리 복사본에서 검사했다.

테스트 전체에서는 새 버전의 상단 측정 출력이 기존 버전과 정확히 동일한지도 확인한다. 결과는 [상단 전용 254장 비교 보고서](../../analysis/theta_huber_upper_only_20261101/report.html)에 기록한다.

일반 Huber와 공통 코드의 근거·라이선스는 기존 버전의 `THIRD_PARTY_NOTICES.md`, `licenses/statsmodels-LICENSE.txt`를 따른다. 비교 버전만 이동할 때에는 기존 공통 소스 및 해당 라이선스도 함께 제공해야 한다.
