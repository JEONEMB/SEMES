# QR 고정 파이프라인 사용법

```powershell
cmake --build build --config Release
New-Item -ItemType Directory -Force docs/validation-runs/manual | Out-Null
.\build\Release\main.exe "data/SSAFY Test Image Set/center.bmp" --output "docs/validation-runs/manual/result"
.\build\Release\main.exe "data/SSAFY Test Image Set/center.bmp" --output "docs/validation-runs/manual/result" --no-gui
```

## 처리 순서

1. Canny 에지 검출.
2. 허프 직선 검출 후 같은 외곽선에 해당하는 조각 병합.
3. 고정 조건으로 플레이트 사각형 한 번 선택.
4. 플레이트 중점 기준 좌·중·우 QR ROI 생성.
5. 각 ROI에 고정 전처리(Gaussian 3×3, sigma 0.6 → 2배 선형 확대 → 흰 여백 32px).
6. 각 QR에 detectAndDecode 한 번 호출. 실패하면 실패 결과를 반환하며 다른 조건으로 재시도하지 않는다.
7. 내용·버전·ECC·마스크·인코딩 모드를 고려해 이상적 모듈 패턴을 재생성하고 관측 패턴과 일치 확인.
8. 모델의 체커보드형 교점 후보를 원본 영상에 투영하고 원본 그레이스케일에서 평균 제거 Fourier 교점 정밀화(11×11, alpha=2.4)를 수행.
9. 국소 코너/명암/이동량 및 평가점을 제외한 재투영 잔차로 품질 확인.

원본 source는 디코딩 전처리에 의해 변경되지 않는다. 프레임마다 동일한 처리 순서를 사용한다. 잘못된 입력, 저장 오류, 검출 실패 판정은 남아 있다.

## 소스 구성

| 파일 | 역할 |
|---|---|
| src/alignment.cpp | 입력 검증과 단계 연결 |
| src/plate_detection.cpp | Canny·허프·직선 병합·사각형 평가 |
| src/qr_roi.cpp | 플레이트 좌표계와 ROI 추출 |
| src/qr_decode.cpp | 고정 전처리와 디코딩 1회 |
| src/qr_model.cpp | 이상적 QR 모듈 모델 재생성·검증 |
| src/qr_features.cpp | 후보 생성, 원본 영상 정밀화, 기하 검증 |
| src/pipeline_visualization.cpp | 이미지 저장과 단계별 표시 공통 처리 |
| include/pipeline_settings.hpp | Fourier 정밀화·디코딩·선 병합 고정 설정 |
| include/qr_refinement.hpp | Fourier 및 영상 형성 모델의 수치 최적화 |
| .clang-format | BSD/Allman, 4칸 들여쓰기, 중괄호 다음 줄 |

기존 alignment(image) 호출, AlignmentOptions 및 PlateDetectionResult 인터페이스는 유지한다. qrcodegen은 MIT 라이선스 외부 코드이므로 BSD 포매팅 대상에서 제외했다.

## 시각화·결과

키를 누르면 다음 단계로 진행한다. `_02_merged.png`는 병합된 선, `_06_decode_input_L/C/R.png`는 고정 디코딩 입력이다. 기존 에지, 허프, 플레이트, ROI, 이상적 모델, 후보, 정밀화 결과 및 특징점 CSV도 유지한다. decodeUsedGaussian은 이제 재시도 여부가 아니라 고정 디코딩 전처리에 Gaussian을 사용했음을 뜻한다. decodeGaussianSigma는 0.6이다.

QrRoiResult의 payload, modelMatched, featuresValid, features를 확인한다. 특징점 ID는 (QR 내용, 모듈 기반 ID)로 해석하며 accepted가 거짓인 좌표는 측정값으로 사용하지 않는다. 정렬 오차 y/t는 아직 계산하지 않으므로 NaN을 반환한다.

종료 코드: 0=세 QR 특징점 처리 성공, 1=입력/저장 오류, 2=플레이트 검출 실패, 3=ROI 오류, 4=디코딩/모델/특징점 실패.

[고정 파이프라인 검증](qr-fixed-pipeline.md) · [이전 재시도 파이프라인 검증 이력](qr-stability-report.md)

## 교점 정밀화 선택

기본값은 `QrRefinementMethod::Fourier`이다. 명시적인 옵션 없이 실행해도 선택한 Fourier 평균 제거 변형이 적용된다.

```powershell
.\build\Release\main.exe <image.bmp> --no-gui --refinement fourier
.\build\Release\main.exe <image.bmp> --no-gui --refinement model
.\build\Release\main.exe <image.bmp> --no-gui --refinement corner
```

이전 방식과 영상 형성 모델은 재현·비교용으로 유지한다. Fourier 설정은 `include/pipeline_settings.hpp`에 모았다. 원본 픽셀의 홀수 각주파수 성분을 최소화하며, 극좌표 영상 재보간은 하지 않는다. 디코딩용 crop과 최종 측정 좌표를 구분한다.

수치·한계는 [비교 보고서](qr-subpixel-comparison/README.md)와 [선정 결과 및 그래프](qr-subpixel-comparison/fourier-selection.md)를 참고한다. 현재 운영 alignment()의 물리적 Y/theta 반환은 NaN이며, 보고된 물리량 성능은 별도 오프라인 평가기의 결과이다.
