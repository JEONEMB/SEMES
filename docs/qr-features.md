# QR 모델 기반 특징점 추출

## 실행

```powershell
cmake --build build --config Release
.\build\Release\main.exe "data/SSAFY Test Image Set/center.bmp" --output "result"
.\build\Release\main.exe "data/SSAFY Test Image Set/center.bmp" --output "result" --no-gui
```

기존 1~5단계 뒤에 QR별 L → C → R 순서로 6~9단계가 실행됩니다. 각 화면에서 키를 누르면 다음 화면으로 진행합니다.

- 06_decode: 디코딩 내용과 QR 외곽.
- 07_model: 재생성한 이상적 QR과 체커보드형 교점 후보.
- 08_candidates: ROI 위로 투영한 초기 후보(청록색).
- 09_refined: 초기점(청록), 통과점(초록), 제외점(빨강), 보정 벡터(노랑). 보기 위해 3배 확대했으며 측정 좌표는 원본 영상 픽셀 단위입니다.
- 10_features: 특징점별 CSV. 파일 접미사는 L/C/R입니다.

## 처리 원리

OpenCV QRCodeDetector로 내용과 정규화된 모듈 행렬을 읽습니다. 실패 시 디코딩 입력만 확대/흰 테두리 추가 후 재시도합니다. C++ QR-Code-generator로 같은 내용·버전의 auto/byte 인코딩, 네 ECC 수준, 여덟 마스크를 비교합니다. 관측 모듈 행렬과의 불일치율 3% 이하이며 차선 후보보다 2%p 넘게 일치하는 패턴만 모델로 채택합니다. 관측 모듈 자체를 이상적 원본으로 취급하지 않습니다. 혼합 세그먼트/ECI 등 현재 재생성 범위로 맞출 수 없으면 명시적으로 실패하며, 인쇄 원본이 확보되면 해당 모듈 행렬을 등록하는 방식으로 확장하는 것이 좋습니다.

모델에서 대각선 흑백이 교차하는 교점만 후보로 사용하고, 모듈 좌표에 기반한 고정 ID를 부여합니다. ID는 QR 내용/모델과 함께 해석해야 합니다. 같은 색 모듈이 붙은 격자점은 후보가 아닙니다.

디코더의 외곽으로 QR 모델→원본 영상 변환을 구하고, 원본 그레이스케일 영상에서 cornerSubPix를 수행합니다. 원근 보정 ROI는 decode/시각화에만 사용합니다. 윈도 반경은 모듈당 픽셀 수의 0.30배(정수 내림), 종료 기준은 40회/0.001px입니다. 종료 기준은 정확도 보장이 아닙니다.

## 결과 및 품질 검사

`PlateDetectionResult.qrRois[i]`에서 payload, decoded, modelMatched, moduleCount, featuresValid, featureStatus와 features에 접근합니다. 각 QrFeature는 모델 좌표, 초기/정밀화된 원본 좌표, 이동량, 최소 고유값, 사분면 대비, leave-one-out 재투영 잔차, localAccepted/accepted를 제공합니다.

초기점 이동량 ≤ 모듈 피치×0.45, 최소 고유값 > 0.0001, 예상 흑백 사분면 대비 > 25를 국소 조건으로 사용합니다. 이후 RANSAC과 각 평가점을 제외하고 계산한 호모그래피로 잔차를 검사합니다. 잔차 한계는 max(0.75px, 모듈 피치×0.15)입니다. 최종 8개 이상이며 가로·세로 분포가 QR의 40% 이상이어야 featuresValid가 참입니다. 이 조건들은 초기 경험적 기준으로서 반복 촬영·알려진 이동을 통한 조정이 필요합니다. 점들 간 일관성은 절대 위치 정확도나 확률적 confidence와 다릅니다.

실패/제외점의 잔차는 NaN일 수 있으며, accepted가 거짓인 refined 좌표는 측정값으로 사용하지 않습니다. QR의 내용/모델/후보 ID가 같은 점끼리 비교해야 합니다. 현재는 실제 Y/θ 정렬 오차를 계산하지 않습니다.

종료 코드: 0=세 QR 특징점 처리 성공, 1=입력/저장 오류, 2=플레이트 검출 실패, 3=ROI 오류, 4=디코딩·모델 대응·특징점 품질 실패. GUI 키 입력은 별도 실행 확인이 필요합니다.

## 외부 라이브러리

Project Nayuki의 [QR-Code-generator](https://github.com/nayuki/QR-Code-generator) C++ 구현(MIT)을 src/qrcodegen.cpp 및 include/third_party/qrcodegen.hpp에 포함했습니다. 두 파일 상단에 원문 라이선스를 보존했습니다. 기존 CMake의 src/*.cpp 재귀 수집으로 자동 빌드되며 별도 설치는 필요 없습니다.

## 추가된 검출 안정화와 검증

플레이트는 >180px 가로선 조건으로 먼저 검출하고, 유효 사각형이 없으면 >100px로 재시도합니다. 기존 QR 디코딩 재시도가 모두 실패한 경우에만 3×3 Gaussian(sigma 0.6, 0.9)을 디코딩 복사본에 적용합니다. `decodeUsedGaussian`과 `decodeGaussianSigma`로 사용 여부를 확인하며, 성공한 Gaussian 입력은 `_06_gaussian_input_L/C/R.png`로 저장됩니다. cornerSubPix와 품질 평가는 원본 그레이스케일을 계속 사용합니다.

[281장 전수 검증 보고서](qr-stability-report.md)에 조건별 성공률, 실패 영상과 ROI 비교, 특징점 유지율, 평균 이동 제거 후 요동, 원시 CSV 및 CPU 병렬 재현 절차를 정리했습니다.
