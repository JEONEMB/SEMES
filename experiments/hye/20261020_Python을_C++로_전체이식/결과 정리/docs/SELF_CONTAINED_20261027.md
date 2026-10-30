# 독립 실행 구조와 검증

최신 사용자 지시: Python 구현을 전부 C++로 대체하고 입력을 복사한 뒤 경로·재실행·재현을 검증한다. 완료 후 원본을 legacy로 이동한다. 전체 대체는 진행 중이다.

| 경로 | 용도 |
|---|---|
| include, src | 공통 C++ 라이브러리 |
| tool, tests, examples | 실행 도구, 검증, 제품 호출 예제 |
| experiments | 과거 실험별 구현과 출처 |
| data/raw/hw | 복사한 원본 촬영 데이터 |
| data/reference | 기존 입력·결과·문서·프런트엔드 비교 자산, 원본 디렉터리 구조 유지 |
| models, config | 모델과 실행 설정 |
| third_party | 복사한 OpenCV·zlib와 출처를 고정한 ECC 커널 |
| resources/web | 프런트엔드 자산 |
| results | 실행마다 새 폴더로 보존하는 결과 |
| build/native | preset으로 새로 구성한 빌드 |

데이터 복사: 6,024개, 19,311,888,142바이트. 원본과 대상의 SHA256을 대조했고 오류 0개다. [복사 명세](../results/self_contained_20261027_001/checks.json). 의존성 545개, 388,433,627바이트도 해시 검증했다. [의존성 명세](../results/vendor_copy_20261027_001/checks.json). 기존 결과 복사본은 새 실행 결과와 구분한다.

`final`에서 실행한다. MSVC와 Windows SDK는 설치된 빌드 도구를 사용한다.

```powershell
cmake --preset native
cmake --build --preset native-release
ctest --preset native-release --output-on-failure
cmake --preset modern
cmake --build --preset modern-release
ctest --preset modern-release --output-on-failure
build/native/Release/semes.exe list
build/native/Release/semes.exe run core.infer-example results/my_new_run
build/native/Release/semes.exe run feedback.rebuild341 results/my_new_training
```

`semes run`은 config/native_programs.json에 선언한 실제 C++ 실행 파일을 호출하며, 입력을 이 폴더 내부에서 찾는다. 출력 경로가 이미 있으면 덮어쓰지 않는다. 위 프로그램 목록은 전체 원본 이식 완료 목록이 아니다.

기본 검증 3개와 OpenCV5 제어 검증 1개 통과. OpenCV5는 third_party/opencv-5.0.0에서 직접 빌드했다. 원본 ZIP SHA256은 E2ABAA1EA2443FB163299369506521518EE02D8E51B88A144A128C4E492123BD다. 재빌드는 `cmake --build third_party/opencv500-build --config Release --target opencv_core opencv_imgproc opencv_imgcodecs opencv_video`를 사용한다. 새 빌드 기록은 results/self_contained_20261027_001/*build*.log에 보존한다.

새 이식 검증:

- analysis_v4/zero_align.py: 12개 축·방법 조합, 원본 표 84개 수치가 소수 셋째 자리 반올림 오차 범위에서 일치. 최대 차이 0.000495508. 그림은 OpenCV/GDI로 새로 렌더링하며 Matplotlib 바이트 일치는 주장하지 않는다. results/zeroalign_native_20261027_003/checks.json.
- 10월26일 motion.py: 8개 시나리오 61번 호출. 식별 가능한 유한 수치 최대 차이 2.42e-11. 순위가 부족해 조건수가 1e12를 넘는 진단값은 수치 일치에서 제외하며 양쪽의 거부·행동은 동일하다. 첫 실패와 변경된 기준을 모두 보존했다. results/motion_native_compare_20261027_002/checks.json.
- lines/extractor.py: 341장 × 48블록 = 16,368개. 유효성 변경 0, 변위 최대 차이 2.71e-7px로 사전 설정 1e-6px 이내. results/line_native_full_20261027_001/parity/checks.json.
- direct/extractor.py: OpenCV3에 새 커널만 연결했을 때 0.02374px 차이를 보였고 그 실패 기록을 보존했다. OpenCV5 전체를 직접 빌드해 연결한 뒤 341장·네 후보 모두 통과했다. 최대 변위 차이 1.37e-12px, 성공·유효성·비유한 값 변경 0. results/direct_native_full_opencv500_20261027_001/parity/checks.json.
- prepare_features.py: 새 C++ 측정으로 생성한 8개 NPZ 배열이 원본과 설정 허용차 1e-6 이내다. results/features_native_20261027_002/parity.json. C:/Temp에서 전체 341장을 다시 처리했을 때 features.npz SHA256이 바이트 단위로 일치했다. results/selfcontained_features_rerun_20261027_001/reproduction.json.
- 이미지 세션: 네 조건·16회 호출에서 원본과 최대 1.47e-7 차이로 통과했다. results/image_session_parity_20261027_001/checks.json.
- 과거 피드백 제어기: 세 정책·18개 시나리오·162회 호출 최대 차이 1.71e-13. results/feedback_parity_20261027_001/checks.json.
- 보정 재학습: 기존 특징 입력으로 새로 계산한 투영·Jacobian·noise가 실제 보존 모델 네 개와 최대 8.84e-10 차이로 통과했다. 원본과 같은 OpenBLAS 단일 스레드와 합산 순서를 고정했다. results/calibrate_full_frozen_inputs_20261027_002/parity.json. 원본 사진부터 새로 측정한 특징으로 다시 학습한 모델은 기저 방향과 수치가 달라질 수 있으며, 그 차이는 results/calibrate_full_fresh_inputs_20261027_002/differences_from_original.json에 보존한다. 이것을 원본 전체 학습 경로의 바이트 일치로 주장하지 않는다.

새 API: image_extract condition image new-output은 QR/위/아래/양쪽의 시각 특징을 생성한다. calibrate training.json new-output은 제공된 학습 행으로 보정 투영과 Jacobian을 계산한다. calibrate_full features.npz manifest.csv new-output은 네 조건을 함께 학습한다. motion_session은 특징 JSON 입력, image_session은 이미지 추출부터 이동 제안·상태 복원까지 연결하는 API다. 현대 영상 경로는 build/modern/bin/Release의 실행 파일을 사용한다.

models/feedback_full341_frozen의 네 모델은 기존 joblib를 JSON으로 일회성 변환했고 투영·Jacobian·noise 값의 정확한 왕복 일치를 확인했다. conversion_manifest.json에 원본과 변환 후 해시가 있다. 저장된 모델 형식 변환이며 C++ 재학습 완료 증거가 아니다. `image_session start state.json image.bmp models/feedback_full341_frozen/qr.json settings.json prior-guarded`처럼 실행한다. settings.json에는 probe_y, probe_t, max_step_y, max_step_t를 명시한다. 실제 실행한 이동량은 `image_session observe state.json next-image.bmp dy dt`로 전달한다. 하드웨어 명령은 전송하지 않는다.

원본 코드가 남아 있는 이유는 전체 이식과 재현 검증이 아직 완료되지 않았기 때문이다. 이 문서나 일부 통과 결과를 전체 대체 검증으로 사용하지 않는다. 341장 자료는 이미 노출된 개발 자료이며 독립 검증·실제 로봇 복귀 성과가 아니다.
