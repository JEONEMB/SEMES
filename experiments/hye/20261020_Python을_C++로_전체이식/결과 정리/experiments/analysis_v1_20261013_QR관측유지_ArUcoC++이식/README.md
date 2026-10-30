# 초기 QR 내부점·ECC 설정 조사 C++ 이식

2026-10-27 재검증에서는 `historical_ecc_tune original-workspace bmp-20261014 new-output [variant-index 0..6]` 진단 옵션을 추가했다. 마지막 인자가 없으면 기존처럼 7개 설정 전부를 계산한다. 인자가 있으면 해당 설정의 원본 선택 68장 전체를 계산하며 `checks.json`에 `variants: 1`과 `selected_variant`를 기록한다. 설정별 새 출력 폴더로 독립 실행한 뒤 0~6 순서로 합치면 전체 476행을 대조할 수 있다. 영상 선택·ECC 조건·계산식·OpenCV 스레드 수는 변경하지 않았다. 잘못된 인덱스는 입력 영상 로딩 전에 거절한다. 이번 실행·대조 근거는 [재검증 보고서](../../results/recheck_20261027_001/README.md)에 있다. 원본 수치와의 동등성 미해결 상태를 해소했다는 뜻은 아니다.

원본은 `analysis_v3/v1_data/qr_only.py`, `ecc_tune.py`다. 두 파일의 생성 관측 시각은2026-10-13 13:05:20.524 KST이며 일괄 보관본 시각으로 취급한다. 실제 실행 시각·원본 작성자를 파일 시각만으로 단정하지 않는다. 이번 C++ 작업자는 현재 채팅의 Codex다.

## 문제·조사·해결

QR 내부의 같은 점을 추적했을 때 어느 점이 안정적인지, 판의 원근 변환 측정에서 흐림·마스크 확장·ECC 내부 필터가 잡음에 어떤 영향을 주는지 조사한 초기 실험이다.

기존 고정 C++ 환경 OpenCV3.4.15에는 원본이 사용하는 `QRCodeDetectorAruco`가 없다. 검출기를 다른 방법으로 바꾸는 대신 공식 OpenCV4.11.0 소스를 `final/build/dependencies`에서 따로 빌드했다. C++4 실행 파일은 OpenCV3 추론 라이브러리와 링크하지 않으며 기존 설치를 변경하지 않았다.

## 유지한 부분

- QR mask: center 한 장에서 ArUco 기반 QR 세 사각형 검출·정수 좌표 마스크.
- QR points: goodFeaturesToTrack 최대300·quality0.01·거리12·block5. 현재 사진에서 새 ID를 만들지 않음.
- Tracking:20×20 템플릿·가로±110/세로±6·NCC·가로 포물선 보간.
- Filtering: 원본282행 전체에서 원점 통과 선형 모델 잔차<2px·최저 NCC>0.8. 이 전체 자료 선택 조건은 과거 실험이며 독립 검증 성적으로 표시하지 않음.
- ECC tuning: 원본 T조건 행 중 매3번째, 일곱 blur/마스크/필터 설정, homography500회·epsilon1e-9·h20 기울기로 환산한 잡음.

## 바꾼 부분

Python/Pillow/NumPy를 native C++ OpenCV4.11과 C++ NPZ 입출력으로 대체했다. 이미지 경로는 명시적으로 받고 원본의 오래된 Downloads 경로를 실행하지 않는다. 저장된 NumPy object 메타데이터는 실행 없는 기호 파서로 읽는다. OpenCV 빌드 최적화·원본 Python 패키지 버전 차이는 별도로 취급한다.

새 결과는 새 폴더에만 기록한다. 원본·DATA·기존 결과·legacy를 수정하지 않는다.

## qr_only.py 검증 완료

2026-10-23 10:16:52.537 KST, `../../results/historical_qr_only_20261023_001/checks.json`:

- 원본 순서·ID의300점 좌표 최대차0px.
- 중복 center 포함282행 전체 측정. 원본 D배열과 최대차0.000190352px; 미리 정한0.001px 대조 범위 이내.
- 선택된263점 ID와 순서가 원본과 같음.
- C++ NPZ를 다시 로딩해 숫자 배열 비교. 전체점·QR 마스크는 원래3840×2160 크기로 별도 저장.

첫 파일 조사에서는 `ecc_tune.py`의 일곱 설정을 담은 숫자 표를 찾지 못했다. 이후2026-10-23 11:11 KST에 [원래 Claude 로그](../provenance_recovery_20261023_01/README.md)에서 소수4자리 출력7개를 복구했다. 로그의 작성 명령 전체 코드가 현재 보관본과 줄바꿈 정규화 후 일치한다. 작성 관측은2026-10-12 15:37 KST, 출력 관측은15:49 KST이며10월13일 보관 시각과 다르다. 당시 NPZ 바이트와 Python/OpenCV 빌드는 아직 동결 근거가 없다.

11:14:25 KST의 `../../results/historical_ecc_tune_20261023_002`는7설정×68행을 실패0으로 계산했다. 원본 주석의67장이 아니라 실제 조건으로 선택된68행을 유지했다. 복구 출력과의 대조 `historical_ecc_tune_parity_20261023_001`에서 잡음7개 모두 소수4자리 반올림 범위 밖이며 최대차0.00712078이다. 설정 값은 같다. 수치 차이 원인이 미해결이므로 완료 체크는 하지 않는다. 실행 당시 보고서의 ‘저장 기준 숫자 없음’은 초기 파일 조사 상태이며, 이후 로그 복구·대조를 이 문서와 별도 결과로 추가했다.

## 실행

```powershell
# final 기준. OpenCV4 선택 빌드가 준비된 경우
cmake -S . -B build -DOpenCV_DIR=C:/opencv/build -DSEMES_OPENCV4_DIR=<final의절대경로>/build/dependencies/opencv411-build
cmake --build build --config Release -- /m:1 /nr:false
.\build\experiments\analysis_v1_20261013_QR관측유지_ArUcoC++이식\Release\historical_qr_only.exe .. ../hw-analysis/data/hw/bmp/20261014 results/qr_only_새이름
.\build\experiments\analysis_v1_20261013_QR관측유지_ArUcoC++이식\Release\historical_ecc_tune.exe .. ../hw-analysis/data/hw/bmp/20261014 results/ecc_tune_새이름
```

공식 버전과 소스는 [OpenCV4.11.0 릴리스](https://github.com/opencv/opencv/releases/tag/4.11.0), 빌드 자산·해시는 `../../config/opencv411_build.json`에서 확인한다.
