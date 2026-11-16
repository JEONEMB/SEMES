# SSAFY 장비 Y, T 측정

[문서 목차](docs/README.md), [빠른 시작](docs/01_quickstart.md), [API와 상태](docs/03_api_and_status.md), [검증 결과](docs/05_validation.md)

2026-11-16 사용자 요청으로 SEMES의 넓은 QR 탐색, 나사 조합별 계산, 영점 및 상태 API를 적용했다. 모델은 SSAFY 사진 1959장의 기존 L2 라벨로 별도 생성했으며 SEMES 계수를 사용하지 않는다. 원본 S15P31A507과 기존 실험은 변경하지 않았다.

## 변경 내용

1. QR 띠 전체 폭을 축소 탐색하고 QR 모양이 정확히 3개인지, 이웃 간격이 맞는지 확인한 뒤 정밀 측정한다. 띠의 세로 여유는 기준 위치 주변 40 px다. 전체 화면 탐색은 아니다.
2. 나사 검출 조합에 맞는 16개 계산식을 사용한다. 좌우 나사가 확보되면 정밀 조합으로 취급하고, 부족하면 대략값과 검토 알림을 반환한다. 부분 가림 의심은 실패로 중단한다.
3. 운영 측정은 영점 설정 후 가능하다. 영점 설정 실패, 초기화, 기준 변경, clear_zero는 영점을 무효화한다.
4. 학습한 Y 또는 T 범위 초과만으로 측정이나 영점 설정을 실패시키지 않는다. W_OUT_OF_RANGE와 측정값을 반환한다. 이 경고만 있으면 Action::Correct, correction_allowed=true로 보정을 허용한다. 경고를 없애거나 학습 범위를 임의 확대하지 않았다.
5. 기준 사진, reference.json, 조합별 계수와 점검식의 SHA-256을 확인한다.

## 운영 API

현재 공개 API는 include/alignment.hpp의 alignment 네임스페이스다. 이전 전역 initialize_joint와 joint API는 legacy에 보존했으며 기본 빌드 대상이 아니다.

```cpp
const auto init = alignment::initialize("data/model");
if (init.status == alignment::Status::Failed) return;
const auto zero = alignment::set_zero_from_photos({center_photo});
if (zero.status == alignment::Status::Failed) return;
const auto result = alignment::measure_alignment(photo);
if (result.correction_allowed) {
    // result.y_mm와 result.t_deg는 추정값이다. 장치 이동 명령으로의 변환은 호출자가 담당한다.
}
```

상태와 보정 허용 여부는 다르다. 영점 설정 결과의 Warning은 실패가 아니지만 알림을 확인해야 한다. 실제 측정의 모든 notices와 correction_allowed를 함께 읽는다. 영점 없는 개발 평가에만 measure_diagnostic을 사용하며 이 결과는 보정을 허용하지 않는다.

## 빌드와 실행

Windows x64, MSVC v141, OpenCV 3.4.15 기준이다. ssafy 폴더에서 실행한다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
./build/Release/measure_one.exe data/model PHOTO CENTER_PHOTO
./build/Release/measure_one.exe data/model PHOTO --diagnostic
```

measure_one의 마지막 인자는 영점 사진 또는 명시적인 --diagnostic이다. 영점 인자를 생략한 운영 측정은 허용하지 않는다.

## 전체 평가와 모델 생성

```powershell
./build/Release/evaluate_photos.exe data/model photos.tsv data/results/new_run 1
./build/Release/train_combinations.exe data/model training.tsv data/results/new_model
```

photos.tsv는 머리줄 없는 id와 사진 경로 TSV다. 평가 도구는 정답을 입력받지 않고 진단 측정만 수행한다. 실제 측정 사진과 결과를 후처리에서 정답과 비교한다. training.tsv는 id, path, y, t 머리줄과 기존 SSAFY 1959장의 라벨을 사용한다. 학습 도구는 누락이나 검출 실패를 조용히 빼지 않고 중단하며 출력 폴더가 이미 있으면 덮어쓰지 않는다. 새 모델은 검증 후 적용한다.

[이번 변경 검증](docs/runtime_migration.md)을 확인한다. 기존 보정 라벨은 영상에서 만든 정답이므로 실제 장비 위치를 독립 계측한 정확도로 해석하지 않는다.

## 이전 버전과 검사

legacy에는 이전 소스, 헤더, 도구, 모델이 보존돼 있다. 이전 final00_tool과 절대 영점 기본 동작은 과거 재현용이며 운영용이 아니다. 변경 전 평가 결과도 이전 버전의 기록이며 현재 문서 목차와 구분한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/check_text_style.ps1
```
