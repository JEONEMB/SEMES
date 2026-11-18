# SSAFY 장비 Y, T 측정 패키지

S15P31A507/experiments/hye/00000_final_Y_T_ssafy를 별도로 복사한 독립 빌드 패키지다. 원본 폴더와 다른 실험은 변경하지 않았다. ssafy2 두날 평균 라벨 모듈이 아닌 기존 ssafy 모듈과 그 모델을 사용한다.

## 구성

| 경로 | 내용 |
|---|---|
| include, src | 원본 측정 API와 구현, 공통 진단 구현 |
| third_party | 원본 ECC 호환 구현 |
| data/model | 원본 assets의 모델, 기준 사진, 장비 설정, 해시 정보 |
| data/evidence | 원본 평가 기록 |
| tools | 측정, 학습, 평가 도구와 문자 검사 |
| docs | 원본 설명, 포팅 이력, 파일 출처 |

## 알고리즘과 입력, 출력

기본 QR 방식은 가우시안 흐림 후 QR 3개 NCC 비교와 포물선 보간이다. QR 가로 이동 1개와 나사 4개의 관측값을 사용하며 기본 모델의 바 기울기 가중치는 0이다. data/model/site.json이 장비 위치와 탐색 범위를 정의한다. 기존 모델 계수, 기준 이미지와 장비 설정은 바이트 그대로 복사했다. health TSV 2개의 주석 문자는 정리하고 provenance.json의 해당 해시만 갱신했다. 점검식 수치는 유지했다. 현재 semes 패키지의 알고리즘이나 모델로 대체하지 않았다.

입력은 SSAFY 장비 사진과 동봉된 기준 자료다. initialize_joint로 자료를 읽고 joint로 Y, T 추정값을 구한다. Y는 mm, SSAFY T는 도 단위이며 장비 부호는 SEMES와 다르다. 출력은 이동 명령이 아니다. 사진을 읽지 못하거나 측정에 실패하면 예외 또는 실패 종료를 반환한다. 경고는 alignment_diagnostics로 확인한다.

동봉 alignment.hpp는 원본 공통 헤더다. 이 패키지가 제공하는 초기화 및 측정은 initialize_joint, joint이고, 헤더에 있는 control, initialize_control, initialize_alignment, alignment의 구현은 포함하지 않는다. 추가 설정 API는 include/joint_reference.hpp를 확인한다.

기본 모델은 기존 절대 영점 기준이다. 당일 영점 변경은 set_joint_zero, 기준 위치 재등록은 set_joint_reference를 사용한다. 각 API의 전제 조건은 원본 설명을 확인한다.

## 빌드

Windows x64, MSVC v141, OpenCV 3.4.15 기준이다. ssafy 폴더에서 실행한다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
cmake --build build --config Release
```

CMake가 PATH에 없다면 Visual Studio Build Tools의 Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe를 사용한다. 원본 저장소는 빌드와 실행에 필요하지 않다.

## 사진 한 장 측정

```powershell
./build/Release/measure_one.exe data/model data/model/center.bmp
```

마지막 인자를 실제 사진 경로로 바꾼다. 이 예제는 기본 절대 영점 모델을 사용하며 당일 영점을 새로 설정하지 않는다. 기준 사진 결과도 반드시 정확한 0, 0이라는 뜻은 아니다.

## 일괄 측정과 평가

```powershell
./build/Release/final00_tool.exe measure data/model photos.tsv data/results/measure
./build/Release/final00_tool.exe predict data/model photos.tsv labels.tsv data/results/predict
```

photos.tsv는 머리줄 없이 id와 사진 경로를 탭으로 구분한다. labels.tsv와 평가 분할 형식, 영점 옵션은 tools/final00_tool.cpp의 사용 설명을 확인한다. 기존 출력 폴더 대신 새 출력 경로를 사용한다. 외부 사진과 정답표는 사용자 입력이며 동봉된 자료만으로 전체 평가를 재현하는 구성은 아니다.

원본 평가 수치는 개발에 사용한 사진 및 보정 라벨에 대한 기록이다. 새 사진의 독립 성능이나 실제 장비 복귀를 보장하지 않는다. 이 분리 작업에서는 모델을 재학습하지 않았다.

## 출처와 검사

docs/source_manifest.tsv에 원본 경로와 복사 전 SHA-256을 기록했다. 소스의 주석과 출력 문구에 있던 금지 문자는 정리했으며 계산식은 유지했다. 원본 설명의 실험 링크는 이전 저장소를 참조하는 이력이며 실행 의존성이 아니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/check_text_style.ps1
```

## 전체 장비 사진 재평가

2026-11-16의 [전체 사진 정확도와 속도 결과](data/results/20261116_full_device/report.md), [현재 SEMES와의 차이 및 보완 후보](docs/comparison_with_semes.md)를 확인한다.

전체 입력 목록의 사진을 순서대로 측정하고 속도를 기록하는 도구를 추가했다. 모델과 측정 알고리즘은 변경하지 않았다.

```powershell
./build/Release/evaluate_photos.exe data/model photos.tsv data/results/new_run 3
```

photos.tsv는 머리줄 없이 id와 UTF-8 사진 경로를 탭으로 구분한다. 마지막 인자는 전체 반복 횟수다. 준비 측정 10회는 속도 집계에서 제외하며 OpenCV 스레드는 1개로 고정한다. 결과는 measurements.tsv와 run.txt다. 정답은 이 실행 파일에 전달하지 않으며 별도 집계에서만 비교한다.
