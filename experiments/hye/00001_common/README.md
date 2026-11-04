# 실험 공통 C++

`experiments/hye/<실험>/실험/CMakeLists.txt`에서 include해 쓰는 헤더와 CMake 함수다.

```cmake
cmake_minimum_required(VERSION 3.15)
project(<이름> LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(<이름> src/main.cpp)
```

- `REPO_DIR`: 저장소 루트. 입력은 `data/clean/` (`hye::load_1014()`, `hye::load_341()`, `hye::load_set()`).
- `EXPERIMENT_DIR`: 그 실험의 `결과/`. 실행하면 `결과/이미지/`, `결과/결과.md`, `결과/보고서.md`를 다시 쓴다. 시각·난수를 출력에 넣지 않아 다시 돌려도 같은 파일이 나온다.
- `include/hye/dataset.hpp`: `data/clean/*/_metadata/manifest.csv` 읽기, 파일명 명령값(Y mm, T deg) 정답, 통과 기준 0.09.
- `include/hye/report.hpp`: 표(마크다운·CSV), 숫자 서식, 평균·RMS·백분위, 원래 결과 대조표(`Parity`).
- `include/hye/plot.hpp`: OpenCV로 그리는 산점도·선·막대, 정답 대비 오차 그림, 사진 확대(기준·측정 상자 표시). 그림 글자는 영문만 된다.
- `include/hye/ecc_compat.hpp`, `src/ecc_compat.cpp`: Python cv2 4.11 이후와 같은 ECC. OpenCV 3.4의 `warpAffine`은 좌표를 1/32 px로 양자화해서 ECC 결과가 0.01 px 수준으로 달라진다. 그래서 ECC 안의 역변환을 정확한 bilinear로 바꿨다. 10-12 QR 세 영역 실험에서 원래 Python 측정과 0.0001 px 안으로 맞았다.

빌드 폴더는 저장소 밖에 둔다 (예: `cmake -S 실험 -B %TEMP%/build/<이름> -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build`).
