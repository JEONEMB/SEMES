# 빠른 시작

## 1. 필요한 환경과 파일

검증 환경은 Windows x64, MSVC v141, OpenCV 3.4.15다. C++17과 CMake 3.15 이상이 필요하다. data/model의 기준 사진, reference.json, coefficients, checks, model_manifest.tsv를 함께 사용한다.

입력은 3840x2160, 8비트 회색, BGR 또는 BGRA 사진이다. 파일 경로를 읽는 read_photo는 회색 영상으로 반환한다. 영상 정답이나 파일명에 적힌 이동량은 추론 입력이 아니다.

## 2. 빌드

저장소 루트에서 실행한다. Visual Studio 설치 경로와 OpenCV 경로는 PC에 맞춘다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CMake가 PATH에 없으면 Visual Studio Build Tools의 Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe를 사용한다. MSVC 빌드는 실행 파일 옆에 OpenCV DLL을 복사한다. Linux 또는 라즈베리파이에서 실행한 결과는 아직 없다.

## 3. 사진 한 장

```powershell
./build/Release/measure_one.exe data/model "C:/photos/current.bmp" "C:/photos/center.bmp"
```

마지막 인자는 현재 설치와 같은 조건에서 촬영한 영점 사진이다. 파일명이 center라는 이유만으로 물리적 영점이 되는 것은 아니다. 영점 설정 실패 시 측정을 시작하지 않는다.

개발용으로 기본 절대 좌표를 확인할 때만 다음을 사용한다.

```powershell
./build/Release/measure_one.exe data/model data/model/center.bmp --diagnostic
```

진단 결과는 보정을 허용하지 않는다. measure_one은 y, t, status, correction_allowed, screws, code를 출력한다. 상태 숫자는 Ok=0, Warning=1, Failed=2다. 종료 코드 0은 도구 실행이 끝났다는 뜻이며 정확도 통과나 보정 허용을 의미하지 않는다.

## 4. 전체 사진 진단

```powershell
./build/Release/evaluate_photos.exe data/model photos.tsv data/results/run_001 1
```

photos.tsv는 머리줄 없이 id와 사진 경로를 탭으로 구분한다. 마지막 인자는 전체 반복 횟수다. 출력 폴더에 measurements.tsv가 있으면 덮어쓰지 않는다.

측정값, 검출 조합, 알림, 읽기 시간과 측정 시간이 measurements.tsv에 저장된다. 초기화 시간과 실행 조건은 run.txt에 기록된다. 준비 측정 10회는 집계에서 제외하고 OpenCV 스레드를 1개로 고정한다. 이 도구는 진단 API를 사용하며 정답을 입력받지 않는다.
