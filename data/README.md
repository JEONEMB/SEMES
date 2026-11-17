# data

원본 이미지는 Git 에 올리지 않는다. 받은 zip 을 `data-automation` 으로 넣으면 bmp/, png/ 로 나눠 정리된다.

## 구조

```
data/
├── README.md
├── data-automation.cpp  zip -> bmp/, png/ 정리 프로그램
├── CMakeLists.txt       위 프로그램 빌드용
├── raw/zip/             받은 zip 과 분할 조각(.z01 ...) 그대로        (Git 제외)
├── bmp/<폴더명>/        원본 BMP. 알고리즘 입력은 항상 여기          (Git 제외)
│   ├── center.bmp
│   ├── Y/
│   └── Theta/
└── png/<폴더명>/        같은 경로의 무손실 PNG. 보기·공유용          (Git 제외)
```

- `<폴더명>`: zip 이름에 날짜(`20261014`, `2026-10-14` 등)가 있으면 그 날짜만 `YYYYMMDD` 로 (예: `zip_20261014.zip` → `20261014`), 없으면 실행할 때 입력한 이름.
- SSAFY 에서 받은 원본처럼 zip 안에 또 (분할) zip 이 들어 있으면 그것까지 풀어서 BMP 를 찾는다 (최대 3단계).
- zip 안의 맨 위 폴더 하나(예: `SSAFY Test Image Set/`)는 벗기고, 그 아래 구조는 그대로 둔다.
- PNG 는 BMP 와 픽셀이 완전히 같다(무손실). 용량이 약 1/4 이라 보기와 공유에 쓴다.

## 빌드

프로젝트 루트에서 [Windows 개발환경 설정](../docs/windows-setup.md)과 같은 옵션으로 빌드한다.

```powershell
cmake -S data -B build/data -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build/data --config Release
```

## 사용

프로젝트 루트에서 실행한다.

```powershell
.\build\data\Release\data-automation.exe
```

1. `zip 경로를 입력하세요:` 에 zip 경로를 넣는다. 탐색기의 "경로로 복사" 결과나 창에 끌어다 놓은 경로도 된다.
2. zip 이름에 날짜가 없으면 `폴더명을 입력하세요:` 가 나온다.
3. 끝나면 `data/bmp/<폴더명>/`, `data/png/<폴더명>/` 에 들어가고, zip 은 `data/raw/zip/` 에 보관된다.

zip 경로를 실행 인자로 주면 묻지 않는다: `.\build\data\Release\data-automation.exe "<zip 경로>"`

zip 은 7-Zip, 반디집(bz.exe), Windows 기본 tar 순서로 찾아서 푼다. 분할 zip(.z01 ...)은 7-Zip 이나 반디집이 있어야 한다.
