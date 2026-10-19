# data

원본 이미지는 Git 에 올리지 않는다. 받은 zip 을 `data_ingest` 로 넣으면 날짜별 폴더로 정리된다.

## 구조

```
data/
├── README.md
├── manifest.csv         처리한 묶음 목록 (Git 추적)
├── raw/zip/             받은 zip 과 분할 조각(.z01 ...) 그대로        (Git 제외)
├── bmp/<YYYYMMDD>/      원본 BMP. 알고리즘 입력은 항상 여기          (Git 제외)
│   ├── center.bmp
│   ├── Y/
│   └── Theta/
└── png/<YYYYMMDD>/      같은 경로의 무손실 PNG. 보기·공유용          (Git 제외)
```

- `<YYYYMMDD>` 는 묶음을 구분하는 날짜. 기본값은 zip 파일 수정일이고 `--date` 로 바꿀 수 있다.
- zip 안의 맨 위 폴더 하나(예: `SSAFY Test Image Set/`)는 벗기고, 그 아래 구조는 그대로 둔다.
- PNG 는 BMP 와 픽셀이 완전히 같다(무손실). 용량이 약 1/4 이라 보기와 공유에 쓴다.

## 빌드

소스는 `tool/data-ingest/`. 프로젝트 루트에서 [Windows 개발환경 설정](../docs/windows-setup.md)과 같은 설정으로 빌드한다.

```powershell
cmake -S tool/data-ingest -B build/data-ingest -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build/data-ingest --config Release
```

## 사용

프로젝트 루트에서 실행한다.

```powershell
# 받은 zip 을 넣고 처리 (분할 조각 .z01 ... 도 raw/zip/ 으로 같이 복사됨)
.\build\data-ingest\Release\data_ingest.exe "C:/Users/<나>/Downloads/SSAFY Test Image Set.zip"

# 날짜 폴더 이름 지정
.\build\data-ingest\Release\data_ingest.exe "<zip 경로>" --date 20261014

# data/raw/zip/ 에 직접 넣어 둔 새 zip 전부 처리 (이미 처리한 것은 건너뜀)
.\build\data-ingest\Release\data_ingest.exe
```

zip 은 7-Zip(`C:\Program Files\7-Zip\7z.exe`), 반디집(`C:\Program Files\Bandizip\bz.exe`) 순서로 찾아서 푼다.
분할 zip(.z01 ...)은 둘 중 하나가 있어야 하고, 분할이 아닌 zip 은 Windows 기본 `tar` 로도 풀린다.

## 묶음 목록

| 날짜 폴더 | 내용 | 장수 |
|---|---|---|
| `20261012` | SSAFY Test Image Set (center 1, Y 80, Θ 200) | 281 |

자세한 값(zip 이름, sha256, 처리 시각)은 `manifest.csv` 를 본다.
