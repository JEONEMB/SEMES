01_prepare_clean: 원본 데이터 정리 도구 (C++)
==========================================

1. 역할
-------
data/raw/zip_20261014.zip -> data/clean/clean_20261014
data/raw/zip_20261020.zip -> data/clean/clean_20261020
data/raw/zip_20261029.zip -> data/clean/clean_20261029
data/raw/[교보재] 세메스 데이터 4차.zip -> data/clean/clean_round4

BMP 픽셀이나 파일 바이트를 가공하지 않고 중첩 분할 ZIP을 풉니다.
공통 최상위 포장 폴더 한 개만 제거하고 촬영 폴더/파일명을 보존합니다.
잘 안 보이거나 추정이 어려운 사진도 제외하지 않습니다. 반복 사진도 보존합니다.
20261014: BMP 281장 (center 1, Y 80, T 200)
20261020: BMP 60장 (center 3, 혼합 57), 동봉 TXT 60개
20261029: BMP 234장 (center 없음, 혼합 Y+0.05 x Θ 78자세 x 3장), 동봉 TXT 234개
4차: BMP 144장 (Y 12값 × T 12값, 각 1장), 동봉 TXT 없음

Python이나 과거 실험 폴더 없이 실행됩니다.
사용 라이브러리: C++17, OpenCV 3.4.15, Windows BCrypt(SHA256).
압축 해제는 Windows 기본 tar.exe를 CreateProcessW로 실행합니다.
7-Zip, 반디집, Python 패키지를 추가 설치하지 않습니다.

2. 빌드
-------
프로젝트 루트 S15P31A507의 PowerShell에서 실행합니다.
팀의 기존 개발환경(MSVC v141 / OpenCV vc15)을 그대로 사용합니다.
상위 CMakeLists.txt와 main 프로그램은 수정하지 않았습니다.

cmake -S tool -B build/tool -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build/tool --config Release

결과: build/tool/Release/01_prepare_clean.exe
OpenCV DLL은 빌드 후 실행 파일 옆으로 자동 복사합니다.
cmake 명령을 찾지 못하면 docs/windows-setup.md의 PATH 설정을 확인하세요.

3. 실행
-------
전체 생성 또는 기존 데이터 검증 (4차 ZIP이 제공되면 포함):
  .\build\tool\Release\01_prepare_clean.exe

한 날짜만:
  .\build\tool\Release\01_prepare_clean.exe --date 20261020

4차만:
  .\build\tool\Release\01_prepare_clean.exe --date round4

기존 데이터 검증만 (없으면 오류, 생성/수정하지 않음):
  .\build\tool\Release\01_prepare_clean.exe --verify-only

자체 검사 (SHA256/CRC32 기준값, 경로 거부, 명령값 파싱, CSV):
  .\build\tool\Release\01_prepare_clean.exe --self-test

설명:
  .\build\tool\Release\01_prepare_clean.exe --help

저장소는 실행 파일의 상위 경로에서 먼저 찾습니다. 다음으로 현재 작업
폴더의 상위 경로와 빌드 시 소스 위치를 확인합니다. 다른 저장소를
명시하려면 --repo-root "저장소 경로"를 사용하세요.
exe를 따로 배포할 때는 같은 폴더의 OpenCV DLL도 함께 전달하세요.

검증용 별도 출력 (기본 clean에 영향을 주지 않음):
  .\build\tool\Release\01_prepare_clean.exe --date 20261020 --output-root "C:/temp/clean_test"

4. 코드 처리 순서
-----------------
zipMembers:
  ZIP 중앙 디렉터리를 읽습니다. 경로 이탈/중복 경로/암호화/심볼릭 링크를
  거부합니다. 원본 파일명, 바이트 수, CRC32를 확보합니다.

extractChecked / runTar:
  Windows tar.exe로 새 임시 폴더에 풉니다. 파일 수/경로/크기를 대조합니다.
  셸이나 Python을 거치지 않고 Unicode 경로를 Windows API로 전달합니다.

normalizeSplit:
  .z01, .z02, ..., .zip 순으로 결합합니다. ZIP 내부의 볼륨 번호와 파일
  위치만 단일 ZIP 기준으로 변경합니다. BMP/TXT 압축 내용은 바꾸지 않습니다.
  현재 두 자료에 필요한 ZIP64 locator 정규화도 처리합니다.
  4GB 이상 오프셋/대형 ZIP64 디렉터리/암호화 ZIP은 지원하지 않습니다.

createClean:
  각 원본 멤버의 크기/CRC32를 대조하고 SHA256을 계산합니다. 파일을 원본
  바이트 그대로 옮긴 뒤 해시를 다시 확인합니다. OpenCV로 BMP의 읽기와
  3840x2160 크기를 검사하고 메타데이터를 만듭니다. 검증이 끝난 폴더만
  최종 위치로 옮깁니다. raw ZIP의 해시도 실행 전후 비교합니다.

verifyExisting:
  이미 clean 폴더가 있으면 압축을 다시 풀거나 덮어쓰지 않습니다.
  원본 ZIP 해시, 전체 BMP/TXT 해시/CRC/크기, 영상 디코딩, 사진별 명령
  목록과 동봉 TXT를 대조합니다. 바뀌거나 빠진 파일이 있으면 오류로 끝납니다.

Scratch:
  매 실행마다 고유 임시 폴더를 만들고, 범위를 확인한 그 폴더만 정리합니다.
  기존 비어 있지 않은 clean 폴더를 지우는 옵션은 없습니다.

5. 출력 메타데이터
------------------
_metadata/manifest.csv
  사진 경로/해시/크기, 파일명 Y/T 명령, 반복 번호, 동봉 TXT의 원시값.

_metadata/file_inventory.csv
  BMP/TXT 전체의 원본 ZIP 멤버 이름, 상대 경로, SHA256, CRC32, 바이트 수.

_metadata/provenance.json
  원본 ZIP/분할 조각의 해시, 개수, 정리 방식과 검증 결과.

README.txt
  데이터 구성과 필드 해석 안내.

CSV는 UTF-8 BOM, 안내와 JSON은 UTF-8입니다.
기존 Python판 clean의 메타데이터도 읽어 검증할 수 있습니다.
이전 파일의 matches_previous_evaluation_bytes 열은 추가 정보로 남고,
새 생성 파일에는 matches_source_archive_bytes를 기록합니다.
새 C++ 도구는 외부 평가 자료와 일치한다고 임의로 주장하지 않습니다.

6. 명령과 측정값
----------------
파일명에서 파싱한 filename_Y_command/filename_T_command와 TXT의
sidecar_Y_Actual_raw/sidecar_T_Actual_raw는 서로 다른 필드입니다.
좌표계/원점/단위를 임의로 맞추거나 보정하지 않습니다.
센터 TXT의 원래 .txt.txt 파일명도 그대로 보존합니다.
알고리즘의 공통 규칙은 저장소 루트의 ALGORITHM_RULES.md를 참고하세요.
