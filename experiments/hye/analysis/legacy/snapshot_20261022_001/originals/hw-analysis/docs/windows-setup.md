# Windows 개발환경 설정

## 요약

- [ ] VS Code와 Git for Windows 설치
- [ ] Visual Studio Build Tools 2026: C++ 데스크톱 개발, Windows SDK, MSVC v141, CMake 도구 설치
- [ ] CMake 사용자 PATH 등록 후 새 PowerShell/VS Code에서 확인
- [ ] 회사와 동일한 OpenCV 3.4.15 설치 파일을 `C:\opencv`에 압축 해제하고 파일 확인
- [ ] VS Code C/C++ 및 CMake Tools 확장 설치
- [ ] 저장소 clone 후 프로젝트 루트에서 Configure → Release 빌드 → 실행

모든 명령은 **PowerShell 기준**입니다. 아래 경로는 팀에서 확인한 설치 위치입니다. 설치 위치가 다르면 PATH와 Configure의 경로도 함께 변경하세요.

## 설치

### 1. VS Code와 Git

[VS Code](https://code.visualstudio.com/download)의 Windows 설치 파일을 실행합니다. 설치 옵션에서 PATH 추가를 선택하면 터미널에서 `code .`로 프로젝트를 열 수 있습니다.

아무것도 설치되지 않은 PC에서 저장소를 clone하려면 [Git for Windows](https://git-scm.com/downloads/win)도 설치합니다. 설치 후 새 PowerShell에서 `git --version`으로 확인하세요.

### 2. Visual Studio Build Tools 2026

[Visual Studio 다운로드](https://visualstudio.microsoft.com/downloads/)에서 **Build Tools for Visual Studio 2026**을 받아 Visual Studio Installer를 실행합니다.

1. **C++를 사용한 데스크톱 개발** 워크로드를 선택합니다.
2. 설치 세부 정보/개별 구성 요소에서 다음을 확인합니다.
   - MSVC C++ x64/x86 빌드 도구와 MSBuild
   - Windows 10 또는 Windows 11 SDK (Windows x64 개발용)
   - **MSVC v141 - VS 2017 C++ x64/x86 빌드 도구 (v14.16)**
   - **Windows용 C++ CMake 도구** (Visual Studio CMake 도구)
3. 팀 기준 설치 경로는 `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools`입니다.

코드 편집용 IDE는 **VS Code**, 실제 C++ 컴파일러는 Build Tools에 포함된 **MSVC**입니다. 이 프로젝트는 OpenCV의 `vc15` 바이너리에 맞춰 **v141(VS 2017 호환)** 도구 집합을 명시합니다. 최신 MSVC만 설치하지 말고 v141도 선택하세요. v14.16 도구 집합의 컴파일러 버전은 MSVC 19.16 계열입니다.

### 3. CMake 확인 및 PATH 등록

먼저 PowerShell에서 확인합니다.

```powershell
cmake --version
where.exe cmake
```

명령을 찾지 못하면 Build Tools에 포함된 파일을 확인합니다.

```powershell
Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
```

`True`이면 설치되어 있고 PATH 등록이 필요합니다. Windows 검색에서 **환경 변수 편집 → 계정의 환경 변수 → 사용자 변수의 Path → 편집 → 새로 만들기**로 다음 **폴더**를 추가합니다.

```text
C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin
```

기존 PowerShell과 VS Code를 완전히 닫고 다시 열어 `cmake --version`, `where.exe cmake`를 재확인합니다. `where.exe cmake`에 위 경로의 실행 파일이 나와야 합니다. 여러 CMake가 나오면 사용할 경로를 사용자 PATH에서 우선 배치하세요.

파일 확인이 `False`이면 Installer의 CMake 도구 선택과 설치 경로를 확인합니다. 별도 [CMake](https://cmake.org/download/)를 설치해도 되지만, **Visual Studio 18 2026 생성기를 지원하는 CMake 4.2 이상**이 필요합니다. 지원 여부는 `cmake --help`의 Generators 목록으로 확인할 수 있습니다.

### 4. OpenCV 3.4.15

회사 환경과 동일한 **`opencv-3.4.15-vc14_vc15.exe`**를 사용합니다. 실행 후 최종 폴더 구조가 `C:\opencv\build`가 되도록 압축을 해제합니다. 중복으로 `C:\opencv\opencv`가 만들어지지 않게 확인하세요. 이 프로젝트는 **x64/vc15** 바이너리를 사용합니다.

```powershell
Test-Path "C:\opencv\build\x64\vc15\bin\opencv_world3415.dll"
Test-Path "C:\opencv\build\x64\vc15\lib\opencv_world3415.lib"
```

두 결과가 모두 `True`여야 합니다. Debug 빌드에는 같은 위치의 `opencv_world3415d.dll`과 `opencv_world3415d.lib`도 필요합니다.

**OpenCV bin은 PATH에 등록하지 않아도 됩니다.** 현재 `CMakeLists.txt`의 빌드 후 복사 설정이 구성에 맞는 DLL을 exe 옆에 자동 복사합니다.

| 구성 | 실행 파일 | 자동 복사되는 DLL |
| --- | --- | --- |
| Release | `build/Release/main.exe` | `opencv_world3415.dll` |
| Debug | `build/Debug/main.exe` | `opencv_world3415d.dll` |

### 5. VS Code 권장 확장

VS Code의 확장 메뉴에서 설치합니다.

- Microsoft C/C++: `ms-vscode.cpptools`
- Microsoft CMake Tools: `ms-vscode.cmake-tools`

확장은 편집·빌드 작업을 돕는 도구이며 **컴파일러, CMake, OpenCV 자체를 대신 설치하지 않습니다.**

## 최초 configure → build → run

저장소의 Clone 메뉴에서 실제 주소를 복사한 다음 실행합니다. 아래 URL 자리표시자는 실제 저장소 주소로 바꾸세요.

```powershell
git clone <저장소-Clone-URL>
Set-Location .\S15P31A507
code .
```

`CMakeLists.txt`가 있는 **프로젝트 루트**의 PowerShell에서 최초 Configure를 실행합니다.

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -T v141 `
  -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" `
  -DOpenCV_DIR="C:/opencv/build"
```

PowerShell 줄바꿈 문자(백틱)는 줄 맨 끝에 두고 뒤에 공백을 넣지 않습니다. 한 줄 명령은 [README](../README.md)의 Quick Start를 참고하세요.

성공 시 로그에서 다음을 확인합니다.

- C++ 컴파일러: **MSVC 19.16 계열**
- `OpenCV ARCH: x64`, `OpenCV RUNTIME: vc15`
- OpenCV **3.4.15** 발견 (`Found OpenCV`)
- `Configuring done`, `Generating done`

이어서 Release 빌드와 결과 확인 후 실행합니다.

```powershell
cmake --build build --config Release
Test-Path .\build\Release\main.exe
Test-Path .\build\Release\opencv_world3415.dll
.\build\Release\main.exe
```

두 파일 확인 결과가 모두 `True`여야 합니다. 별도 DLL 복사나 OpenCV PATH 등록은 필요 없습니다.

Debug 실행이 필요하면:

```powershell
cmake --build build --config Debug
.\build\Debug\main.exe
```

Debug에서는 `opencv_world3415d.dll`이 자동 복사됩니다.

## build 초기화

프로젝트 루트인지 확인하고 실행 중인 프로그램을 종료한 뒤 실행합니다. 다음 명령은 이 프로젝트의 생성된 `build` 폴더만 삭제합니다.

```powershell
Remove-Item -Recurse -Force build
```

`main.exe` 또는 DLL이 사용 중이라 삭제되지 않으면 아래 순서로 처리합니다. `taskkill`은 같은 이름으로 실행 중인 모든 `main.exe`를 강제 종료하므로 해당 프로그램인지 확인하세요.

```powershell
taskkill /F /IM main.exe
Remove-Item -Recurse -Force build
```

삭제 후 반드시 위의 **Configure → build → run**을 다시 실행합니다. Configure 없이 바로 `cmake --build build`를 실행하면 `build is not a directory` 오류가 발생합니다.

## 문제 해결

| 증상 | 확인 및 해결 |
| --- | --- |
| `cmake` 명령을 찾지 못함 | CMake 파일 존재 확인 → 사용자 PATH에 bin 폴더 추가 → PowerShell/VS Code 재시작 → `cmake --version`, `where.exe cmake` 확인 |
| `Visual Studio 18 2026` 생성기를 찾지 못함 | `cmake --help` 확인. 오래된 CMake가 PATH에서 먼저 선택되었는지 확인하고 지원 버전 사용 |
| OpenCV vc15와 컴파일러 호환/검색 문제 | x64/vc15 설치 파일과 v141(v14.16) 설치 확인. `-A x64 -T v141`, `OpenCV_DIR=C:/opencv/build`로 build 초기화 후 Configure 재실행 |
| 실행 시 OpenCV DLL 누락 | 구성별 DLL 이름과 exe 옆 파일 확인. 원본 vc15/bin에 파일이 있는지 확인하고 해당 구성으로 다시 빌드. Debug DLL과 Release DLL을 이름만 바꿔 사용하지 않기 |
| main.exe 사용 중이라 build 삭제 실패 | 해당 프로그램 종료 또는 `taskkill /F /IM main.exe` 후 다시 삭제 |
| `build is not a directory` | build 삭제 후 Configure를 생략한 상태. 최초 Configure부터 다시 실행 |

생성기 및 인스턴스 선택 참고: [CMake Visual Studio 18 2026 문서](https://cmake.org/cmake/help/latest/generator/Visual%20Studio%2018%202026.html).
