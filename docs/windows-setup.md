# Windows 개발환경 설정

## 설치

- **VS Code**: C/C++ 확장 설치.
- **CMake**: 설치 시 PATH에 추가. `Visual Studio 18 2026` 생성기를 지원하는 버전 사용.
- **Visual Studio Build Tools 2026**: C++ 빌드 도구, **MSVC v141 (VS 2017) x64/x86 toolset**, Windows SDK 설치. 아래 명령은 `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools` 설치 기준.
- **OpenCV 3.4.15**: `opencv-3.4.15-vc14_vc15.exe`를 실행해 `C:\opencv`에 압축 해제. **x64/vc15** 바이너리 사용.

## 최초 configure → build → run

VS Code에서 프로젝트를 열고, 프로젝트 루트의 PowerShell에서 순서대로 실행합니다. `build`를 삭제한 경우에도 configure부터 다시 실행합니다.

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -T v141 `
  -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" `
  -DOpenCV_DIR="C:/opencv/build"

cmake --build build --config Release
.\build\Release\main.exe
```

PowerShell 줄바꿈 문자(백틱)는 줄 맨 끝에 두고, 뒤에 공백을 넣지 않습니다. 한 줄 명령은 [README](../README.md)의 Quick Start를 참고하세요.

빌드 후 `opencv_world3415.dll`이 실행 파일 옆으로 자동 복사됩니다. 별도 DLL 복사나 OpenCV PATH 등록은 필요 없습니다.

Debug 실행이 필요하면:

```powershell
cmake --build build --config Debug
.\build\Debug\main.exe
```

Debug에서는 `opencv_world3415d.dll`이 자동 복사됩니다.

## build 초기화

실행 중인 프로그램을 종료한 뒤, 프로젝트 루트에서 실행합니다.

```powershell
Remove-Item -Recurse -Force .\build
```

이후 위의 **configure → build → run**을 다시 실행합니다. configure 없이 바로 build하면 `build is not a directory` 오류가 발생합니다.

`main.exe` 또는 DLL이 사용 중이라 삭제되지 않으면, 실행 중인 `main.exe`를 강제 종료한 뒤 다시 삭제합니다.

```powershell
taskkill /F /IM main.exe
Remove-Item -Recurse -Force .\build
```
