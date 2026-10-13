# S15P31A507

## Quick Start (Windows x64)

VS Code, CMake, Visual Studio Build Tools 2026(MSVC v141), OpenCV 3.4.15(`C:\opencv`, vc15)가 설치된 환경에서 프로젝트 루트의 PowerShell로 실행합니다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build --config Release
.\build\Release\main.exe
```

설치 항목, PowerShell 줄바꿈 명령, 빌드 초기화는 [Windows 개발환경 설정](docs/windows-setup.md)을 참고하세요.
