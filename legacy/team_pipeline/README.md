# S15P31A507

## Quick Start (Windows x64)

VS Code, CMake, Visual Studio Build Tools 2026(MSVC v141), OpenCV 3.4.15(`C:\opencv`, vc15)가 설치된 환경에서 프로젝트 루트의 PowerShell로 실행합니다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build --config Release
.\build\Release\main.exe "data/SSAFY Test Image Set/center.bmp"
```

처음 설치하는 팀원은 설치 체크리스트부터 진행하세요. 전체 설치 과정, PowerShell 줄바꿈 명령, 빌드 초기화와 문제 해결은 [Windows 개발환경 설정](docs/windows-setup.md)을 참고하세요.


## 문서와 검증 결과

- [문서 목록](docs/README.md)
- [고정 파이프라인 사용법](docs/qr-features.md)
- [고정 파이프라인 검증](docs/qr-fixed-pipeline.md)
- 수동 실행으로 생성한 PNG·CSV는 docs/validation-runs/ 아래에 저장합니다. 저장 경로 예시는 사용법을 참고하세요.
