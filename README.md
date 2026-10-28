# S15P31A507

## 개발 규칙과 모듈 구성

작업 전 [AGENTS.md](AGENTS.md)와 [알고리즘 공통 규칙](ALGORITHM_RULES.md)을 확인합니다. 팀장이 전달한 헤더 선언·CPP 구현 분리 지침과 현재 입출력 계약은 [모듈 구성 지침](docs/team-module-layout.md)에 정리했습니다.

데이터 정제 도구의 빌드·실행은 [tool/README.txt](tool/README.txt), Git 작업은 [Git 컨벤션](docs/gitConvention.md)을 참고하세요.

## Quick Start (Windows x64)

VS Code, CMake, Visual Studio Build Tools 2026(MSVC v141), OpenCV 3.4.15(`C:\opencv`, vc15)가 설치된 환경에서 프로젝트 루트의 PowerShell로 실행합니다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build --config Release
.\build\Release\main.exe joint "data/clean/clean_20261014/center.bmp"
```

처음 설치하는 팀원은 설치 체크리스트부터 진행하세요. 전체 설치 과정, PowerShell 줄바꿈 명령, 빌드 초기화와 문제 해결은 [Windows 개발환경 설정](docs/windows-setup.md)을 참고하세요.

## joint / control 모듈

공개 함수 선언은 [alignment.hpp](include/alignment.hpp)에 있습니다. `Alignment_t`의 y,t는 추정값이며 로봇 명령이 아닙니다.

- [1차 joint 구현·사용법](experiments/hye/00000_final_Y_T_joint/README.md)
- [2차 QR536 control 구현·검증 도구](experiments/hye/00000_final_Y_T%20control/README.md)
- [호출 예제](src/main.cpp): 사진 읽기, 초기화, 모드에 따른 함수 호출

빌드에 쓸 1차·2차 모듈은 [experiments/selected_modules.cmake](experiments/selected_modules.cmake)의 두 경로로 고릅니다. 이 파일만 고치면 다음 빌드에서 해당 모듈과 `assets/`가 사용됩니다.

원래 `alignment(image)`는 `joint(image)`로 연결했습니다. 오류는 예외로 전달하고 추가 진단은 `alignment_diagnostics()`로 읽습니다.

