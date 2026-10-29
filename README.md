> 2026-10-30 상태: 아래 C++는 29일 θ→Y 순차 보정 작성본입니다. 341장 점검에서 명령 불가 311장이 관측됐으며 최종 정렬 성공을 입증하지 못했습니다. 현재 목표는 이미지 1장의 보정량으로 한 번에 Y ±0.045mm·θ ±0.005°에 도달하는 것입니다. 20회 제한은 코드 범위 밖으로 변경됐지만 기존 코드에는 남아 있습니다. 상·하단 전체 구조물 특징으로의 교체와 새 요구 반영은 미수행입니다.

# S15P31A507

## Quick Start (Windows x64)

VS Code, CMake, Visual Studio Build Tools 2026(MSVC v141), OpenCV 3.4.15(`C:\opencv`, vc15)가 설치된 환경에서 프로젝트 루트의 PowerShell로 실행합니다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools" -DOpenCV_DIR="C:/opencv/build"
cmake --build build --config Release
# Reference must be the first-data center.bmp. Input is one current image.
.\build\Release\main.exe --reference "C:\data\center.bmp" --config configs/theta_alignment.json --y-config configs/y_alignment.json --output result.json "C:\data\current.bmp"
ctest --test-dir build -C Release --output-on-failure
```

처음 설치하는 팀원은 설치 체크리스트부터 진행하세요. 전체 설치 과정, PowerShell 줄바꿈 명령, 빌드 초기화와 문제 해결은 [Windows 개발환경 설정](docs/windows-setup.md)을 참고하세요.


## LSJ 순차 정렬

`dev`의 `Alignment_t alignment(cv::Mat&)` 형식을 유지한 θ→Y 순차 보정입니다. Y는 QR별 기존 목표 범위와 촬영 구간별 기울기를 사용합니다. 초기화, 실제 이동 완료 통지와 20회 제한, 반환 상태, 검증 범위는 [Y·순차 정렬 구현 문서](experiments/LSJ/y_controller/README.md)를 확인하세요. `--y-config`를 생략하면 기존 θ 단독 실행입니다.
