# Native — 알고리즘 실행기 (임시)

Unity '알고리즘 확인' 화면(GitLab dev `verify_web` 을 옮긴 것)이 띄우는 알고리즘 실행기.

| 파일 | 출처 |
| --- | --- |
| `ksh_alignment_1101/` | GitLab dev `legacy/KSH/Alignment` (김성현, 2026-11-01 커밋 929b8e3) 그대로 복사. 받은 시점 dev = bae0526 |
| `align_worker_verify_web_original.cpp` | GitLab dev `verify_web/native/align_worker.cpp` 원본 (참고용) |
| `align_worker.cpp` | 위 원본을 Windows·11-01 API 에 맞춘 것 |
| `CMakeLists.txt` | 실행기 빌드 |

- 용도: 작동 여부 시험용 임시. verify_web 이 실제로 쓰는 김성현 A507 버전(라즈베리파이 `/home/pi/KSH_Algorithm/Code`)은 GitLab 에 없어 따로 받아야 함.
- 11-01 버전과 A507 버전 차이: `alignment_reference_info()` 없음, 폴더 구조(`ksh/` vs `src/modules/`) 다름.
- 상태: OpenCV 3.4.15 미설치로 **아직 빌드·실행 안 함**. 실행기가 없으면 Unity 는 verify_web 과 같은 모의 추론(실제 오차 + 잡음)을 쓴다.
- 입력 사진: 장비 카메라 연결 전에는 Unity 가상 카메라(SEMES 화각) 화면 3840×2160 흑백. 연결 후에는 실제 카메라 화면.
