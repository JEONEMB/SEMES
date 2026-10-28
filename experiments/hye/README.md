# 실험 안내

실험 이름 규칙과 평가 기준은 [ALGORITHM_RULES.md](../../ALGORITHM_RULES.md)를 따른다. 기존01/02/03 이름은 파일시스템의 생성 시각을 사용해 시분초로 변경했다. [이름 대응표](directory_rename_20261028.json)에 이전·현재 이름을 보존했다. 스냅샷 안의 당시 경로와 해시는 과거 기록이므로 변경하지 않았다.

- [joint 이식 기록](20261028_132007_joint_QR_상하선_Cpp이식/README.md)
- [과거 미채택 조건부Y 기록](20261028_132007_control_T0_조건부Y_신규모델_미채택/README.md)
- [QR536에 선을 더한 비교 계획](20261028_140538_control_QR536_상하선_비교/PROTOCOL.md)
- [P12와J204 새 학습 비교](20261028_141257_control_P12_vs_J204/README.md)

지정된 전달 경로인 `00000_final_Y_T control`과 `00000_final_Y_T_joint`는 유지한다. 새 비교 결과가 나와도 자동 교체하지 않는다. 빌드에 사용할 모듈은 [selected_modules.cmake](../selected_modules.cmake)에서 경로로 고른다.

2026-10-28 실험 전체를 `experiments/`에서 `experiments/hye/`로 옮겼다. 현재 쓰는 README·CMake의 경로는 고쳤고, `results/`·스냅샷·manifest·log 안의 경로는 당시 기록이라 바꾸지 않았다. 폴더명을 바꾸기 전 생성한 CMake 빌드 폴더에는 이전 경로가 캐시돼 있을 수 있다. 해당 실험을 다시 빌드할 때는 새 빌드 디렉터리로 구성한다. 기존 빌드·결과를 삭제하지 않는다.
