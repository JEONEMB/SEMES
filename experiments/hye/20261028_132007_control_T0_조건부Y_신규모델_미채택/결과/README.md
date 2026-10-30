# 2026-10-28 / 02 / T=0 조건부Y 신규 모델 / 미채택

**이 모델은 사용자가 지정한 ‘초기에100%가 나온 기존 모델’이 아니다.**
기존 `control()` 전달 대상으로 잘못 연결했던 신규 실험이며, 비교·추적용으로 보존한다.
초기100% 모델의 확인·연결은 별도 작업으로 남아 있다.

## 보존 내용

아래 `snapshot/`과 `snapshot_manifest.json`은 [실험 방법](../실험) 폴더에 있다.

- `snapshot/module/src/control.cpp`: 신규 조건부Y 추론과 추정T 게이트.
- `snapshot/module/tool/train_control.cpp`: T=0 사진81장으로 새 계수를 만든 C++ 도구.
- `snapshot/module/assets/control_model.yml`: 그 도구가 생성한 파라미터.
- `snapshot/module/README.md`, `CMakeLists.txt`: 당시 설명과 빌드 연결.
- `snapshot/repository_context/`: 공개 헤더·호출부·검사 도구 및 당시 joint/control 재생 결과.
- `snapshot_manifest.json`: 원본 대응과 SHA-256.

12개 파일의 사본을 검증했다. 공동 관측 구현과 기준 영상 등 당시 공통 자료는
[01 joint 이식 기록](../../20261028_132007_joint_QR_상하선_Cpp이식/결과/README.md)에 보존했다.

## 결과의 의미

당시 T=0인84장 중Y0.09 통과84/84는81장 학습사진과 이미 노출된센터3장을 포함한 개발 재생 결과다.
이는 초기100% 모델의 성적도, 신규 독립 평가도, 실제 두 단계 정렬 성공률도 아니다.

기존 전달 폴더를 지우거나 빌드 경로를 끊지 않았다. 해당 위치의 구현은 아직 이 신규 모델이므로
‘기존100% control 연결 완료’로 보고하거나 전달하지 않는다.
스냅샷 원문에 있는 상대 링크·빌드 경로는 원래 저장소 위치를 기준으로 한다.

현재 채택 기준은 [ALGORITHM_RULES.md](../../../../ALGORITHM_RULES.md)를 따른다.
