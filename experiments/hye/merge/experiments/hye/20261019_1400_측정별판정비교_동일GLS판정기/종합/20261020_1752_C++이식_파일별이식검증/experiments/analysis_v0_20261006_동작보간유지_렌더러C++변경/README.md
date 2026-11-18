# 20261006 로봇 동작 개념 애니메이션

원본은 루트의 `make_robot_motion.py`다. 관측된 생성·수정 시각은 모두 2026-10-06 17:04:35.534 KST다. 파일 시각만으로 원본 작성자를 확정하지 않는다. C++ 이식은 이 채팅의 Codex가 수행했다.

## 문제와 방법

실제 기구 구조가 확인되지 않은 상태에서 접근, QR 촬영, Y/Θ 보정, 진입의 네 단계를 개념도로 설명하는 코드다. 측정 또는 평가 모델이 아니다. 원본의 미확인 구조에 관한 설명과 ‘추정’ 표기를 유지했다.

## 유지한 부분

960×560 화면, 논리 프레임 70개, 프레임당 100ms, 총 7초, 네 단계의 전환 시각과 보간식, 로봇·선반·QR·카메라 위치와 한글 문구를 유지했다. 수치상 동작은 `frames.csv`로 따로 기록한다.

## 바꾼 부분

Pillow 대신 C++ OpenCV와 Windows GDI로 도형과 한글을 그린다. GIF는 Windows WIC로 기록한다. 고정 256색 팔레트를 사용하며, Pillow의 팔레트 최적화·동일 프레임 병합·픽셀 단위 래스터 결과와는 다르다. 새 출력은 지정한 새 폴더에만 생성한다.

WIC의 프레임 지연 값은 [Microsoft 메타데이터 사양](https://learn.microsoft.com/en-us/windows/win32/wic/-wic-native-image-format-metadata-queries)의 `/grctlext/Delay`를 사용한다. 픽셀을 쓰기 전에 지연 정보를 설정해야 실제 출력에 반영되는 것을 실행 후 디코딩으로 확인했다.

## 검증

2026-10-22 17:19:22 KST 실행의 `../../results/historical_robot_motion_002/checks.json`에서 독립 WIC 디코더가 70프레임과 7,000ms를 확인했다. 단계별 PNG도 별도로 저장하고 보정 단계의 한글·도형 배치를 확인했다. `historical_robot_motion_001`은 지연 메타데이터 검증에 실패한 초기 실행이며 덮어쓰지 않았다.

```powershell
# final 기준
.\build\experiments\analysis_v0_20261006_동작보간유지_렌더러C++변경\Release\historical_robot_motion.exe results/robot_새이름
```

원본은 현재 위치에 유지하며 `legacy`에 추가하거나 이동하지 않는다.
