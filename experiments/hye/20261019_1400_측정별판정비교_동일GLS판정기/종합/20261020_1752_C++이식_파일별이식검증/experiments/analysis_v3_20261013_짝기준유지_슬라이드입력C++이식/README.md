# 슬라이드의 짝 기준·입력·연결 패널 C++ 이식

원본 `analysis_v3/slide_data.py`, `slide_panels.py`는 생성 관측2026-10-13 13:37:34/13:38:31 KST다. 실제 수정 시각과 작성 도구 호출은 [전체 파일 목록](../../전체_파일_목록.md)에 기록한다. 이번 이식 작업자는 현재 채팅의 Codex다.

## 문제·조사·해결

좌·중·우·좌중·좌우·중우·좌중우 QR 가로 위치 중 어느 값을 같은 짝 기준으로 삼는지에 따라 구분 불가 수가 달라진다.7가지 평균 기준과 상대 축20001개 지점을 조사하는 minimax 최적 짝을 C++로 계산한다. v2 세 방법과 v3 세 방법의 기존 판정도 같은 형식으로 모은다. 총14방법이다.

이 원본은 center를 포함한81/201행을 보간에 사용한다. 앞선 `draw_twin_*`의 center 제외 보간과 분리했다. 판정 임계값0.0449/0.0274/0.0478px, 범위 밖 처리, 원점부터 같은 크기의±명령을 모두 구분 못하는 범위와 바깥부터 모두 구분하는 범위, 최적 짝20001개 격자를 유지한다. 명령값 비교 순서가 실제 촬영 시간순이라는 주장은 하지 않는다. 규칙의 단일 원본은 [루트 문서](../../../../../99999.trash/20261022_이전작업_스냅샷/실험%20방법/snapshot_20261022_001/originals/ALGORITHM_RULES.md)다.

## 유지·변경

Python 모듈의 사전·pickle 대신 native C++ 자료구조와 `methods.json`을 사용한다. v2/v3 판정 입력은 이미 원본 수치·판정과 대조한 C++ CSV다. `slide_data.cpp`는 판정3920개·짝·일부 방법의 추정값·범위 요약을 생성한다. 원본14개 패널 PNG의 테두리 색을 읽어 화면에 표시된2240개 판정 플래그를 독립 대조한다. 이는 정확한 선 끝점 소수 좌표까지 PNG에서 복원했다는 의미는 아니다.

패널은 원본과 같은900×900 고정 crop→96×96 캐시→44×44 썸네일, 흰색 혼합0.65, sz44/gap4/mid170, 파란 선과30구간 빨간 점선, Y81/T81 표시를 유지한다. OpenCV/GDI의 글꼴·테두리·샘플 픽셀은 Pillow와 다르다. 원래3948×332 배치 아래28px 설명을 추가해3948×360으로 저장하며 배율 약0.04889배·명령값 보간·T 단위 미확정을 명시한다.

## 실행·검증

2026-10-23 11:00:12/11:01:21 KST의 `../../results/historical_slide_data_20261023_001/checks.json`, `../../results/historical_slide_panels_20261023_001/checks.json`에 근거를 남겼다.

- 원본 PNG14개의 표시 판정2240개와 차이0.
- v1 좌중우 Y33/T29, 최적 짝Y45/T41, G1 Y12/T8, O3+C5 Y1/T1.
- 패널14개 PNG 재디코딩·크기 검사, 원본 사진161개 직접 읽기·썸네일2268개 배치.
- 전체 패널 contact sheet를 눈으로 확인했다. 기존 사진과 결과를 덮어쓰지 않는다.

`build_ppt.py`는 이 입력 외에 더 늦은 `zero_offset_photos.py` 등의 결과도 참조하는 현재 수정본이다. 생성 관측 시각만 보고 이전 실행본이라고 섞지 않고, 미이식 입력을 먼저 해결한 뒤 발표자료를 이어서 이식한다.

```powershell
# final 기준. 각각 새 출력 이름
$bin = './build/experiments/analysis_v3_20261013_짝기준유지_슬라이드입력C++이식/Release'
& "$bin/historical_slide_data.exe" .. results/historical_v2_judge_001/decisions.csv results/historical_v3_evaluate_001/predictions.csv results/슬라이드입력_새이름
& "$bin/historical_slide_panels.exe" .. ../hw-analysis/data/hw/bmp/20261014 results/슬라이드입력_새이름/methods.json results/슬라이드패널_새이름
```
