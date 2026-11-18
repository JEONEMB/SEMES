# Y/T 대응 짝 연결 그림의 C++ 이식

원본은 `analysis_v3/draw_twin_lines.py`(생성 관측2026-10-13 13:30:36 KST), `draw_twin_photos.py`(13:34:13 KST)다. 실제 수정 시각·해시·작업 도구 쓰기 호출은 [전체 파일 목록](../../전체_파일_목록.md)에 구분한다. 이번 C++ 작업자는 현재 채팅의 Codex다.

## 문제·조사·해결

QR 가로 평균 위치가 같은 Y/T 사진이 과거 판정에서 구분되지 않는 범위를 보여준다. 사진 순서가 아니라 명령값 위치와 보간으로 만든 대응 관계다. 원본의 v1 판정용 임시 `Yo.pkl`, `To.pkl`은 현재 없어서, 기존 `22_YΘ구분표_v1.xlsx`의 판정을 읽고 이미 이식한 C++ v1 규칙1의280개 판정과 모두 같은지 확인했다.

원본 그림의 좌표는 `judge.pkl`에서 만들던280장 합친 표를 CSV로 재구성한다. 이 표에는 center가 없지만 v1 Excel의 보간은 center를 포함한다. 따라서 원점을 가로지르는2행은 Excel과 최대0.0126263 차이가 있고, 나머지 보간은 최대4.44e-16 차이다. **그림의 원래 center 제외 보간을 유지**하며, Excel의 보간값으로 대체하지 않는다. 판정 플래그는 전부 같다. v2-a2는 기존 C++ 규칙2와1960개 원본 판정 대조가 끝난 입력을 사용한다.

## 유지·변경

Y80장/T200장 명령 격자와 center 표시, 평균 QR 가로 이동 보간, v1·v2-a2 플래그, 파란 실선Y→T/빨간 점선T→Y 의미를 유지했다. 사진 그림은 동일한 C QR 중심 주변900×900 영역, area 축소, 구분 가능 사진의 흰색 혼합0.6, 굵은 테두리, T±0.4 및±1.0 두 표시 범위를 유지한다.

Matplotlib/Pillow 대신 C++ OpenCV3.4.15·Windows GDI로 그린다. 폰트 위치·안티앨리어싱·미리보기 크기·테두리 렌더링은 바뀌며 픽셀 단위 같은 그림이라고 주장하지 않는다. 현재 규칙에 따라 축소 배율·T 단위 미확정·명령값 보간임을 그림에 표시한다. 당시의 ‘QR 최고’ 표현은 ‘과거 비교’로 명시했다. 알고리즘 기준은 [루트 문서](../../../../../99999.trash/20261022_이전작업_스냅샷/실험%20방법/snapshot_20261022_001/originals/ALGORITHM_RULES.md)다.

## 실행·확인

2026-10-23 10:45:57/10:46:36 KST 실행 근거는 각각 `../../results/historical_twin_lines_20261023_001/checks.json`, `../../results/historical_twin_photos_20261023_001/checks.json`이다.

- v1 짝 수Y33/T29, v2-a2 Y15/T16.
- 연결도2640×1560, 사진 확대도13242×2169, 전체도13989×1879. 저장 PNG를 다시 디코딩해 크기를 검사했다.
- 사진281개를 직접 읽어 두 패널에324개/564개 썸네일을 배치했다. 원본의 사진·center 중복 표시 수와 같다.
- 판정·보간 대조와 별도로 연결도 및 사진 확대도 미리보기를 눈으로 확인했다. 생성 결과는 새 폴더에만 저장하며 원본·DATA·legacy를 수정하지 않는다.

```powershell
# final 기준. 출력 폴더는 새 이름
$bin = './build/experiments/analysis_v3_20261013_대응짝유지_그림렌더러C++변경/Release'
& "$bin/historical_twin_lines.exe" .. ../hw-analysis/data/hw/bmp/20261014 results/historical_v2_judge_001/decisions.csv results/짝연결도_새이름
& "$bin/historical_twin_photos.exe" .. ../hw-analysis/data/hw/bmp/20261014 results/historical_v2_judge_001/decisions.csv results/사진연결도_새이름
```
