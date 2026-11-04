# 20261012 QR 이동 미리보기 영상

원본은 `video_output/make_preview.py`다. 관측된 생성·수정 시각은 2026-10-12 15:49:11.620 KST다. 해당 시각만으로 원본 작성자를 배정하지 않는다. C++ 이식은 현재 채팅의 Codex가 수행했다. 원본은 제자리에 유지한다.

## 문제·조사·해결

center와 단일축 사진에서 QR 세 개가 움직이는 양을 영상으로 표시한다. 원본은 세 QR의 110×110 템플릿을 310×310 창에서 NCC로 찾고, 정수 픽셀 이동과 파일명 명령값을 함께 표시했다. 이 측정은 초기 미리보기용이며 630점 경계 측정이나 Y/T 예측 모델과 구별한다.

C++에서도 같은 사진·템플릿·탐색 창·0.85 점수 기준으로 이동량을 계산한다. 추적 실패는 성공한 프레임으로 대체하지 않고 실행 오류로 남긴다.

## 유지한 부분

1920×1080 화면, 원본 크기의 1400×840 QR 영역, center 기준선, 현재 위치, 정수 이동 표시, 전체 사진 미리보기와 색상을 유지했다. T −1.00~+1.00 후 Y −2.00~+2.00 순서이며 두 center는 각각 10프레임이다. 총 300프레임·20fps·15초다.

## 바꾼 부분

Python·ffmpeg 프로세스 대신 C++와 Windows Media Foundation으로 H.264 MP4를 기록한다. 원본 libx264 CRF22와 새 8Mbit/s 인코딩은 압축 결과가 다르다. 하단에 명령값 순서라는 설명을 추가하여 실제 촬영 시간 순 로봇 궤적으로 오해하지 않도록 했다.

Media Foundation 호출 순서는 [Microsoft Sink Writer 예제](https://learn.microsoft.com/en-us/windows/win32/medfound/tutorial--using-the-sink-writer-to-encode-video)를 참고했다. 새 출력 폴더만 허용하며 원본 영상은 덮어쓰지 않는다.

## 실행과 확인

2026-10-22 17:31:00.426 KST의 `../../results/historical_video_preview_001/checks.json`에서 별도 OpenCV 디코더가 300프레임·20fps·1920×1080을 확인했다. 첫 프레임의 압축 전후 채널 평균 절대차는 2.054/255다. 디코딩한 첫 화면과 구간별 PNG의 배치를 확인했다. 원본 영상과의 픽셀 동일성은 주장하지 않는다.

`frames.csv`에는 프레임별 파일명·명령값·시간·세 QR의 dx/dy/점수를 기록한다. 이는 재생 시간이며 원래 촬영 시각이 아니다.

```powershell
# final 기준. 새 출력 폴더 사용
.\build\experiments\analysis_v1_20261012_영상순서유지_H264C++변경\Release\historical_video_preview.exe ../hw-analysis/data/hw/bmp/20261014 results/video_새이름
```

DATA와 원본을 이동하지 않으며 `legacy`에도 추가하지 않는다.
