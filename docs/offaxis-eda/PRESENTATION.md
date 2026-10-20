# 혼합 데이터 EDA 발표 자료

[20장 PowerPoint](SEMES_02_혼합데이터EDA_정리본_v1.pptx)

사용자가 제공한 SEMES_01_데이터EDA_정리본_v4.pptx의 크기·폰트·사진/그래프와 설명 표 구성을 따른 새 데이터 요약이다.
새 60장을 중심으로 기초 정보, 샘플 배치, 기준점 차이, 특징점/ECC, 이동 경로, 반복 변동, 밝기, QR별 차이,
additive residual, 좌표 보류 검증과 19개 좌표 전수 수치를 정리했다. 물리 모델의 실패 결과도 부록에 포함한다.
촬영일은 확인되지 않아 분석일 2026-10-20으로 표시한다.

표와 주요 데이터 차트는 편집 가능한 PowerPoint 객체다. 차트의 숫자는 표시용 유효숫자 10자리로 기록하며,
원래 정밀도의 계측값과 raw residual은 기존 CSV에 그대로 남긴다. 전수 표의 값은 새 기준 3장 평균 대비다.

`brightness.csv`는 추가 산출물이다. ECC 가로/세로 이동량으로 이동한 QR 중심을 정수 픽셀에 반올림하고,
각 204×216 영역의 원본 R 채널 평균과 RGB 채널 일치 여부를 저장한다. 위치별 인쇄 패턴과 테두리 영향이
섞이므로 센서 잡음이나 조도 자체의 독립 계측으로 해석하지 않는다.

```powershell
python tool/offaxis/brightness.py --root "data/SSAFY Test Image Set" --output docs/offaxis-eda
```

PPT의 각 슬라이드 노트에 근거 CSV를 기록했다. 상세 모델 정의와 raw residual은 README.md를 참조한다.
