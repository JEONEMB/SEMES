# SSAFY 패키지 작업 규칙

상위 AGENTS.md의 문자 규칙을 적용한다. 원본 실험과 다른 알고리즘이나 모델을 임의로 혼합하지 않는다.

완료 전에 다음 명령으로 텍스트를 검사한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/check_text_style.ps1
```

실행 결과는 tools/check_text_style.ps1 -Path data/results로 별도 검사한다.
