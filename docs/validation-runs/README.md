# 로컬 검증 실행 결과

자동 생성 PNG·CSV는 이 디렉터리 아래에 저장합니다. 실행별로 하위 폴더를 만들어 사용하며, 생성 결과는 Git에서 제외됩니다.

```powershell
New-Item -ItemType Directory -Force docs/validation-runs/manual | Out-Null
.\build\Release\main.exe "data/SSAFY Test Image Set/center.bmp" --output "docs/validation-runs/manual/result" --no-gui
```

기존 루트의 result_* 파일은 manual/로 이동했습니다. 보고서에서 사용하는 선별된 이미지는 ../qr-stability/images/에 보관하며, 통계와 보고서는 계속 Git으로 관리합니다.