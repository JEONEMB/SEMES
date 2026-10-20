# Y–theta 새 데이터 분석 (진행 중)

2026-10-20. 새 이미지 복사 완료 후 전체 341장 확인: 기존 281장,
혼합 좌표 19곳 × 반복 3장 = 57장, 새 기준 반복 3장.
좌표/파일명/sidecar 값/이미지 크기/SHA256은 sampling.csv에 보존한다.
파일명은 명목값이며 sidecar의 Y_Actual/T_Actual도 독립 외부 실측 정답으로 인증된 것은 아니다.
θ 단위는 기존 PPT처럼 도로 가정하며 별도 기구 로그의 확인이 필요하다.

## 재현

```powershell
python -m pip install -r tool/offaxis/requirements.txt
python tool/offaxis/inventory.py --root "data/SSAFY Test Image Set" --output docs/offaxis-eda
```

원본 이미지를 변경하거나 필터링하지 않는다. 파일 수정 시각은 촬영 시각으로 간주하지 않는다.
새 기준 3장의 실제 기록 원점을 별도로 유지한다. 기존 center와 새 기준 차이를 raw 값과 함께 보고한다.
θ=2.00은 기존 단독 θ 범위 ±1.00 밖이므로 additive baseline의 외삽 진단과 구분한다.

## 기준 자료와 방법 차이

첨부 SEMES_01_데이터EDA_정리본_v4.pptx 24장을 읽었다.
4쪽은 전체 모서리 755점, 대각선 제외 630점(L212/C210/R208)을 사용한다.
5쪽 ECC는 QR 라벨+8px의 204×216 패치, template 위치 탐색 후 affine ECC이다.
17쪽은 2×2 밝기 경계 모델 fitting 진단을 설명한다.
630점은 진단 통과 여부와 무관하게 고정하며 탈락 마스크 결과를 별도로 비교한다.
본문 9쪽과 부록 20쪽의 방법 간 한 칸 차이 수치는 집계 정의가 달라 재측정 결과와 별도 취급한다.

저장소 조사 기준 커밋은 ae9a3de (feature/shin/y-thetaEstimation)이다.
src/qr_features.cpp, include/qr_refinement.hpp, docs/qr-subpixel-comparison/README.md,
docs/qr-stability/scripts/evaluate.cpp와 tool/qr-subpixel/validate.cpp를 참고했다.
기존 코드는 checkerboard X 교점 중심이므로 PPT의 비대각선 L 모서리 630점과 같은 집합이 아니다.
기존 plate/decoder Gaussian 경로를 이번 원본 계측의 무필터 경로로 부르지 않는다.
기존 ECC 표기의 대부분은 QR error correction이며 영상 정합 ECC 구현은 검색에서 발견하지 못했다.

참고 원문: [Fourier corner 논문](https://www.tnt.uni-hannover.de/papers/data/1335/main.pdf),
[OpenCV ECC API](https://docs.opencv.org/doc/doxygen/html/dc/d6b/group__video__track.html).
Fourier X 교점 대칭성 목적함수를 L 모서리에 그대로 적용하지 않는다.
ECC의 gaussFiltSize는 1로 지정하여 기본 Gaussian 5×5를 사용하지 않는다.

## Git

docs/gitConvention.md를 Git 작업 전에 확인했다. 실제 통합/원격 기본 브랜치는 dev이며
문서의 develop은 존재하지 않는다. 실제 통합 브랜치 dev에서 feature/offaxis-eda를 생성했다.
참고 조사한 기능 브랜치의 운영 코드를 통합 브랜치에 임의로 옮기지 않고 분석 도구를 독립 추가한다.
push 시에는 사용자 지정 Jira 생성·활성 sprint·Fibonacci SP 절차를 먼저 수행해야 한다.
