# QR 자세 보정 납품·실증 결과

- [최신 프로젝트 전체 보고서와 C++14 포팅 가이드](qr-pose-correction-and-porting.pdf)
- [C++14 실행기·빌드·연동 안내](../../../experiments/shin/qr_only_multiple_exe/native/README.md)

아래 문서는 초기 Python 실행과 순차 재생의 검증 기록이다. C++14 포팅판의 실행 명령과 지원 환경은 위 가이드에 있다.

- [결과와 한계 보고서 PDF](ssafy-rig-qr-delivery-validation.pdf)
- [SSAFY 납품용 실행과 외부 목표 설정](ssafy-delivery.md)
- [Rig 실행과 라즈베리 파이 연동](rig-demo.md)
- [실행기·외부 설정·캘리브레이션 모델](../../../experiments/shin/qr_only_multiple_exe/README.md)

## 보류 검증

SSAFY 혼합 자세 287장 중 216장(75.3%)에서 Y ±0.045 mm, θ ±0.01°를 동시에 만족했다. 평균 오차 ± 표준편차는 Y +0.0012 ± 0.0434 mm, θ -0.0002 ± 0.0078°였다. Windows 프레임 추론 중앙값은 6.09 ms였다.

Rig 정·역방향 재생에서 Y = 0인 QR 응답을 사용하면 3,360개 시작점에서 재생 θ 기준에 최대 6회로 도달했다. 재생 θ 오차는 +0.00010 ± 0.00130°였다. 실제 Y와 두 축의 동시 복귀율은 미계측이다. 실기에서는 첫 Y 이동 직후와 최종 θ 보정 직후에 Y를 각각 재고 최종 θ도 독립 계측한다.

| 검증 자료 | 내용 |
| --- | --- |
| [SSAFY 정확도](validation/ssafy-accuracy.json), [프레임별 예측](validation/ssafy-grouped-and-y-level-predictions.csv) | 동일 자세 그룹 보류와 혼합 자세 Y 값 전체 보류 |
| [SSAFY 추론 시간](validation/ssafy-timing.json) | 287장, Windows 단일 스레드 |
| [Rig 혼합 자세](validation/rig-qr-pose-runtime-metrics.json) | θ 구간, Y 구간, 촬영 방향 보류 |
| [Rig θ 재생](validation/sequential-theta-replay.json) | 정·역방향의 QR 위치 편향 민감도와 모의 θ 오차 |

## Jira

[핵심 알고리즘 구현 및 검증 에픽](https://ssafy.atlassian.net/browse/S15P31A507-2) · [납품·실증 스토리](https://ssafy.atlassian.net/browse/S15P31A507-1211)

- [SSAFY 실행기 및 외부 목표 설정](https://ssafy.atlassian.net/browse/S15P31A507-1214)
- [Rig 순차 보정과 스테이지 연동](https://ssafy.atlassian.net/browse/S15P31A507-1212)
- [오프라인 평가와 보고서](https://ssafy.atlassian.net/browse/S15P31A507-1213)
- [SSAFY 수신측 검증](https://ssafy.atlassian.net/browse/S15P31A507-1215)
- [Rig 실기 독립 계측](https://ssafy.atlassian.net/browse/S15P31A507-1216)
