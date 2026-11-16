# 이전 구현에서 전환

## 현재 기본 구성

현재 루트의 include, src, tools, data/model이 운영 대상이다. legacy는 이전 final00 실험 구현과 모델을 보존한 폴더이며 기본 빌드가 사용하지 않는다.

| 이전 구현 | 현재 구성 |
|---|---|
| initialize_joint, joint 전역 함수 | alignment::initialize, alignment::measure_alignment |
| 별도 AlignmentDiagnostics 조회 | Result에 모든 notices와 correction_allowed 포함 |
| 기본 절대 영점으로 바로 측정 | 운영은 영점 설정 후 측정, 개발용 measure_diagnostic 별도 |
| 고정 QR 탐색창 | 가로 전체 QR 띠에서 개수와 간격 확인 |
| 나사 4개 모두 필요 | 검출 조합별 16개 식, 좌우 확보 여부 판단 |
| model7.tsv와 health.tsv | coefficients와 checks의 조합별 파일 |

SSAFY의 기존 1959장 L2 라벨과 새 관측값으로 모델을 생성했다. SEMES의 계수나 장비 위치를 사용하지 않았다. 기존 S15P31A507 실험 폴더도 변경하지 않았다.

## 학습 범위 정책의 최종 상태

초기 전환에서는 범위 경고가 Review와 보정 금지로 이어졌다. 이후 사용자 요청으로 W_OUT_OF_RANGE만 있을 때 Correct와 보정 허용으로 수정했다. 3.00 mm만 특별 허용하는 예외가 아니라 Y와 T의 학습 범위 초과 전체에 적용한다.

정확도가 떨어질 수 있다는 경고와 원래 측정값을 반환한다. 실제 검출 실패, 영점 미설정, 좌우 정보 부족, 값 일치 점검 초과는 유지한다. 모델 범위를 임의 확대하거나 측정값을 한계값으로 잘라내지 않았다.

## 보존 자료

legacy의 모델, 코드와 data/evidence는 이전 실험 기록이다. 그 성적과 현재 조합 모델의 성적을 혼합하지 않는다. 현재 재현 범위와 알려진 실패 사진은 [검증 결과](05_validation.md)를 따른다.
