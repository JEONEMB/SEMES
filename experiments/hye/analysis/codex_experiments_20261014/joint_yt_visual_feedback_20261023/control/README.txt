실제 로봇 복귀 시험이 아닌 가정 시뮬레이션입니다.

정적 사진341장으로부터 학습한 경험적 영상 반응식을 사용했습니다. 초기관측도
실제 사진을 다시 넣은 것이 아니라, 기존341개 명령값을 초기상태로 삼아 별도의
plant가 생성한 특징입니다. 연속 촬영, 모터 이동, 정착 시간, 물리적 복귀오차를
측정하지 않았습니다. 모든341장은 이미 개발에 노출됐습니다.

API
  calibration = joblib.load('full_calibration_both.joblib')['calibration']
  controller = calibration.controller('fixed_jacobian')
  controller.start(calibration.project(raw_feature), zeros(12))
  u = controller.next_action()  # 제안값만 반환; 하드웨어 호출 없음
  controller.observe(calibration.project(next_raw_feature), executed_relative_u)

controller.py 온라인 메서드에는 현재 절대Y/T, 파일명, 시행 번호가 없습니다.
오프라인 Calibration.fit은 학습 사진과 학습 명령을 사용합니다. 조건별 full341
교정4개는 추론용 학습 산출물이며, OOF 성적이나 독립 검증이 아닙니다.

관측조건은 QR만54D, QR+위78D, QR+아래78D, QR+양쪽102D입니다.
학습 fold 안에서 선형/이차 반응5방향과 반복잡음7방향을12D로 투영하고 잡음
공분산으로 정규화합니다. 주제어기는 학습한 선형Jacobian을 사용합니다.
Plant는 분리된 heldout 사진/명령으로 별도 full quadratic을 학습합니다.
동일 학습계수를 정방향·역방향에 공유하지 않습니다. 그러나 이는 물리적으로
검증된 simulator가 아닙니다. 반복잡음은20261020의 실제 반복조건 잔차이며,
날짜간 차이와 전체 모델 잔차를 모두 재현한 잡음이라고 가정하지 않습니다.

주 실험은 fixed Jacobian / 작은2probe+Broyden / 작은3probe+Broyden입니다.
probe는(.05,0),(0,.01), 세 번째(-.05,-.01). 최대단계(.25,.25), 최대20회.
초기사진0턴이고 probe와 0이동 확인 촬영도 예산에 포함합니다. 실제 로봇에서
허용된 이동크기나 검증된 안전한 한계라는 뜻이 아닙니다. T 물리단위는 미확인.

시나리오4개
1. 이상조건: plant 절편을 강제로0, 잡음·구동오차 없음.
2. 절편0, heldout 반복사진 잔차를 잡음으로 사용.
3. plant가 별도로 학습한 절편 유지, 구동행렬[[.85,.08],[-.04,1.12]], 반복잡음.
4. 3번에 프레임당 정규화영상 RMS .05 drift, 반전 시 [.02,.005] backlash 추가.
3/4번의 절편도 실제 원점을 따로 계측한 값이 아니라 경험식의 적합 절편입니다.

중단은 영상 RMS<=2.5와 추정국소보정량<=.03이 연속2회 충족될 때입니다.
평가자는 별도로 실제 시뮬레이션 상태 |Y|,|T|<=.09를 검사합니다. 둘은 다르며
false_visual_stops를 숨기지 않습니다. 최대20회에서 기준 미달도 분모에 남습니다.
단계 중 heldout 명령 convex hull 밖에 나가면 extrapolated를 기록합니다.
전체자료 범위+.25를 넘으면 평가자가 중단하며, 이것은 실제 안전영역이 아닙니다.

주 실험 결과
- 이상조건 fixed Jacobian은 네 관측 모두341/341이나 smooth plant 가정의 결과입니다.
- 작은probe의 '다른 축과 구별되는 반응/차분잡음'은 QR .33~.44,
  선추가 .41~.67로1미만입니다. 큰 전체영상 이동이 보여도 축 분리 신호는 약합니다.
- 절편+구동오차+잡음에서 fixed 통과는 QR178,위239,아래250,양쪽254 /341입니다.
- 같은조건 작은2probe+Broyden은61,167,149,162 /341로 나빠졌습니다.
- Plant 조건평균 적합RMS는 정규화단위 약3~8.9이며 반복잡음RMS 약.88~1.88보다
  큽니다. 모델이 실제 영상 반응의 상당 부분을 설명하지 못하므로 실제 성공률로
  읽을 수 없습니다. 생산 cycle time도 측정하지 않았습니다.

보조 실험 exploratory/
주 결과를 본 뒤 추가했음을 명시하고 원본 주 결과27개 파일 SHA를 보존했습니다.
후보는 RLS feedback-only, 큰2probe(.20,.04), 큰3probe뿐이며 한 번 평가했습니다.
weighted secant를 최근12개 누적하고, 두 축의 충분한 excitation/SNR1.5, 잡음바닥을
넘는 잔차 개선, 기존Jacobian 대비 응답변화50%이내를 모두 만족해야 갱신합니다.
정규화RLS는 최근 pair로 normal equation을 다시 풀며 원 논문 알고리즘의 완전한
복제는 아닙니다. 정책·시나리오를 성적에 맞춰 후속 튜닝하지 않았습니다.
- 절편+구동오차+잡음, 양쪽: fixed254→큰2probeRLS262 /341.
  그러나 Y최대오차 .165083→.249604로 커져 전반적으로 더 좋다고 할 수 없습니다.
- 순수반복잡음, 양쪽: 작은2probeBroyden232→큰2probeRLS340 /341,
  기존 fixed341 /341. 잡음으로 나쁜Jacobian을 채택하는 현상은 줄었습니다.
- 이 탐색 결과도 실제20회 복귀 성공률 또는 새 사진 일반화 근거가 아닙니다.

simulation_cases.csv와 trajectories.npz는 모든 실패·정지·구간이탈을 보존합니다.
simulation_summary.csv는 평균/최대 잔여오차, 동시통과수, 횟수, 잘못된영상정지,
실패와 구간이탈을 집계합니다. results_verification.json과 independent_review.json은
코드/분할/산술 검증이며, 독립 로봇 실험 인증이 아닙니다. research.json에5개1차자료.
