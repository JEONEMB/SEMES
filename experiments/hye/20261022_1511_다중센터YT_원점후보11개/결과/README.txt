센터 후보 + Y/T 동시 추정 · 고정630점 별도 실험

결과: report.html
주의: 구현은 완료했지만 실제 자동 보정에 충분한 정확도가 나오지는 않았습니다.
최종 개발 절차의341장4-fold OOF:
Y 평균/최대 절대오차0.112240/0.754249 mm
T 평균/최대 절대오차0.019840/0.132601 (물리 단위 미확정)
두 축 허용오차0.09 동시 통과169/341 (49.6%)
전체 학습 후341장 재대입:191/341 (56.0%)
이 수치는 축 분류 정확도가 아닙니다.
센터 후보를 늘려도 Y/T 분리와 회귀 표면 오차가 충분히 해결되지 않았습니다.

추론 방식
고정630구조모서리 → erf 경계 직접 맞춤 → 전체점 QR별 공간 집계
→ 학습된11개 원점 각각에서 Y/T를 함께 최적화 → 최소비용 원점+두축 선택.
현재 사진 파일명/정답/trial을 후보 선택에 넣지 않습니다.
원점 후보: 고정center0 + 기존6학습trial 절편 +10월20일 촬영군 절편1
+실제center1~3의 측정 특징3 =11.
이11개는 전체 학습 배포용 후보 수입니다. 시험center를 빼거나 내부모델이
학습 후보만을 선택한 OOF fold는11/10/8/10개 후보입니다.
반복번호1/2/3을 서로 다른 원점의 독립trial이라고 가정하지 않았습니다.

초기 탐색 변경은 따로 기록했습니다.
±130px NCC 실패 시 ±260/512로 확장하며, 이후630 ID/경계 목적함수/
9x9+11x11 진단/전체점 회귀는 유지합니다. 기존T2의12장이 복구돼341장 모두 측정.
측정엔진은 이전 speed 실험의 Numba 컴파일 LM을 복사한 것입니다.
모델은 추정 좌표 중 유효하지 않은 점도 포함합니다.

실행 (이 폴더에서)
  python engine.py --image "원본.bmp" --out prediction.json
반환: Y,T,center_id,후보수,비용간격,대안추정,모호성,유효점수,단계별시간.
T는미확정단위. 반환값은 로봇제어명령이 아닌 영상기반 이동추정치입니다.
파일명라벨을 바꿔도 입력픽셀이같으면 추론값은 같습니다.
프로세스를 상주시켜Engine 객체를 재사용하세요. 패키지로드가 매번드는CLI는
캐시가있는 새프로세스3회에서약2.2초, 상주한 이미지추론은중앙0.0747초였습니다.
Numba 첫JIT는별도시간이필요합니다. 사진결과캐시는사용하지않았습니다.
카메라/네트워크/로봇이동·정착은측정에포함되지않습니다.
기존8765웹과8766속도API는이새모델로교체하지않았습니다.

재현
  python extract.py      # 원본341장630점 재측정
  python evaluate.py     # 고정4-fold + 내부3-fold 선택 +전체학습
  python test_audit.py   # 누출/ID/반복그룹/저장복원/합성식별성 검사
  python benchmark.py   # 최종모델341장 원본바이트 실제추론
  python diagnose.py    # 센터와이동의식별성진단
  python report.py
학습/Evaluation 라벨은 학습과사후평가에만 사용합니다.
모델의predict(X)에는파일명/라벨/trial 인자가없습니다.
구조측정은현재영상과고정기준만사용합니다.

데이터및평가
데이터경로: ../hw-analysis/data/hw/bmp
기존280장의시험fold그대로유지. 복합19조건 반복3장을같은fold에묶음.
center4장시험fold별1장. 해당시험center를학습후보에서제외.
기존210장뿐아니라해당fold학습의복합/center도추가되어
예전단일축전용OOF와학습데이터양이다릅니다.
선택점수는내부시험의실패수→최악오차→평균오차합으로사전기록.
같은fold반복개발로새독립검증이아닙니다.
고정center0은교정입력이므로독립시험사진이아닙니다.
재대입의center1~3도교정후보로제공하므로독립성능근거가아닙니다.
기존Y-or-T참조모델은기존단일축280장학습고정모델이므로
새모델OOF와정식독립동일분할성적처럼비교하지않습니다.
기존엄격초기화실패12장은분모에포함하며평균/최대오차는성공329장기준.

산출물
assets/: 고정630구조/기준좌표/기존모델 스냅샷
model.npz: 전체학습센터+YT모델
results/manifest.csv,measurements.npz: 341장목록,630점좌표/진단/시간
results/p12_initial/: 첫P12공유표면결과와모델보존
results/expanded_features/: QR별/전체변위비교결과와모델보존
results/local_surfaces/: 최종국소표면후보추가절차의전체결과
  summary.csv: 평균/최대오차,두축동시통과율
  predictions.csv,failures.csv: 모든사진예측및실패
  splits.json,selection.json,protocol.json,audit.json: fold와학습선택근거
  fold*_model.npz: 학습분할별모델
  final_centers.json: 전체학습원점후보/설정
  end_to_end.json: 341장실추론출력및시간
  checks.json: 누출방지/630ID/복합반복그룹검사
  identifiability.json: 학습반응유사성/조건수/모호성
  diagnostic_oracle_group.csv: 정답group를준반사실진단,운영성적아님
  comparison.png,structure_630_T2.png: 결과및전점시각화
최종결과보다성적이좋은이전단계를사후선택해최종성적으로바꾸지않았습니다.
모든실험은기존코드/모델/결과를덮어쓰지않는이폴더에저장했습니다.
프로젝트합의기준단일원본: ../ALGORITHM_RULES.md

의존성
기존Python numpy/scipy/opencv/pandas/sklearn/threadpoolctl,
Numba/llvmlite는 ../motion_speed_630_20261022/deps 를재사용합니다.
다른환경에서는별도가상환경에이패키지들을설치하세요.
