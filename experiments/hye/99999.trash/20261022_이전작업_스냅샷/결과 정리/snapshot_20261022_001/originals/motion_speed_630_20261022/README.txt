Motion speed 630 · 별도 속도 최적화 실험 (2026-10-22)

결과 보기: report.html 또는 http://127.0.0.1:8766/
기존 motion_web_20261022, 모델 및 측정 결과는 변경하지 않았습니다.
기준 문서: ../ALGORITHM_RULES.md

대표3장, 한 장씩 순차 처리: 중앙값48.7558초 →0.2381초, 약204.8배.
원본280장 재측정: 중앙값0.1421초, p95 0.1636초, 최대0.1804초.
상주API 별도시험4회: 추론0.230~0.253초, 로컬HTTP왕복0.311~0.346초.
측정 세션과 시스템 부하가 다르므로 이 수치를 하나의 조건처럼 섞지 않습니다.
결과 캐시/저장된 좌표를 추론에 재사용하지 않았습니다.

같은 고정630개 구조 ID와 모델을 사용합니다.
NCC →9x9 erf 경계 soft_l1 맞춤(창 재정렬 최대3회) →11x11 진단
→전체630점 동일가중 P12 →기존 학습된 Y/T 이차곡선 비용 비교.
좌표/흐림/밝기/대비의 경계와 진단 임계값을 유지했습니다.
SciPy TRF의 수치미분 대신 Numba 해석미분/경계 제한 감쇠LM/4스레드 점 병렬화.
원래 계산기와 다른 최적화 계산기이므로 수치 동일성을 주장하지 않습니다.
예측 최대차0.000545. 좌표 최대차0.191256px. 유효성 진단6132/176400건 변경.
축/부호 및 통과/실패 사진 목록은280장 모두 같고 둘다275/280장 통과.
Y최대오차0.255573892→0.255609445mm, T최대오차0.05 유지.
원래5건 오판정과0.09 초과 문제는 해결된 것이 아닙니다.
전체 학습 재대입 비교이며 새로운4-fold OOF/현장 독립 검증은 아닙니다.

실행 (현재 PC의 기존 Python 패키지 + 이 폴더 deps 사용)
  python serve.py --port 8766
서버가 ready를 출력한 뒤 요청하세요.
Numba 첫 JIT는 시험 약28초. 컴파일 캐시가 있던 서버 시작은 약4.12초.
한 번 로드한 상주 프로세스를 재사용하므로 초기화는 사진마다 반복하지 않습니다.
기존 웹의8765포트는 그대로 두고 별도8766포트를 사용합니다.

API
  GET http://127.0.0.1:8766/health
  POST http://127.0.0.1:8766/predict
  Content-Type: application/octet-stream
  body: 원본 이미지 바이트 (3840x2160, 32MB 이하)
  반환: axis,value,Y,T,points,valid,timings,cache_hit(false) 등.
  파일명/라벨은 전달하지 않습니다. T 물리 단위 미확정.
  반환값은 이동 추정치이며 로봇 제어 명령이 아닙니다.
  장비 좌표계/보정 부호/카메라 획득/전송/정착은 별도 연동 및 검증이 필요합니다.

단독 CLI (프로세스 초기화가 매번 포함되므로 실시간 연동에는 상주API 권장)
  python predict.py --image "원본.bmp" --output prediction.json
예제 클라이언트
  python client_example.py --image "원본.bmp"

재현
  python bench_one.py --variant baseline --file Y/Y+0.50.bmp
  python bench_one.py --variant compiled --file Y/Y+0.50.bmp
  python benchmark.py
  python verify280.py
  python test_math.py
  python make_report.py
고정 입력 경로: engine.py ROOT (hw-analysis/data/hw/bmp/20261014).
다른 PC의 패키지는 가상환경을 이 폴더 안에 만들고 requirements.txt로 설치하세요.
첫 실행은 컴파일되므로 이후 한 장 처리 시간과 구분하세요.
측정 시 기존 일괄 작업/다른 CPU 집중 작업을 동시에 실행하지 마세요.

파일
  baseline.py: 기존 측정기를 복사해 단계별 계측만 추가
  optimized.py,fast_fit.py: 해석미분 SciPy 중간 비교 (최종 채택 아님)
  compiled.py: 최종 컴파일 경계 모델 계산기
  engine.py: 고정기하/학습모델, 이미지 바이트 추론, 단계별 시간
  assets/: 고정630구조/center좌표/기존학습모델 스냅샷
  results/paired_timings.json: 동일3장 단독 전후 시간
  results/compiled280.csv: 원본280장 새추론/정답오차/기존결과차이
  results/compiled_failures.csv: 오판정/허용오차실패5장
  results/service_checks.json: 실제API4회, 잘못된입력, 반복사진재계산 검사
  results/compiled280_summary.json: 전체결과와수치차이

카메라부터 실제 로봇 보정 완료까지의 지연이나 실시간 마감시간을 보장한 결과는 아닙니다.
