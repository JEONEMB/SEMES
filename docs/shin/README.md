# 신윤석 실험 전달 기록

- T 단독 추정: `experiments/shin/t_estimation`. 공개 선언은 include/, 구현은 src/, 실제 추정 라이브러리는 modules/에 보존했다. 원본 feature/shin/T-estimation의 9fee400 기준이며 최신 8장 제외 재평가까지 포함한다.
- 자세 추정: `experiments/shin/pose_alignment`. 실행 도구는 src/, 공개 헤더는 include/, 정합 구현과 모델은 modules/에 있다. 575장 평가와 학습 결과를 보존했다.
- 두 실험은 서로 다른 라벨 정책을 사용한다. T 재평가는 8장 제외, 자세 추정은 라벨 1장 정정 및 4장 가중치 감소다. 결과를 직접 같은 조건의 순위로 비교하지 않는다.
- T 최대 1 tick 목표는 미달이며 전수 재대입 결과를 독립 검증으로 해석하지 않는다.

각 실험 폴더를 독립 CMake 프로젝트로 빌드한다:

```powershell
cmake -S experiments/shin/t_estimation -B experiments/shin/t_estimation/build -DOpenCV_DIR=C:/opencv/build -DOpenCV_RUNTIME=vc15
cmake --build experiments/shin/t_estimation/build --config Release
cmake -S experiments/shin/pose_alignment -B experiments/shin/pose_alignment/build -DOpenCV_DIR=C:/opencv/build -DOpenCV_RUNTIME=vc15
cmake --build experiments/shin/pose_alignment/build --config Release
```

EDA 기록은 docs/shin/의 세션 분석·원점 분석·v3 감사표에 보존했다. 이미지/영상과 대형 중간 배열은 포함하지 않았다. 과거 보고서의 개인 절대 경로는 출처 기록이며 실행 의존성이 아니다. Python 분석 스크립트는 과거 EDA 재현 참고 자료이고 전달 알고리즘은 C++이다.

최신 판단: 세션 차분은 바이어스 영향을 제거하지만 절대 원점을 식별하지 못한다. 증강은 예측 안정화 후보일 뿐 물리적 참값의 증거가 아니다. 모델 예측으로 전체 라벨을 교체할 근거는 부족하며, 보류 자세와 미사용 관측 부위로 후보를 교차검사해야 한다.

Jira 연동 시도는 Atlassian Rovo 서버의 403(app is not installed)로 실패했다. 이슈 및 story point는 생성되지 않았다.
