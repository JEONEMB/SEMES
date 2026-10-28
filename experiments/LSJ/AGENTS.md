# LSJ 작업 범위

- 이 폴더는 기존 분석 자료다. 추가 이미지 분석·통계 검정은 사용자가 지정한 범위에서만 수행한다.
- 현재 요구사항과 후속 정정은 CURRENT_CONTEXT.md를 따른다. 과거 보고서의 초기 결론을 최신 합의로 오인하지 않는다.
- 향후 알고리즘 개발은 저장소 dev의 include/alignment.hpp, src/alignment.cpp, src/main.cpp와 루트 CMake 스켈레톤을 기반으로 한다.
- 기존 Alignment_t 및 alignment(cv::Mat&) 계약을 먼저 확인하며 Python 연구 코드를 최종 C++ 구현으로 간주하지 않는다.
- 공통 소스나 팀원 코드 변경은 요청받은 개발 범위에서만 한다. 원본 데이터·기존 측정 결과를 덮어쓰지 않는다.
- 작업 브랜치는 LSJ. Jira 작업 키를 해당 커밋 메시지에 정확히 포함하고 push 후 원격 포함 여부를 확인한다.
- Codex는 코드·분석·Git 작업 및 전달 YAML, Claude는 Jira 관리·코드 검증을 담당한다.
- 전달 YAML은 예시 형식을 복사한 새 파일로 만들며 작업 완료 여부와 Git/Jira 반영 여부를 구분한다.
