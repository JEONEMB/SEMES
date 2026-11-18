# 2026-11-02 게시 안내

10/30~11/1에 수행한 연구·구현·검증 기록을 Jira 작업 키에 대응시켜 보관한다. 이번 Git 반영에서는 알고리즘을 다시 설계하거나 촬영본 전체를 재분석하지 않았다.

## 현재 결과

- 새 θ 작성본: `../algorithm/theta_huber_20261101/`
- 상단 전용 비교본: `../algorithm/theta_huber_upper_only_20261101/`
- 기준 모델의 고정 254장 평가: 허용오차 ±0.005° 이내 82장(32.2835%).
- 상단 전용 비교 평가: 같은 254장 중 86장(33.8583%). 두 작성본 모두 목표 정확도 미달이다.
- `alignment()`의 `t`는 θ 보정 각도이며 `y=0`은 Y 명령이 없다는 뜻이다. Y 정렬 성공을 뜻하지 않는다.
- 루트의 10/29 알고리즘과 다른 팀원 코드는 수정하지 않았다. 최신 작성본은 이 LSJ 전용 폴더의 독립 CMake 대상으로 사용한다.

## 기능별 코드 위치

`algorithm/theta_huber_20261101/src/`의 `edge_points.cpp`는 경계점 선택, `huber_line.cpp`는 직선 피팅, `theta_decoder.cpp`는 두 각도의 θ 환산, `theta_pipeline.cpp`는 전체 연결을 담당한다. 상단 전용 모델은 `algorithm/theta_huber_upper_only_20261101/src/upper_theta.cpp`에 있다.

상단 전용 빌드는 공통 경계점·Huber·입출력 소스를 형제 폴더인 `theta_huber_20261101`에서 사용한다. 두 폴더와 라이선스를 함께 제공해야 한다. 다른 팀원 폴더에 의존하지 않는다.

현재 팀의 다른 브랜치에는 변위 특징 또는 현재 자세 추정값을 반환하는 모듈도 있다. 우리 각도 특징·보정 명령과 의미가 같지 않으며 이번 반영에서 이를 연결하거나 공유 API를 바꾸지 않았다.

## 빌드와 실행

팀 기준 C++17 / Windows x64 / MSVC v141 / OpenCV 3.4.15를 사용한다. PowerShell에서 **팀 저장소 루트** 기준으로 실행한다. OpenCV_DIR만 실제 설치 위치로 지정한다.

```powershell
cmake -S experiments/LSJ/algorithm/theta_huber_20261101 -B build-lsj-theta -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/Users/SSAFY/opencv/build
cmake --build build-lsj-theta --config Release -- /m:1
ctest --test-dir build-lsj-theta -C Release --output-on-failure

cmake -S experiments/LSJ/algorithm/theta_huber_upper_only_20261101 -B build-lsj-upper -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/Users/SSAFY/opencv/build
cmake --build build-lsj-upper --config Release -- /m:1
ctest --test-dir build-lsj-upper -C Release --output-on-failure
```

```powershell
.\build-lsj-theta\Release\theta_alignment.exe experiments/LSJ/algorithm/theta_huber_20261101/config/calibration.json 'C:\실제\이미지.bmp'
.\build-lsj-upper\Release\theta_upper_alignment.exe experiments/LSJ/algorithm/theta_huber_upper_only_20261101/config/calibration.json 'C:\실제\이미지.bmp'
```

두 CLI의 실행 입력은 이미지 한 장과 설정 파일이다. 실시간 추론에는 학습 사진·TXT 라벨이 필요하지 않다. 당시 README의 `algorithm/…`, `analysis/…` 상대 경로 예제를 그대로 쓰려면 작업 폴더를 `experiments/LSJ`로 둔다. 개인 설치 경로가 들어 있는 과거 `build.cmd`와 연구 스크립트는 사용 환경에 맞게 경로를 준비해야 한다.

## 이번에 수행한 게시 검사

상세 결과는 [validation.json](validation.json)과 같은 폴더의 로그를 따른다.

- 사본과 원본의 SHA-256, 파일 추적 여부, `experiments/LSJ/` 내부 변경 범위 확인.
- 새 위치에서 상·하단 θ 버전, 상단 전용 버전, 공통 직선 추출 도구 구성·빌드.
- θ 버전들의 기존 CTest와 공통 직선 피팅·설정·수치 처리·Gaussian 제외 조건 회귀 검사 실행.
- 기존 Python 연구 코드의 문법 확인. 연구 스크립트의 이미지 분석·학습은 실행하지 않음.
- 254장 평가, 기준 영상 픽셀 변경 시험, 상관·민감도 분석을 이번에 다시 실행한 것은 아님.

공통 도구의 `pipeline_contract_test`는 기존 코드가 `configs/methods.json`을 상대 경로로 읽는다. 첫 실행은 저장소 루트에서 호출해 실패했으며, 해당 프로젝트 폴더인 `experiments/LSJ/analysis/reference_line_methods_108_20261030/`를 작업 폴더로 지정한 재실행에서는 통과했다. 소스는 변경하지 않았고 최초 실행 기록도 보존했다. 해당 도구 검사를 수동으로 실행할 때도 이 작업 폴더를 사용한다.

합성·코드 계약 검사 통과는 실제 장비 정렬 성공이나 ±0.005° 정확도 달성을 의미하지 않는다.

## 보관 범위와 재현 한계

[파일 manifest](../artifact_manifest_20261102.json)에 최종 사본의 원본 경로·해시를 기록했다. [제외 목록](exclusions.json)에는 다음 자료를 Git에 포함하지 않았음을 기록한다.

- 원본 촬영 이미지·압축 파일, 실행 바이너리·빌드·가상환경·브라우저 캐시.
- 대용량 Gaussian 원시 반환값과 표본별 전체 피팅 표, NumPy 원시 배열.
- 원문 논문 PDF·추출 전문. 근거 URL·기존 출처 manifest는 보존한다.

코드·설정·요약 결과·보고서·그림·기존 검증 기록은 함께 제공한다. 제외된 원시값을 읽는 상세 탐색 기능이나 전체 연구 재실행은 해당 입력을 별도로 준비해야 한다. 과거 JSON·로그 안의 절대 경로는 당시 출처이며 팀원의 PC에 자동으로 존재한다는 뜻이 아니다.

구형 108조합 보고서는 보완 전 측정 결과를 보존한 기록이다. 소스는 후속 공통 보완과 Gaussian 제외 조건 제거까지 포함하므로, 과거 수치를 현재 소스로 새로 생성했다고 해석하지 않는다. `before/`, 변경 패치, 작업별 보고서에서 버전을 구분한다.

## Jira / Git

작업 키별 보고서는 `../reports/`, 매핑은 [jira_issue_map_20261102.yaml](../jira_issue_map_20261102.yaml)에 있다. 개발 기준은 `LSJ`, 이번 작업 브랜치는 `S15P31A507-966-967-20261102-analysis-theta`다. 대상 MR은 `LSJ`이며 `dev`나 `master`로 반영하지 않는다. 실제 MR·머지 해시는 최종 Claude 전달 YAML에 기록한다.
