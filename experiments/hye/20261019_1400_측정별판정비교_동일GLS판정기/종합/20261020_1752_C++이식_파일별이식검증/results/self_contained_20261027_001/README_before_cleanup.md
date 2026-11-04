# SEMES C++ 재개발

2026-10-27 최신 지시: 전체 Python 구현을 C++로 대체하고 데이터·의존성·실행 자산을 이 폴더에 복사한다. 전체 대체와 재현 검증을 마친 다음 기존 코드를 `legacy`로 이동한다. 현재 진행 상태와 새 실행 방법은 [독립 실행 구조와 검증](docs/SELF_CONTAINED_20261027.md)을 기준으로 한다. 기존 검증은 [재검증 결과](results/recheck_20261027_001/README.md)에 보존한다.

`final`은 C++ 코드, 모델, 실험별 출처, 복사한 데이터, 의존성, 새 검증 결과를 모으는 작업 폴더다. 데이터는 `data/raw`, 기존 실험 입력과 비교 기준은 `data/reference`, 의존성은 `third_party`를 사용한다.

원본은 전체 검증을 마칠 때까지 보존한다. 과거의 최근 실험 보류·복사 금지 지시는 최신 전체 대체 지시로 대체됐다. 저장된 결과의 복사는 해당 결과를 생성하는 알고리즘의 C++ 이식으로 세지 않는다.

오래된 실험의 현재 작업은 [실험별 안내](experiments/README.md)에서 확인한다. 아래 630점 경로는 최신 지시 전에 구현·검증한 기반이며, 현재 진행 순서는 과거 실험을 우선한다. 커밋·푸시는 사용자 최신 답변에 따라 미루고 `final` 파일 정리만 진행한다.

**[전체 파일 목록과 C++ 완료 체크리스트](전체_파일_목록.md)** 에서 파일 하나씩 확인한다. 2026-10-23 10:00 KST 기준 원본21,614개·프로젝트 소스444개를 기록했다. 각 소스의 연관 입력·출력·공유 코드, 원본 시각·해시, 확인된 쓰기 호출, C++ 위치·검증 근거를 연결한다. 데이터·문서·결과·의존성 전체 경로는 목록 하단의 비소스 파일표에 있다. 최신 체크 수는 이 목록이 기준이며 전체 이식은 진행 중이다.

10월23일 작업의 시각별 내용은 [작업 기록](docs/WORK_LOG_20261023.md)에 보존한다. 10월27일 전체 소스478개를 다시 집계했으며 검증 완료41개(기존 C++15개, 이식26개), 미완료437개다. [현재 목록 집계](experiments/file_list_20261027_full_001/checks.json)를 기준으로 한다.

알고리즘 규칙의 단일 원본은 [ALGORITHM_RULES.md](../ALGORITHM_RULES.md)다. 기존 규칙이나 결과를 이 폴더에서 다시 정의하지 않는다.

## 폴더

| 위치 | 내용 |
|---|---|
| `include`, `src` | C++ 추론·학습·입출력 라이브러리와 CLI |
| `models` | 출처를 고정한 구조 ID·모델 자산 |
| `config` | 고정 분할과 실험 설명 |
| `tool` | 목록·작성 흔적·보존·평가·대조 도구, 기존 C++ EDA |
| `examples` | 제품에서 이미지 바이트로 추론하는 C++ 호출 예제 |
| `experiments` | 원본 파일 시각·해시·작성 흔적·실험별 변경점·이식 원장 |
| `results` | 실행마다 별도 디렉터리에 저장한 새 결과 |
| `data/raw`, `data/reference` | 해시 검증한 원본 데이터·기존 실험 자산 복사본 |
| `third_party`, `resources` | 복사한 네이티브 의존성과 프런트엔드 자산 |
| `../legacy/snapshot_20261022_001` | 이전 보존 사본. 전체 대체 검증 후 새 이동 명세를 별도로 생성 |

## 빌드

이 PC에서 검증한 환경은 Visual Studio Build Tools 2026의 MSVC 14.51, Windows SDK 10.0.26100, C++17, OpenCV 3.4.15다. OpenCV3·OpenCV4·zlib는 `third_party`에 복사했으며 기본 빌드는 그 경로를 우선 사용한다. 제품 실행 경로는 C++이고 Python 프로세스를 호출하지 않는다. `results`의 Python 감사 스크립트는 원본 대조·일회성 자산 변환용이며 제품 실행 코드가 아니다. 새 빌드는 `cmake --preset native`, `cmake --build --preset native-release`, `ctest --preset native-release`를 사용한다. 아래 기존 세부 명령보다 [최신 실행 안내](docs/SELF_CONTAINED_20261027.md)를 우선한다.

PowerShell에서 `final` 디렉터리를 기준으로 실행한다. `cmake`가 PATH에 없으면 이 PC의 `C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin`에 있는 실행 파일을 사용한다.

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DOpenCV_DIR=C:/opencv/build -DSEMES_ZLIB_SOURCE=C:/opencv/sources/3rdparty/zlib
cmake --build build --config Release -- /m:1 /nr:false
ctest --test-dir build -C Release --output-on-failure
```

실행 파일은 `build/Release`에 생성된다. 기존 EDA는 `build/tool/hw/eda/Release/hw_eda.exe`다. OpenCV DLL은 사용하는 실행 파일 옆에 자동 복사한다.

초기 ArUco·픽셀 G1은 별도 OpenCV4.11 소스 빌드를 사용한다. 준비된 빌드를 선택하려면 위 configure 명령에 `-DSEMES_OPENCV4_DIR=<final절대경로>/build/dependencies/opencv411-build`를 추가한다. OpenCV3와4를 같은 실행 파일에 링크하지 않는다. 버전·다운로드 해시·모듈·빌드 기록은 [config/opencv411_build.json](config/opencv411_build.json)과 [초기 QR 실험](experiments/analysis_v1_20261013_QR관측유지_ArUcoC++이식/README.md)에 있다. 팀원의 PC에 이 의존성을 자동 설치하는 단계는 아직 없다.

## 630점 추론

```powershell
.\build\Release\main.exe reference models/edge630 ../hw-analysis/data/hw/bmp/20261014/center.bmp
.\build\Release\main.exe predict models/edge630 ../hw-analysis/data/hw/bmp/20261014/center.bmp ../hw-analysis/data/hw/bmp/20261014/Y/Y+0.50.bmp results/example_새이름
```

출력 디렉터리는 매번 새 이름이어야 한다. `prediction.json`과 630점의 좌표·변위·진단을 담은 `measurement.csv`를 기록한다. `Engine::infer`에는 이미지 바이트만 들어간다. 파일명 정답과 시행 ID는 추론 인터페이스에 없다.

`examples/product.cpp`가 같은 라이브러리를 사용하는 제품 연결 예제다. 이 단계는 C++ 제품 연결이며, 기존 웹의 업로드·ZIP·재개·시각화 UI 전체를 C++로 재구현한 것은 아니다.

## 검증 결과

| 검증 | 결과 | 저장 위치 |
|---|---|---|
| 고정 ID·영역·기준 좌표 | 630점 확인, 원본 center 좌표 최대차 4.55e-13px | `results/parity280_001/checks.json` |
| 저장된 Python 측정에서 P12 재계산 | 최대차 6.88e-13 | 같은 파일 |
| 저장된 Python 측정에서 동결 모델 추론 | 최대차 4.71e-14, 축 변경 0 | 같은 파일 |
| 실제 280장 C++ LM 추론 대 기존 Python LM | 예측 최대차 7.94e-7, 축 변경 0, 사진별 유효점 수 차이 0 | 같은 파일 |
| 실제 280장 동결 모델 재대입 | 처리 실패 0, 축·부호 275/280, 두 축 허용오차 동시 통과 276/280 | `results/full280_001` |
| 처리시간 | 중앙값 약 0.209초, 평균 약 0.210초 | `results/full280_001/summary.json` |
| 학습 210장·시험 70장의 고정 4-fold | 각 학습 분할 안에서 곡선·공분산 재학습 | `results/cpp_oof4_001` |
| 센터 후보·동시 추정 모델 | 저장된 341장 관측의 예측 재현 최대차 7.38e-13, 후보 선택 차이 0 | `results/joint_replay341_002` |

재대입과 OOF를 서로 다른 폴더에 보관한다. 센터 후보 모델의 재현은 저장 관측·학습 모델을 이용한 재생이며 새 영상 측정 결과가 아니다. 그 모델의 341장 동시 허용오차 통과는 191장으로, 기존 단일축 모델의 대체 모델로 채택하지 않았다.

C++ 측정은 기존 `compiled.py`의 경계 제한 LM을 이식했다. 웹 원본의 SciPy TRF와는 수치해석기가 다르다. 280장 전체에서 TRF와의 좌표 차이 최대값은 약 0.1913px, 진단 유효 여부 차이는 6,132점이다. 이 차이를 숨기거나 TRF의 완전 동등 이식으로 표시하지 않는다.

## 실험과 시간 구분

[실험별 안내](experiments/README.md)에서 원본 경로별 설명을 확인한다. 각 실험에 문제·조사·해결 방법·유지한 부분·바꾼 부분·구분 기준을 기록하며, 개별 소스의 생성/수정 시각, SHA256, 관측된 작성 도구 호출을 따로 보관한다. `catalogue_001` 등 앞선 실행 기록은 덮어쓰지 않는다.

`authorship_20261023_01`은 Claude와 Codex의 직접 쓰기 도구 호출을 추가 조사한 기록이다. 확인하지 못한 작성자는 `unknown`이다. 파일 시각만으로 작업자를 배정하지 않는다. 같은 경로를 두 작업자가 다룬 기록도 지우지 않는다. 셸로 작성한 파일이나 남아 있지 않은 세션은 이 자료만으로 확정할 수 없다.

센터 후보 실험의 16:02 예측표와 16:11 모델은 다른 버전이다. `joint_replay341_001`은 그 짝이 맞지 않았던 비교로 보존했고, `joint_replay341_002`는 `results/local_surfaces/predictions.csv`와 맞춘 비교다.

## 원본 보존과 남은 작업

`legacy/snapshot_20261022_001`에 원본 5,003개를 복사하고 해시를 대조했다. 기존 데이터·라이브러리·캐시·루트 규칙 문서는 위치를 유지했다. 목록 작성 중 다른 작업이 수정한 파일 3개는 같은 스냅샷에 섞지 않고 `issues.csv`에 기록했다. PowerPoint 잠금 파일 3개도 최초 목록의 `errors.txt`에 기록했다.

원본을 삭제하거나 이동하지 않았다. 최신 지시에 따라 앞으로도 `legacy`에 추가하거나 원본을 이동하지 않는다. [진행 상태](docs/MIGRATION_STATUS.md)에 미완료 범위를 명시한다.

커밋·푸시를 재개할 때의 저장소 경계, 팀 브랜치 규약, 단계별 커밋 단위는 [Git 작업 순서](docs/GIT_WORKFLOW.md)에 정리했다. 현재는 파일 정리만 진행한다.
