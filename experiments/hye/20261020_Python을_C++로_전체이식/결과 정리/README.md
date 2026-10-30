# SEMES C++ 작업 공간

Python 구현을 C++로 전부 대체하고, 데이터·모델·의존성을 이 폴더 안에서 읽도록 정리하는 작업이다. 전체 이식과 재현 검증을 마친 뒤 기존 코드를 `legacy`로 이동한다. 커밋·푸시는 보류한다.

현재 원본 소스 478개 중 검증 완료 50개, 미완료 428개다. 복사한 데이터와 저장된 결과를 코드 이식으로 세지 않는다. [전체 파일별 체크리스트](../실험%20방법/전체_파일_목록.md)와 [현재 집계](../실험%20방법/experiments/file_list_20261027_full_004/checks.json)를 기준으로 한다.

| 경로 | 내용 |
|---|---|
| `include`, `src` | C++ 측정·모델·학습·제어 라이브러리 |
| `tool`, `tests`, `examples` | 실행·검증 도구와 호출 예제 |
| `experiments` | 과거 실험별 구현과 원본 출처 |
| `data/raw/hw` | 해시 검증한 촬영 데이터 복사본 |
| `data/reference` | 과거 입력·결과·문서 비교 자산 |
| `models`, `config` | 고정 모델, 분할, 상대 경로 이미지 목록, 실행 설정 |
| `third_party`, `resources` | 고정 네이티브 의존성과 프런트엔드 자산 |
| `results` | 실행별 새 결과와 감사 기록 |
| `build` | 생성된 빌드 파일 |

`final`에서 실행한다. Visual Studio Build Tools 2026과 Windows SDK를 사용한다.

```powershell
cmake --preset native
cmake --build --preset native-release
ctest --preset native-release --output-on-failure
cmake --preset modern
cmake --build --preset modern-release
ctest --preset modern-release --output-on-failure
build/native/Release/semes.exe list
build/native/Release/semes.exe run core.infer-example results/my_new_run
build/native/Release/semes.exe run feedback.rebuild341 results/my_new_training
```

`cmake`가 PATH에 없으면 `C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin`의 실행 파일을 사용한다. `semes`는 실행 파일 위치에서 작업 공간을 찾으며 외부 작업 디렉터리에서도 동작한다. 출력 폴더는 매번 새 이름을 사용한다.

데이터 6,024개(19.3GB)의 복사와 해시 검증, 내부 의존성 빌드, 341장 영상 측정·네 보정 모델의 전체 재실행을 확인했다. 재실행한 특징 파일과 네 모델 파일의 SHA256이 이전 C++ 실행과 정확히 같았다. [최신 구조·검증 기록](docs/SELF_CONTAINED_20261027.md)에서 결과별 허용차와 원본 대비 차이를 확인한다.

제품 실행 파일은 Python을 호출하지 않는다. `results`의 Python 스크립트는 원본을 대조하는 감사와 일회성 자산 변환용이며 제품 실행 경로에 들어가지 않는다. 새로 측정한 자료를 재학습한 모델과 과거 저장 모델의 수치 차이를 별도로 보존한다. 341장은 이미 노출된 개발 자료이며 독립 검증이나 실제 로봇 복귀 성과가 아니다.

알고리즘 규칙은 현재 [상위 ALGORITHM_RULES.md](../../20261022_이전작업_스냅샷/실험%20방법/snapshot_20261022_001/originals/ALGORITHM_RULES.md)를 따른다. [이전 감사](results/recheck_20261027_001/README.md)와 [10월23일 기록](docs/WORK_LOG_20261023.md)은 당시 상태의 기록이다. 전체 대체가 아직 끝나지 않았으므로 기존 코드는 보존 중이다.