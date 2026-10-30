# 2026-10-27 final 재검증

**판정: 전체 대체 완료가 아니므로 기존 코드를 legacy로 이동하지 않는다.** 실행 가능한 C++ 경로는 재현되지만, 완료되지 않은 이식 범위와 원본 폴더 의존성, 원본 Python 수치와의 차이가 남아 있다. 사용자가 요청한 조건부 이동의 전제가 충족되지 않았다. 추가 승인 문제가 아니다.

최종 집계: C++ 재실행 34개 경로·비교 파일 1,386개 모두 재현 통과. ECC 7개 설정·476행도 완료했다. 원본 Python 대비 별도 동등성 검사 3건은 실패했다. [전체 판정 JSON](audit_summary.json), [C++ 재현 상세](reproduction_comparison.json).

## 확인한 범위

| 항목 | 결과 | 근거 |
|---|---|---|
| 프로젝트 소스 새 빌드 | 별도 `build-recheck-20261027`에서 Release 기본 전체 타깃 빌드 성공. OpenCV4 선택 실험 포함 | `configure.log`, `build.log`, `commands.json` |
| 기본 테스트 | CTest 2/2 통과 | `ctest.log`, `ctest_after_fix.log` |
| 기본 빌드에서 제외된 포맷 도구 | 별도 타깃 컴파일 통과. 소스를 수정하는 포맷 실행은 하지 않음 | `excluded_tool_build.json` |
| 630점 280장 새 영상 추론 | 실패 0, 축·부호 275/280, 두 축 허용오차 동시 통과 276/280. 이전 C++ 수치 재현 | `full280`, `reproduction_comparison.json` |
| 4-fold 분할 내부 재학습 | 예측·실패 목록·학습 모델·요약 수치 재현 | `oof4`, `reproduction_comparison.json` |
| 원본 Python LM 대비 280장 | 예측 최대차 약 7.94e-7, 축 변경 0, 유효점 수 불일치 0 | `parity280/checks.json` |
| 특징 집계 31개 설정 | 저장 281장 관측에서 기존 특징 재계산 대조 통과 | `aggregation/checks.json` |
| 센터 후보 모델 341장 | 새 영상 측정 및 저장 관측 재생 각각 이전 C++ 결과 재현. 재생의 원본 예측 최대차 약 7.38e-13 | `joint_fresh341`, `joint_replay341` |
| CWD 의존성 | 프로젝트 상위 디렉터리에서 절대경로 입력으로 추론 성공 | `supplement_commands.json`, `other_cwd` |
| 수정 후 기본 실행 파일 | `build/Release/main.exe`의 Y+0.50 재추론에서 예측·630점 측정값 차이 0 | `normal_predict_final_check.json` |
| EDA 281장·60장 | CSV 7개 수치 최대차 0 | `eda_commands.json`, `eda_comparison.json` |
| 과거 실험 경로 | 초기 회귀·판정·특징 조합·곡선 GN·실제 G1 281장·표·그림·슬라이드 입력 등 기존 C++ 결과 재현 | `commands.json`, `reproduction_comparison.json` |
| ECC 설정 전수 | 7설정×68장=476행, 처리 실패 0. 이전 C++ 잡음·homography 수치 재현 | `ecc_tune_parallel`, `ecc_parallel_commands.json` |
| GIF·동영상 | GIF 70프레임/7초, MP4 300프레임/20fps/15초. 이전 C++ 결과와 디코딩한 모든 프레임의 픽셀 차이 0 | `artifact_integrity.json`, `decoded_media_comparison.json` |
| 생성 파일 | PNG·GIF·MP4 디코딩, XLSX·PPTX ZIP CRC 및 XML 파싱: 58개 실패 0 | `artifact_integrity.json` |
| 원본 보존 | 검증 전·중·후 데이터 690개·과거 원본 소스 28개 해시 변경 0, 미확인 0 | `preservation/checks.json`, `preservation_after/checks.json`, `preservation_final/checks.json` |

새 빌드는 프로젝트 소스를 다시 컴파일했다. 외부 OpenCV3.4.15 및 기존 OpenCV4.11 의존성 빌드는 재사용했다. 다른 PC에서 의존성까지 처음 설치하는 검증은 아니다. 테스트·관리 도구 전부의 모든 옵션을 실행한 검증도 아니다. 보관·포맷·작성자 조사처럼 원본 또는 외부 로그를 대상으로 하는 관리 작업은 이 감사의 실행 범위에서 제외했다.

수치 재현 비교는 CSV·JSON의 계산값에 절대차 1e-9 기준을 사용하고, 실행 시각·처리 시간·새 출력 경로·해시는 별도로 취급했다. PNG는 디코딩한 픽셀을 비교했다. 입력 보존 해시는 별도 검사했다. C++ 재실행 간 재현과 과거 Python 저장 결과 대비 동등성은 서로 다른 판정이다. 이번 재현은 같은 개발 자료에 대한 확인이며 독립 성능 검증이나 실제 로봇 복귀 검증이 아니다.

## 원본 이동을 막는 항목

1. **완료 범위 부족.** 10월23일 원본 소스 444개에서 현재 478개로 늘었다. 추가 34개·기존 파일 변경 1개·삭제 0개다. `port_progress.json`의 44개 항목 중 완료 표시는 38개이며, 나머지 440개 전체의 대체 완료 근거가 없다. 최신 10월26일 이동 제어 실험도 포함돼 있지 않다. [현재 목록 변화](inventory_drift.json), [소스별 이동 판정 478행](archive_readiness.csv).
2. **원본 디렉터리를 런타임 입력으로 사용.** 과거 실험이 `analysis_agent/qr_measurements.csv`, `analysis_v2/homography_features.csv`, `analysis_v3/v1_data/*.npz` 및 루트 CSV·Excel을 읽는다. EDA는 `hw-analysis/data`와 `hw-analysis/docs`도 읽는다. 이들 디렉터리를 통째로 legacy로 옮기면 현재 인자·고정 경로가 끊어진다. [코드의 경로 참조 목록](external_path_references.json).
3. **원본 Python 수치와 미해결 차이.** 초기 affine ECC 측정 281장은 원본 대비 최대 약 0.0363159px, homography 281장은 약 0.0349125px 차이다. 이전에 사용한 1e-4 기준의 원본 동등성 검사에서 각각 실패했다. 기존 C++ 결과의 재현 자체는 통과했다. [affine 대조](original_affine_parity/checks.json), [homography 대조](original_homography_parity/checks.json). 초기 84점 추적에도 dx 약 0.0038681px/dy 약 0.0064013px 차이가 남는다.

   ECC 설정 비교도 복구된 원본 4자리 출력과 잡음 최대차 약 0.00712078이 남아 기존 5e-5 기준 검사에서 실패했다. [잡음 대조](original_ecc_noise_parity/checks.json). 630점 C++ LM은 Python LM과 대조한 것이며 SciPy TRF의 완전 동등 이식은 아니다. TRF 대비 좌표 최대차 약 0.191256px, 유효성 진단 차이 6,132점도 그대로 기록했다. [630점 대조](parity280/checks.json).

완료 38개의 원본 SHA256, C++ 대상 경로, 근거 파일 존재를 검사했고 누락·해시 변경은 없었다. 이는 전체 478개의 완료를 뜻하지 않는다. [이식 기록 무결성](migration_integrity.json).

## 이번에 수정한 코드

- EDA에 `--ref-summary <summary.yml>`를 추가했다. 새 결과는 `final/results`에 쓰면서 기준 결과는 기존 `hw-analysis/docs`에서만 읽던 연결을 보완했다. 이번 1차 실행의 summary를 2차 분석에 직접 넣어 CSV 3개와 summary가 기존 결과와 일치하는지 확인했다. 옵션 값 누락·없는 파일은 측정 전에 거절한다. 기존 `--ref <묶음>`만 쓰는 방식은 호환성을 위해 기존 위치를 읽는다. [현재 실행 안내](../../../실험%20방법/tool/hw/eda/README.md), [명시적 기준 재현](eda_explicit_reference_comparison.json), [실행·입력 거절](eda_fix_commands.json).
- ECC 진단에 선택 설정 인덱스 `0..6`을 추가했다. 인자가 없을 때 기존 7개 설정을 그대로 계산한다. 설정별 68장을 독립 실행해 합칠 수 있도록 했으며 계산식·영상 선택·수렴 조건·스레드 수는 바꾸지 않았다. 잘못된 인덱스 거절을 확인했다. [입력 검사](ecc_variant_input_checks.json). 최초 순차 ECC 실행은 병렬 전수 검증으로 대체하기 위해 의도적으로 중단했고, 그 기록과 불완전 출력 `ecc_tune`은 보존했다. 이 중단을 알고리즘 오류로 집계하거나 성공으로 표시하지 않는다.
- `main inspect`가 NPZ 경로를 읽기 전에 정확히 한 개의 파일 인자를 확인하도록 수정했다. 인자 누락·과다 입력의 오류 종료와 정상 NPZ 검사를 확인하고 CTest를 다시 실행했다. [CLI 입력 검사](main_cli_commands.json).
- README·진행 상태·정책에 이번 사용자의 조건부 이동 지시와 현재 완료 범위를 기록했다. 기존 규칙과 원본 실험 결과는 변경하지 않았다.

검증용 Python 스크립트는 이 폴더의 `scripts`에 모았다. C++ 제품 추론 경로에서 Python을 실행하지 않는다. 원본 Python 프로그램 전체를 다시 실행한 검증은 아니며, 실제 C++ 계산과 보존된 원본 관측·예측·출력을 대조했다. 웹 전체 기능과 미이식 실험을 C++로 대체했다고 주장하지 않는다.

## 보존 및 다음 정리 조건

기존 코드·데이터·기존 결과를 이동하거나 삭제하지 않았고 legacy에도 추가하지 않았다. 새 실행은 모두 새 결과 디렉터리에 기록했다. 원본을 이동하려면 먼저 미완료 범위를 처리하고, 필요한 관측·모델·문서를 명시적 입력으로 분리하고, 원본 경로를 사용할 수 없는 상태에서도 해당 실행이 통과하는지 확인해야 한다. 그다음 원본별 SHA256 이동 목록과 경로 변경을 함께 기록하고 이동 후 재검증해야 한다.

실제 명령과 종료 코드는 `commands.json`, `supplement_commands.json`, `eda_commands.json`, `eda_fix_commands.json`, `ecc_parallel_commands.json`, `original_parity_commands.json`에 보존한다. 최초 오류·불완전 실행도 덮어쓰지 않는다.
