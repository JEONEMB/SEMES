# Y 보정과 θ→Y 순차 실행

작성일: 2026-10-29. `origin/dev` d64a62fb19daabf911b44620e84147bdb773e929의 `Alignment_t {double y; double t;}`와 `alignment(cv::Mat&)` 형식을 유지합니다. 현재 코드는 `.tmp/theta_implementation_20261029/payload`에 작성한 변경본이며, 팀 저장소 반영·커밋·push는 수행하지 않았습니다.

## 채택한 Y 기준

사용자가 2026-10-29 선택한 **4번 제외 OLS·HAC 공통 범위**를 `configs/y_alignment.json`에 그대로 옮겼습니다. 추가 ±0.045mm 허용대나 고정 중점 목표를 만들지 않았습니다. 아래 숫자의 전체 정밀도는 JSON에 보존되어 있습니다.

| QR | 하한(px) | 상한(px) |
|---|---:|---:|
| L | 1641.5084992273016 | 1642.0520375163053 |
| C | 1899.5189918887024 | 1900.0669405323601 |
| R | 2158.5222280477115 | 2159.072710092858 |

이 범위는 기존 연구에서 추적한 QR 영역 중심의 **가로 좌표**입니다. 새로 검출한 QR 도형의 기하학적 중심이나 QR 전체 면적이 아닙니다. 입력은 원본 3840×2160, 좌상단 픽셀 중심 x=0, 오른쪽이 양수입니다. 기준 영상의 ROI 중심은 `(x0+x1-1)/2`이고 현재 좌표는 여기에 ECC 아핀 중심 변위 `dx`를 더한 값입니다.

목표 범위 출처는 `analysis/zero_equivalence/without_group4/equivalence_results.json`의 `intersection_of_primary_and_sensitivity`입니다. 범위에 반영된 구간은 2·3·5·6·7이고, Y 변화율은 `analysis/first_y_slopes/y_slope_results.json`의 2·3·6·7번 QR별 기울기 12개입니다. θ 계산은 기존 4·5번 선 각도 변화율을 계속 사용합니다. 새 회귀나 동등성 검정은 수행하지 않았습니다. [provenance.json](provenance.json)에 원본 해시와 사용자 선택을 기록했습니다.

## 계산 순서

1. 현재 이미지의 θ를 기존 방법으로 측정합니다. θ 보정이 필요하면 Y를 계산하지 않고 θ만 반환합니다.
2. θ 조건을 만족하면 QR 세 개를 각자의 템플릿으로 ZNCC→ECC 아핀 정합합니다. 중앙 QR 하나의 이동량을 세 QR에 복제하지 않습니다.
3. 세 좌표가 각자의 기존 범위 안에 있으면 완료입니다.
4. 범위 밖이면 각 QR과 각 촬영 구간의 기울기 `a`에 대해 `low <= x + a*dY <= high`를 만족하는 이동 구간을 계산합니다. 기울기가 모두 양수이므로 각 구간은 `[(low-x)/a, (high-x)/a]`입니다.
5. 이 12개 구간의 교집합 안에 있는 0.05mm 정수배 명령을 찾고, 여러 개면 절댓값이 가장 작은 명령을 반환합니다. 최종 후보가 12개 예상 좌표 조건을 모두 만족하는지 다시 확인합니다.
6. 실제 이동·재촬영 후 새로운 이미지로 θ와 Y 조건을 다시 확인합니다. 예상 위치만으로 완료 처리하지 않습니다.

네 촬영 구간의 기울기를 모두 가능한 경우로 다루는 것은 제어 설계 선택입니다. 확률이나 신뢰구간을 새로 구한 것이 아니며, 물리적 정확도를 보증하지 않습니다. 계산 오차 방지용 약 5.5e-11px 비교 여유만 있으며 운영 허용대를 넓히지 않습니다.

## 반환 상태

| 상태 | 뜻 |
|---|---|
| `move_theta` / `move_theta_extrapolated` | θ 명령만 반환. 후자는 기울기 자료의 ±1° 밖 추정 |
| `move_y` | θ 조건 충족 후 Y 명령만 반환 |
| `aligned` | 현재 이미지에서 θ와 세 QR의 Y 조건 모두 충족 |
| `no_common_y_interval` | 12개 조건을 함께 만족하는 연속 Y 이동 구간이 없음 |
| `no_reachable_y_tick` | 연속 구간은 있으나 0.05mm 단위 명령이 없음 |
| `theta_unresolved_at_tick_resolution` | 기존 θ 계산이 이동 단위 안에서 해결되지 않음 |
| `measurement_failed` | CLI에서 잘못된 크기·빈 이미지·정합 실패 등을 기록 |
| `move_limit_reached` | 세션에서 실제 완료된 이동이 20회라 추가 명령 차단 |

실패를 `{0,0}` 정렬 완료로 바꾸지 않습니다. 원래 반환 형식에 상태 필드가 없으므로 `alignment()`는 실패 시 예외를 던지고, 자세한 진단이 필요한 호출자는 `SequentialAligner::estimate()`의 `AlignmentDecision`을 사용합니다.

## 스켈레톤에 연결하기

```cpp
// One call at the start of a new positioning task, not on every frame.
configureAlignment(referenceImage,
                   "configs/theta_alignment.json",
                   "configs/y_alignment.json");

// The external controller supplies a newly captured image each time.
Alignment_t command = alignment(currentImage);
// command.y: nominal Y mm, command.t: nominal robot theta degrees.
// In combined mode (0,0) means both measured criteria passed.

// After the external controller actually finishes a NONZERO movement:
acknowledgeAlignmentMoveExecuted();
// Then capture a fresh frame and call alignment(newImage).
```

사용 기호는 제공된 파일 라벨의 Y/θ 부호입니다. 실제 제어기 축 부호와의 연결은 장비 연동 코드에서 동일 좌표계를 사용해야 합니다. 원본 1차 `center.bmp`를 초기화에 사용합니다. 다른 기준 사진을 넣고 같은 절대 목표 범위를 그대로 쓰면 좌표 기준이 달라집니다.

현재 코드는 기계 통신·이동·촬영을 수행하지 않습니다. 이동 완료 통지는 호출자가 실제 성공을 확인한 뒤 호출해야 합니다. 통지 전 중복 명령과 통지 중복을 거부합니다. Y·θ를 합쳐 완료 이동 20회까지 허용하며, 20번째 이동 뒤에는 새 이미지의 완료 판정은 허용하지만 21번째 명령은 막습니다. 프레임마다 재초기화하면 횟수가 리셋되므로 초기화는 정렬 작업 시작 때 한 번만 합니다. 전역 스켈레톤 어댑터는 동시 호출을 지원하지 않습니다.

두 인자 `configureAlignment(reference, thetaConfig)`는 이전 θ 단독 호환용입니다. 전체 순차 동작과 세션 횟수 관리는 위의 세 인자 초기화를 사용합니다.

## 파일 실행과 테스트

프로젝트 루트에서:

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\main.exe --reference "C:\data\center.bmp" --config configs/theta_alignment.json --y-config configs/y_alignment.json --output result.json "C:\data\current.bmp"
```

`--input-list paths.txt`는 UTF-8 파일의 한 줄당 이미지 경로를 읽습니다. CLI의 여러 입력은 **서로 독립된 오프라인 이미지**이며 실제 이동 순서로 취급하지 않습니다. 이 모드에서는 이동 완료 횟수를 세지 않습니다. 일부 입력이 측정 실패 또는 명령 산출 불가이면 나머지도 처리해 JSON을 저장하고 종료 코드 1을 반환합니다. 초기화·인자 오류는 2, 모든 입력 처리 성공은 0입니다. `--y-config` 생략 시 θ 단독입니다.

네이티브 통합 테스트는 `y_alignment_test.exe configs/theta_alignment.json configs/y_alignment.json reference_path.txt`로 실행합니다. 마지막 파일은 UTF-8로 원본 center.bmp 절대 경로 한 줄만 담습니다. 테스트가 메모리에서 5px 이동한 입력을 만들며 원본 이미지를 저장하거나 변경하지 않습니다.

## 확인한 범위

MSVC v141 / OpenCV 3.4.15 / x64 Release로 컴파일했습니다. θ·Y 두 CTest 묶음이 통과했고, Y 테스트는 공통 구간·양/음 이동 단위·경계 포함·세 QR 불일치·네 번째 기울기 적용·θ 우선·실제 이동 통지·20회 제한을 확인합니다. 원본 center.bmp와 알려진 5px 이동 테스트에서 QR 좌표, Y 방향, 스켈레톤 연결이 통과했습니다. 실제 θ 이미지 두 장은 CLI의 θ 우선 분기 확인에만 사용했습니다. 전체 촬영 데이터 정확도 분석이나 장비 수렴 검증은 아닙니다.

상세 결과는 [validation_summary.json](validation_summary.json)에 있습니다. 작업 폴더의 `verify_y_implementation.py`로 테스트 및 출처 일치 검사를 재현할 수 있습니다. 실제 장비 보정 정확도, 전체 데이터 성능, 20회 이내 물리적 수렴은 미검증입니다.
