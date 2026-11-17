# 00004_final_Y_T — QR·바·나사 Huber66 전달 후보

2026-11-11 사용자 요청으로 직전 평가의 원본 정확도 후보를 이 폴더에 패키징했다. `experiments/selected_modules.cmake`의 joint 선택을 이 모듈로 연결한다. 기존 control은 유지한다. 목표·평가 판정·추론 입력 제한의 원본은 [ALGORITHM_RULES.md](../../../ALGORITHM_RULES.md)이며, `final`이라는 폴더 이름은 현장 안정성 검증 완료를 뜻하지 않는다.

## 포함한 알고리즘과 이유

| 단계 | 적용 | 이유 |
|---|---|---|
| 기준 | 동봉 center.bmp·geometry.json·line_protocol.json 고정 | 평가와 실행의 기준/좌표를 일치시킨다. 새 기계에서 기준만 교체하면 기존 모델의 성능을 보장하지 않는다. |
| QR | QR3개 NCC 초기 위치 → ECC homography → 각3×3 투영 위치의 dx/dy, 총54값 | 평균 이동과 QR 내부 형상 변화를 함께 남긴다. 27개 위치는 독립 검출한27개 코너가 아닌 가상 투영 위치다. |
| 바 | 상·하 각24구간 수직 프로파일 정합 → 최소제곱 평균 이동/기울기, 총4값 | 영상에서 선의 법선 방향 변화를 추출한다. 접선 방향 물질점을 추적한다고 해석하지 않는다. |
| 나사 | **나사4개**, 각91×91 패치 NCC → gain/offset을 맞추는 이동 GN, x/y 총8값 | 작은 상대 이동을 측정한다. 나사8개가 아니며 십자 위치는 패치 기준점이다. 물리 중심의 절대 정답 검증은 아니다. |
| 특징 | QR54 + 바4 + 나사8 =66 | 직전 712/715 후보의 특징 순서와 동일하다. |
| 추정 | train 표준화 → ridge1e-7 선형식 → 두 축 공유 Huber 가중6회 | 큰 학습 잔차의 영향을 낮춘 후보가 원본 개발 평가에서 높았다. 옛 final66의 큰 잔차를 더 가중하는 방식과 다르다. |
| 실패/진단 | 예외, measurement_valid, outside_training_range | 실패를0,0으로 숨기지 않는다. 측정 유효 또는 범위 내라는 표시는 자세 안정성 보증이 아니다. |

회귀 가중은 `r=max(abs(eY)/0.045,abs(eT)/0.01)`, `w=min(1,1/max(r,1e-12))`이며 반복마다 평균가중을1로 정규화한다. 절편은 ridge 규제에서 제외한다. 이미지/숫자 증강 없이 학습했다.

이 패키지의 기본 출력은 Huber66이다. 전문가 중앙값·나사 없는 경로 전환을 조용히 섞지 않았다. 비교 후보들은 원본 성능과 최대오차에서 손실이 남았으므로 실험 성적만 evidence에 함께 기록했다.

## 입력·출력과 사용

- 입력:3840×2160, uint8, gray/BGR/BGRA 한 장. 입력 영상은 수정하지 않는다.
- 초기화:`initialize_joint("experiments/hye/00004_final_Y_T/assets")`를 실행하고 `joint(image)` 또는 기존 `alignment(image)`를 호출한다.
- 출력:`Alignment_t{y,t}` 추정값. 로봇 이동 명령이 아니다. 학습 파일명 Y/T 좌표계이며 T를 임의로 도 단위로 바꾸거나 부호를 반전하지 않는다.
- 추론은 파일명·촬영일·정답을 받지 않는다. 초기화는 완전히 성공한 뒤 교체하므로 실패한 재초기화가 이전 모델을 없애지 않는다.
- 필요한 파일:`assets/center.bmp`, `geometry.json`, `line_protocol.json`, `huber_linear66.yml`, `provenance.json`. 주요4개 파일의 SHA-256을 초기화에서 검사한다. 다른 모델/기준을 넣으려면 별도 재검증이 필요하다.
- 함수 선언은 저장소 `include/alignment.hpp`, 구현은 `src/*.cpp`. 이 폴더의 `include/final04.hpp`는 진단/학습 도구용 내부 API다.
- 내부 실행은 Python, 개인 경로, 다른 실험 폴더에 의존하지 않는다. 원본 사진은 재현 감사에만 별도 공급한다.

```cpp
#include "alignment.hpp"
// img: 호출부에서 준비한3840x2160 uint8 영상
initialize_joint("experiments/hye/00004_final_Y_T/assets");
Alignment_t estimate = joint(img);
AlignmentDiagnostics diagnosis = alignment_diagnostics();
// 실패는 std::exception으로 전달되므로 호출부가 처리한다.
```

## 성능과 한계

평가 대상은 개발에 노출된715장(기존571+4차144). 각 outer fold의 학습사진으로만 표준화·계수를 학습한 OOF 성적이다. **동봉 모델은 같은 학습식으로 전체715장을 학습한 배포 계수**이며, 그 모델을 전체 학습자료에 다시 채점한 점수를 아래 성적으로 제시하지 않는다. 이상사진을 제외하거나 라벨을 수정하지 않았다.

| 개발 평가 | 동시 통과 | Y 평균 / p95 / 최대 | T 평균 / p95 / 최대 |
|---|---:|---:|---:|
| 혼합이미지80:20,5fold |712/715 (99.58%)|0.010339 /0.030486 /0.044992|0.004813 /0.006405 /1.677200|
| 동일날짜+자세 그룹5fold |710/715 (99.30%)|0.010348 /0.031506 /0.044292|0.004832 /0.006587 /1.676583|

| 동일 혼합OOF의 숫자 교란 | 동시 통과 |
|---|---:|
| 네 나사 x에 모두+0.5px |0/715|
| 네 나사 y에 모두+0.5px |654/715|
| 한 나사 x+3px (번호를 순환) |0/715|
| 나사별 독립좌표잡음 표준편차0.5px |168/715|

공통 x편향0.5px가 Y약−0.14/T약+0.03으로 증폭된다. 따라서 높은 원본 통과율을 나사 흔들림 안정성으로 해석하면 안 된다. 최대T 사진은 `Y-0.8 T-0.96.bmp`(ID702)이며 OOF예측T+0.7172다. 원본QR와 나사의 이동도 양의T 사진에 가깝고 캐시/원본 바이트는 일치한다. 명령과 실제 촬영시점 상태 불일치가 의심되나 실제축 로그 없이 라벨오류로 확정하지 않았다. `evidence/worst_T_case.json`에 기록했다.

실제 같은 자세5회 재촬영, 시간 진동, 신규 기계, 실제 로봇 복귀는 미검증이다. 오검출·공통편향 대응까지 해결한 최종판이라는 주장은 하지 않는다.

## 빌드·실행

팀 기준은 [Windows x64/MSVC v141/OpenCV3.4.15 vc15](../../../docs/windows-setup.md)다. 저장소 루트에서 기존 CMake로 빌드하면 이 joint가 선택된다. 이 모듈만 검증하려면 PowerShell:

```powershell
cmake -S experiments/hye/00004_final_Y_T/tool/standalone -B build/final04 -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR="C:/opencv/build"
cmake --build build/final04 --config Release
ctest --test-dir build/final04 -C Release --output-on-failure
.\build\final04\joint\Release\final04_tool.exe predict experiments/hye/00004_final_Y_T/assets "C:/photos/current.bmp"
```

DLL은 빌드 후 실행파일 옆으로 복사한다. OpenCV 버전/비Windows 환경의 명시적 감사 옵션은 성능 검증 면제가 아니라 비교시험용이다.

이번 실제 빌드 환경은macOS arm64/OpenCV3.4.15다. 해당 환경에서는 `-DSEMES_ALLOW_UNVALIDATED_PLATFORM=ON`으로 구성하고 `build/final04/joint/final04_tool`을 실행한다. **Windows/MSVC v141에서 실행했다고 주장하지 않는다.** 팀 PC에서 동봉 계약검사와 원본사진 수치대응을 확인해야 한다.

### 동봉 캐시로 재현

`final04_release`는 C++17 도구다. 아래 입력파일은 모두 동봉되어 있으며 기존 결과가 있으면 덮어쓰지 않고 중단한다. 출력은 새 경로를 사용한다. Windows는 해당 Release의`.exe`로 실행한다.

```text
final04_release verify evidence/training_features.tsv evidence/image_oof.tsv NEW_image_oof.tsv
final04_release verify evidence/training_features.tsv evidence/group_oof.tsv NEW_group_oof.tsv
final04_release train evidence/training_features.tsv NEW_model.yml NEW_training_meta.json
final04_release audit assets evidence/training_features.tsv ORIGINAL_IMAGE_MANIFEST.tsv NEW_image_audit.tsv
```

명령은 이 모듈 폴더 기준이다. 실제 실행파일 경로를 앞에 붙인다. audit manifest는 `id<TAB>path` 헤더 뒤에 사진ID와 로컬 원본경로를 적는다. 원본SHA는 training_features.tsv와 일치해야 한다. 이 manifest는 채점/감사용이며 추론 함수에 파일명 라벨을 넘기지 않는다.

`final04_tool repeat5`에는 같은 자세에서 독립 촬영한5장을 넣는다. 동일 바이트 파일 재사용을 거부하지만 같은 자세/실제5회 촬영 여부는 호출자가 확보해야 한다. `final04_overlay`에는 `ASSETS IMAGE NEW_OUTPUT_DIR`를 주어 QR·바·나사 오버레이를 저장할 수 있다(출력 규격은 tool/overlay.cpp 참조).

## 이번 패키지 검증

- C++ 독립 빌드 및 계약검사1/1 통과:초기화,입력불변성,색변환,실패시진단,실패한재초기화,빈/잘못된영상,특징순서,모델스키마.
- 배포 학습식으로 혼합/그룹 OOF를 다시 계산:712/715·710/715 재현, 기존 baseline66의 모든 예측과 최대차0.
- 전수학습 모델 저장→재로드의 예측 대응 통과.
- 기존/4차를 포함한16개 원본사진에서 이미지→66특징→예측을 캐시와 대조. 상세값은 `evidence/image_parity.tsv` 및 로그. 이는16장 이식 감사이며 새 독립정확도 평가가 아니다.

코드·assets·작은 수치근거를 이 커밋에 포함한다. raw ZIP/전체촬영이미지, 다른 실험 폴더·브라우저 산출물은 포함하지 않는다. `artifact_sha256.json`은 전달 파일 무결성 목록이다. 전체 관측 캐시가 아닌 선택된715×66 수치와 라벨/원본해시만 `training_features.tsv`에 포함했다.
