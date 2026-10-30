# joint_center_630_20261022 실험 출처

## 문제

복수 원점 후보와 Y/T를 함께 추정

## 조사

P12·QR18·QR36·raw1260, 공유 곡면과 후보별 곡면, 탐색 반경 확대

## 해결 방법

관측 특징→학습된 후보별 비용→두 축 동시 출력

## 유지한 부분

고정 구조 ID와 경계 목적함수, 정답·시행을 추론에 전달하지 않음

## 바꾼 부분

동결 모델 NPZ 로딩과 특징·후보 선택을 C++로 이식

## 다른 실험과의 구분

16:02 결과와 16:11 모델을 섞으면 불일치. local_surfaces의 같은 시점 예측표와 비교해야 함

## 시간과 작성 흔적

관측된 최초 파일 생성: 2026-10-22T15:57:24.477+09:00. 관측된 최종 파일 수정: 2026-10-22T16:14:48.061+09:00. 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.

### benchmark.py

- 원본: `joint_center_630_20261022/benchmark.py`
- 생성 KST: 2026-10-22T16:08:38.284+09:00
- 수정 KST: 2026-10-22T16:14:48.061+09:00
- SHA256: `ac8e3a2267f38ec0fa0767307657ae430e2d9e1086f1efb641fd868e2f9ae922`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### diagnose.py

- 원본: `joint_center_630_20261022/diagnose.py`
- 생성 KST: 2026-10-22T16:07:37.667+09:00
- 수정 KST: 2026-10-22T16:13:16.978+09:00
- SHA256: `9f544779ef3f9c44bbfa2e9ade9587e2480f95aa14991ca078970b1d539c39ad`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### engine.py

- 원본: `joint_center_630_20261022/engine.py`
- 생성 KST: 2026-10-22T16:08:38.280+09:00
- 수정 KST: 2026-10-22T16:14:36.166+09:00
- SHA256: `4afe5e60b739311913c44619a423594b445242df1640e0598bc0cea629726904`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### evaluate.py

- 원본: `joint_center_630_20261022/evaluate.py`
- 생성 KST: 2026-10-22T16:01:46.925+09:00
- 수정 KST: 2026-10-22T16:11:23.059+09:00
- SHA256: `49c648701e223965ac359fa016358bed0ddc513f782b691189828292417992f0`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### extract.py

- 원본: `joint_center_630_20261022/extract.py`
- 생성 KST: 2026-10-22T15:58:12.821+09:00
- 수정 KST: 2026-10-22T15:58:12.822+09:00
- SHA256: `bac78433ef42e7abdd54432eb59ec1b9f555441fb907d7794e5d48028bd11354`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### features.py

- 원본: `joint_center_630_20261022/features.py`
- 생성 KST: 2026-10-22T16:04:21.799+09:00
- 수정 KST: 2026-10-22T16:04:21.800+09:00
- SHA256: `04ab5869d4e63f167b55edf423b2f903745e31d35b5e3b6570bcc8b217097758`
- 관측된 작업자: unknown
- 이식 상태: `ported_feature_maps_validated_QR36_341_replay`
- C++ 위치: `src/alignment.cpp`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### local_model.py

- 원본: `joint_center_630_20261022/local_model.py`
- 생성 KST: 2026-10-22T16:11:22.922+09:00
- 수정 KST: 2026-10-22T16:11:22.923+09:00
- SHA256: `6b37e554790cfe6da023d514b94ec9089509a67a1526eddf8db0dff29182cba8`
- 관측된 작업자: unknown
- 이식 상태: `frozen_model_inference_ported_training_pending`
- C++ 위치: `src/joint.cpp`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### measurement.py

- 원본: `joint_center_630_20261022/measurement.py`
- 생성 KST: 2026-10-22T15:57:24.481+09:00
- 수정 KST: 2026-10-22T15:57:24.655+09:00
- SHA256: `bde63f26cb0da672598b0b2f695b0ae976a045abd8bae8984e261530eba2578a`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### model.py

- 원본: `joint_center_630_20261022/model.py`
- 생성 KST: 2026-10-22T15:59:32.123+09:00
- 수정 KST: 2026-10-22T16:11:23.058+09:00
- SHA256: `5aa62391ac9930abf70700d266cf3a3b5adca739423b0c0d97007e2d2a7c88c6`
- 관측된 작업자: unknown
- 이식 상태: `frozen_model_inference_ported_training_pending`
- C++ 위치: `src/joint.cpp`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### reference.py

- 원본: `joint_center_630_20261022/reference.py`
- 생성 KST: 2026-10-22T15:57:24.477+09:00
- 수정 KST: 2026-10-22T15:11:05.666+09:00
- SHA256: `91b93e305276055665de2428b7d5088e9cbb91fccca6a9ae373a45081e331ecf`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

## 보존된 결과와 문서

- `joint_center_630_20261022/assets/corners.json` (2026-10-22T10:25:42.267+09:00)
- `joint_center_630_20261022/assets/model.json` (2026-10-21T17:23:50.354+09:00)
- `joint_center_630_20261022/assets/provenance.json` (2026-10-22T10:25:42.290+09:00)
- `joint_center_630_20261022/results/audit.json` (2026-10-22T16:02:06.251+09:00)
- `joint_center_630_20261022/results/expanded_features/audit.json` (2026-10-22T16:04:45.134+09:00)
- `joint_center_630_20261022/results/expanded_features/diagnostic_oracle_group.csv` (2026-10-22T16:07:40.757+09:00)
- `joint_center_630_20261022/results/expanded_features/end_to_end.json` (2026-10-22T16:09:08.893+09:00)
- `joint_center_630_20261022/results/expanded_features/failures.csv` (2026-10-22T16:04:45.099+09:00)
- `joint_center_630_20261022/results/expanded_features/final_centers.json` (2026-10-22T16:04:45.135+09:00)
- `joint_center_630_20261022/results/expanded_features/identifiability.json` (2026-10-22T16:07:40.769+09:00)
- `joint_center_630_20261022/results/expanded_features/predictions.csv` (2026-10-22T16:04:45.088+09:00)
- `joint_center_630_20261022/results/expanded_features/protocol.json` (2026-10-22T16:04:24.757+09:00)
- `joint_center_630_20261022/results/expanded_features/selection.json` (2026-10-22T16:04:45.133+09:00)
- `joint_center_630_20261022/results/expanded_features/splits.json` (2026-10-22T16:04:24.759+09:00)
- `joint_center_630_20261022/results/expanded_features/summary.csv` (2026-10-22T16:04:45.131+09:00)
- `joint_center_630_20261022/results/failures.csv` (2026-10-22T16:02:06.218+09:00)
- `joint_center_630_20261022/results/final_centers.json` (2026-10-22T16:02:06.252+09:00)
- `joint_center_630_20261022/results/local_surfaces/audit.json` (2026-10-22T16:11:41.958+09:00)
- `joint_center_630_20261022/results/local_surfaces/diagnostic_oracle_group.csv` (2026-10-22T16:13:19.863+09:00)
- `joint_center_630_20261022/results/local_surfaces/failures.csv` (2026-10-22T16:11:41.921+09:00)
- `joint_center_630_20261022/results/local_surfaces/final_centers.json` (2026-10-22T16:11:41.959+09:00)
- `joint_center_630_20261022/results/local_surfaces/identifiability.json` (2026-10-22T16:13:19.874+09:00)
- `joint_center_630_20261022/results/local_surfaces/predictions.csv` (2026-10-22T16:11:41.911+09:00)
- `joint_center_630_20261022/results/local_surfaces/protocol.json` (2026-10-22T16:11:26.109+09:00)
- `joint_center_630_20261022/results/local_surfaces/selection.json` (2026-10-22T16:11:41.956+09:00)
- `joint_center_630_20261022/results/local_surfaces/splits.json` (2026-10-22T16:11:26.111+09:00)
- `joint_center_630_20261022/results/local_surfaces/summary.csv` (2026-10-22T16:11:41.954+09:00)
- `joint_center_630_20261022/results/manifest.csv` (2026-10-22T15:58:59.235+09:00)
- `joint_center_630_20261022/results/p12_initial/audit.json` (2026-10-22T16:02:06.251+09:00)
- `joint_center_630_20261022/results/p12_initial/failures.csv` (2026-10-22T16:02:06.218+09:00)
- `joint_center_630_20261022/results/p12_initial/final_centers.json` (2026-10-22T16:02:06.252+09:00)
- `joint_center_630_20261022/results/p12_initial/predictions.csv` (2026-10-22T16:02:06.209+09:00)
- `joint_center_630_20261022/results/p12_initial/protocol.json` (2026-10-22T16:02:05.503+09:00)
- `joint_center_630_20261022/results/p12_initial/selection.json` (2026-10-22T16:02:06.250+09:00)
- `joint_center_630_20261022/results/p12_initial/splits.json` (2026-10-22T16:02:05.504+09:00)
- `joint_center_630_20261022/results/p12_initial/summary.csv` (2026-10-22T16:02:06.248+09:00)
- `joint_center_630_20261022/results/predictions.csv` (2026-10-22T16:02:06.209+09:00)
- `joint_center_630_20261022/results/protocol.json` (2026-10-22T16:02:05.503+09:00)
- `joint_center_630_20261022/results/selection.json` (2026-10-22T16:02:06.250+09:00)
- `joint_center_630_20261022/results/splits.json` (2026-10-22T16:02:05.504+09:00)
- `joint_center_630_20261022/results/summary.csv` (2026-10-22T16:02:06.248+09:00)
