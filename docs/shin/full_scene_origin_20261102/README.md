# 특징 신뢰도와 공통 원점 재검증 v2

## 사진에서 사용 영역 확인

- `output/pdf/T_axis_used_features_photo_atlas.pdf`: 사진 표시 6쪽 PDF.
- `features_on_original_photo.png`: 원본 전체 사진에 사용 영역 표시.
- `features_central_zoom.png`, `features_background_zoom.png`: 실제 부위 확대.
- `features_A_all_patches.png`: 원점 생성에 쓰인 A 특징 22개.
- `features_B_all_patches.png`: 생성에서 보류한 검증 B 특징 17개.
- `features_QR_actual_corners.png`: QR 212+210+208=630개 모서리.
- `annotated_feature_ID_map.csv`: 그림 ID와 원본 픽셀 좌표, 역할 대응.

분홍 A = 원점 생성, 초록 B = 보류 검증, 파랑 = QR 실제 모서리, 노랑 = 상부 선의 법선 측정 표본입니다. 자연 특징 네모는 41×41 범위이며 2px 간격 21×21 밝기 표본을 정합했습니다. 패치 중심 자체가 꼭짓점이라는 뜻이 아닙니다.

## 분석 결과

`output/pdf/T_axis_origin_quality_revalidation_v2.pdf`가 10쪽 상세 보고서입니다.

최초 후보 154개 중 보정 파일럿의 측정 품질로 75개를 전수 추적했습니다. 방향별 정보 행렬, gain/offset 제거 후 공분산, 정·역방향 오차, 광도 잔차를 함께 검사했습니다. 후보 생김새가 아닌 측정 안정성으로 채택했습니다. 모든 영상에서 90% 이상 엄격 검사를 통과한 후보는 37개입니다. 모델별 최종 후보 선택은 학습 영상 내 유효율 85% 기준이며 각 영상 관측은 개별 엄격 기준을 다시 통과해야 합니다.

원점 생성 A와 검증 B는 기준·유효 target 영상 모두에서 측정 패치 겹침 0건입니다. 오른쪽 B 관측은 원점/바이어스 목적함수에 넣지 않았습니다. 학습 자세에서 별도 반응식을 보정한 뒤 보류 자세에서 B 위치를 검사했습니다. 이것은 물리적 원점 참값 검증과는 다릅니다.

기존 자료 575장 중 라벨 의심 4장을 제외했습니다. 특징 선택용 고정 파일럿과 같은 자세의 반복을 모두 보정에 남긴 20장 외, 551장을 동일 자세 묶음 및 연속 자세 블록으로 보류했습니다. 비교의 공분산·특징 선택은 매 fold의 학습 관측에서만 추정했습니다.

외부 특징은 보류 오른쪽 부위 설명을 개선했으나 QR 오차와 상충합니다. 특히 center3 기준 사진을 제외한 사후 검증의 오른쪽 잔차가 일반 보류 95백분위를 넘었습니다. 따라서 `standard_origin_v2.json`은 **검증 미충족 후보**로 표시했습니다. 원본 파일명·라벨·기존 모델은 수정하지 않았습니다. 이전 v1의 개선율을 물리적 원점 정확도라고 해석하지 마세요.

## 재현

현재 워크스페이스의 기존 Lanczos 재측정 캐시와 MSVC v141/CMake, bundled Python 및 `.analysis_packages`가 필요합니다. `reproduce.ps1`은 네이티브 측정기 빌드부터 PDF 생성까지 순서대로 수행합니다. 원본 이미지는 저장소 외부 지정 경로에서 읽기만 합니다.

좌표·실측치 및 검사 결과: `measurements.npz`, `quality.npz`, `natural_measurements.csv`, `directional_feature_quality.csv`, `strict_cv_predictions.csv`, `contiguous_cv_predictions.csv`, `heldout_reference_checks.csv`.

새 단일 기준의 잠정 QR 630점과 자연 특징은 `standard_origin_v2_QR630.csv`, `standard_origin_v2_natural.csv`입니다. 가정과 검증 실패 상태를 무시하고 확정된 실제 좌표로 취급하면 안 됩니다.
