# 원본 PPT 분석 방법 기준 재계산 (v2)

θ 단위는 사용자 확인에 따라 **degree(°)**로 확정했다.

앞선 v1은 배치 위주로 맞추고 일부 통계량도 달랐다. 이번에는 755점 전체 재계측, 2창 재중심화, 고정 630점 전체 평균, 한 장 center 기준, OLS 절편·기울기·표본 σ, 간격별 한 칸 이동, 공통/개별 residual 분해로 다시 계산했다.

| 원본 장 | 항목 | 이번 계산 | 산출물 |
|---|---|---|---|
| 6 | 절편 | dx−slope·command | zero_intercepts.png |
| 7–8 | 이동 경로 | dx–dy, 화면 세로 100배 | path_new_*.png |
| 9 | 한 칸 | 동일 point ID의 이웃 이미지 차이, 간격별 μ±3σ | steps.png |
| 11 | 시각/밝기 | 파일 수정 시각 대리축 / 원본 QR 평균 | timing_brightness.png |
| 12 | 노이즈 | 점별 OLS residual → 630점 공통 → 개별 σ | noise_decomposition.png |
| 17–18 | 진단 | 755점 2창 재중심화, 공개된 통과 기준 | fit_group_maps.png |
| 19 | 점 지도 | 공통 제거 후 점별 σ 지도 | point_noise_map.png |
| 20 | 방법 비교 | 755 / 630 / 통과점 / ECC | step_statistics.csv |
| 21–22 | 판정 정확도 | 원본 코드·P12·분할 미제공으로 재현 불가 | 제한 명시 |
| 23 | 기준점 | affine 평가점을 (1,.75)px 변경 | reference_point_perturbation.csv |
| 24 | 패치 | 명시한 두 ROI의 affine ECC 비교 | patch_sensitivity.png |

## 핵심 기준

- 신규 기준은 `기준 위치/center1.bmp` 한 장이다. center2/3 평균으로 기준을 바꾸지 않는다.
- 고정 630점 평균은 QR별 균등 평균이 아닌 **630점 전체를 동일 가중**한다. ECC는 3QR 패치 중심 이동의 평균이다.
- σ는 ddof=1 표본 표준편차이다. 기존 v1의 반복 RMS와 바꾸어 읽지 않는다.
- 점별 `d_ij = a_j+b_j*q_i+e_ij`를 적합한 후 `c_i = mean_630(e_ij)`, `u_ij=e_ij-c_i`로 분해한다. 공통 흔들림은 std(c), 점 하나 흔들림은 std(u)이다. 직선 잔차에는 비선형도 섞인다.
- 새 fixed-axis 그룹은 분석 목적의 그룹이다. 실제 연속 촬영 trial이라고 단정하지 않는다.
- 새 간격은 불규칙하다. 동일한 명령 간격끼리만 한 칸 μ·σ·3σ 초과 여부를 계산한다.
- 원본 이미지는 smoothing/denoising하지 않았다. raw residual과 진단 탈락점도 보존한다.

## 원본 수치 대조와 재현 한계

기준 영상 통과는 이번 600/755, 원본 PPT는 597/755이다. 같은 문서화 기준을 복원했으나 원본의 미공개 optimizer 설정까지 동일하다고 주장하지 않는다.

원본 PPT는 fitting 세부 초기화·반복 수·허용 오차, ECC Gaussian 설정, 판 전체 밝기 ROI, 고정창 좌표, P12 정의와 저장 분할을 제공하지 않는다. 해당 항목은 차이를 명시하며 원본 수치를 맞추기 위해 tuning하지 않았다. 센서 잡음이나 판정 정확도를 임의로 재현했다고 표시하지 않는다.

## 재현

```powershell
python tool/offaxis/ppt_extract.py --root "data/SSAFY Test Image Set" --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2
python tool/offaxis/ppt_patch_compare.py --root "data/SSAFY Test Image Set" --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2
python tool/offaxis/ppt_analysis.py --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2
python tool/offaxis/ppt_figures.py --root "data/SSAFY Test Image Set" --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2
python tool/offaxis/ppt_report.py --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2
```

`points.csv.gz`는 재계측 raw fitting, `displacements.csv.gz`는 두 세션의 기준점·변위, `point_linear_residuals.csv.gz`는 공통/개별 분해 전후의 residual이다. 그래프별 집계는 일반 CSV로 제공한다. 파일 시각은 촬영 로그 대신 사용한 대리 지표이다.

## 참고 브랜치 확인

인증 후 `origin/chore/hw-merge` (ef01a82d98d16a077c5ad67443ae4bd2192aa9f5)를 확인했다. `src/alignment.cpp`는 y와 theta를 0으로 반환하는 placeholder이며 원본 PPT 분석 코드나 CSV는 없다. `chore/hye/hw-merge` 이름의 원격 브랜치는 확인되지 않았다.

대각선 125점은 fitting 진단에서 모두 탈락했다. 이 점들의 작은 직선 잔차 σ는 부정확한 위치 모델이나 초기점 고정에 의해서도 생기므로 높은 계측 정확도의 근거로 해석하지 않는다.
