# 촬영 세션·원점 EDA

`report.html`을 열면 전체 해석, 6개 그림, 14개 세션 후보 통계와 CSV 링크를 볼 수 있습니다.

- 원본 575장, 기존 OOF 예측 571장, 기준 사진 4장.
- CreationTime은 2026-11-02 복사 시점으로 판단되어 촬영 구분에 사용하지 않았습니다.
- 촬영 세션은 확인된 사실이 아닌 수정 시각의 120초 초과 공백에 따른 후보입니다.
- Lanczos-3 + 해석 미분 + 경계 제한 감쇠 Gauss–Newton으로 575장을 전수 재측정했습니다.
- SHA-256 전수 일치, 실패 0, 기존 104특징 최대차이 0.
- 원점의 실제 영상 차이는 있으나 T 원점 이동량의 안정적인 환산에는 실패했습니다.
- 4개의 라벨 이상을 기록했습니다. 원본 및 기존 예측은 변경하지 않았습니다.

## 재현

현재 워크스페이스와 bundled Python, `.analysis_packages`, 기존 빌드 산출물이 필요합니다.

```powershell
# 메타데이터/Actual 재검사, 원본 해시 전수 대조, 특징 재측정 결과 대조, 통계/그림/보고서 재생성
& .\analysis\session_eda_20261102\reproduce.ps1

# 원본 전수 특징 재추출부터 다시 실행
& .\analysis\session_eda_20261102\reproduce.ps1 -Remeasure
```

`filesystem_metadata.csv`는 최초 작업 시 읽은 원본 CreationTimeUtc/LastWriteTimeUtc 스냅샷입니다.
`remeasured/`에는 재추출한 575장 특징과 630점·120선 표본 캐시를 저장했습니다.
원본 이미지는 저장소 외부 지정 경로에서 읽기만 했습니다.

`reference_origin_comparison.csv`의 상당 Y/T와 `origin_sensitivity.csv`는 불안정성을 확인하기 위한 진단 계산입니다. 확정된 원점 보정값이나 물리적 참값으로 사용하면 안 됩니다.
`samples_with_errors.csv`는 기존 method 7 OOF 결과입니다. 이번 EDA로 새 모델의 독립 평가를 수행한 것은 아닙니다.
