# 저장소 밖 분석·실험 코드 통합 (2026-10-30)

> 2026-10-30 이후 폴더 이름과 배치는 [README](README.md)의 실험 목록이 기준이다. 아래 표의 새 이름은 그 전 단계 이름이다.

10-06 ~ 10-29에 저장소 밖 `projects/2026-10_SEMES/`에서 한 분석·실험의 코드와 문서를 옮겼다. 실험마다 `experiments/hye/<날짜>_<실험>/` 폴더를 만들고, 그 안을 `실험/`(코드·빌드·계획)과 `결과/`(README·보고서·로그·결과)로 나눴다. 원래 폴더 이름과 새 이름은 맨 아래 표에 있다.

## 넣은 것과 뺀 것

- 넣음: `.py .cpp .hpp .h .c .cmake .md .txt .ps1 .bat .toml .yml .yaml`, `CMakeLists.txt`, `CMakePresets.json` 중 1MB 이하
- 뺌: 영상(BMP·PNG·MP4), 측정·모델 파일(npz·joblib·json·csv), 발표자료(pptx), 1MB 넘는 파일, `data/`, `build*/`, `deps/`(파이썬 패키지 사본), `third_party/`의 OpenCV·OpenBLAS 사본
- `final/third_party/ecc5`(ECC 호환 소스)는 넣었다
- `hw-analysis`는 같은 저장소의 옛 clone이라 커밋된 적 없는 `tool/hw`, `docs/hw`, `docs/validation-runs`만 넣었다
- 원래 폴더에 있던 하위 `.gitignore`는 원래 배치 기준이라 넣지 않았다

뺀 결과·데이터·모델은 저장소 밖 `projects/2026-10_SEMES/00000.legacies/<원래 폴더명>/`로 원래 구조째 옮겨 두었다. 그래서 여기 있는 스크립트는 대부분 이 폴더에서 바로 실행되지 않는다. 문서 안의 절대 경로도 원래 위치를 가리킨다.

## C++ 이식이 끝난 원본

`final/config/port_progress.json`에서 `cpp_complete`인 원본 55개는 여기에 넣지 않았다. 원본은 저장소 밖 `projects/2026-10_SEMES/00000.legacies/`로 옮겼고, 대응하는 C++ 구현은 [20261020_Python을_C++로_전체이식](20261019_1400_측정별판정비교_동일GLS판정기/종합/20261020_1752_C++이식_파일별이식검증) 아래(`experiments/analysis_v*`, `src`, `include`, `tool`)에 있다. 원본이 빠져서 같은 폴더의 다른 Python 스크립트가 import에 실패할 수 있다.

| 원래 폴더 | 옮긴 원본 수 |
|---|---|
| `make_robot_motion.py` (루트) | 1 |
| `video_output` | 1 |
| `analysis_agent` | 3 |
| `analysis_v2` | 4 (폴더 전체) |
| `analysis_v3` | 17 |
| `analysis_v4` | 1 (폴더 전체) |
| `hw-analysis/tool/hw/eda/src` | 15 |
| `codex_experiments_20261014` | 13 |

## 주요 폴더

| 폴더 | 내용 |
|---|---|
| `analysis_v2` ~ `analysis_v8`, `analysis_*_20261019` | 10-12 ~ 10-19 초기 분석 (QR 측정, Y/Θ 구분, 특징점·패치 이동장) |
| `eda_*_20261022`, `experiment_documentation_20261021` | 10-20 ~ 10-22 EDA와 실험 문서 |
| `codex_experiments_20261014` | 10-14 ~ 10-29 joint/control 실험 (구조점630, ECC, 위·아래 선, 337/341, 구조점 vs ECC 비교) |
| `joint_center_630_20261022`, `motion_*_20261022` | 구조점630 기준과 동작·속도 실험 |
| `final` | Python → C++ 전체 이식 작업 공간 (체크리스트 `final/전체_파일_목록.md`) |
| `legacy` | 10-22 이전 스냅샷 |

## 원래 폴더와 새 이름

| 원래 위치 (`projects/2026-10_SEMES/`) | 새 폴더 |
|---|---|
| `루트 낱개 문서` | [20261012_초기_질문·계획·리포트_문서](20261012_초기_질문·계획·리포트_문서/) |
| `analysis_agent` | [20261012_QR3영역_YΘ구분](20261012_1549_QR3개구분_ECC선형회귀) |
| `analysis_grid_corners_20261019` | [20261019_격자코너_분석슬라이드](20261019_1316_모서리추적_경계모델서브픽셀) |
| `analysis_judge_compare_20261019` | [20261019_측정방법별_YΘ판정비교](20261019_1400_측정별판정비교_동일GLS판정기) |
| `analysis_v3` | [20261013_v3_픽셀정합GN과_발표그림](20261013_v3_픽셀정합GN과_발표그림/) |
| `analysis_v5` | [20261014_v5_단계별_측정방법](20261014_1339_이동무늬판정_패치이동장) |
| `analysis_v6` | [20261014_v6_QR특징_분석](20261014_1440_특징점구분_특징점이동장) |
| `analysis_v7` | [20261014_v7_센터기준_판정조합](20261014_1557_QR2장판정_아핀18값GLS) |
| `analysis_v7b` | [20261014_v7b_센터와현재2장_YΘ판정](20261014_1559_0근처축오판_2단계판정) |
| `analysis_v8` | [20261014_v8_잡음모델과_CRLB](20261014_1654_점잡음원인_강체검사합성) |
| `codex_experiments_20261014/audit` | [20261014_기존분석_감사](20261014_1616_T005오분류_후보비용v7b비교) |
| `codex_experiments_20261014/branch_curves_4fold_20261021` | [20261021_분기곡선_4fold](20261021_1759_후보별곡선_독립변형곡선) |
| `codex_experiments_20261014/center_fixed_20261015` | [20261015_고정센터_기준](20261015_1402_센터0점고정_절편0곡선) |
| `codex_experiments_20261014/center_patch_v2_20261021` | [20261021_센터패치_v2](20261021_1723_센터단일기준_630패치GN추적) |
| `codex_experiments_20261014/classifier_4fold_20261021` | [20261021_분류기_4fold](20261021_1741_분류기판정_LDA와SVC) |
| `codex_experiments_20261014/colleague_brief_20261015` | [20261015_동료공유_요약자료](20261014_1717_280장전수_고정모델재실행) |
| `codex_experiments_20261014/decoded_grid_corners_20261015` | [20261015_QR격자코너_디코딩](20261015_1816_finder만사용_사각형66점) |
| `codex_experiments_20261014/direct_alignment` | [20261014_QR3개_영상직접정합](20261014_1622_QR직접정합_픽셀P12정합) |
| `codex_experiments_20261014/evaluate_archives_20261022` | [20261022_보관결과_재평가](20261022_1601_새자료적용_고정모델적용) |
| `codex_experiments_20261014/finder_only_20261015` | [20261015_파인더패턴만_사용](20261015_1816_finder만사용_사각형66점) |
| `codex_experiments_20261014/forward_gn_20261021` | [20261021_전방모델_GN측정](20261021_1807_GN기울기방식_현재대칭기울기) |
| `codex_experiments_20261014/full280` | [20261014_280장_전체평가](20261014_1717_280장전수_고정모델재실행) |
| `codex_experiments_20261014/fullfield_4fold_20261021` | [20261021_전체이동장_4fold](20261021_1756_전체점직접_1260차원SVD축소) |
| `codex_experiments_20261014/gls_comparison_20261014` | [20261014_GLS_비교](20261014_1621_공분산가중효과_GLS_WLS_비교) |
| `codex_experiments_20261014/green_feature_map_20261015` | [20261015_특징점지도_그림](20261015_1816_finder만사용_사각형66점) |
| `codex_experiments_20261014/hybridfield_4fold_20261021` | [20261021_혼합이동장_4fold](20261021_1758_요약과전체혼합_P12전체점혼합) |
| `codex_experiments_20261014/joint_struct_vs_ecc_lines_20261029` | [20261029_132926_joint_구조점vsECC_상하선](20261029_1329_구조점대ECC_같은fold비교) |
| `codex_experiments_20261014/joint_yt_20261022` | [20261022_joint_YT동시추정_첫실험](20261022_1519_YT동시추정_GLS이차순방향) |
| `codex_experiments_20261014/joint_yt_disentangle_20261022` | [20261022_joint_YT_분리학습](20261022_1631_YT반응분리_사진쌍분리학습) |
| `codex_experiments_20261014/joint_yt_erroraudit_20261022` | [20261022_joint_YT_오차감사](20261022_1712_실패10장누출_누출감사재선택) |
| `codex_experiments_20261014/joint_yt_generalization_20261022` | [20261022_joint_YT_일반화_구조점630](20261022_1608_새사진일반화_탐색확장국소회귀) |
| `codex_experiments_20261014/joint_yt_improved_20261022` | [20261022_joint_YT_개선](20261022_1529_동시추정오차_비선형트리증강) |
| `codex_experiments_20261014/joint_yt_motion_logic_20261026` | [20261026_joint_YT_이동로직](20261026_1410_이동전후제어_탐색반응갱신) |
| `codex_experiments_20261014/joint_yt_reboot_20261022` | [20261022_joint_YT_재시작_직접정합](20261022_1740_관측재설계_QR직접정합) |
| `codex_experiments_20261014/joint_yt_visual_feedback_20261023` | [20261023_joint_ECC와상하선_337](20261023_1756_가로선효과_ECC54위아래선) |
| `codex_experiments_20261014/likelihood_4fold_20261021` | [20261021_우도모델_4fold](20261021_1753_후보별잡음_가우스우도) |
| `codex_experiments_20261014/measurement` | [20261014_측정_추출과_정밀화](20261014_1618_같은점추적_고정ID_LK_GN) |
| `codex_experiments_20261014/modeling` | [20261014_모델링_비교](20261014_1621_일반모델선택_중첩CV_모델비교) |
| `codex_experiments_20261014/multiref_4fold_20261021` | [20261021_다중기준영상_4fold](20261021_1733_원점후보4분할_48설정중첩선택) |
| `codex_experiments_20261014/patch_windows_20261021` | [20261021_패치창_측정](20261021_1746_GN창크기_창흐림6종비교) |
| `codex_experiments_20261014/pixel_weighted_gn_20261015` | [20261015_픽셀가중_GN](20261014_1621_픽셀잡음가중_역분산가중GN) |
| `codex_experiments_20261014/ppt_six_origin_20261021` | [20261021_발표_6원점](20261019_1400_측정별판정비교_동일GLS판정기) |
| `codex_experiments_20261014/prepare_clean_20261028` | [20261028_원본정리_Python](20261026_1410_이동전후제어_탐색반응갱신) |
| `codex_experiments_20261014/prepare_clean_cpp_20261028` | [20261028_원본정리_C++](20261026_1410_이동전후제어_탐색반응갱신) |
| `codex_experiments_20261014/presentation_build` | [20261014_발표자료_차트복원](20261014_1717_280장전수_고정모델재실행) |
| `codex_experiments_20261014/release_qr536_20261021` | [20261021_QR536_배포후보](20261021_1810_QR만통과확인_내부536점고정) |
| `codex_experiments_20261014/release_screw_fallback_20261021` | [20261021_나사대체_배포후보](20261021_1804_나사추가검증_QR630+나사고정) |
| `codex_experiments_20261014/repository_rules_20261028` | [20261028_QR이동추정_공통규칙](20261026_1410_이동전후제어_탐색반응갱신) |
| `codex_experiments_20261014/robust_gn_20261021` | [20261021_강건GN_측정](20261021_1750_픽셀이상치_강건GN가중) |
| `codex_experiments_20261014/robustness_research_20261014` | [20261014_강건성_조사](20261014_1717_280장전수_고정모델재실행) |
| `codex_experiments_20261014/spline_4fold_20261021` | [20261021_스플라인_4fold](20261021_1801_곡선형태_평활스플라인) |
| `codex_experiments_20261014/structural_motion_20261015` | [20261015_구조점_이동그림](20261015_1838_모서리정밀화_경계모델적합) |
| `codex_experiments_20261014/subpixel_grid_corners_20261015` | [20261015_부화소_격자코너](20261015_1838_모서리정밀화_경계모델적합) |
| `codex_experiments_20261014/t005_v7b_20261015` | [20261015_T0.05_v7b_검증](20261014_1616_T005오분류_후보비용v7b비교) |
| `codex_experiments_20261014/trial_all_points_20261015` | [20261015_전체점_시도그림](20261014_1717_280장전수_고정모델재실행) |
| `codex_experiments_20261014/trial_direct_reference_20261015` | [20261015_직접기준_시도그림](20261014_1717_280장전수_고정모델재실행) |
| `codex_experiments_20261014/trial_paths_12groups_20261015` | [20261015_12그룹_경로그림](20261014_1717_280장전수_고정모델재실행) |
| `codex_experiments_20261014 의 낱개 스크립트` | [20261014_codex_공용스크립트](20261014_1619_두장추론모델_6후보0점모델) |
| `eda_630_inventory_20261022` | [20261022_EDA_구조점630_목록](20261022_1736_630점목록_점별유효반응) |
| `eda_all_points_20261022` | [20261022_EDA_전체점](20261022_1541_전점노이즈_점별직선겹침) |
| `eda_capture_inventory_20261022` | [20261022_EDA_촬영목록](20261022_1743_촬영기록정리_파일시각정렬) |
| `eda_linearity_20261020_20261022` | [20261022_EDA_20261020_선형성](20261022_1528_선형성노이즈_구간별직선적합) |
| `eda_trial_origins_20261022` | [20261022_EDA_시도별원점](20261022_1750_trial원점_절편외삽역산) |
| `eda_y_trials_20261022` | [20261022_EDA_Y시도](20261022_1548_Y지그재그_시행별절편) |
| `experiment_documentation_20261021` | [20261021_실험문서화](20261021_1741_분류기판정_LDA와SVC) |
| `final` | [20261020_Python을_C++로_전체이식](20261019_1400_측정별판정비교_동일GLS판정기) |
| `hw-analysis` | [20261020_EDA_도구_hw](20261019_1400_측정별판정비교_동일GLS판정기) |
| `joint_center_630_20261022` | [20261022_센터후보_YT동시추정_구조점630](20261022_1511_다중센터YT_원점후보11개) |
| `legacy` | [20261022_이전작업_스냅샷](99999.trash) |
| `motion_speed_630_20261022` | [20261022_동작속도_구조점630](20261022_1511_추론속도_Numba해석LM) |
| `motion_web_20261022` | [20261022_Motion_Studio_웹](20261021_1810_QR만통과확인_내부536점고정) |
| `output` | [20261015_ECC_실측그림](20261015_1402_센터0점고정_절편0곡선) |
| `presentation_eda_20261019` | [20261019_EDA_발표자료](20261015_1838_모서리정밀화_경계모델적합) |
| `video_output` | [20261015_영상_라벨](20261012_1549_QR3개구분_ECC선형회귀) |
| `wafer_composite_20261015` | [20261015_웨이퍼_합성그림](20261015_1838_모서리정밀화_경계모델적합) |
