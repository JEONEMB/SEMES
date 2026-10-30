"""One-time ledger maintenance; not a replacement runtime."""
from pathlib import Path
import json,hashlib
final=Path(__file__).resolve().parents[2];root=final.parent
path=final/'config/port_progress.json';p=json.loads(path.read_text(encoding='utf-8-sig'))
entries=[
('analysis_agent/measure_qr.py','experiments/analysis_v1_20261012_3QR측정유지_회귀C++이식/src/measure_qr.cpp',True,'cpp_ported_OpenCV5_IPP_281_images_all25_numeric_columns_exact','results/historical_ecc_opencv500_ipp_20261027_001/parity_all/checks.json'),
('analysis_v3/v1_data/track_all.py','experiments/analysis_v1_20261013_고정점분할유지_선형계산C++이식/src/track_all.cpp',True,'cpp_ported_OpenCV5_IPP_detect84_points_282_rows_exact','results/historical_track_opencv500_ipp_detect_20261027_003/parity.json'),
('codex_experiments_20261014/joint_yt_reboot_20261022/metrics.py','include/tolerance_metrics.hpp',True,'cpp_ported_6_cases_full_failure_denominator','results/tolerance_metrics_parity_20261027_001/checks.json'),
('codex_experiments_20261014/joint_yt_visual_feedback_20261023/predict.py','tool/predict_observation.cpp',True,'cpp_ported_16_raw_image_predictions_max2.22e-8','results/predict_observation_parity_20261027_001/checks.json'),
('codex_experiments_20261014/joint_yt_visual_feedback_20261023/evaluate_static.py','tool/evaluate_static.cpp',True,'cpp_ported_380_fits_20_models_2728_predictions_24_summaries','results/evaluate_static_native_20261027_001/prediction_parity/checks.json'),
('codex_experiments_20261014/joint_yt_visual_feedback_20261023/models.py','include/observation_model.hpp',False,'static_fit_predict_6_configs_verified_sequence_optimizer_remaining','results/observation_fit_parity_20261027_001/checks.json'),
('analysis_v4/zero_align.py','tool/historical_zeroalign.cpp',True,'cpp_ported_12_models_84_rounded_metrics','results/zeroalign_native_20261027_003/checks.json'),
('analysis_v2/extract_homography.py','experiments/analysis_v2_20261013_특징분할유지_통계C++이식/src/extract_homography.cpp',True,'cpp_ported_OpenCV5_281_images_exact_original_float32_fields','results/historical_homography_opencv500_20261027_001/parity_original_float32/checks.json'),
('codex_experiments_20261014/joint_yt_generalization_20261022/forward_model.py','include/forward_gls.hpp',True,'cpp_ported_weighted_forward_GLS_6_configs_prediction_tolerance_1e-6','results/forward_gls_parity_20261027_001/checks.json'),
('codex_experiments_20261014/joint_yt_motion_logic_20261026/motion.py','src/motion.cpp',True,'cpp_ported_61_calls_8_scenarios_singular_diagnostic_excluded','results/motion_native_compare_20261027_002/checks.json'),
('codex_experiments_20261014/joint_yt_visual_feedback_20261023/lines/extractor.py','src/line_observer.cpp',True,'cpp_ported_341_images_16368_blocks_1e-6px_tolerance','results/line_native_full_20261027_001/parity/checks.json'),
('codex_experiments_20261014/joint_yt_reboot_20261022/direct/extractor.py','src/direct_observer.cpp',True,'cpp_ported_OpenCV5_341_images_4_candidates_max_1.37e-12px','results/direct_native_full_opencv500_20261027_001/parity/checks.json'),
('codex_experiments_20261014/joint_yt_visual_feedback_20261023/image_observations.py','include/image_observer.hpp',True,'cpp_ported_OpenCV5_4_conditions_16_image_session_calls','results/image_session_parity_20261027_001/checks.json'),
('codex_experiments_20261014/joint_yt_visual_feedback_20261023/prepare_features.py','tool/prepare_features.cpp',True,'cpp_ported_341_rows_8_NPZ_arrays_1e-6_tolerance','results/features_native_20261027_002/parity.json'),
('codex_experiments_20261014/joint_yt_motion_logic_20261026/session.py','tool/image_session.cpp',True,'cpp_ported_4_conditions_16_calls_frozen_training_model_JSON_persistence','results/image_session_parity_20261027_001/checks.json'),
('codex_experiments_20261014/joint_yt_visual_feedback_20261023/control/controller.py','include/calibration.hpp',True,'cpp_ported_native_calibration_and_3_feedback_policies_162_calls','results/feedback_parity_20261027_001/checks.json'),
('codex_experiments_20261014/joint_yt_visual_feedback_20261023/control/save_full_calibration.py','tool/calibrate_full.cpp',True,'cpp_ported_4_retrained_models_match_original_artifacts_1e-8','results/calibrate_full_frozen_inputs_20261027_002/parity.json'),
('codex_experiments_20261014/joint_yt_visual_feedback_20261023/feedback_session.py','tool/feedback_image_session.cpp',True,'cpp_ported_4_conditions_3_policies_36_image_calls','results/feedback_image_session_parity_20261027_001/checks.json')]
for source,destination,complete,status,evidence in entries:
    if not (root/source).is_file():raise FileNotFoundError(source)
    p[source]=dict(status=status,destination=destination,cpp_complete=complete,source_sha256=hashlib.sha256((root/source).read_bytes()).hexdigest(),validation=status,report='docs/SELF_CONTAINED_20261027.md',evidence=[evidence])
p['codex_experiments_20261014/joint_yt_visual_feedback_20261023/control/controller.py']['evidence'].append('results/calibrate_full_frozen_inputs_20261027_002/parity.json')
p['codex_experiments_20261014/joint_yt_visual_feedback_20261023/control/controller.py']['additional_destinations']=['include/feedback.hpp']
path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
