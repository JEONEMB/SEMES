# Native static observation models

include/observation_model.hpp implements the original six static fit/predict configurations: degree1/2, shrink0.01/0.1, and either all dates or mixed-date training. Training rows are weighted by date; inference receives features only. Actual selected training indices are retained. All six configurations were compared with the original implementation: maximum prediction difference1.58e-7 with indices exactly equal. The sequence optimizer in the same original models.py file remains unported; that file is therefore partial.

tool/evaluate_static.cpp performs all preserved inner/outer splits, checks group separation, fits380 candidate/final models, chooses20 outer/full models, and emits2728 predictions plus24 summary rows. Every chosen configuration, pass count, failure count and per-image pass flag matches the preserved result. Maximum Y prediction difference1.10e-7 and T difference1.38e-8, under the declared1e-6 tolerance. results/evaluate_static_native_20261027_001 contains the original-result comparisons.

tool/predict_observation.cpp replaces predict.py using the native image extractor and a JSON model. Four saved models were converted with exact numeric round trips into models/observation_full341_frozen; this conversion is not retraining evidence. The independent native fitter is tested separately. Sixteen actual image/condition predictions match the original API within2.22e-8. results/predict_observation_parity_20261027_001/checks.json.

include/tolerance_metrics.hpp preserves the full denominator when measurements or truth are nonfinite. Six tests cover exact predictions, mixed errors, threshold boundaries and complete/partial failure. Counts are exact and scalar errors agree within1e-12. results/tolerance_metrics_parity_20261027_001/checks.json.

All inputs are already exposed development data. These comparisons establish implementation parity and do not establish independent field accuracy.
