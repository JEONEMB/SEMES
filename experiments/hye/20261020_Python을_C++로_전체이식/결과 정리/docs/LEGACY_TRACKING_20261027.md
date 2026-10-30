# Historical tracking parity

The original affine, homography and point-tracking protocols remain separate experiments. Their former OpenCV3 and OpenCV4 mismatch records are preserved. They are historical measurements, including QR-outside points where the original experiment used them; they do not change the adopted observation restrictions.

OpenCV5 with the original Intel IPP2026.0 kernels is built under third_party/opencv500-ipp-build. Apps use the modern-ipp preset. This runtime is kept separate from the already verified modern image-feedback build.

- analysis_agent/measure_qr.py: 281 images, all25 numeric columns exactly equal to the preserved CSV, including NCC scores. results/historical_ecc_opencv500_ipp_20261027_001/parity_all/checks.json.
- analysis_v2/extract_homography.py: 281 images. ECC score is exact. Restoring fields to their original float32 dtype yields exact equality for all remaining numeric columns. Decimal CSV representations differ by at most3.75e-6px. The stricter double comparison failure and historical1e-4 test are preserved alongside the exact dtype comparison. results/historical_homography_opencv500_20261027_001/parity_original_float32/checks.json.
- analysis_v3/v1_data/track_all.py: center-only detection finds the original84 points. All282 rows, including the original duplicate center, match the saved NPY dx/dy/score exactly. NumPy2 float32 promotion is reproduced through subpixel interpolation and search-window subtraction before adding float64 point offsets. results/historical_track_opencv500_ipp_frozen_20261027_003/checks.json. Detection and frozen-point runs produce identical points.csv and measurements.csv: results/historical_track_opencv500_ipp_detect_20261027_003/parity.json.

Builds without the corresponding IPP kernels are not substituted for these verified historical recipes. Earlier failures show why the dependency version and arithmetic order are part of the experiment provenance.
