"""Evaluate theta correction when Y is known to remain fixed.

Odd theta settings calibrate the image displacement. Even settings are held out.
The fit passes through center (zero theta, zero measured displacement).
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
data = pd.read_csv(ROOT / "qr_measurements.csv")
theta = data[data.axis == "T"].copy()

for i in range(3):
    theta[f"x{i}"] = (
        theta[f"q{i}_dx"]
        + (theta[f"q{i}_a00"] - 1) * 90
        + theta[f"q{i}_a01"] * 96
    )
theta["qr_mean_shift_px"] = theta[[f"x{i}" for i in range(3)]].mean(axis=1)
theta["step"] = np.rint(theta.value.abs() * 100).astype(int)

train = theta[theta.step % 2 == 1]
test = theta[theta.step % 2 == 0].copy()
slope = np.dot(train.qr_mean_shift_px, train.value) / np.dot(
    train.qr_mean_shift_px, train.qr_mean_shift_px
)
test["theta_pred"] = slope * test.qr_mean_shift_px
test["theta_abs_error"] = (test.theta_pred - test.value).abs()

# This conversion describes equivalent QR image motion, not physical theta-axis travel.
theta_px_per_unit = 80.06
y_px_per_mm = 14.25
test["qr_y_equivalent_error_mm"] = (
    test.theta_abs_error * theta_px_per_unit / y_px_per_mm
)
test[
    [
        "file", "value", "qr_mean_shift_px", "theta_pred", "theta_abs_error",
        "qr_y_equivalent_error_mm",
    ]
].to_csv(ROOT / "theta_only_holdout.csv", index=False)

for label, subset in [
    ("all", test),
    ("abs(theta)<=0.25", test[test.value.abs() <= 0.25]),
    ("abs(theta)<=0.10", test[test.value.abs() <= 0.10]),
]:
    print(
        label, "n", len(subset),
        "theta MAE", subset.theta_abs_error.mean(),
        "theta max", subset.theta_abs_error.max(),
        "QR Y-equivalent MAE mm", subset.qr_y_equivalent_error_mm.mean(),
        "QR Y-equivalent max mm", subset.qr_y_equivalent_error_mm.max(),
    )
