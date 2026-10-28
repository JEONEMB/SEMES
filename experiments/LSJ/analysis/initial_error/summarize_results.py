"""Summarize image-space consistency; this does not measure physical ground truth."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "measurements.json").read_text(encoding="utf-8"))
MARKERS = ["L", "C", "R"]


def fit(x, y, degree):
    coeff = np.polynomial.polynomial.polyfit(x, y, degree)
    residual = y - np.polynomial.polynomial.polyval(x, coeff)
    return {"coefficients_ascending": coeff.tolist(), "zero_prediction_px": float(coeff[0]),
            "rms_residual_px": float(np.sqrt(np.mean(residual**2))),
            "max_abs_residual_px": float(np.max(np.abs(residual)))}, residual


def main():
    rows = DATA["measurements"]
    failures = [(r["file"], region, method) for r in rows for region, values in r["regions"].items()
                for method in ["affine", "translation"] if "error" in values[method]]
    if failures:
        raise RuntimeError(f"Registration failures: {failures}")
    qr_cc = [r["regions"][m]["affine"]["correlation"] for r in rows for m in MARKERS]
    summary = {"image_count": len(rows), "unique_file_hashes": len({r["sha256"] for r in rows}),
               "min_qr_affine_ecc": min(qr_cc), "registration_failures": failures,
               "limitations": DATA["limitations"], "axes": {}}
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for col, axis in enumerate(["Y", "Theta"]):
        selected = sorted((r for r in rows if r["axis"] == axis), key=lambda r: r["label"])
        x = np.array([r["label"] for r in selected])
        dx = np.array([[r["regions"][m]["affine"]["dx"] for m in MARKERS] for r in selected])
        dy = np.array([[r["regions"][m]["affine"]["dy"] for m in MARKERS] for r in selected])
        translation = np.array([[r["regions"][m]["translation"]["dx"] for m in MARKERS] for r in selected])
        ncc = np.array([[r["regions"][m]["ncc_dx"] for m in MARKERS] for r in selected])
        mean = dx.mean(axis=1)
        fits = {}
        for degree in [1, 2, 3]:
            fits[str(degree)], _ = fit(x, mean, degree)
        slope = fits["1"]["coefficients_ascending"][1]
        local_limits = [0.1, 0.25, 0.5, 1.0, 2.0] if axis == "Y" else [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
        local = {}
        for limit in local_limits:
            mask = np.abs(x) <= limit + 1e-8
            detail, _ = fit(x[mask], mean[mask], 1)
            detail["samples"] = int(mask.sum())
            detail["center_offset_command_equivalent"] = -detail["zero_prediction_px"] / slope
            local[str(limit)] = detail
        residuals = np.column_stack([fit(x, dx[:, i], 1)[1] for i in range(3)])
        residual_mean = residuals.mean(axis=1)
        backgrounds = {}
        for name in ["background_left", "background_right", "background_upper"]:
            values = np.array([r["regions"][name]["affine"]["dx"] for r in selected])
            detail, residual = fit(x, values, 1)
            detail["residual_correlation_with_qr_mean"] = float(np.corrcoef(residual_mean, residual)[0, 1])
            backgrounds[name] = detail
        nearest = np.argsort(np.abs(x))[:10]
        nearest = nearest[np.argsort(x[nearest])]
        fit1, residual = fit(x, mean, 1)
        top = np.argsort(np.abs(residual))[-8:][::-1]
        axis_summary = {
            "nonzero_image_count": len(selected), "label_range": [float(x.min()), float(x.max())],
            "label_unit": "mm" if axis == "Y" else "degree",
            "global_polynomial_fits_excluding_center": fits,
            "local_linear_fits_excluding_center": local,
            "residual_rms_command_equivalent": fits["1"]["rms_residual_px"] / abs(slope),
            "marker_residual_correlation": np.corrcoef(residuals.T).tolist(),
            "differential_residual_rms_px": float(np.sqrt(np.mean((residuals-residual_mean[:, None])**2))),
            "affine_minus_translation_mean_dx_rms_px": float(np.sqrt(np.mean((mean-translation.mean(axis=1))**2))),
            "affine_minus_ncc_mean_dx_rms_px": float(np.sqrt(np.mean((mean-ncc.mean(axis=1))**2))),
            "vertical_displacement_range_px": [float(dy.mean(axis=1).min()), float(dy.mean(axis=1).max())],
            "backgrounds": backgrounds,
            "near_zero": [{"file": selected[i]["file"], "label": float(x[i]), "mean_dx_px": float(mean[i])} for i in nearest],
            "largest_linear_residuals": [{"file": selected[i]["file"], "residual_px": float(residual[i]),
                                           "command_equivalent": float(residual[i]/slope)} for i in top],
        }
        summary["axes"][axis] = axis_summary
        unit = "mm" if axis == "Y" else "deg"
        grid = np.linspace(x.min(), x.max(), 300)
        ax = axes[0, col]
        ax.scatter(x, mean, s=12, color="#087f8c", label="Mean of 3 QR patches")
        ax.plot(grid, np.polynomial.polynomial.polyval(grid, fit1["coefficients_ascending"]), color="#c05b22", lw=1, label="Linear fit (center excluded)")
        ax.scatter([0], [0], marker="*", color="#d03c46", s=90, label="Single center reference", zorder=4)
        ax.set(title=f"{axis}: displacement vs filename label", xlabel=f"Nominal {axis} ({unit})", ylabel="Horizontal displacement (px)")
        ax.legend(fontsize=8)
        ax = axes[1, col]
        for i, marker in enumerate(MARKERS):
            ax.plot(x, residuals[:, i], marker=".", ms=3, lw=0.7, label=marker)
        ax.axhline(0, color="gray", lw=0.7)
        ax.scatter([0], [-fit1["zero_prediction_px"]], marker="*", color="#d03c46", s=90, zorder=4)
        ax.set(title=f"Residual from each marker's linear trend; mean RMS={fit1['rms_residual_px']:.3f} px",
               xlabel=f"Nominal {axis} ({unit})", ylabel="Residual (px)")
        ax.legend(fontsize=8)
        ax = axes[2, col]
        limit = 0.25 if axis == "Y" else 0.05
        mask = np.abs(x) <= limit+1e-8
        local_fit = local[str(limit)]
        ax.scatter(x[mask], mean[mask], s=30, color="#087f8c", label="Neighbor images")
        for xi, yi in zip(x[mask], mean[mask]):
            ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(3, 5), textcoords="offset points", fontsize=7)
        small_grid = np.linspace(-limit, limit, 100)
        ax.plot(small_grid, np.polynomial.polynomial.polyval(small_grid, local_fit["coefficients_ascending"]), color="#c05b22", lw=1)
        ax.scatter([0], [0], marker="*", color="#d03c46", s=120, zorder=4, label="Center reference")
        ax.set(title=f"Near zero: fitted intercept={local_fit['zero_prediction_px']:+.3f} px",
               xlabel=f"Nominal {axis} ({unit})", ylabel="Horizontal displacement (px)")
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle("Initial SEMES images: internal consistency, not absolute physical accuracy\n281 unique input files; shared center excluded from trend fitting", fontsize=13)
    fig.savefig(ROOT / "consistency.png", dpi=160)
    plt.close(fig)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
