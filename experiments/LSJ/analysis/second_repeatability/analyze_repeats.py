"""Same-label image variation: translation ECC versus subpixel phase correlation.

Only reads original/extracted BMPs. Reports image-space displacements, not
mechanical accuracy. A 0.001-pixel numerical grid is NOT an accuracy guarantee.
No affine/corner fitting and no physical Y/theta inversion are performed.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import fourier_shift


HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / "second_data_inspection" / "inventory.json"
# Outer white labels in second-data center1.bmp, checked visually.
BASE_BOXES = {"L": (1549, 968, 189, 201), "C": (1807, 969, 188, 201),
              "R": (2065, 970, 189, 201)}
METHODS = ("ecc_translation", "phase_correlation")
PAIRS = ((0, 1), (0, 2), (1, 2))


def read_image(root, row):
    raw = (root / row["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError(f"Inventory hash mismatch: {row['file']}")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2160, 3840):
        raise ValueError(f"Unexpected image: {row['file']}")
    return image


def crop(image, roi):
    x0, y0, x1, y1 = roi
    return image[y0:y1, x0:x1]


def locate_rois(reference, first, margin=12):
    """ZNCC only locates each group's ROI; it is not a third reported estimator."""
    result = {}
    for marker, (x, y, w, h) in BASE_BOXES.items():
        seed = [x-margin, y-margin, x+w+margin, y+h+margin]
        sx, sy = 230, 20
        search = first[seed[1]-sy:seed[3]+sy, seed[0]-sx:seed[2]+sx]
        scores = cv2.matchTemplate(search, crop(reference, seed), cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(scores)
        if mx in (0, scores.shape[1]-1) or my in (0, scores.shape[0]-1) or score < .8:
            raise ValueError(f"Unreliable ROI localization: {marker}, {score}")
        dx, dy = mx-sx, my-sy
        result[marker] = {"xyxy": [seed[0]+dx, seed[1]+dy, seed[2]+dx, seed[3]+dy],
                          "localization_zncc": float(score)}
    return result


def ecc_translation(a, b):
    """Return target-minus-reference displacement (right/down positive)."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    margin = 12
    scores = cv2.matchTemplate(b, a[margin:-margin, margin:-margin], cv2.TM_CCOEFF_NORMED)
    _, score, _, peak = cv2.minMaxLoc(scores)
    if any(v in (0, 2*margin) for v in peak):
        raise ValueError("ECC initial match reached +/-12 px boundary")
    coarse = np.array(peak, dtype=np.float32)-margin
    warp = np.eye(2, 3, dtype=np.float32)
    warp[:, 2] = coarse
    value, warp = cv2.findTransformECC(
        a/255., b/255., warp, cv2.MOTION_TRANSLATION,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-8), None, 5)
    return {"dx": float(warp[0, 2]), "dy": float(warp[1, 2]),
            "ecc": float(value), "initial_zncc": float(score)}


def phase_correlation(a, b):
    """Phase-only spectrum, Hann window, local inverse DFT at 0.01 then 0.001 px.

    Implements the inverse Fourier sum directly in two small search grids.
    F(b)*conj(F(a)) gives target-minus-reference displacement, unlike APIs that
    return the correction needed to align the target to the reference.
    """
    a, b = a.astype(np.float64), b.astype(np.float64)
    # Same 5x5 Gaussian scale as ECC; mean removal and Hann reduce patch edges.
    a = cv2.GaussianBlur(a, (5, 5), 0)
    b = cv2.GaussianBlur(b, (5, 5), 0)
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2((a-a.mean())*window)
    fb = np.fft.fft2((b-b.mean())*window)
    cross = fb*np.conj(fa)
    amplitude = np.abs(cross)
    cross = np.divide(cross, amplitude, out=np.zeros_like(cross),
                      where=amplitude > amplitude.max()*1e-12)
    response = np.fft.ifft2(cross).real
    y, x = np.unravel_index(np.argmax(response), response.shape)
    y = float(y if y <= a.shape[0]//2 else y-a.shape[0])
    x = float(x if x <= a.shape[1]//2 else x-a.shape[1])
    if max(abs(x), abs(y)) >= 12:
        raise ValueError("Phase correlation displacement exceeds same-label search scale")
    fy, fx = np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1])
    for radius, step in ((.75, .01), (.015, .001)):
        offsets = np.arange(-round(radius/step), round(radius/step)+1)*step
        ys, xs = y+offsets, x+offsets
        ey = np.exp(2j*np.pi*np.outer(ys, fy))
        ex = np.exp(2j*np.pi*np.outer(fx, xs))
        local = (ey @ cross @ ex).real / a.size
        iy, ix = np.unravel_index(np.argmax(local), local.shape)
        if iy in (0, len(ys)-1) or ix in (0, len(xs)-1):
            raise ValueError("Local DFT refinement reached its search boundary")
        y, x = float(ys[iy]), float(xs[ix])
    return {"dx": x, "dy": y, "phase_peak": float(local[iy, ix])}


def vector(result):
    return np.array([result["dx"], result["dy"]], dtype=float)


def measure_pair(a, b):
    return {"ecc_translation": ecc_translation(a, b),
            "phase_correlation": phase_correlation(a, b)}


def vector_summary(values):
    v = np.asarray(values, float)
    return {"count": len(v), "mean_xy_px": v.mean(axis=0).tolist(),
            "rms_xy_px": np.sqrt(np.mean(v*v, axis=0)).tolist(),
            "max_abs_xy_px": np.max(np.abs(v), axis=0).tolist(),
            "rms_distance_px": float(np.sqrt(np.mean(np.sum(v*v, axis=1)))),
            "max_distance_px": float(np.max(np.linalg.norm(v, axis=1)))}


def group_summary(pair_rows):
    summary = {}
    for method in METHODS:
        means, closures, marker_results = [], [], {}
        for marker in BASE_BOXES:
            displacements = [vector(pair_rows[i]["regions"][marker][method]) for i in range(3)]
            # A is only the coordinate origin, not an independently known truth.
            positions = np.array([[0., 0.], displacements[0], displacements[1]])
            closure = displacements[0] + displacements[2] - displacements[1]
            closures.append(closure)
            marker_results[marker] = {"pair_displacements_xy_px": [v.tolist() for v in displacements],
                "positions_relative_to_A_xy_px": positions.tolist(),
                "sample_std_xy_px_n3": positions.std(axis=0, ddof=1).tolist(),
                "max_pair_distance_px": float(np.max(np.linalg.norm(displacements, axis=1))),
                "closure_xy_px": closure.tolist()}
        for pair in pair_rows:
            means.append(np.mean([vector(pair["regions"][m][method]) for m in BASE_BOXES], axis=0))
        positions = np.array([[0., 0.], means[0], means[1]])
        summary[method] = {"markers": marker_results, "mean_three_qr_pair_displacements_xy_px": [v.tolist() for v in means],
            "mean_three_qr_max_pair_distance_px": float(np.max(np.linalg.norm(means, axis=1))),
            "mean_three_qr_sample_std_xy_px_n3": positions.std(axis=0, ddof=1).tolist(),
            "max_marker_closure_distance_px": float(np.max(np.linalg.norm(closures, axis=1)))}
    differences = [vector(pair["regions"][m]["phase_correlation"])-vector(pair["regions"][m]["ecc_translation"])
                   for pair in pair_rows for m in BASE_BOXES]
    summary["phase_minus_ecc"] = vector_summary(differences)
    return summary


def synthetic_check(sources):
    rng = np.random.default_rng(20261021)
    shifts = [(0., 0.), (.02, -.03), (.05, .07), (.25, -.4), (1.2, -.7), (-.5, .3)]
    records = []
    for name, large in sources:
        # Shift a larger crop then measure its interior to reduce periodic-edge effects.
        a = large[32:-32, 32:-32].astype(float)
        spectrum = np.fft.fftn(large.astype(float))
        for dx, dy in shifts:
            shifted = np.fft.ifftn(fourier_shift(spectrum, (dy, dx))).real[32:-32, 32:-32]
            for variant in ("shift_only", "gain_offset_noise"):
                b = shifted if variant == "shift_only" else shifted*1.02+2+rng.normal(0, 1, shifted.shape)
                b = np.clip(np.rint(b), 0, 255).astype(np.uint8)
                measured = measure_pair(a, b)
                records.append({"source": name, "variant": variant, "known_dx": dx, "known_dy": dy,
                    "measurements": measured,
                    "errors_xy_px": {m: (vector(measured[m])-np.array([dx, dy])).tolist() for m in METHODS}})
    summary = {variant: {m: vector_summary([r["errors_xy_px"][m] for r in records if r["variant"] == variant])
                        for m in METHODS} for variant in ("shift_only", "gain_offset_noise")}
    return {"scope": "Numerical sign/sensitivity checks under a Fourier-shift image model, not physical calibration.",
            "noise": "gain 1.02, offset 2 DN, Gaussian sigma 1 DN, uint8 quantization; fixed seed 20261021",
            "summary": summary, "trials": records}


def plot_results(result, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    groups = result["groups"]
    labels = ["기준 위치" if g["is_reference"] else f"Y {g['label_y_mm']:g} / θ {g['label_theta_degree']:g}" for g in groups]
    xs = np.arange(len(groups))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    colors = ["#1467ad", "#db7428"]
    for method, color, short in zip(METHODS, colors, ["ECC 평행이동", "위상 상관"]):
        vals = [g["summary"][method]["mean_three_qr_max_pair_distance_px"] for g in groups]
        axes[0, 0].plot(xs, vals, "o-", label=short, color=color, ms=4)
        vals = [g["summary"][method]["max_marker_closure_distance_px"] for g in groups]
        axes[1, 1].plot(xs, vals, "o-", label=short, color=color, ms=4)
    axes[0, 0].set_title("동일 조건 3장 사이 최대 거리 · QR 3개 평균 변위")
    axes[0, 0].set_ylabel("영상 위치 차이 (px)")
    axes[1, 1].set_title("A→B + B→C - A→C 일관성 · QR별 최대")
    axes[1, 1].set_ylabel("경로 간 불일치 (px)")
    for ax in (axes[0, 0], axes[1, 1]):
        ax.set_xticks(xs, labels, rotation=75, ha="right", fontsize=8)
        ax.legend()
    all_regions = [p["regions"][m] for g in groups for p in g["pairs"] for m in BASE_BOXES]
    for coordinate, color in zip(("dx", "dy"), colors):
        x = [r["ecc_translation"][coordinate] for r in all_regions]
        y = [r["phase_correlation"][coordinate] for r in all_regions]
        axes[0, 1].scatter(x, y, s=12, alpha=.7, label=coordinate, color=color)
    low, high = axes[0, 1].get_xlim()
    axes[0, 1].plot([low, high], [low, high], "k--", lw=.8)
    axes[0, 1].set(title="두 방법의 QR별 변위 비교 · 180쌍", xlabel="ECC (px)", ylabel="위상 상관 (px)")
    axes[0, 1].legend()
    differences = [g["summary"]["phase_minus_ecc"]["max_distance_px"] for g in groups]
    axes[1, 0].bar(xs, differences, color="#52677e")
    axes[1, 0].set(title="조건별 최대 방법 간 차이 · QR별", ylabel="위상 상관 - ECC의 거리 (px)")
    axes[1, 0].set_xticks(xs, labels, rotation=75, ha="right", fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("2차 데이터: 동일 좌표 기록의 영상 위치 편차\n방법 간 일치는 실제 장비 정확도의 검증이 아님", fontsize=15)
    fig.savefig(HERE / "repeatability_comparison.png", dpi=150)
    plt.close(fig)
    if not panels:
        return
    fig, axs = plt.subplots(5, 4, figsize=(16, 7), constrained_layout=True)
    for ax, (name, image, boxes) in zip(axs.flat, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        for box in boxes:
            x0, y0, x1, y1 = box
            ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="red", lw=.7))
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.savefig(HERE / "roi_check.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Development smoke run; output marked partial.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cv2.setNumThreads(2)
    start = time.perf_counter()
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    root = Path(inventory["summary"]["extracted_root"])
    grouped = defaultdict(list)
    for row in inventory["images"]:
        grouped[(row["label_y_mm"], row["label_theta_degree"])].append(row)
    items = sorted(grouped.items())
    if args.limit:
        items = items[:args.limit]
    center_row = next(r for r in inventory["images"] if Path(r["file"]).name == "center1.bmp")
    center = read_image(root, center_row)
    groups, panels, sources = [], [], []
    for gi, ((y, theta), rows) in enumerate(items):
        rows = sorted(rows, key=lambda r: r["file"])
        if len(rows) != 3 or len({(r["Y_Actual"], r["T_Actual"]) for r in rows}) != 1:
            raise ValueError("Expected three records with identical Actual coordinates")
        images = [read_image(root, r) for r in rows]
        rois = locate_rois(center, images[0])
        pairs = []
        for i, j in PAIRS:
            regions = {}
            for marker, roi in rois.items():
                a, b = crop(images[i], roi["xyxy"]), crop(images[j], roi["xyxy"])
                regions[marker] = measure_pair(a, b)
                regions[marker]["brightness_difference_dn"] = float(b.mean()-a.mean())
            pairs.append({"i": i, "j": j, "file_a": rows[i]["file"], "file_b": rows[j]["file"], "regions": regions})
        name = rows[0]["condition"]
        group = {"condition": name, "is_reference": rows[0]["is_reference"], "label_y_mm": y,
                 "label_theta_degree": theta, "recorded_actual": [rows[0]["Y_Actual"], rows[0]["T_Actual"]],
                 "files": [{k: r[k] for k in ("file", "sha256", "archive_modified_time_no_timezone")} for r in rows],
                 "rois": rois, "pairs": pairs, "summary": group_summary(pairs)}
        groups.append(group)
        x0 = min(r["xyxy"][0] for r in rois.values())-8
        y0 = min(r["xyxy"][1] for r in rois.values())-8
        x1 = max(r["xyxy"][2] for r in rois.values())+8
        y1 = max(r["xyxy"][3] for r in rois.values())+8
        panels.append((name, images[0][y0:y1, x0:x1],
                       [[r["xyxy"][0]-x0, r["xyxy"][1]-y0, r["xyxy"][2]-x0, r["xyxy"][3]-y0] for r in rois.values()]))
        if gi == 0 or (y, theta) == (1., 2.):
            for marker, roi in rois.items():
                x0, y0, x1, y1 = roi["xyxy"]
                sources.append((f"{name}/{marker}", images[0][y0-32:y1+32, x0-32:x1+32]))
        summary = group["summary"]
        print(f"{gi+1}/{len(items)} {name}: max mean-QR pair distance ECC={summary[METHODS[0]]['mean_three_qr_max_pair_distance_px']:.6f}, phase={summary[METHODS[1]]['mean_three_qr_max_pair_distance_px']:.6f} px", flush=True)
    all_region_results = [p["regions"][m] for g in groups for p in g["pairs"] for m in BASE_BOXES]
    differences = [vector(r["phase_correlation"])-vector(r["ecc_translation"]) for r in all_region_results]
    summary = {"condition_count": len(groups), "image_count": len(groups)*3, "qr_pair_count": len(all_region_results),
               "phase_minus_ecc": vector_summary(differences),
               "min_ecc": min(r["ecc_translation"]["ecc"] for r in all_region_results),
               "partial_run": bool(args.limit)}
    for method in METHODS:
        by_max = sorted(groups, key=lambda g:g["summary"][method]["mean_three_qr_max_pair_distance_px"], reverse=True)
        summary[method] = {"max_mean_three_qr_pair_distance_px": by_max[0]["summary"][method]["mean_three_qr_max_pair_distance_px"],
            "worst_condition": by_max[0]["condition"],
            "max_qr_pair_distance_px": max(np.linalg.norm(vector(r[method])) for r in all_region_results),
            "max_marker_closure_distance_px": max(g["summary"][method]["max_marker_closure_distance_px"] for g in groups)}
    print("Running known-shift numerical checks...", flush=True)
    checks = synthetic_check(sources)
    result = {"method": {"units": "native image pixels, target-minus-reference, x right/y down",
              "ecc": "translation only; ZNCC integer initialization +/-12 px, max 200 iterations, epsilon 1e-8, Gaussian 5x5",
              "phase": "phase-only cross spectrum; Gaussian 5x5, subtract mean, Hann; inverse DFT local grids 0.01 and 0.001 px",
              "roi": "center1 white QR boxes plus 12 px, located per group using ZNCC +/-230 x and +/-20 y; fixed within each group",
              "aggregation": "3 QR estimates retained individually; their arithmetic mean reports common apparent displacement",
              "software": {"cv2": cv2.__version__, "numpy": np.__version__}},
              "summary": summary, "groups": groups, "synthetic_checks": checks,
              "limitations": ["Three original images per group; the three pairs are not independent repeat captures.",
                "Capture protocol and Actual-field provenance remain unspecified.",
                "Both methods use the same image evidence and translation model; agreement does not establish physical accuracy.",
                "Numerical 0.001 px phase grid is not a demonstrated 0.001 px accuracy.",
                "No Y/theta physical inversion or pass/fail tolerance is applied.",
                "The two methods do not test affine or perspective changes.",
                "Synthetic trials test only the stated rendering/noise model and are not physical calibration."]}
    result["elapsed_seconds"] = time.perf_counter()-start
    destination = HERE / ("smoke_results.json" if args.limit else "repeatability_results.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.limit:
        plot_results(result, panels)
    print(json.dumps({"summary": summary, "synthetic_checks": checks["summary"], "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
