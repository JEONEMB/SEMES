"""Figures from the C++ results (report only; no measurement is computed here).

Reads results/*.json and intermediates, the four original BMPs (read-only, hash-checked), and writes figures/.
Python/OpenCV is used only for reading images and drawing.
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from fit_point_views import fit_point_view, rms_label

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_CAND = "#8fb8de"     # candidates (light)
C_EXCL = "#bdbdbd"     # excluded
C_SEL = "#ff9f1c"      # selected support
C_OUT = "#9b59b6"      # trimmed by fit
C_FINAL = "#e3172b"    # final line
C_AUX = "#1f77b4"      # auxiliary (hough line, initial corners)

EXT_WINDOWS = {"upper_structure": [(400, 640), (3000, 3240)], "lower_structure": [(400, 640), (3000, 3240)]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    lm = json.loads((RES / "line_measurements.json").read_text(encoding="utf-8"))
    targets = json.loads((ROOT / "configs/targets.json").read_text(encoding="utf-8"))["targets"]
    manifest = json.loads((ROOT / "configs/inputs_manifest.json").read_text(encoding="utf-8"))
    images = {}
    for r in manifest["references"]:
        raw = np.fromfile(r["path"], dtype=np.uint8)
        assert hashlib.sha256(raw.tobytes()).hexdigest() == r["sha256"], r["path"]
        images[r["reference_id"]] = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    recs = {r["measurement_id"]: r for r in lm["records"]}
    return lm, recs, {t["id"]: t for t in targets}, manifest, images


def inter(rec):
    return json.loads((RES / rec["intermediate_json"]).read_text(encoding="utf-8"))


def show(ax, img, x0, x1, y0, y1, title=None, cmap="gray"):
    ax.imshow(img[y0:y1 + 1, x0:x1 + 1], cmap=cmap, vmin=0, vmax=255, extent=(x0 - .5, x1 + .5, y1 + .5, y0 - .5),
              interpolation="nearest", aspect="equal")
    ax.set_xlim(x0 - .5, x1 + .5); ax.set_ylim(y1 + .5, y0 - .5)
    if title: ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def overlay_mask(ax, rec_images, name, x0, x1, y0, y1, color=(0, 1, 1), alpha=.45):
    info = rec_images.get(name)
    if not info: return
    m = cv2.imdecode(np.fromfile(RES / info["path"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    ox, oy = info["origin_xy"]
    rgba = np.zeros(m.shape + (4,)); rgba[..., :3] = color; rgba[..., 3] = (m > 0) * alpha
    ax.imshow(rgba, extent=(ox - .5, ox + m.shape[1] - .5, oy + m.shape[0] - .5, oy - .5), interpolation="nearest")
    ax.set_xlim(x0 - .5, x1 + .5); ax.set_ylim(y1 + .5, y0 - .5)


def draw_line(ax, rec, xa, xb, **kw):
    if not rec.get("line") or rec["slope_dy_dx"] is None: return
    m, b = rec["slope_dy_dx"], rec["intercept_y_px"]
    ax.plot([xa, xb], [m * xa + b, m * xb + b], color=C_FINAL, lw=kw.pop("lw", 1.6), **kw)


def signed_dist(rec, x, y):
    (px, py), (dx, dy) = rec["line"]["point_xy"], rec["line"]["direction_xy"]
    return -(np.asarray(x) - px) * dy + (np.asarray(y) - py) * dx


def fail_text(ax, rec):
    if rec["status"] != "ok":
        ax.text(.5, .5, f"실패: {rec['failure_category']}\n{rec['failure_reason']}", transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="white", bbox=dict(facecolor="#b00020", alpha=.85, edgecolor="none"), wrap=True)


def fit_points(it):
    view = fit_point_view(it["intermediates"])
    return np.array(view["points"], dtype=float).reshape(-1, 2), np.array(view["display_mask"], dtype=bool)


def scatter_points(ax, rec, it, x0, x1, y0, y1):
    """Actual last-fit input when recorded; legacy results show final classification only."""
    iv = it["intermediates"]
    mid = rec["method_id"][1:]
    if mid in ("01", "05", "06") and "columns" in iv:
        cols = iv["columns"]
        xs = np.array([c[0] for c in cols], float); ys = np.array([c[1] for c in cols], float)
        ok = (xs >= x0) & (xs <= x1)
        ax.scatter(xs[ok], ys[ok], s=6, marker="s", facecolors="none", edgecolors=C_CAND, lw=.6, label="거친 경계 화소(후보)")
    if mid == "02" and "band_points" in iv:
        bp = iv["band_points"]
        xs = np.array([b[0] for b in bp], float); ys = np.array([b[1] for b in bp], float)
        st = np.array([b[6] for b in bp])
        ok = (xs >= x0) & (xs <= x1) & (st != "selected")
        ax.scatter(xs[ok], ys[ok], s=3, color=C_CAND, label="Devernay 점(제외)")
    if mid == "03" and "voting_pixels" in iv:
        vp = np.array(iv["voting_pixels"], float).reshape(-1, 3)
        ok = (vp[:, 0] >= x0) & (vp[:, 0] <= x1)
        ax.scatter(vp[ok & (vp[:, 2] == 0), 0], vp[ok & (vp[:, 2] == 0), 1], s=4, marker="s", color=C_CAND, label="투표 Canny 화소")
    pts, flag = fit_points(it)
    point_view = fit_point_view(iv)
    if len(pts):
        ok = (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
        ax.scatter(pts[ok & flag, 0], pts[ok & flag, 1], s=4, color=C_SEL, label=point_view['label'])
        ax.scatter(pts[ok & ~flag, 0], pts[ok & ~flag, 1], s=10, marker="x", color=C_OUT, lw=.8, label=point_view['excluded_label'])
        if point_view['used'] is not None and point_view['final_inlier'] is not None:
            changed = ok & (flag != np.array(point_view['final_inlier'], dtype=bool))
            if np.any(changed):
                ax.scatter(pts[changed, 0], pts[changed, 1], s=24, facecolors='none', edgecolors=C_AUX,
                           label='사용 여부와 최종 거리 분류가 다른 점')


def draw_segments(ax, it, x0, x1, y0, y1, lw_sel=2.0):
    segs = it["intermediates"].get("segments", [])
    for s in segs:
        xa, ya, xb, yb, _, _, _, _, st = s
        if max(xa, xb) < x0 or min(xa, xb) > x1: continue
        if st == "selected": ax.plot([xa, xb], [ya, yb], color=C_SEL, lw=lw_sel, alpha=.9)
        elif st.startswith("excluded:not_in_consensus"): ax.plot([xa, xb], [ya, yb], color=C_CAND, lw=1.2)
        else: ax.plot([xa, xb], [ya, yb], color=C_EXCL, lw=.8)


def residual_panel(ax, rec, it, t):
    pts, flag = fit_points(it)
    if rec["status"] != "ok" or not len(pts):
        ax.set_title("지지점 잔차: 없음", fontsize=8); fail_text(ax, rec); return
    d = signed_dist(rec, pts[:, 0], pts[:, 1])
    ax.scatter(pts[flag, 0], d[flag], s=2, color=C_SEL)
    ax.scatter(pts[~flag, 0], d[~flag], s=6, marker="x", color=C_OUT, lw=.6)
    for a, b in t.get("exclude_x", []): ax.axvspan(a, b, color="gray", alpha=.25, hatch="//")
    ax.axhline(0, color=C_FINAL, lw=1)
    rms = rec["fit_rms_px"]
    ax.set_title(f"최종선 대비 수직 잔차 | {rms_label(rec)} {'-' if rms is None else f'{rms:.3f}'} px", fontsize=8)
    ax.set_xlabel("원본 x (px)", fontsize=7); ax.set_ylabel("잔차 (px)", fontsize=7); ax.tick_params(labelsize=6)
    lim = max(2.5, float(np.percentile(np.abs(d), 99.5)) * 1.1) if len(d) else 2.5
    ax.set_ylim(lim, -lim)


def segment_residual_panel(ax, rec, it, t):
    """All detected segments drawn in (x, distance-to-final-line) space."""
    if rec["status"] != "ok":
        # No final line: draw the detected segments in original (x, y) with the band, y axis stretched.
        segs = it["intermediates"].get("segments", [])
        y0, y1 = t["band_y"]
        for s in segs:
            xa, ya, xb, yb, _, _, _, _, st = s
            col = C_SEL if st == "selected" else C_CAND if st.startswith("excluded:not_in_consensus") or st.startswith("excluded:occluded") else C_EXCL
            ax.plot([xa, xb], [ya, yb], color=col, lw=1.4)
        for a, b in t.get("exclude_x", []): ax.axvspan(a, b, color="gray", alpha=.25, hatch="//")
        ax.axhline(y0 - .5, color="k", ls=":", lw=.8); ax.axhline(y1 + .5, color="k", ls=":", lw=.8)
        ax.set_ylim(y1 + 20, y0 - 20)
        ax.set_title(f"검출 선분 {len(segs)}개, 원본 좌표 (세로 확대 표시: 기울기 과장, 각도 판독 금지)\n하늘=가림/짧음 탈락, 회색=각도 등 탈락", fontsize=7)
        ax.set_xlabel("원본 x (px)", fontsize=7); ax.tick_params(labelsize=6)
        fail_text(ax, rec); return
    n = {"selected": 0, "cons": 0, "other": 0}
    for s in it["intermediates"].get("segments", []):
        xa, ya, xb, yb, _, _, _, _, st = s
        da, db = signed_dist(rec, [xa, xb], [ya, yb])
        if abs(da) > 40 and abs(db) > 40: continue
        if st == "selected": ax.plot([xa, xb], [da, db], color=C_SEL, lw=2.2); n["selected"] += 1
        elif st.startswith("excluded:not_in_consensus"): ax.plot([xa, xb], [da, db], color=C_CAND, lw=1.4); n["cons"] += 1
        else: ax.plot([xa, xb], [da, db], color=C_EXCL, lw=.9); n["other"] += 1
    for a, b in t.get("exclude_x", []): ax.axvspan(a, b, color="gray", alpha=.25, hatch="//")
    ax.axhline(0, color=C_FINAL, lw=1)
    ax.set_title(f"검출 선분(최종선 대비 거리 공간): 선택 {n['selected']} / 일관성 탈락 {n['cons']} / 필터 탈락 {n['other']}", fontsize=8)
    ax.set_xlabel("원본 x (px)", fontsize=7); ax.set_ylabel("최종선까지 거리 (px)", fontsize=7); ax.tick_params(labelsize=6)
    ax.set_ylim(20, -20)


def method_panel(ax, rec, it, img, t, view):
    """Method-specific intermediate (column 3)."""
    m = rec["method_id"]; fam = m[1:]
    iv = it["intermediates"]
    x0, x1, y0, y1 = view
    if fam == "01":
        ex = [e for e in iv.get("profile_examples", []) if e.get("state") == "selected"]
        if not ex: ax.set_title("프로파일 예시 없음", fontsize=8); return
        e = ex[len(ex) // 2]
        ys = np.array(e["y_samples"]); fs = np.array(e["sobel_samples"])
        yy = np.linspace(ys.min(), ys.max(), 200)
        ax.plot(ys, fs, "o", color=C_AUX, ms=4, label="Sobel 표본(극성 곱)")
        ax.plot(yy, e["A"] * np.exp(-(yy - e["mu"]) ** 2 / (2 * e["sigma"] ** 2)), color=C_FINAL, lw=1.4, label="Gaussian 피팅")
        ax.axvline(e["mu"], color=C_SEL, ls="--", lw=1, label=f"μ={e['mu']:.3f}")
        ax.set_title(f"실제 프로파일 예시 x={e['x']} | A={e['A']:.1f}, σ={e['sigma']:.2f} px", fontsize=8)
        ax.set_xlabel("원본 y (px)", fontsize=7); ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    elif fam == "02":
        bp = [b for b in iv.get("band_points", []) if b[6] == "selected" and x0 <= b[0] <= x1]
        if not bp: ax.set_title("선택 점 없음", fontsize=8); fail_text(ax, rec); return
        cx, cy = bp[len(bp) // 2][0], bp[len(bp) // 2][1]
        a, b = int(round(cx)) - 6, int(round(cy)) - 4
        show(ax, img, a, a + 12, b, b + 8)
        sel = [p for p in bp if a <= p[0] <= a + 12]
        ax.scatter([p[2] for p in sel], [p[3] for p in sel], s=60, marker="s", facecolors="none", edgecolors=C_AUX, lw=1.2, label="정수 화소(반올림)")
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=14, color=C_SEL, label="Devernay 서브픽셀 점")
        for p in sel: ax.plot([p[2], p[0]], [p[3], p[1]], color=C_AUX, lw=.8)
        ax.set_title("정수 경계 화소 vs 서브픽셀 점 (원본 화소 확대)", fontsize=8); ax.legend(fontsize=6, loc="lower right")
    elif fam == "03":
        show(ax, img, x0, x1, y0, y1)
        overlay_mask(ax, rec["intermediate_images"], "canny_edges", x0, x1, y0, y1, color=(.6, .6, .6), alpha=.5)
        overlay_mask(ax, rec["intermediate_images"], "canny_polarity_band_filtered", x0, x1, y0, y1, color=(0, .8, 1), alpha=.7)
        for c in iv.get("hough_candidates", [])[:30]:
            th = np.radians(c["theta_normal_deg"]); r = c["rho_original_px"]
            ys_ = (r - np.array([x0, x1]) * np.cos(th)) / np.sin(th)
            ax.plot([x0, x1], ys_, color=C_AUX if c["state"] == "selected" else C_EXCL, lw=1.4 if c["state"] == "selected" else .6,
                    ls="--" if c["state"] == "selected" else "-")
        cs = iv.get("hough_candidates", [])[:4]
        txt = "\n".join(f"#{c['rank']} θ={c['theta_normal_deg']:.2f}° 득표={c['votes']:.0f} {c['state'].split(':')[0]}" for c in cs)
        ax.text(.01, .02, txt, transform=ax.transAxes, fontsize=5.5, color="white", va="bottom", bbox=dict(facecolor="black", alpha=.6, edgecolor="none"))
        ax.set_title("Canny(회색)·투표 화소(하늘)·허프 후보선(점선=선택)", fontsize=8)
    elif fam in ("04", "07", "08", "09"):
        name = {"04": "lsd_line_support_regions", "07": "ed_edge_map", "08": "fld_internal_canny_recomputed"}.get(fam)
        show(ax, img, x0, x1, y0, y1)
        if name: overlay_mask(ax, rec["intermediate_images"], name, x0, x1, y0, y1, color=(0, .8, 1), alpha=.45)
        sp = np.array(iv.get("selected_support_pixels", []), float).reshape(-1, 2)
        ok = (sp[:, 0] >= x0) & (sp[:, 0] <= x1) if len(sp) else []
        if len(sp): ax.scatter(sp[ok, 0], sp[ok, 1], s=5, marker="s", facecolors="none", edgecolors="#2ca02c", lw=.5)
        draw_segments(ax, it, x0, x1, y0, y1)
        lab = {"04": "LSD 지지 영역", "07": "ED 경계 화소", "08": "FLD 내부 Canny(재계산 표시)", "09": "(ELSED 경계 지도 미저장)"}[fam]
        ax.set_title(f"{lab}(하늘)·선분(회색 제외/파랑 일관성 탈락/주황 선택)·지지 화소(녹)", fontsize=7)
    elif fam == "05":
        ex = [e for e in iv.get("window_examples", []) if e.get("state") == "selected"]
        if not ex: ax.set_title("Zernike 창 예시 없음", fontsize=8); fail_text(ax, rec); return
        e = ex[len(ex) // 2]
        w = np.array(e["window"]); h = w.shape[0] // 2
        ax.imshow(w, cmap="gray", extent=(e["x"] - h - .5, e["x"] + h + .5, e["coarse_y"] + h + .5, e["coarse_y"] - h - .5), interpolation="nearest")
        ax.add_patch(plt.Circle((e["x"], e["coarse_y"]), h + .5, fill=False, color=C_AUX, lw=1))
        psi = np.radians(e["psi_deg"])
        ax.add_patch(FancyArrowPatch((e["x"], e["coarse_y"]), (e["x"] + 2.5 * np.cos(psi), e["coarse_y"] + 2.5 * np.sin(psi)), color=C_SEL, arrowstyle="->", mutation_scale=10))
        ax.plot(e["x"], e["coarse_y"], "s", ms=6, mfc="none", mec=C_AUX)
        ax.plot(e["u"], e["v"], "x", color=C_FINAL, ms=8, mew=2)
        ax.set_title(f"7×7 창(단위원) x={e['x']}: ψ={e['psi_deg']:.1f}°, Q_E={e['Q_E']:.4f}, ℓ={e['l_ramp']:.4f}, 이동 {e['shift_px']:.3f}px", fontsize=7)
        ax.tick_params(labelsize=6)
    elif fam == "06":
        ex = [e for e in iv.get("window_examples", []) if e.get("state") == "selected"]
        if not ex: ax.set_title("PAE 창 예시 없음", fontsize=8); fail_text(ax, rec); return
        e = ex[len(ex) // 2]
        w = np.array(e["window_rows_top_to_bottom"]); m_ = w.shape[0] // 2
        ax.imshow(w, cmap="gray", extent=(e["x"] - 1.5, e["x"] + 1.5, e["edge_pixel_y"] + m_ + .5, e["edge_pixel_y"] - m_ - .5), interpolation="nearest", aspect="equal")
        xx = np.linspace(-1.5, 1.5, 50)
        ax.plot(e["x"] + xx, e["edge_pixel_y"] + e["a"] + e["b"] * xx + e["c"] * xx ** 2, color=C_FINAL, lw=1.5)
        ax.plot(e["x"], e["edge_pixel_y"], "s", ms=7, mfc="none", mec=C_AUX)
        ax.set_title(f"3×9 창 x={e['x']}: A={e['A']:.1f}, B={e['B']:.1f}, a={e['a']:.3f}, b={e['b']:.3f}, c={e['c']:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)


def qr_specific(ax, rec, it, img, t, crop):
    fam = rec["method_id"]
    iv = it["intermediates"]
    x0, x1, y0, y1 = crop
    show(ax, img, x0, x1, y0, y1)
    if fam == "Q10":
        overlay_mask(ax, rec["intermediate_images"], "otsu_binary", x0, x1, y0, y1, color=(0, .8, 1), alpha=.35)
        c = np.array(iv.get("selected_contour", []), float).reshape(-1, 2)
        if len(c): ax.plot(np.r_[c[:, 0], c[:1, 0]], np.r_[c[:, 1], c[:1, 1]], color=C_AUX, lw=.8)
        tr = np.array(iv.get("top_run", []), float).reshape(-1, 3)
        if len(tr): ax.scatter(tr[tr[:, 2] == 1, 0], tr[tr[:, 2] == 1, 1], s=3, color=C_SEL)
        pv = np.array(iv.get("polygon_vertices", []), float).reshape(-1, 2)
        if len(pv): ax.plot(pv[:, 0], pv[:, 1], "o", ms=5, mfc="none", mec="#2ca02c")
        ax.set_title("Otsu 이진화(하늘)·외곽 윤곽(파랑)·상단 변 사용점(주황)·근사 꼭짓점(녹)", fontsize=7)
    elif fam == "Q11":
        cc = iv.get("corner_candidates", [])
        for c in cc:
            col = {"outer_top_left_pattern": "#2ca02c", "outer_top_right_pattern": "#17becf"}.get(c["pattern"], C_EXCL)
            ax.plot(*c["xy"], "o", ms=3, color=col)
        for c in iv.get("top_corners", []):
            ax.plot(*c["initial_xy"], "o", ms=7, mfc="none", mec=C_AUX, mew=1.2)
            ax.plot(*c["refined_xy"], "x", ms=8, color=C_FINAL, mew=1.6)
        ax.set_title("코너 후보(회색)·좌상 패턴(녹)·우상 패턴(청록)·초기(파랑 원)→정밀화(빨강 x)", fontsize=7)
    elif fam == "Q12":
        overlay_mask(ax, rec["intermediate_images"], "otsu_binary_inv", x0, x1, y0, y1, color=(0, .8, 1), alpha=.25)
        for f in iv.get("finder_candidates", []):
            q = np.array(f["refined_corners_xy"]); q = np.vstack([q, q[:1]])
            ax.plot(q[:, 0], q[:, 1], color="#17becf", lw=1)
            q = np.array(q)
            ax.plot([q[0, 0], q[2, 0]], [q[0, 1], q[2, 1]], color="#17becf", lw=.5, ls=":")
            ax.plot([q[1, 0], q[3, 0]], [q[1, 1], q[3, 1]], color="#17becf", lw=.5, ls=":")
            ax.plot(*f["center_xy"], "+", ms=10, color="#ffd60a", mew=1.8)
        ax.set_title("파인더 외곽(청록)·대각선(점선)·대각선 교점 중심(노랑 +)", fontsize=7)
    else:
        for name in rec["intermediate_images"]:
            overlay_mask(ax, rec["intermediate_images"], name, x0, x1, y0, y1, color=(0, .8, 1), alpha=.35)
        if fam[1:] in ("04", "07", "08", "09"): draw_segments(ax, it, x0, x1, y0, y1, lw_sel=1.5)
        scatter_points(ax, rec, it, x0, x1, y0, y1)
        ax.set_title("QR ROI 전체: 중간 산출물(하늘)·후보·선택 지지점", fontsize=7)


def keypoint_zoom(ax, rec, it, img):
    """Q11/Q12: zoom on the two points that define the line."""
    iv = it["intermediates"]
    if rec["method_id"] == "Q11":
        pts = [(c["initial_xy"], c["refined_xy"]) for c in iv.get("top_corners", [])]
        title = "상단 두 코너 확대: 초기(파랑 원) → cornerSubPix(빨강 x)"
    else:
        a = iv.get("assignment", {})
        fc = iv.get("finder_candidates", [])
        pts = [(fc[a[k]]["center_xy"], fc[a[k]]["center_xy"]) for k in ("A_top_left", "B_top_right") if k in a and a[k] < len(fc)]
        title = "A·B 파인더 중심 확대(노랑 +). 두 점 연결선 잔차 0은 정확도 근거 아님"
    if not pts:
        ax.set_title("핵심점 없음", fontsize=8); fail_text(ax, rec); return
    ax.axis("off"); ax.set_title(title, fontsize=7)
    for k, (p0, p1) in enumerate(pts[:2]):
        sub = ax.inset_axes([0.02 + 0.5 * k, 0.05, 0.46, 0.85])
        cx, cy = int(round(p1[0])), int(round(p1[1]))
        show(sub, img, cx - 8, cx + 8, cy - 8, cy + 8)
        sub.plot(*p0, "o", ms=9, mfc="none", mec=C_AUX, mew=1.2)
        sub.plot(*p1, "x" if rec["method_id"] == "Q11" else "+", ms=10, color=C_FINAL if rec["method_id"] == "Q11" else "#ffd60a", mew=2)
        draw_line(sub, rec, cx - 8, cx + 8, lw=1)


def stage_figure(rid, mid, recs, targets, img):
    ext = mid.startswith("E")
    tids = ["upper_structure", "lower_structure"] if ext else ["qr_L", "qr_C", "qr_R"]
    fig, axes = plt.subplots(len(tids), 4, figsize=(19, 4.1 * len(tids)), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for row, tid in enumerate(tids):
        rec = recs[f"{rid}_{mid}_{tid}"]; it = inter(rec); t = targets[tid]
        ax = axes[row]
        ang = rec["image_angle_deg"]
        head = f"{tid} | {rec['status']}" + ("" if ang is None else f" | 각도 {ang:+.5f}° m={rec['slope_dy_dx']:+.7f} b={rec['intercept_y_px']:.3f}")
        if ext:
            y0, y1 = t["band_y"][0] - 10, t["band_y"][1] + 10
            for k, (a, b) in enumerate(EXT_WINDOWS[tid]):
                show(ax[k], img, a, b, y0, y1, f"{head if k == 0 else ''}\n{'좌' if k == 0 else '우'}측 가시 구간 원본 확대 x={a}~{b} (가로:세로 1:1)")
                ax[k].axhline(t["band_y"][0] - .5, color="white", lw=.5, ls=":"); ax[k].axhline(t["band_y"][1] + .5, color="white", lw=.5, ls=":")
                if mid[1:] in ("04", "07", "08", "09"): draw_segments(ax[k], it, a, b, y0, y1)
                scatter_points(ax[k], rec, it, a, b, y0, y1)
                draw_line(ax[k], rec, a, b)
                if rec["status"] != "ok": fail_text(ax[k], rec)
            method_panel(ax[2], rec, it, img, t, (EXT_WINDOWS[tid][0][0], EXT_WINDOWS[tid][0][0] + 120, y0, y1))
            if mid[1:] in ("04", "07", "08", "09"): segment_residual_panel(ax[3], rec, it, t)
            else: residual_panel(ax[3], rec, it, t)
        else:
            rx0, ry0, rx1, ry1 = t["roi_xyxy"]
            qr_specific(ax[0], rec, it, img, t, (rx0, rx1 - 1, ry0, ry1 - 1))
            ax[0].set_title(head + "\n" + ax[0].get_title(), fontsize=7)
            draw_line(ax[0], rec, rx0, rx1 - 1, lw=1.2)
            xa, xb = t["x_range"][0] - 12, t["x_range"][1] + 12
            show(ax[1], img, xa, xb, 950, 990, "상단 테두리 원본 확대(1:1), 점선=탐색 band")
            ax[1].axhline(t["band_y"][0] - .5, color="white", lw=.5, ls=":"); ax[1].axhline(t["band_y"][1] + .5, color="white", lw=.5, ls=":")
            if mid[1:] in ("04", "07", "08", "09"): draw_segments(ax[1], it, xa, xb, 950, 990)
            scatter_points(ax[1], rec, it, xa, xb, 950, 990)
            if mid == "Q11":
                for c in it["intermediates"].get("top_corners", []):
                    ax[1].plot(*c["initial_xy"], "o", ms=7, mfc="none", mec=C_AUX); ax[1].plot(*c["refined_xy"], "x", ms=8, color=C_FINAL, mew=1.6)
            draw_line(ax[1], rec, xa, xb)
            if rec["status"] != "ok": fail_text(ax[1], rec)
            if mid == "Q12":
                ax[1].set_title("Q12는 바깥 테두리가 아닌 파인더 중심선이므로 이 창 밖에 위치(참고용 창)", fontsize=7)
            if mid in ("Q10", "Q11", "Q12"):
                if mid == "Q12":
                    show(ax[2], img, rx0, rx1 - 1, ry0, ry1 - 1, "A-B 중심 연결선(빨강) — 테두리와 다른 특징")
                    for f in it["intermediates"].get("finder_candidates", []):
                        ax[2].plot(*f["center_xy"], "+", ms=10, color="#ffd60a", mew=1.8)
                    draw_line(ax[2], rec, rx0, rx1 - 1)
                elif mid == "Q10":
                    show(ax[2], img, rx0, rx1 - 1, ry0, ry1 - 1, "Otsu 이진화 결과(실제 산출물)")
                    overlay_mask(ax[2], rec["intermediate_images"], "otsu_binary", rx0, rx1 - 1, ry0, ry1 - 1, color=(1, 1, 0), alpha=.5)
                else:
                    keypoint_zoom(ax[2], rec, it, img)
                if mid == "Q10":
                    residual_panel(ax[3], rec, it, t)
                elif mid == "Q12":
                    keypoint_zoom(ax[3], rec, it, img)
                else:
                    ax[3].axis("off")
                    ax[3].text(.02, .5, "두 코너를 잇는 직선은 항상 생기므로\n잔차가 정의되지 않습니다(fit RMS = null).", fontsize=8, va="center")
            else:
                method_panel(ax[2], rec, it, img, t, (xa, xb, 950, 990))
                if mid[1:] in ("04", "07", "08", "09"): segment_residual_panel(ax[3], rec, it, t)
                else: residual_panel(ax[3], rec, it, t)
    handles = [Line2D([], [], color=C_CAND, marker="s", ls="", mfc="none", label="후보(옅은 색)"),
               Line2D([], [], color=C_EXCL, lw=1, label="제외된 선분/후보"),
               Line2D([], [], color=C_SEL, marker="o", ls="", label="선택된 지지점/선분"),
               Line2D([], [], color=C_OUT, marker="x", ls="", label="피팅 trim 제외"),
               Line2D([], [], color=C_FINAL, lw=2, label="최종 직선")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8)
    fig.suptitle(f"{rid} · {mid}: 실제 실행 산출물로 그린 추출 과정 (원본 좌표, 가로:세로 1:1; 잔차 패널은 각도와 무관)", fontsize=11)
    out = FIG / "methods" / f"{mid}_{rid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=95)
    plt.close(fig)
    return out


def targets_figure(targets, images):
    img = images["R01"]
    fig, ax = plt.subplots(figsize=(15, 8.8), layout="constrained")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(-.5, 3839.5, 2159.5, -.5))
    colors = {"upper_structure": "#ff6438", "lower_structure": "#00aedb"}
    for tid in ("upper_structure", "lower_structure"):
        t = targets[tid]; c = colors[tid]
        (x0, x1), (y0, y1) = t["x_range"], t["band_y"]
        ax.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1, y1 - y0 + 1, fill=False, ec=c, lw=1.5))
        for a, b in t["exclude_x"]:
            ax.add_patch(Rectangle((a - .5, y0 - .5), b - a + 1, y1 - y0 + 1, fc="gray", alpha=.55, hatch="////", ec="gray"))
        ax.text(x0 + 10, y0 - 22, f"{tid}: {t['name_ko']} 1개 — 탐색 y={y0}~{y1}, 가림 x={t['exclude_x'][0][0]}~{t['exclude_x'][0][1]}(빗금) 제외, 양쪽 점을 하나의 직선으로",
                color=c, fontsize=9, bbox=dict(facecolor="black", alpha=.6, edgecolor="none"))
    for tid in ("qr_L", "qr_C", "qr_R"):
        t = targets[tid]; x0, y0, x1, y1 = t["roi_xyxy"]
        ax.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fill=False, ec="#39ff14", lw=1.2))
        (a, b), (c0, c1) = t["x_range"], t["band_y"]
        ax.add_patch(Rectangle((a - .5, c0 - .5), b - a + 1, c1 - c0 + 1, fill=False, ec="#ff2bd6", lw=1.2))
        ax.text(x0, y1 + 30, tid, color="#39ff14", fontsize=9, bbox=dict(facecolor="black", alpha=.6, edgecolor="none"))
    ax.set_title("R01 center.bmp: 측정 대상 위치 (축척 그대로, 원본 좌표). 주황=상단 구조물, 하늘=하단 구조물, 녹=QR ROI, 분홍=QR 바깥 테두리 상단 경계 탐색 범위", fontsize=10)
    ax.set_xlabel("원본 x (px)"); ax.set_ylabel("원본 y (px)")
    out = FIG / "targets_R01_overview.png"
    fig.savefig(out, dpi=110); plt.close(fig)

    # Zoom + column-mean profiles showing the target transition (dark above -> bright below).
    fig, axes = plt.subplots(3, 4, figsize=(19, 10.5), layout="constrained")
    specs = [("upper_structure", 300, 620), ("upper_structure", 3000, 3320), ("lower_structure", 300, 620), ("lower_structure", 3000, 3320)]
    for k, (tid, a, b) in enumerate(specs):
        t = targets[tid]; y0, y1 = t["band_y"]
        ax = axes[0, k]
        show(ax, img, a, b, y0 - 25, y1 + 25, f"{tid} x={a}~{b} 원본 확대(1:1). 점선=탐색 band")
        ax.axhline(y0 - .5, color=colors[tid], ls="--", lw=1); ax.axhline(y1 + .5, color=colors[tid], ls="--", lw=1)
        pr = img[y0 - 25:y1 + 26, a:b + 1].astype(float).mean(axis=1)
        axes[1, k].plot(np.arange(y0 - 25, y1 + 26), pr, color=colors[tid])
        axes[1, k].axvspan(y0, y1, color=colors[tid], alpha=.12)
        axes[1, k].set_title("열 평균 밝기 vs y: band 안에서 y 증가 시 어두움→밝음 경계가 주 대상", fontsize=8)
        axes[1, k].set_xlabel("원본 y (px)", fontsize=7); axes[1, k].tick_params(labelsize=6)
    for k, tid in enumerate(("qr_L", "qr_C", "qr_R")):
        t = targets[tid]; x0, y0, x1, y1 = t["roi_xyxy"]
        ax = axes[2, k]
        show(ax, img, x0, x1 - 1, y0, y1 - 1, f"{tid} ROI {t['roi_xyxy']} (x1,y1 미포함)")
        (a, b), (c0, c1) = t["x_range"], t["band_y"]
        ax.add_patch(Rectangle((a - .5, c0 - .5), b - a + 1, c1 - c0 + 1, fill=False, ec="#ff2bd6", lw=1.2))
        ax.annotate("주 선: 바깥 흰 테두리 상단 경계\n(위 어두운 판 → 아래 흰 테두리)", xy=((a + b) / 2, (c0 + c1) / 2), xytext=(x0 + 5, y0 + 120),
                    color="#ff2bd6", fontsize=7, arrowprops=dict(arrowstyle="->", color="#ff2bd6"), bbox=dict(facecolor="black", alpha=.6, edgecolor="none"))
    t = targets["qr_L"]; (a, b), (c0, c1) = t["x_range"], t["band_y"]
    pr = img[945:1000, a:b + 1].astype(float).mean(axis=1)
    axes[2, 3].plot(np.arange(945, 1000), pr, color="#ff2bd6"); axes[2, 3].axvspan(c0, c1, color="#ff2bd6", alpha=.12)
    axes[2, 3].set_title("qr_L 열 평균 밝기 vs y: y≈953 밝음→어두움(판 상단, 반대 극성), y≈967 어두움→밝음(주 대상), y≈980 파인더(반대 극성)", fontsize=7)
    axes[2, 3].set_xlabel("원본 y (px)", fontsize=7); axes[2, 3].tick_params(labelsize=6)
    fig.suptitle("R01 대상 선정 근거: 원본 확대와 밝기 프로파일 (실제 영상)", fontsize=11)
    out2 = FIG / "targets_R01_zoom_profiles.png"
    fig.savefig(out2, dpi=95); plt.close(fig)

    # Same fixed regions on all four references (no per-image adjustment).
    fig, axes = plt.subplots(4, 4, figsize=(19, 10), layout="constrained")
    for r, rid in enumerate(["R01", "R02", "R03", "R04"]):
        im = images[rid]
        for k, (tid, a, b) in enumerate([("upper_structure", 300, 560), ("lower_structure", 300, 560), ("qr_L", None, None), ("qr_R", None, None)]):
            t = targets[tid]; ax = axes[r, k]
            if a is None:
                (a, b) = t["x_range"]; a -= 15; b += 15
                show(ax, im, a, b, 945, 995, f"{rid} {tid} 상단 테두리")
            else:
                y0, y1 = t["band_y"]
                show(ax, im, a, b, y0 - 12, y1 + 12, f"{rid} {tid} x={a}~{b}")
            y0, y1 = t["band_y"]
            ax.axhline(y0 - .5, color="#ff2bd6", ls="--", lw=.8); ax.axhline(y1 + .5, color="#ff2bd6", ls="--", lw=.8)
    fig.suptitle("네 기준 이미지에 같은 고정 탐색 범위(분홍 점선)를 적용: 대상 경계가 범위 안에 있는지 확인용 (조정 없음)", fontsize=11)
    out3 = FIG / "targets_4refs_same_regions.png"
    fig.savefig(out3, dpi=90); plt.close(fig)
    return [out, out2, out3]


def viewer_assets(images):
    out = FIG / "viewer"; out.mkdir(parents=True, exist_ok=True)
    crops = {"upper": (0, 672, 3839, 756), "lower": (0, 1825, 3839, 1925), "qr": (1520, 935, 2295, 1200)}
    for rid, im in images.items():
        prev = cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", prev, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(out / f"{rid}_preview.jpg"))
        for name, (x0, y0, x1, y1) in crops.items():
            cv2.imencode(".png", im[y0:y1 + 1, x0:x1 + 1])[1].tofile(str(out / f"{rid}_{name}.png"))
    return crops


def main():
    lm, recs, targets, manifest, images = load()
    FIG.mkdir(exist_ok=True)
    targets_figure(targets, images)
    crops = viewer_assets(images)
    (FIG / "viewer" / "crops.json").write_text(json.dumps(crops), encoding="utf-8")
    mids = sorted({r["method_id"] for r in recs.values()})
    only = sys.argv[1:]
    for mid in mids:
        if only and mid not in only: continue
        for rid in ["R01", "R02", "R03", "R04"]:
            stage_figure(rid, mid, recs, targets, images[rid])
        print("figures", mid, flush=True)
    for r in manifest["references"]:
        assert sha(r["path"]) == r["sha256"]
    print("done; inputs unchanged")


if __name__ == "__main__":
    main()
