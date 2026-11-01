# -*- coding: utf-8 -*-
"""Stage 2 — 1차 데이터: Y 이동 vs Θ 회전 비교 (핵심).

1) QR 꼭짓점 12개 서브픽셀 검출 → 2) Center 정의 → 3) 특징 → 4) 선형성 → 5) 짝지은 차영상
6) 변위장(패치별 이동) 비교 → 7) 분리도 순위 → 8) QR 정보 그래프 + 동기화 영상
특징 계산은 모든 세트에 대해 하고 (Stage 3 재사용), 분석은 1차 세트(y_sweep·theta_sweep)로 한다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

import media
from center_def import diag_center, save_definition
from common import (CENTER, FIGURES, TABLES, TOL_T_DEG, TOL_Y_UM, cache_get, cache_put, read_gray, setup_matplotlib,
                    write_csv)
from qr_geom import find_qr_boxes, subpixel_corners

QR = ("L", "C", "R")
FEATURE_DOC = {
    # 이름: (설명, 단위)
    "plate_cx": ("플레이트 중심 x (QR 3개 대각선 교점 평균)", "px"),
    "plate_cy": ("플레이트 중심 y", "px"),
    "plate_cx_C": ("가운데 QR 중심 x", "px"),
    "gap_LC": ("QR 간격 L–C (중심 x 차)", "px"),
    "gap_CR": ("QR 간격 C–R", "px"),
    "span_LR": ("QR 간격 L–R", "px"),
    "curv": ("곡률 cxL + cxR - 2·cxC (간격 비대칭)", "px"),
    "w_diff_RL": ("가로 폭 차 R - L", "px"),
    "w_ratio_RL": ("가로 폭 비 R / L", "-"),
    "w_mean": ("가로 폭 평균 (3개)", "px"),
    "roll_mean": ("기울기 평균 (위·아래 변 각도)", "°"),
}
for q in QR:
    FEATURE_DOC.update({
        f"cx_{q}": (f"QR {q} 중심 x", "px"), f"cy_{q}": (f"QR {q} 중심 y", "px"),
        f"w_{q}": (f"QR {q} 가로 폭 (위·아래 변 평균)", "px"), f"h_{q}": (f"QR {q} 세로 높이 (왼·오른 변 평균)", "px"),
        f"area_{q}": (f"QR {q} 면적", "px²"), f"roll_{q}": (f"QR {q} 기울기", "°"),
        f"trap_lr_{q}": (f"QR {q} 사다리꼴 (왼 변 - 오른 변)", "px"), f"trap_tb_{q}": (f"QR {q} 사다리꼴 (위 변 - 아래 변)", "px"),
        f"cdiff_x_{q}": (f"QR {q} 중심 정의 차 x (대각선 - 평균)", "px"),
    })


# ---------------------------------------------------------------- 1) 꼭짓점
def corners_one(path: str, sha1: str) -> dict:
    c = cache_get(sha1, "corners")
    if c is not None:
        return c
    g = read_gray(path)
    boxes = find_qr_boxes(g)
    out = {"ok": False}
    if boxes:
        res = [subpixel_corners(g, b) for b in boxes]
        if all(r is not None for r in res):
            out = {"ok": True, "corners": [r["corners"].tolist() for r in res],
                   "edge_resid": [r["edge_resid_px"] for r in res]}
    cache_put(sha1, "corners", out)
    return out


# ---------------------------------------------------------------- 3) 특징
def features_from_corners(C: np.ndarray) -> dict:
    """C : (3, 4, 2) — QR L·C·R × 꼭짓점 TL, TR, BR, BL."""
    f = {}
    cen = []
    for k, q in enumerate(QR):
        tl, tr, br, bl = C[k]
        top, right, bottom, left = (np.linalg.norm(tr - tl), np.linalg.norm(br - tr), np.linalg.norm(br - bl),
                                    np.linalg.norm(bl - tl))
        cd, cm = diag_center(C[k]), C[k].mean(0)
        cen.append(cd)
        x, y = C[k][:, 0], C[k][:, 1]
        f.update({f"cx_{q}": cd[0], f"cy_{q}": cd[1], f"cmean_x_{q}": cm[0], f"cmean_y_{q}": cm[1],
                  f"cdiff_x_{q}": cd[0] - cm[0],
                  f"w_{q}": (top + bottom) / 2, f"h_{q}": (left + right) / 2,
                  f"side_top_{q}": top, f"side_right_{q}": right, f"side_bottom_{q}": bottom, f"side_left_{q}": left,
                  f"area_{q}": 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))),
                  f"roll_{q}": np.degrees(0.5 * (np.arctan2(*(tr - tl)[::-1]) + np.arctan2(*(br - bl)[::-1]))),
                  f"trap_lr_{q}": left - right, f"trap_tb_{q}": top - bottom})
    cen = np.array(cen)
    f.update({"plate_cx": cen[:, 0].mean(), "plate_cy": cen[:, 1].mean(), "plate_cx_C": cen[1, 0], "plate_cy_C": cen[1, 1],
              "gap_LC": cen[1, 0] - cen[0, 0], "gap_CR": cen[2, 0] - cen[1, 0], "span_LR": cen[2, 0] - cen[0, 0],
              "curv": cen[0, 0] + cen[2, 0] - 2 * cen[1, 0],
              "w_diff_RL": f["w_R"] - f["w_L"], "w_ratio_RL": f["w_R"] / f["w_L"],
              "w_mean": (f["w_L"] + f["w_C"] + f["w_R"]) / 3, "roll_mean": (f["roll_L"] + f["roll_C"] + f["roll_R"]) / 3})
    return f


def compute_all_features(cat: pd.DataFrame) -> pd.DataFrame:
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda r: corners_one(r.path, r.sha1), cat.itertuples()))
    rows = []
    for r, c in zip(cat.itertuples(), res):
        base = {"set_id": r.set_id, "file": r.file, "kind": r.kind, "cmd_y_mm": r.cmd_y_mm, "cmd_t_deg": r.cmd_t_deg,
                "shot": r.shot, "shot_order": r.shot_order, "time_used": r.time_used, "approach_y": r.approach_y,
                "approach_t": r.approach_t, "corners_ok": c["ok"]}
        if c["ok"]:
            C = np.array(c["corners"])
            base.update(features_from_corners(C))
            for k, q in enumerate(QR):
                for j, nm in enumerate(("TL", "TR", "BR", "BL")):
                    base[f"k_{q}_{nm}_x"], base[f"k_{q}_{nm}_y"] = C[k, j]
                base[f"edge_resid_{q}"] = float(np.mean(c["edge_resid"][k]))
        rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4) 선형성
def linearity(f1: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    rows = []
    ref = f1[f1.kind == "reference"]
    for sweep, xcol, unit in (("y_sweep", "cmd_y_mm", "mm"), ("theta_sweep", "cmd_t_deg", "°")):
        d = pd.concat([f1[f1.kind == sweep], ref]).sort_values(xcol)
        x = d[xcol].values
        for ft in feats:
            y = d[ft].values.astype(float)
            p1 = np.polyfit(x, y, 1); r1 = y - np.polyval(p1, x)
            p2 = np.polyfit(x, y, 2); r2 = y - np.polyval(p2, x)
            ss = ((y - y.mean()) ** 2).sum()
            row = {"feature": ft, "sweep": sweep, "unit_x": unit, "slope_per_unit": p1[0], "intercept": p1[1],
                   "r2_linear": 1 - (r1 ** 2).sum() / ss if ss > 0 else np.nan, "resid_std_linear": r1.std(ddof=2),
                   "quad_coef": p2[0], "resid_std_quad": r2.std(ddof=3),
                   "resid_lag1_corr": float(np.corrcoef(r2[:-1], r2[1:])[0, 1]) if len(r2) > 3 else np.nan}
            if sweep == "y_sweep":                             # 두 번 나눠 찍은 Y 스윕: 0.1 mm 배수 / 나머지
                mult = np.isclose(np.round(x * 10) / 10, x) & (x != 0)
                odd = ~np.isclose(np.round(x * 10) / 10, x)
                row["resid_pass_diff"] = float(r2[odd].mean() - r2[mult].mean()) if mult.any() and odd.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 7) 분리도
def static_noise(fall: pd.DataFrame, feats: list[str], max_shift_px: float | None = 0.03) -> tuple[pd.Series, int]:
    """2차 세트 같은 지령값 연속 촬영 쌍 → 잡음 σ (차이 std / √2).
    max_shift_px 지정 : 플레이트 이동이 그 미만인 쌍만 (순수 측정 잡음),  None : 모든 쌍 (다시 이동해 찍은 쌍 포함 = 재위치 반복성)."""
    m = fall[(fall.kind == "mixed") & fall.corners_ok].sort_values(["cmd_y_mm", "cmd_t_deg", "shot"])
    diffs = []
    for _, g in m.groupby(["cmd_y_mm", "cmd_t_deg"]):
        g = g.sort_values("shot")
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if max_shift_px is None or (abs(a.plate_cx - b.plate_cx) < max_shift_px and abs(a.plate_cy - b.plate_cy) < max_shift_px):
                diffs.append(b[feats].astype(float).values - a[feats].astype(float).values)
    D = np.array(diffs)
    if len(D) == 0:
        return pd.Series(np.nan, index=feats), 0
    return pd.Series(np.sqrt((D ** 2).mean(0) / 2), index=feats), len(D)


def separability(lin: pd.DataFrame, noise_static: pd.Series, n_pairs: int, feats: list[str], key="plate_cx",
                 noise_repeat: pd.Series | None = None, n_repeat: int = 0) -> tuple[pd.DataFrame, dict]:
    L = lin.set_index(["feature", "sweep"])
    kY, kT = L.loc[(key, "y_sweep"), "slope_per_unit"], L.loc[(key, "theta_sweep"), "slope_per_unit"]
    t_eq = kY / kT                                          # Y 1 mm 와 같은 화면 이동을 만드는 Θ (°)
    rows = []
    for ft in feats:
        aY, aT = L.loc[(ft, "y_sweep"), "slope_per_unit"], L.loc[(ft, "theta_sweep"), "slope_per_unit"]
        disc = aT - aY * kT / kY                              # 플레이트 중심을 고정한 채 Θ 를 1° 바꿀 때 특징 변화
        s_fit = float(np.sqrt(0.5 * (L.loc[(ft, "y_sweep"), "resid_std_quad"] ** 2 + L.loc[(ft, "theta_sweep"), "resid_std_quad"] ** 2)))
        s_st = float(noise_static.get(ft, np.nan))
        s_rp = float(noise_repeat.get(ft, np.nan)) if noise_repeat is not None else np.nan
        rows.append({"feature": ft, "desc": FEATURE_DOC.get(ft, ("", ""))[0], "unit": FEATURE_DOC.get(ft, ("", ""))[1],
                     "per_Y_1mm": aY, "per_T_eq_deg": aT * t_eq, "per_T_1deg": aT,
                     "signal_equal_shift": aY - aT * t_eq,            # 화면 이동이 같은 Y 1 mm 와 Θ t_eq° 의 특징 차
                     "signal_per_0.01deg": abs(disc) * 0.01,
                     "noise_static_sigma": s_st, "noise_repeat_sigma": s_rp, "noise_fit_sigma": s_fit,
                     "snr_static_0.01deg": abs(disc) * 0.01 / s_st if s_st > 0 else np.nan,
                     "snr_fit_0.01deg": abs(disc) * 0.01 / s_fit if s_fit > 0 else np.nan,
                     "theta_res_static_deg": s_st / abs(disc) if disc != 0 else np.inf,
                     "theta_res_repeat_deg": s_rp / abs(disc) if disc != 0 else np.inf,
                     "theta_res_fit_deg": s_fit / abs(disc) if disc != 0 else np.inf})
    df = pd.DataFrame(rows).sort_values("theta_res_static_deg").reset_index(drop=True)
    df["rank_static"] = np.arange(1, len(df) + 1)
    df["rank_fit"] = df.theta_res_fit_deg.rank().astype(int)
    df["margin_vs_tol_static"] = TOL_T_DEG / df.theta_res_static_deg
    df["margin_vs_tol_fit"] = TOL_T_DEG / df.theta_res_fit_deg
    info = {"kY_px_per_mm": kY, "kT_px_per_deg": kT, "theta_equal_shift_deg_per_mm": t_eq, "static_pairs": n_pairs,
            "repeat_pairs": n_repeat, "key": key}
    return df, info


def combined_resolution(f1: pd.DataFrame, lin: pd.DataFrame, noise: pd.Series, feats: list[str]) -> dict:
    """여러 특징을 함께 쓸 때 (Y, Θ) 의 이론 분해능 — 대각 잡음 가정 (특징 간 상관 무시 → 낙관적)."""
    L = lin.set_index(["feature", "sweep"])
    A = np.array([[L.loc[(f, "y_sweep"), "slope_per_unit"], L.loc[(f, "theta_sweep"), "slope_per_unit"]] for f in feats])
    W = np.diag(1 / np.maximum(noise[feats].values.astype(float), 1e-9) ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    return {"features": feats, "sigma_Y_um": float(np.sqrt(cov[0, 0]) * 1000), "sigma_T_deg": float(np.sqrt(cov[1, 1]))}


# ---------------------------------------------------------------- 5) 짝지은 차영상
def _shift_img(g: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(g.astype(np.float32), M, (g.shape[1], g.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def paired_diff(cat1: pd.DataFrame, f1: pd.DataFrame, t_eq: float, plt) -> list[dict]:
    out = []
    th = f1[f1.kind == "theta_sweep"]
    for y_mm in (1.0, 2.0):
        ry = f1[(f1.kind == "y_sweep") & np.isclose(f1.cmd_y_mm, y_mm)].iloc[0]
        rt = th.iloc[(th.plate_cx - ry.plate_cx).abs().argmin()]          # 화면 이동이 가장 비슷한 Θ 사진
        gy = read_gray(cat1[cat1.file == ry.file].path.iloc[0]).astype(np.float32)
        gt = read_gray(cat1[cat1.file == rt.file].path.iloc[0]).astype(np.float32)
        dx, dy = ry.plate_cx - rt.plate_cx, ry.plate_cy - rt.plate_cy
        gt_s = _shift_img(gt, dx, dy)
        x0 = int(ry.cx_L - 160); x1 = int(ry.cx_R + 160); y0 = int(ry.plate_cy - 150); y1 = int(ry.plate_cy + 150)
        d = (gy - gt_s)[y0:y1, x0:x1]
        fig, ax = plt.subplots(figsize=(12, 4.2))
        im = ax.imshow(d, cmap="RdBu_r", vmin=-40, vmax=40, extent=(x0, x1, y1, y0))
        fig.colorbar(im, ax=ax, fraction=0.02, label="밝기 차 (Y 사진 - 이동 맞춘 Θ 사진)")
        for q in QR:
            xs = [ry[f"k_{q}_{n}_x"] for n in ("TL", "TR", "BR", "BL", "TL")]
            ys = [ry[f"k_{q}_{n}_y"] for n in ("TL", "TR", "BR", "BL", "TL")]
            ax.plot(xs, ys, "k-", lw=0.6)
        ax.set_title(f"짝지은 차영상: {ry.file} vs {rt.file} (중심 이동 차 {dx:+.2f}, {dy:+.2f} px 보정 후)")
        p = FIGURES / f"s2_paired_diff_Y{y_mm:+.0f}mm.png"
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        # 열 방향 평균 → 어느 x 에서 차이가 나는가
        out.append({"y_file": ry.file, "t_file": rt.file, "t_deg": rt.cmd_t_deg, "align_dx_px": dx, "align_dy_px": dy,
                    "diff_rms_band": float(np.sqrt((d ** 2).mean())), "figure": str(p)})
    return out


# ---------------------------------------------------------------- 6) 변위장 (패치별 위상 상관)
def patch_flow(cat1: pd.DataFrame, f1: pd.DataFrame, plt, step: int = 32, size: int = 64) -> pd.DataFrame:
    """기준 영상 QR 띠의 패치마다, 각 사진에서 같은 패치의 이동(dx, dy)을 위상 상관으로 구하고
    Y 스윕은 Y 에 대해, Θ 스윕은 Θ 에 대해 회귀해 '단위당 변위장' 을 만든다."""
    ref = f1[f1.kind == "reference"].iloc[0]
    g0 = read_gray(cat1[cat1.file == ref.file].path.iloc[0]).astype(np.float32)
    xs = np.arange(int(ref.cx_L - 110), int(ref.cx_R + 110) - size, step)
    ys = np.arange(int(ref.plate_cy - 100), int(ref.plate_cy + 100) - size + 1, step)
    win = cv2.createHanningWindow((size, size), cv2.CV_32F)
    sub = f1[f1.kind.isin(["y_sweep", "theta_sweep", "reference"])]

    X0, X1, Y0, Y1 = int(xs[0]) - 200, int(xs[-1]) + size + 200, int(ys[0]) - 40, int(ys[-1]) + size + 40   # 잘라 둘 범위

    def load(r):
        g = read_gray(cat1[cat1.file == r.file].path.iloc[0])
        ix, iy = int(round(r.plate_cx - ref.plate_cx)), int(round(r.plate_cy - ref.plate_cy))   # 거친 이동 (플레이트 중심)
        return ix, iy, g[Y0 + iy:Y1 + iy, X0 + ix:X1 + ix].astype(np.float32)

    def one(r, ix, iy, crop):
        res = []
        for y in ys:
            for x in xs:
                a = g0[y:y + size, x:x + size]
                b = crop[y - Y0:y - Y0 + size, x - X0:x - X0 + size]
                (sx, sy), resp = cv2.phaseCorrelate(np.ascontiguousarray(a), np.ascontiguousarray(b))   # 창 함수 인자는 OpenCV 5 에서 결과가 불안정해 쓰지 않음
                res.append((r.file, x + size / 2, y + size / 2, ix + sx, iy + sy, resp))
        return res

    # OpenCV 5 : 다른 스레드에서 OpenCV(imdecode 등)가 도는 동안 phaseCorrelate 를 부르면 0 을 돌려준다 (확인됨)
    # → 스레드로 QR 띠만 잘라 모두 읽은 뒤, 스레드가 끝나고 나서 상관을 순차 계산
    rows_ = list(sub.itertuples())
    with ThreadPoolExecutor(8) as ex:
        loaded = list(ex.map(load, rows_))
    allr = [x for r, (ix, iy, crop) in zip(rows_, loaded) for x in one(r, ix, iy, crop)]
    P = pd.DataFrame(allr, columns=["file", "px", "py", "dx", "dy", "resp"]).merge(sub[["file", "kind", "cmd_y_mm", "cmd_t_deg"]], on="file")
    rows = []
    for (px, py), g in P.groupby(["px", "py"]):
        r = {"px": px, "py": py, "resp_med": g.resp.median()}
        for sweep, col in (("y_sweep", "cmd_y_mm"), ("theta_sweep", "cmd_t_deg")):
            d = g[g.kind.isin([sweep, "reference"])]
            r[f"{sweep}_dx_slope"] = np.polyfit(d[col], d.dx, 1)[0]
            r[f"{sweep}_dy_slope"] = np.polyfit(d[col], d.dy, 1)[0]
        rows.append(r)
    F = pd.DataFrame(rows)
    return F


def plot_flow(F: pd.DataFrame, info: dict, plt) -> str:
    t_eq = info["theta_equal_shift_deg_per_mm"]
    F = F[F.resp_med > 0.2].copy()
    uy, vy = F.y_sweep_dx_slope, F.y_sweep_dy_slope                    # Y 1 mm 당
    ut, vt = F.theta_sweep_dx_slope * t_eq, F.theta_sweep_dy_slope * t_eq   # 같은 화면 이동의 Θ
    fig, axs = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, u, v, title in ((axs[0], uy - uy.mean(), vy - vy.mean(), "Y 1 mm 당 변위 - 평균 이동"),
                            (axs[1], ut - ut.mean(), vt - vt.mean(), f"Θ {t_eq:.3f}° (같은 화면 이동) 당 변위 - 평균 이동"),
                            (axs[2], uy - ut, vy - vt, "차이 (Y - Θ) : Y·Θ 를 가르는 신호")):
        sc = ax.scatter(F.px, F.py, c=u, cmap="RdBu_r", vmin=-0.08, vmax=0.08, s=90, marker="s")
        ax.quiver(F.px, F.py, u, -v, angles="xy", scale=0.6, width=0.0025)
        ax.invert_yaxis(); ax.set_title(title); ax.set_ylabel("y (px)")
        fig.colorbar(sc, ax=ax, fraction=0.02, label="가로 변위 (px)")
    axs[-1].set_xlabel("x (px)")
    p = FIGURES / "s2_flow_compare.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    # 열 평균 단면
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for lab, u, c in (("Y 1 mm", uy, "tab:blue"), (f"Θ {t_eq:.3f}°", ut, "tab:orange")):
        s = pd.Series(u.values, index=F.px).groupby(level=0).mean()
        ax.plot(s.index, s.values, ".-", color=c, label=lab)
    ax.set(xlabel="x (px)", ylabel="가로 변위 (px)", title="QR 띠 가로 위치별 가로 변위 (같은 평균 이동, 기울기 차 = 원근 신호)")
    ax.legend()
    p2 = FIGURES / "s2_flow_profile.png"
    fig.tight_layout(); fig.savefig(p2); plt.close(fig)
    return p


# ---------------------------------------------------------------- 그림
def plot_feature_vs_cmd(f1: pd.DataFrame, feats: list[str], info: dict, plt) -> list[str]:
    t_eq = info["theta_equal_shift_deg_per_mm"]
    out = []
    ref = f1[f1.kind == "reference"]
    for chunk in (feats[:6], feats[6:12], feats[12:18]):
        if not chunk:
            continue
        fig, axs = plt.subplots(len(chunk), 1, figsize=(11, 2.3 * len(chunk)))
        axs = np.atleast_1d(axs)
        for ax, ft in zip(axs, chunk):
            y = f1[f1.kind == "y_sweep"]; t = f1[f1.kind == "theta_sweep"]
            ax.scatter(y.cmd_y_mm, y[ft], s=8, c="tab:blue", label="Y 스윕 (x = Y mm)")
            ax.scatter(t.cmd_t_deg / t_eq, t[ft], s=8, c="tab:orange", label=f"Θ 스윕 (x = Θ / {t_eq:.3f})")
            ax.scatter([0], ref[ft], s=60, facecolors="none", edgecolors="k")
            ax.set_ylabel(f"{ft}\n({FEATURE_DOC.get(ft, ('', ''))[1]})", fontsize=8)
        axs[0].legend(fontsize=8, ncol=2); axs[0].set_title("특징 vs 지령값 — 가로축은 같은 화면 이동 기준 (Y mm, Θ 환산)")
        axs[-1].set_xlabel("Y (mm) / Θ 환산 (같은 플레이트 중심 이동)")
        fig.tight_layout(); p = FIGURES / f"s2_feature_vs_cmd_{len(out) + 1}.png"; fig.savefig(p); plt.close(fig); out.append(str(p))
    return out


def plot_separability(sep: pd.DataFrame, plt) -> str:
    s = sep.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(s.feature, s.theta_res_static_deg, color="tab:green", label="측정 잡음 기준 (같은 자세 반복)")
    ax.scatter(s.theta_res_repeat_deg, s.feature, color="tab:blue", zorder=3, label="재위치 반복 기준 (같은 지령값 다시 이동)")
    ax.barh(s.feature, s.theta_res_fit_deg, color="none", edgecolor="tab:red", label="적합 잔차 기준 (자세·라벨 오차 포함)")
    ax.axvline(TOL_T_DEG, color="k", ls="--", label="판정 기준 0.01°")
    ax.set_xscale("log"); ax.set_xlabel("단일 특징의 Θ 분해능 (°, 작을수록 좋음)")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))   # 로그 눈금의 유니코드 마이너스(글꼴 없음) 대신 일반 숫자
    ax.set_title("Y·Θ 분리도 순위 — 플레이트 중심을 고정한 채 Θ 만 바뀔 때의 특징 변화 / 잡음")
    ax.legend(fontsize=8)
    p = FIGURES / "s2_separability_rank.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return str(p)


def plot_qr_info(fall: pd.DataFrame, tag: str, plt) -> str:
    d = fall[fall.set_id == tag].sort_values("shot_order")
    items = [("cx_L", "cx_C", "cx_R", "QR 중심 x (px)"), ("cy_L", "cy_C", "cy_R", "QR 중심 y (px)"),
             ("w_L", "w_C", "w_R", "가로 폭 (px)"), ("gap_LC", "gap_CR", None, "QR 간격 (px)"),
             ("roll_L", "roll_C", "roll_R", "기울기 (°)"), ("trap_lr_L", "trap_lr_C", "trap_lr_R", "사다리꼴 왼-오른 (px)")]
    fig, axs = plt.subplots(len(items), 2, figsize=(15, 2.2 * len(items)), sharex="col")
    for i, (a, b, c, lab) in enumerate(items):
        for col, (x, xl) in enumerate((("shot_order", "촬영 순번"), ("plate_cx", "플레이트 중심 x (px)"))):
            ax = axs[i, col]
            for f, cl in zip((a, b, c), ("tab:blue", "tab:green", "tab:red")):
                if f is None:
                    continue
                v = d[f] - (d[f].iloc[0] if "cx" in f else 0)
                ax.plot(d[x], v, ".", ms=3, color=cl, label=f)
            ax.set_ylabel(lab + (" (첫 사진 대비)" if "중심 x" in lab else ""), fontsize=8)
            if i == len(items) - 1:
                ax.set_xlabel(xl)
        axs[i, 0].legend(fontsize=7, ncol=3)
    axs[0, 0].set_title(f"{tag}: QR 정보 — 촬영 순번 기준"); axs[0, 1].set_title("플레이트 중심 이동 기준")
    p = FIGURES / f"qr_info_{tag}.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- 8) 동기화 영상
class _Overlay:
    def __init__(self, feats: pd.DataFrame):
        self.f = feats.set_index("file")

    def __call__(self, img, r):
        if r.file not in self.f.index or not self.f.loc[r.file, "corners_ok"]:
            return
        s = img.shape[1] / 3840
        row = self.f.loc[r.file]
        for q, c in zip(QR, ((255, 128, 0), (0, 200, 0), (0, 0, 255))):
            pts = np.array([[row[f"k_{q}_{n}_x"], row[f"k_{q}_{n}_y"]] for n in ("TL", "TR", "BR", "BL")]) * s
            cv2.polylines(img, [pts.astype(np.int32)], True, c, 1, cv2.LINE_AA)
        cv2.circle(img, (int(row.plate_cx * s), int(row.plate_cy * s)), 3, (0, 0, 255), -1)

    def inset(self, ins, r, rect, size):
        if r.file not in self.f.index or not self.f.loc[r.file, "corners_ok"]:
            return
        x0, y0, x1, y1 = rect
        sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)
        row = self.f.loc[r.file]
        for q, c in zip(QR, ((255, 128, 0), (0, 200, 0), (0, 0, 255))):
            pts = np.array([[(row[f"k_{q}_{n}_x"] - x0) * sx, (row[f"k_{q}_{n}_y"] - y0) * sy] for n in ("TL", "TR", "BR", "BL")])
            cv2.polylines(ins, [np.round(pts * 16).astype(np.int32)], True, c, 1, cv2.LINE_AA, shift=4)
            cx, cy = (row[f"cx_{q}"] - x0) * sx, (row[f"cy_{q}"] - y0) * sy
            cv2.drawMarker(ins, (int(cx), int(cy)), c, cv2.MARKER_CROSS, 10, 1)


def _graph_factory(d: pd.DataFrame, cols: list[tuple[str, str]], plt):
    """프레임 i 까지의 점을 그리는 그래프 함수 (A-2 / B)."""
    cache = {}

    def graph(i, w, h):
        key = (w, h)
        if key not in cache:
            fig, axs = plt.subplots(len(cols), 1, figsize=(w / 100, h / 100), dpi=100, sharex=True)
            axs = np.atleast_1d(axs)
            for ax, (c, lab) in zip(axs, cols):
                ax.plot(d.shot_order, d[c], color="0.85", lw=0.8)
                ax.set_ylabel(lab, fontsize=8); ax.tick_params(labelsize=7)
            fig.tight_layout(pad=0.5); fig.canvas.draw()
            bg = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(), cv2.COLOR_RGB2BGR)
            tr = [lambda x, y, T=ax.transData: (int(T.transform((x, y))[0]), int(h - T.transform((x, y))[1])) for ax in axs]
            plt.close(fig)
            cache[key] = (bg, tr)
        bg, tr = cache[key]
        im = bg.copy()
        for k, (c, _) in enumerate(cols):
            for j in range(i + 1):
                x, y = tr[k](d.shot_order.iloc[j], d[c].iloc[j])
                cv2.circle(im, (x, y), 2, (200, 80, 0), -1)
            x, y = tr[k](d.shot_order.iloc[i], d[c].iloc[i])
            cv2.circle(im, (x, y), 6, (0, 0, 230), 2)
        return im
    return graph


def sync_videos(cat: pd.DataFrame, fall: pd.DataFrame, tag: str, kinds: list[str], suffix: str, plt) -> dict:
    c = cat[(cat.set_id == tag) & cat.kind.isin(kinds + ["reference"])].copy()
    c = c.sort_values("shot_order").reset_index(drop=True)
    c["shot_order_orig"] = c.shot_order
    c["shot_order"] = np.arange(1, len(c) + 1)
    f = fall.set_index("file").loc[c.file].reset_index()
    f["shot_order"] = c.shot_order.values
    ref = f.iloc[0]
    inset = (int(ref.cx_L - 160), int(ref.plate_cy - 140), int(ref.cx_R + 160), int(ref.plate_cy + 140))
    graph = _graph_factory(f, [("plate_cx", "플레이트 x (px)"), ("w_diff_RL", "폭 R-L (px)"), ("trap_lr_C", "C 사다리꼴 (px)")], plt)
    return media.shooting_video(c, tag, inset_rect=inset, overlay=_Overlay(fall), graph=graph, suffix=suffix)


# ---------------------------------------------------------------- 실행
def run_all(sets, no_video=False) -> dict:
    plt = setup_matplotlib()
    cat = pd.read_csv(TABLES / "catalog.csv", encoding="utf-8-sig")
    fall = compute_all_features(cat)
    write_csv(fall, TABLES / "features.csv")
    feats = [k for k in FEATURE_DOC if k in fall.columns]

    # C-1 center_by_image
    cols = ["set_id", "file", "kind", "cmd_y_mm", "cmd_t_deg"] + [f"{a}_{q}" for q in QR for a in ("cx", "cy", "cmean_x", "cmean_y")] + ["plate_cx", "plate_cy", "plate_cx_C", "plate_cy_C"]
    write_csv(fall[cols], CENTER / "center_by_image.csv")

    f1 = fall[(fall.set_id == fall[fall.kind == "theta_sweep"].set_id.iloc[0]) & fall.corners_ok]
    tag1 = f1.set_id.iloc[0]
    cat1 = cat[cat.set_id == tag1]
    lin = linearity(f1, feats)
    write_csv(lin, TABLES / "linearity.csv")
    noise, npairs = static_noise(fall, feats)
    noise_rp, nrp = static_noise(fall, feats, max_shift_px=None)
    sep, info = separability(lin, noise, npairs, feats, noise_repeat=noise_rp, n_repeat=nrp)
    write_csv(sep, TABLES / "separability.csv")
    top = [f for f in sep.feature if f not in ("plate_cx", "plate_cy", "plate_cx_C") and not f.startswith(("cx_", "cy_"))][:6]
    comb = combined_resolution(f1, lin, noise, ["plate_cx"] + top)
    comb_rp = combined_resolution(f1, lin, noise_rp, ["plate_cx"] + top)
    comb_all = combined_resolution(f1, lin, noise, [f for f in feats if f not in ("plate_cx_C", "span_LR", "w_mean", "roll_mean", "plate_cy") and np.isfinite(noise.get(f, np.nan))])
    save_definition({"diag_vs_mean_x_px": {q: {"mean": float(fall[f"cdiff_x_{q}"].mean()), "std": float(fall[f"cdiff_x_{q}"].std())} for q in QR},
                     "reference_images": list(cat[cat.kind == "reference"].file)})

    out = {"n_corner_ok": int(fall.corners_ok.sum()), "n_total": len(fall), "sep_info": info,
           "combined_top": comb, "combined_top_repeat_noise": comb_rp, "combined_all": comb_all}
    out["fig_feature"] = plot_feature_vs_cmd(f1, ["plate_cx", "curv", "w_diff_RL", "trap_lr_L", "trap_lr_C", "trap_lr_R",
                                                   "w_L", "w_C", "w_R", "gap_LC", "gap_CR", "roll_mean",
                                                   "trap_tb_C", "h_C", "plate_cy", "area_C"], info, plt)
    out["fig_sep"] = plot_separability(sep, plt)
    out["paired"] = paired_diff(cat1, f1, info["theta_equal_shift_deg_per_mm"], plt)
    F = patch_flow(cat1, f1, plt)
    write_csv(F, TABLES / "patch_flow.csv")
    out["fig_flow"] = plot_flow(F, info, plt)
    for tag in fall.set_id.unique():
        out[f"qr_info_{tag}"] = plot_qr_info(fall, tag, plt)
    if not no_video:
        out["video_Y"] = sync_videos(cat, fall, tag1, ["y_sweep"], "sync_Y", plt)
        out["video_T"] = sync_videos(cat, fall, tag1, ["theta_sweep"], "sync_T", plt)
    return out
