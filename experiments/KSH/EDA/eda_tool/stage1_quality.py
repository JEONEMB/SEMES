# -*- coding: utf-8 -*-
"""Stage 1 — 영상 화질과 환경: QR 영역 밝기·대비·포화·초점, 촬영 잡음, 반사 영역, 이상 사진 경고."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import media
from common import (FIGURES, TABLES, cache_get, cache_put, mad, read_gray, rel, setup_matplotlib, write_csv)
from qr_geom import find_qr_boxes

QR_NAMES = ("L", "C", "R")
METRICS = ("mean", "std", "sat_pct", "focus")


def _immerkaer_sigma(g: np.ndarray) -> float:
    """단일 영상 잡음 σ (Immerkær 1996, 라플라시안 차분 마스크)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    r = cv2.filter2D(g.astype(float), -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6.0)


def measure_one(path: str, sha1: str, force: bool = False) -> dict:
    c = None if force else cache_get(sha1, "stage1")
    if c is not None:
        return c
    g = read_gray(path)
    out = {"img_mean": float(g.mean()), "img_sat_pct": float((g >= 250).mean() * 100),
           "img_noise_sigma": _immerkaer_sigma(g[::2, ::2])}
    boxes = find_qr_boxes(g)
    out["qr_found"] = boxes is not None
    if boxes:
        for name, b in zip(QR_NAMES, boxes):
            x, y, w, h = b["x"], b["y"], b["w"], b["h"]
            m = int(0.08 * w)                                      # 가장자리(플레이트 경계) 제외
            roi = g[y + m:y + h - m, x + m:x + w - m]
            out[f"qr{name}_cx"], out[f"qr{name}_cy"] = b["cx"], b["cy"]
            out[f"qr{name}_mean"] = float(roi.mean())
            out[f"qr{name}_std"] = float(roi.std())
            out[f"qr{name}_sat_pct"] = float((roi >= 250).mean() * 100)
            out[f"qr{name}_focus"] = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    cache_put(sha1, "stage1", out)
    return out


def _flags(df: pd.DataFrame) -> pd.Series:
    """세트 내 중앙값 ± 3 MAD 를 벗어난 항목을 '항목(값)' 문자열로 모은다."""
    cols = ["img_mean", "img_noise_sigma"] + [f"qr{q}_{m}" for q in QR_NAMES for m in ("mean", "std", "focus")]
    # 3 MAD 만 쓰면 아주 고른 데이터에서 0.2 % 밝기 차도 경고가 된다 → 실무 최소 편차를 함께 요구
    floor = {"mean": 0.02, "std": 0.02, "focus": 0.10, "noise_sigma": 0.10}
    parts = {i: [] for i in df.index}
    for c in cols:
        x = df[c].astype(float)
        med, s = np.nanmedian(x), mad(x)
        if not np.isfinite(s) or s == 0:
            continue
        rel_floor = next(v for k, v in floor.items() if c.endswith(k))
        for i in x.index[((x - med).abs() > 3 * s) & ((x - med).abs() > rel_floor * abs(med))]:
            parts[i].append(f"{c}={x[i]:.1f}(중앙 {med:.1f})")
    for i in df.index[~df.qr_found.astype(bool)]:
        parts[i].append("QR검출실패")
    return pd.Series({i: "; ".join(v) for i, v in parts.items()})


# ---------------------------------------------------------------- 연속 촬영 차영상 (2차 세트 같은 지령값 3장)
def noise_from_repeats(cat: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    rep = cat[cat.kind == "mixed"].groupby(["cmd_y_mm", "cmd_t_deg"])
    best = None
    for (y, t), g in rep:
        g = g.sort_values("shot")
        imgs = [read_gray(p).astype(np.float32) for p in g.path]
        for i in range(len(imgs) - 1):
            a, b = imgs[i], imgs[i + 1]
            (dx, dy), resp = cv2.phaseCorrelate(a[800:1350, 1300:2550], b[800:1350, 1300:2550])
            d = a - b
            wall = d[150:450, 200:1000]                           # 위쪽 평탄 영역 (나사·모서리 없음)
            band = d[940:1200, 1400:2500]                         # QR 띠 (Θ 2° 이동 포함)
            rows.append({"set_id": tag, "cmd_y_mm": y, "cmd_t_deg": t, "pair": f"{int(g.shot.iloc[i])}-{int(g.shot.iloc[i + 1])}",
                         "shift_dx_px": dx, "shift_dy_px": dy,
                         "noise_wall_sigma": float(wall.std() / np.sqrt(2)), "noise_band_sigma": float(band.std() / np.sqrt(2)),
                         "dt_s": (pd.to_datetime(g.time_used.iloc[i + 1]) - pd.to_datetime(g.time_used.iloc[i])).total_seconds()})
            if best is None or abs(dx) + abs(dy) < best[0]:
                best = (abs(dx) + abs(dy), d, (y, t, rows[-1]["pair"]))
    df = pd.DataFrame(rows)
    if best is not None:
        plt = setup_matplotlib()
        _, d, (y, t, pr) = best
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
        im = axs[0].imshow(d[600:1500, 1100:2750], cmap="RdBu_r", vmin=-8, vmax=8)
        axs[0].set_title(f"{tag} Y{y:+.2f} T{t:+.2f} ({pr}) 차영상 — QR 주변 (밝기 단위)")
        fig.colorbar(im, ax=axs[0], fraction=0.025)
        axs[1].hist(d[150:450, 200:1000].ravel(), bins=61, range=(-15, 15), color="0.4")
        axs[1].set(title=f"평탄 영역 차이 분포 (σ_차 / √2 = {d[150:450, 200:1000].std() / np.sqrt(2):.2f})", xlabel="밝기 차")
        fig.tight_layout(); fig.savefig(FIGURES / f"s1_noise_map_{tag}.png"); plt.close(fig)
    return df


# ---------------------------------------------------------------- 반사 영역 (포화 빈도)
def reflection_map(cat: pd.DataFrame, tag: str, step: int = 4) -> dict:
    paths = list(cat.path)
    sat = None
    mx = None
    with ThreadPoolExecutor(8) as ex:
        for g in ex.map(lambda p: read_gray(p)[::step, ::step], paths):
            s = (g >= 250).astype(np.float32)
            sat = s if sat is None else sat + s
            mx = g if mx is None else np.maximum(mx, g)
    freq = sat / len(paths)
    plt = setup_matplotlib()
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
    axs[0].imshow(mx, cmap="gray"); axs[0].set_title(f"{tag}: 사진별 최대 밝기 (세트 전체)")
    im = axs[1].imshow(freq * 100, cmap="magma", vmin=0, vmax=100)
    axs[1].set_title("포화(≥250) 빈도 % — 반사 영역")
    fig.colorbar(im, ax=axs[1], fraction=0.025)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(FIGURES / f"s1_reflection_{tag}.png"); plt.close(fig)
    # QR 띠 안 포화 화소 비율
    y0, y1, x0, x1 = 940 // step, 1200 // step, 1400 // step, 2500 // step
    return {"set_id": tag, "sat_any_pct_image": float((freq > 0).mean() * 100),
            "sat_any_pct_qr_band": float((freq[y0:y1, x0:x1] > 0).mean() * 100),
            "sat_always_pct_image": float((freq > 0.9).mean() * 100)}


def plot_trend(q: pd.DataFrame) -> None:
    plt = setup_matplotlib()
    for tag, g in q.groupby("set_id"):
        g = g.sort_values("shot_order")
        fig, axs = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        for ax, m, lab in zip(axs, METRICS, ("평균 밝기", "대비 (표준편차)", "포화 화소 %", "초점 (라플라시안 분산)")):
            for qn, c in zip(QR_NAMES, ("tab:blue", "tab:green", "tab:red")):
                ax.plot(g.shot_order, g[f"qr{qn}_{m}"], ".-", ms=3, lw=0.6, color=c, label=f"QR {qn}")
            w = g[g.quality_warning.fillna("") != ""]
            for x in w.shot_order:
                ax.axvline(x, color="orange", alpha=0.35, lw=1)
            ax.set_ylabel(lab)
        axs[0].legend(ncol=3, fontsize=8)
        axs[0].set_title(f"{tag}: 촬영 순번별 QR 영역 화질 (주황 세로선 = 경고 사진)")
        axs[-1].set_xlabel("촬영 순번")
        fig.tight_layout(); fig.savefig(FIGURES / f"s1_quality_trend_{tag}.png"); plt.close(fig)
    # 1차 vs 2차
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, m, lab in zip(axs, METRICS, ("평균 밝기", "대비", "포화 %", "초점")):
        data = [q[q.set_id == s][[f"qr{n}_{m}" for n in QR_NAMES]].values.ravel() for s in sorted(q.set_id.unique())]
        ax.boxplot(data, tick_labels=sorted(q.set_id.unique())); ax.set_title(f"QR 영역 {lab}")
    fig.tight_layout(); fig.savefig(FIGURES / "s1_quality_trend_compare.png"); plt.close(fig)


def run_all(sets, no_video=False, force=False) -> dict:
    cat = pd.read_csv(TABLES / "catalog.csv", encoding="utf-8-sig")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda r: measure_one(r.path, r.sha1, force), cat.itertuples()))
    q = pd.concat([cat[["set_id", "file", "kind", "cmd_y_mm", "cmd_t_deg", "shot", "shot_order", "time_used",
                        "approach_y", "approach_t", "path"]].reset_index(drop=True), pd.DataFrame(res)], axis=1)
    q["quality_warning"] = ""
    for s, idx in q.groupby("set_id").groups.items():
        q.loc[idx, "quality_warning"] = _flags(q.loc[idx])
    write_csv(q.drop(columns=["path"]), TABLES / "quality.csv")
    plot_trend(q)

    noise = pd.concat([noise_from_repeats(cat[cat.set_id == s], s) for s in cat.set_id.unique()
                       if (cat[cat.set_id == s].kind == "mixed").any()], ignore_index=True)
    write_csv(noise, TABLES / "noise_repeats.csv")
    refl = pd.DataFrame([reflection_map(cat[cat.set_id == s], s) for s in cat.set_id.unique()])
    write_csv(refl, TABLES / "reflection_summary.csv")

    out = {"n_warn": int((q.quality_warning != "").sum()), "warn_files": q[q.quality_warning != ""][["set_id", "file", "quality_warning"]].values.tolist()}
    if not no_video:
        boxes_cache = {}
        for s in q.set_id.unique():
            g = q[q.set_id == s].copy()
            c0 = cat[cat.set_id == s]
            ref = c0[c0.kind == "reference"].iloc[0]
            b = find_qr_boxes(read_gray(ref.path))
            inset = (int(b[0]["x"] - 60), int(min(x["y"] for x in b) - 40), int(b[2]["x"] + b[2]["w"] + 60), int(max(x["y"] + x["h"] for x in b) + 40)) if b else None
            g["approach_y"], g["approach_t"] = c0.approach_y.values, c0.approach_t.values
            out[f"video_{s}"] = media.shooting_video(g, s, inset_rect=inset, warn_col="quality_warning", suffix="shooting_quality")
    return out
