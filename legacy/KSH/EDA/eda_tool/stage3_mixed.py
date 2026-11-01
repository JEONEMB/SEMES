# -*- coding: utf-8 -*-
"""Stage 3 — 2차 데이터: Y·Θ 혼합 이동에서 1차 단서가 성립하는지 (중첩 검사), 날짜 간 원점 차이 (C-2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from legacy.KSH.EDA.eda_tool.common import CENTER, FIGURES, TABLES, TOL_T_DEG, TOL_Y_UM, setup_matplotlib, write_csv

KEY_FEATS = ["plate_cx", "w_diff_RL", "w_L", "w_R", "curv", "gap_LC", "gap_CR", "plate_cy", "w_mean", "trap_lr_C"]


def _design(y, t, terms):
    cols = {"1": np.ones_like(y), "Y": y, "T": t, "T2": t ** 2, "Y2": y ** 2, "YT": y * t}
    return np.column_stack([cols[k] for k in terms])


def set1_model(f1: pd.DataFrame, feat: str):
    """1차 세트(한 축씩)로 f = c + aY·Y + aT·Θ + qT·Θ² 를 맞춘다 (Y² 는 1차 적합에서 무시할 만큼 작음)."""
    d = f1[f1.kind.isin(["y_sweep", "theta_sweep", "reference"])]
    X = _design(d.cmd_y_mm.values, d.cmd_t_deg.values, ["1", "Y", "T", "T2"])
    beta, *_ = np.linalg.lstsq(X, d[feat].values, rcond=None)
    return beta


def superposition(f1: pd.DataFrame, f2: pd.DataFrame, info: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1차 모델로 2차를 예측 (기준 영상 평균으로 절편만 다시 맞춤) → 잔차 분석, 2차 단독 적합의 교차항."""
    ref2 = f2[f2.kind == "reference"]
    mix = f2[f2.kind == "mixed"].copy()
    rows, per = [], []
    for ft in KEY_FEATS:
        b = set1_model(f1, ft)
        pred = _design(mix.cmd_y_mm.values, mix.cmd_t_deg.values, ["1", "Y", "T", "T2"]) @ b
        offset = ref2[ft].mean() - b[0]                           # 날짜 간 원점 차 (기준 영상 기준)
        resid = mix[ft].values - (pred + offset)
        mix[f"resid_{ft}"] = resid
        # 2차 단독 적합 (교차항 포함)
        X2 = _design(mix.cmd_y_mm.values, mix.cmd_t_deg.values, ["1", "Y", "T", "T2", "YT"])
        b2, res2, *_ = np.linalg.lstsq(X2, mix[ft].values, rcond=None)
        r2 = mix[ft].values - X2 @ b2
        s2 = r2.std(ddof=X2.shape[1])
        cov = s2 ** 2 * np.linalg.inv(X2.T @ X2)
        inr = mix.cmd_t_deg <= 1.0 + 1e-9
        rows.append({"feature": ft, "set1_aY": b[1], "set1_aT": b[2], "set1_qT": b[3],
                     "set2_aY": b2[1], "set2_aT": b2[2], "set2_qT": b2[3], "set2_cross_YT": b2[4],
                     "set2_cross_YT_se": float(np.sqrt(cov[4, 4])), "set2_cross_t": float(b2[4] / np.sqrt(cov[4, 4])),
                     "set2_resid_std": s2, "date_offset": offset,
                     "pred_resid_rms_all": float(np.sqrt((resid ** 2).mean())),
                     "pred_resid_rms_in_range": float(np.sqrt((resid[inr.values] ** 2).mean())),
                     "pred_resid_rms_T2deg": float(np.sqrt((resid[~inr.values] ** 2).mean())) if (~inr).any() else np.nan,
                     "pred_resid_mean_T2deg": float(resid[~inr.values].mean()) if (~inr).any() else np.nan})
    S = pd.DataFrame(rows)
    kY, kT = info["kY_px_per_mm"], info["kT_px_per_deg"]
    S["cross_effect_px_at_Y1_T2"] = S.set2_cross_YT * 1.0 * 2.0
    return S, mix


def centers(fall: pd.DataFrame, q: pd.DataFrame, info: dict) -> pd.DataFrame:
    """모든 세트의 기준 영상을 촬영 시각 순으로 비교 (첫 기준 영상 대비)."""
    kY, kT = info["kY_px_per_mm"], info["kT_px_per_deg"]
    refs = fall[fall.kind == "reference"].sort_values("time_used")
    refs = refs.merge(q[["set_id", "file", "img_mean", "qrC_mean", "qrC_focus"]], on=["set_id", "file"], how="left")
    base = refs.iloc[0]
    rows = []
    for _, r in refs.iterrows():
        dx, dy = r.plate_cx - base.plate_cx, r.plate_cy - base.plate_cy
        rows.append({"set_id": r.set_id, "file": r.file, "time": r.time_used, "plate_cx": r.plate_cx, "plate_cy": r.plate_cy,
                     "dx_px": dx, "dy_px": dy, "dx_as_Y_um": dx / kY * 1000, "dx_as_T_deg": dx / kT,
                     "w_diff_RL": r.w_diff_RL, "w_mean": r.w_mean, "scale_ppm": (r.w_mean / base.w_mean - 1) * 1e6,
                     "img_mean": r.img_mean, "qrC_focus": r.qrC_focus})
    D = pd.DataFrame(rows)
    # 세트별 요약 (C-2)
    srows = []
    prev = None
    for s, g in D.groupby("set_id", sort=False):
        o = {"set_id": s, "n_reference": len(g), "origin_cx_px": g.plate_cx.mean(), "origin_cy_px": g.plate_cy.mean(),
             "repeat_sd_cx_px": g.plate_cx.std(ddof=1) if len(g) > 1 else np.nan,
             "repeat_sd_as_Y_um": g.plate_cx.std(ddof=1) / kY * 1000 if len(g) > 1 else np.nan,
             "repeat_sd_as_T_deg": g.plate_cx.std(ddof=1) / kT if len(g) > 1 else np.nan,
             "repeat_range_cx_px": g.plate_cx.max() - g.plate_cx.min(),
             "w_mean_px": g.w_mean.mean(), "w_diff_RL_px": g.w_diff_RL.mean(), "img_mean": g.img_mean.mean()}
        if prev is not None:
            d = o["origin_cx_px"] - prev["origin_cx_px"]
            o.update({"vs_prev_dx_px": d, "vs_prev_dy_px": o["origin_cy_px"] - prev["origin_cy_px"],
                      "vs_prev_as_Y_um": d / kY * 1000, "vs_prev_as_T_deg": d / kT,
                      "vs_prev_scale_ppm": (o["w_mean_px"] / prev["w_mean_px"] - 1) * 1e6,
                      "vs_prev_w_diff_RL_px": o["w_diff_RL_px"] - prev["w_diff_RL_px"],
                      "vs_prev_brightness": o["img_mean"] - prev["img_mean"]})
        srows.append(o); prev = o
    return D, pd.DataFrame(srows)


def plots(f1, f2, mix, S, D, info, plt, tag2: str = "") -> list[str]:
    out = []
    # 범위 밖
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.add_patch(plt.Rectangle((-1, -2), 2, 4, color="tab:blue", alpha=0.08, label="1차 세트 범위 (Θ ±1°, Y ±2 mm, 한 축씩)"))
    ax.plot([-1, 1], [0, 0], color="tab:blue", lw=3, alpha=0.5); ax.plot([0, 0], [-2, 2], color="tab:blue", lw=3, alpha=0.5)
    m = f2[f2.kind == "mixed"]
    out_r = (m.cmd_t_deg.abs() > 1 + 1e-9) | (m.cmd_y_mm.abs() > 2 + 1e-9)
    ax.scatter(m.cmd_t_deg[~out_r], m.cmd_y_mm[~out_r], c="tab:green", s=50, label=f"2차 범위 안 ({(~out_r).sum()}장)")
    ax.scatter(m.cmd_t_deg[out_r], m.cmd_y_mm[out_r], c="tab:red", marker="x", s=70, label=f"2차 범위 밖 ({out_r.sum()}장, {out_r.mean() * 100:.0f} %)")
    ax.set(xlabel="지령 Θ (°)", ylabel="지령 Y (mm)", title="2차 세트 지령값 vs 1차 학습 범위", xlim=(-1.2, 2.3), ylim=(-2.2, 2.2))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); p = FIGURES / f"s3_out_of_range_{tag2}.png"; fig.savefig(p); plt.close(fig); out.append(str(p))

    # 중첩 잔차
    feats = ["plate_cx", "w_diff_RL", "curv", "w_mean"]
    fig, axs = plt.subplots(len(feats), 2, figsize=(13, 3 * len(feats)))
    for i, ft in enumerate(feats):
        for j, (x, xl) in enumerate((("cmd_t_deg", "지령 Θ (°)"), ("cmd_y_mm", "지령 Y (mm)"))):
            ax = axs[i, j]
            sc = ax.scatter(mix[x], mix[f"resid_{ft}"], c=mix["cmd_y_mm" if j == 0 else "cmd_t_deg"], cmap="viridis", s=30)
            ax.axhline(0, color="k", lw=0.8)
            ax.set(xlabel=xl, ylabel=f"{ft} 잔차 (px)")
            fig.colorbar(sc, ax=ax, fraction=0.03, label="Y (mm)" if j == 0 else "Θ (°)")
        r = S[S.feature == ft].iloc[0]
        axs[i, 0].set_title(f"{ft}: 1차 모델 예측 잔차 RMS {r.pred_resid_rms_all:.3f} px (범위 안 {r.pred_resid_rms_in_range:.3f}), 교차항 t = {r.set2_cross_t:.1f}", fontsize=9)
    fig.tight_layout(); p = FIGURES / f"s3_superposition_residuals_{tag2}.png"; fig.savefig(p); plt.close(fig); out.append(str(p))

    # 원점 비교
    kY = info["kY_px_per_mm"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    t = pd.to_datetime(D.time)
    lab = [f"{f.split('/')[-1]}\n{tt:%m-%d %H:%M}" for f, tt in zip(D.file, t)]
    axs[0].bar(range(len(D)), D.dx_as_Y_um, color=["tab:blue" if s.startswith("set1") else "tab:orange" for s in D.set_id])
    axs[0].axhspan(-TOL_Y_UM, TOL_Y_UM, color="tab:green", alpha=0.08)
    axs[0].set_xticks(range(len(D))); axs[0].set_xticklabels(lab, fontsize=8)
    axs[0].set(ylabel="플레이트 중심 x 차 → Y 환산 (µm)", title="기준 영상 원점 (첫 기준 대비)")
    axs[1].bar(range(len(D)), D.scale_ppm, color="0.5"); axs[1].set_xticks(range(len(D))); axs[1].set_xticklabels(lab, fontsize=8)
    axs[1].set(ylabel="QR 폭 평균 변화 (ppm)", title="배율 (거리·초점)")
    axs[2].bar(range(len(D)), D.img_mean - D.img_mean.iloc[0], color="0.5"); axs[2].set_xticks(range(len(D))); axs[2].set_xticklabels(lab, fontsize=8)
    axs[2].set(ylabel="영상 평균 밝기 차", title="밝기")
    fig.tight_layout(); p = FIGURES / "s3_center_compare.png"; fig.savefig(p); plt.close(fig); out.append(str(p))
    return out


def run_all(sets, no_video=False) -> dict:
    import json
    from legacy.KSH.EDA.eda_tool.common import RESULTS
    plt = setup_matplotlib()
    f = pd.read_csv(TABLES / "features.csv", encoding="utf-8-sig")
    q = pd.read_csv(TABLES / "quality.csv", encoding="utf-8-sig")
    info = json.loads((RESULTS / "run_log.json").read_text(encoding="utf-8")).get("stage2", {}).get("sep_info")
    if info is None:
        raise RuntimeError("Stage 2 결과(sep_info)가 없습니다. Stage 2 를 먼저 실행하세요.")
    f = f[f.corners_ok]
    tag1 = f[f.kind == "theta_sweep"].set_id.iloc[0]
    f1 = f[f.set_id == tag1]
    out = {"figs": [], "out_of_range_frac": {}}
    D, C = centers(f, q, info)
    write_csv(D, TABLES / "drift.csv")
    write_csv(C, CENTER / "center_by_set.csv")
    S_all, R_all = [], []
    for tag2 in [s for s in f.set_id.unique() if (f[f.set_id == s].kind == "mixed").any()]:
        f2 = f[f.set_id == tag2]
        S, mix = superposition(f1, f2, info)
        S.insert(0, "set_id", tag2)
        S_all.append(S)
        R_all.append(mix[["set_id", "file", "cmd_y_mm", "cmd_t_deg", "shot"] + [c for c in mix.columns if c.startswith("resid_")]])
        out["figs"] += plots(f1, f2, mix, S, D, info, plt, tag2)
        m = f2[f2.kind == "mixed"]
        out["out_of_range_frac"][tag2] = float(((m.cmd_t_deg.abs() > 1 + 1e-9) | (m.cmd_y_mm.abs() > 2 + 1e-9)).mean())
        if not no_video:
            from legacy.KSH.EDA.eda_tool.stage2_features import sync_videos
            cat = pd.read_csv(TABLES / "catalog.csv", encoding="utf-8-sig")
            fall = pd.read_csv(TABLES / "features.csv", encoding="utf-8-sig")
            out[f"video_{tag2}"] = sync_videos(cat, fall, tag2, ["mixed"], "sync", plt)
    if S_all:
        write_csv(pd.concat(S_all, ignore_index=True), TABLES / "superposition.csv")
        write_csv(pd.concat(R_all, ignore_index=True), TABLES / "superposition_residuals.csv")
    return out
