# -*- coding: utf-8 -*-
"""Stage 4 — 라벨(지령값) 신뢰도.

배경 시차 추정(Center_check, 벽 랜드마크 사용)을 '정답 확인 도구' 로만 써서
지령값과의 차이, 접근 방향(백래시), 같은 지령값 재이동 흩어짐, 원점 불확도를 정리한다.
배경 시차 자체의 오차가 있으므로, 세션 안에서는 '라벨 오차 상한' 으로 해석한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import FIGURES, REPO_DIR, TABLES, TOL_T_DEG, TOL_Y_UM, load_set_config, setup_matplotlib, write_csv


def _load_reference(sets) -> pd.DataFrame:
    frames = []
    for d in sets:
        cfg = load_set_config(d)
        p = cfg.get("label_reference_csv")
        if not p:
            continue
        r = pd.read_csv(REPO_DIR / p, encoding="utf-8-sig")
        r["set_id"] = cfg["set_id"]
        r["basename"] = r.file.str.split("/").str[-1]
        frames.append(r.rename(columns={"estY_mm": "refY_mm", "estT_deg": "refT_deg", "train": "ref_train",
                                        "sdY_um": "ref_sdY_um", "sdT_deg": "ref_sdT_deg"})
                      [["set_id", "basename", "refY_mm", "refT_deg", "ref_train", "ref_sdY_um", "ref_sdT_deg"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_table(cat: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    c = cat.copy()
    c["basename"] = c.file.str.split("/").str[-1]
    t = c.merge(ref, on=["set_id", "basename"], how="left")
    # 지령값 - 배경 시차 추정 = 라벨이 실제 자세보다 얼마나 앞서 있는지 (배경 시차 오차 포함)
    t["dY_um"] = (t.cmd_y_mm - t.refY_mm) * 1000
    t["dT_deg"] = t.cmd_t_deg - t.refT_deg
    # Y 스윕 촬영 묶음: 0.1 mm 배수(첫 묶음) / 나머지(두 번째 묶음)
    y = t.cmd_y_mm.round(4)
    t["y_pass"] = np.where(t.kind != "y_sweep", "", np.where(np.isclose(np.round(y * 10) / 10, y), "pass1(0.1배수)", "pass2(나머지)"))
    return t


def backlash_did(t: pd.DataFrame) -> dict:
    """Y 스윕 이중차분: 음수 Y 는 묶음마다 접근 방향이 다르고(pass1 -, pass2 +) 양수 Y 는 같다(둘 다 +).
    (pass2 - pass1)_음수 - (pass2 - pass1)_양수 = 접근 방향 차(+ 접근 - - 접근) 추정, 학습/검증·시간 효과는 상쇄."""
    y = t[(t.kind == "y_sweep") & t.dY_um.notna()]
    out = {}
    for side, m in (("neg", y.cmd_y_mm < 0), ("pos", y.cmd_y_mm > 0)):
        g = y[m]
        # 묶음마다 의도한 접근 방향만 사용 (예: 두 번째 묶음 −1.95 는 +1.95 에서 건너와 − 접근 → 제외)
        want1, want2 = ("-", "+") if side == "neg" else ("+", "+")
        a = g[g.y_pass.str.startswith("pass1") & (g.approach_y == want1)]
        b = g[g.y_pass.str.startswith("pass2") & (g.approach_y == want2)]
        out[f"{side}_n_pass1"], out[f"{side}_n_pass2"] = len(a), len(b)
        for col, unit in (("dY_um", "um"), ("dT_deg", "deg")):
            d = b[col].mean() - a[col].mean()
            se = np.sqrt(b[col].var(ddof=1) / len(b) + a[col].var(ddof=1) / len(a))
            out[f"{side}_{col}_pass2_minus_pass1"] = d
            out[f"{side}_{col}_se"] = se
        out[f"{side}_approach_pass1"] = ",".join(sorted(a.approach_y.unique()))
        out[f"{side}_approach_pass2"] = ",".join(sorted(b.approach_y.unique()))
    for col in ("dY_um", "dT_deg"):
        out[f"backlash_{col}"] = out[f"neg_{col}_pass2_minus_pass1"] - out[f"pos_{col}_pass2_minus_pass1"]
        out[f"backlash_{col}_se"] = float(np.hypot(out[f"neg_{col}_se"], out[f"pos_{col}_se"]))
    # 라벨 차이는 '지령 - 실제' 이므로 + 접근 - (- 접근) 의 실제 위치 차는 부호 반대
    out["approach_plus_minus_actual_Y_um"] = -out["backlash_dY_um"]
    out["approach_plus_minus_actual_T_deg"] = -out["backlash_dT_deg"]
    return out


def reapproach(t: pd.DataFrame) -> pd.DataFrame:
    """2차 세트 같은 지령값 3장: 각 장의 배경 시차 추정이 묶음 평균에서 얼마나 벗어나는지 (재이동 흩어짐)."""
    m = t[(t.kind == "mixed") & t.refY_mm.notna()].copy()
    g = m.groupby(["cmd_y_mm", "cmd_t_deg"])
    m["devY_um"] = (m.refY_mm - g.refY_mm.transform("mean")) * 1000
    m["devT_deg"] = m.refT_deg - g.refT_deg.transform("mean")
    return m


def pair_noise(t: pd.DataFrame) -> dict:
    """같은 지령값 연속 쌍을 '정지 쌍'(플레이트 이동 < 0.03 px, 장비가 움직이지 않음)과 '재이동 쌍'으로 나눠
    배경 시차 도구 자체 잡음과 재위치 흩어짐을 분리한다 (차이 sd / √2)."""
    f = pd.read_csv(TABLES / "features.csv", encoding="utf-8-sig")[["set_id", "file", "plate_cx"]]
    m = t[t.kind == "mixed"].merge(f, on=["set_id", "file"]).sort_values(["cmd_y_mm", "cmd_t_deg", "shot"])
    st, mv = [], []
    for _, g in m.groupby(["cmd_y_mm", "cmd_t_deg"]):
        g = g.sort_values("shot")
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            d = ((b.refY_mm - a.refY_mm) * 1000, b.refT_deg - a.refT_deg)
            (st if abs(b.plate_cx - a.plate_cx) < 0.03 else mv).append(d)
    st, mv = np.array(st), np.array(mv)
    tool = np.sqrt((st ** 2).mean(0) / 2) if len(st) else np.array([np.nan, np.nan])
    raw = np.sqrt((mv ** 2).mean(0) / 2) if len(mv) else np.array([np.nan, np.nan])
    net = np.sqrt(np.maximum(raw ** 2 - tool ** 2, 0))
    return {"n_static": len(st), "n_moved": len(mv), "tool_Y_um": float(tool[0]), "tool_T_deg": float(tool[1]),
            "moved_raw_Y_um": float(raw[0]), "moved_raw_T_deg": float(raw[1]), "net_Y_um": float(net[0]), "net_T_deg": float(net[1])}


def run_all(sets, no_video=False) -> dict:
    plt = setup_matplotlib()
    cat = pd.read_csv(TABLES / "catalog.csv", encoding="utf-8-sig")
    ref = _load_reference(sets)
    if ref.empty:
        return {"skipped": "label_reference_csv 없음"}
    t = build_table(cat, ref)
    write_csv(t.drop(columns=["path"]), TABLES / "label_check.csv")
    out = {"n_matched": int(t.refY_mm.notna().sum()), "n_total": len(t)}

    # 1) 세션 안 차이 (1차 세트, 배경 시차 학습에 쓰지 않은 홀수 번째)
    s1 = t[(t.kind.isin(["y_sweep", "theta_sweep"])) & (t.ref_train == 0) & t.refY_mm.notna()]
    rows = []
    for (sid, kind), g in s1.groupby(["set_id", "kind"]):
        obsY, obsT = np.sqrt((g.dY_um ** 2).mean()), np.sqrt((g.dT_deg ** 2).mean())
        thY, thT = g.ref_sdY_um.median(), g.ref_sdT_deg.median()
        rows.append({"set_id": sid, "kind": kind, "n": len(g), "rms_dY_um": obsY, "rms_dT_deg": obsT,
                     "ref_theory_sdY_um": thY, "ref_theory_sdT_deg": thT,
                     "label_bound_Y_um": float(np.sqrt(max(obsY ** 2 - thY ** 2, 0))),
                     "label_bound_T_deg": float(np.sqrt(max(obsT ** 2 - thT ** 2, 0))),
                     "mean_dY_um": g.dY_um.mean(), "mean_dT_deg": g.dT_deg.mean()})
    within = pd.DataFrame(rows)

    # 2) 접근 방향 (백래시)
    bl = backlash_did(t)
    # 3) 재이동 흩어짐 (2차)
    ra = reapproach(t)
    ra_sum = {"n": len(ra), "sd_Y_um": float(ra.devY_um.std(ddof=1) * np.sqrt(3 / 2)), "sd_T_deg": float(ra.devT_deg.std(ddof=1) * np.sqrt(3 / 2))}
    pn = pair_noise(t)
    ra_sum.update(pn)
    by_shot = ra.groupby("shot")[["devY_um", "devT_deg"]].mean()
    ra_sum.update({f"shot{int(k)}_mean_devY_um": float(v.devY_um) for k, v in by_shot.iterrows()})
    ra_sum.update({f"shot{int(k)}_mean_devT_deg": float(v.devT_deg) for k, v in by_shot.iterrows()})
    # 4) C-3 원점 불확도 (기준 영상)
    refs = t[(t.kind == "reference") & t.refY_mm.notna()][["set_id", "file", "time_used", "refY_mm", "refT_deg"]].copy()
    refs["refY_um"] = refs.refY_mm * 1000
    origin = {}
    for sid, g in refs.groupby("set_id"):
        origin[sid] = {"n": len(g), "mean_Y_um": float(g.refY_um.mean()), "mean_T_deg": float(g.refT_deg.mean()),
                       "sd_Y_um": float(g.refY_um.std(ddof=1)) if len(g) > 1 else np.nan,
                       "sd_T_deg": float(g.refT_deg.std(ddof=1)) if len(g) > 1 else np.nan,
                       "range_Y_um": float(g.refY_um.max() - g.refY_um.min()), "range_T_deg": float(g.refT_deg.max() - g.refT_deg.min())}
    # 5) 세션 계통 (2차 세트, 1차 모델 그대로) — 라벨인지 배경 시차 모델인지 구분 불가
    s2 = t[(t.kind == "mixed") & t.refT_deg.notna()]
    sess = {}
    if len(s2):
        p = np.polyfit(s2.cmd_t_deg, s2.dT_deg, 1)
        sess = {"mean_dY_um": float(s2.dY_um.mean()), "mean_dT_deg": float(s2.dT_deg.mean()),
                "dT_slope_per_deg": float(p[0]), "rms_dT_deg": float(np.sqrt((s2.dT_deg ** 2).mean()))}

    # 최종 라벨 불확도 (세션 안, 같은 접근 방향 기준) : 원점 ⊕ 재이동 흩어짐  /  접근 방향이 섞이면 백래시 추가
    o2 = origin.get(s2.set_id.iloc[0], {}) if len(s2) else {}
    oY = o2.get("sd_Y_um", np.nan); oT = o2.get("sd_T_deg", np.nan)
    rY, rT = pn["net_Y_um"], pn["net_T_deg"]                          # 도구 잡음을 뺀 재위치 흩어짐
    oY = float(np.sqrt(max(oY ** 2 - pn["tool_Y_um"] ** 2, 0))) if np.isfinite(oY) else oY   # 원점 반복도 도구 잡음 제거
    oT = float(np.sqrt(max(oT ** 2 - pn["tool_T_deg"] ** 2, 0))) if np.isfinite(oT) else oT
    uY_same = float(np.sqrt(np.nansum([oY ** 2, rY ** 2])))
    uT_same = float(np.sqrt(np.nansum([oT ** 2, rT ** 2])))
    bY, bT = abs(bl["backlash_dY_um"]) / 2, abs(bl["backlash_dT_deg"]) / 2      # 방향이 섞이면 ±절반
    uY_mix = float(np.hypot(uY_same, bY)); uT_mix = float(np.hypot(uT_same, bT))
    unc = pd.DataFrame([
        {"component": "참고: 배경 시차 도구 자체 잡음 (정지 쌍)", "Y_um": pn["tool_Y_um"], "T_deg": pn["tool_T_deg"]},
        {"component": "원점(기준 영상 반복) sd, 도구 잡음 제거", "Y_um": oY, "T_deg": oT},
        {"component": "같은 지령값 재이동 흩어짐 sd, 도구 잡음 제거", "Y_um": rY, "T_deg": rT},
        {"component": "접근 방향 차의 절반 (방향 섞일 때)", "Y_um": bY, "T_deg": bT},
        {"component": "합계 — 접근 방향 통일 시", "Y_um": uY_same, "T_deg": uT_same},
        {"component": "합계 — 접근 방향 섞일 때", "Y_um": uY_mix, "T_deg": uT_mix},
        {"component": "참고: 세션 안 라벨 오차 상한 (1차 Θ 스윕)", "Y_um": within[within.kind == "theta_sweep"].label_bound_Y_um.mean(),
         "T_deg": within[within.kind == "theta_sweep"].label_bound_T_deg.mean()},
        {"component": "참고: 세션 간 계통 (2차 평균 차, 라벨/모델 구분 불가)", "Y_um": sess.get("mean_dY_um", np.nan), "T_deg": sess.get("mean_dT_deg", np.nan)},
    ])
    unc["Y_pct_of_tol"] = unc.Y_um.abs() / TOL_Y_UM * 100
    unc["T_pct_of_tol"] = unc.T_deg.abs() / TOL_T_DEG * 100
    write_csv(unc, TABLES / "label_uncertainty.csv")
    write_csv(within, TABLES / "label_within_session.csv")
    out.update({"within": within.to_dict("records"), "backlash": bl, "reapproach": ra_sum, "origin": origin,
                "session": sess, "uncertainty": unc.to_dict("records")})

    # 그림: 백래시
    y = t[(t.kind == "y_sweep") & t.dY_um.notna()]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, col, lab in ((axs[0], "dY_um", "지령 Y - 배경 시차 Y (µm)"), (axs[1], "dT_deg", "지령 Θ - 배경 시차 Θ (°)")):
        for (ps, ap), g in y.groupby(["y_pass", "approach_y"]):
            ax.scatter(g.cmd_y_mm, g[col], s=22, label=f"{ps}, 접근 {ap}", marker="o" if ps.startswith("pass1") else "^")
        ax.axhline(0, color="k", lw=0.8); ax.set(xlabel="지령 Y (mm)", ylabel=lab)
    axs[0].axhspan(-TOL_Y_UM, TOL_Y_UM, color="tab:green", alpha=0.06)
    axs[1].axhspan(-TOL_T_DEG, TOL_T_DEG, color="tab:green", alpha=0.06)
    axs[0].legend(fontsize=8)
    axs[0].set_title(f"Y 스윕: 접근 방향 차 (이중차분) {bl['approach_plus_minus_actual_Y_um']:+.1f} ± {bl['backlash_dY_um_se']:.1f} µm")
    axs[1].set_title(f"Θ 쪽: {bl['approach_plus_minus_actual_T_deg']:+.4f} ± {bl['backlash_dT_deg_se']:.4f}°")
    fig.tight_layout(); p = FIGURES / "s4_backlash.png"; fig.savefig(p); plt.close(fig)

    # 그림: 라벨 불확도
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
    u = unc.iloc[::-1]
    axs[0].barh(u.component, u.Y_pct_of_tol, color="tab:blue"); axs[0].axvline(100, color="k", ls="--")
    axs[0].set(xlabel="판정 기준(±90 µm) 대비 %", title="Y 라벨 불확도")
    axs[1].barh(u.component, u.T_pct_of_tol, color="tab:orange"); axs[1].axvline(100, color="k", ls="--")
    axs[1].set(xlabel="판정 기준(±0.01°) 대비 %", title="Θ 라벨 불확도"); axs[1].set_yticklabels([])
    fig.tight_layout(); p2 = FIGURES / "s4_label_uncertainty.png"; fig.savefig(p2); plt.close(fig)

    # 재이동 그림
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for s, g in ra.groupby("shot"):
        ax.scatter(g.devT_deg, g.devY_um, label=f"({int(s)})번째 촬영", s=30)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set(xlabel="Θ 편차 (°, 묶음 평균 대비)", ylabel="Y 편차 (µm)", title="2차 세트 같은 지령값 3장의 배경 시차 추정 편차 (재이동 흩어짐)")
    ax.legend(fontsize=8)
    fig.tight_layout(); p3 = FIGURES / "s4_reapproach.png"; fig.savefig(p3); plt.close(fig)
    out["figs"] = [str(p), str(p2), str(p3)]
    return out
