# -*- coding: utf-8 -*-
"""Stage 0 — 데이터 목록 점검: 파일 이름 규칙 해석, 해시·해상도, 촬영 시각 판단, 촬영 순서·접근 방향."""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from legacy.KSH.EDA.eda_tool.common import (DATA_ROOT, FIGURES, TABLES, file_sha1, cache_get, cache_put,
                    load_set_config, rel, setup_matplotlib, write_csv)


# ---------------------------------------------------------------- 파일 이름 해석
def _parse_files(set_dir: Path, cfg: dict) -> tuple[list[dict], list[str]]:
    rows, seen = [], set()
    for rule in cfg["filename_rule"]:
        for p in sorted(set_dir.glob(rule["glob"])):
            if p in seen or not p.is_file():
                continue
            rec = {"path": p, "kind": rule["kind"], "cmd_y_mm": rule.get("y"), "cmd_t_deg": rule.get("t"), "shot": 1}
            ok = True
            if "dir_regex" in rule:
                m = re.match(rule["dir_regex"], p.parent.name)
                if not m:
                    continue
                for k, v in m.groupdict().items():
                    rec[{"y": "cmd_y_mm", "t": "cmd_t_deg"}.get(k, k)] = float(v) if k in "yt" else int(v)
            if "name_regex" in rule:
                m = re.search(rule["name_regex"], p.stem)
                if not m:
                    ok = False
                else:
                    for k, v in m.groupdict().items():
                        rec[{"y": "cmd_y_mm", "t": "cmd_t_deg"}.get(k, k)] = float(v) if k in "yt" else int(v)
            if ok:
                seen.add(p)
                rows.append(rec)
    all_imgs = {p for ext in ("*.bmp", "*.png", "*.tif", "*.tiff", "*.jpg") for p in set_dir.rglob(ext)}
    unparsed = sorted(rel(p) if False else str(p.relative_to(set_dir)) for p in all_imgs - seen)
    return rows, unparsed


def _motor_txt(img: Path) -> Path | None:
    for c in (img.with_suffix(".txt"), img.with_name(img.stem + ".txt.txt")):
        if c.exists():
            return c
    return None


def _read_motor(txt: Path) -> tuple[float, float]:
    lines = [l for l in txt.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    y, t = lines[1].split()[:2]
    return float(y), float(t)


def _ts(x: float) -> str:
    return dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 촬영 시각 판단
def _time_quality(times: pd.Series, cmd_order_key: pd.Series) -> dict:
    t = (times - pd.Timestamp("1970-01-01")).dt.total_seconds()     # pandas 버전마다 내부 단위가 달라 초로 직접 변환
    ties = int(t.duplicated(keep=False).sum())
    span_s = float(t.max() - t.min())
    srt = np.sort(t.values)
    gaps = np.diff(srt)
    return {"n": len(t), "same_second_files": ties, "span_min": round(span_s / 60, 1),
            "median_gap_s": float(np.median(gaps)) if len(gaps) else 0.0,
            "max_gap_min": round(float(gaps.max()) / 60, 1) if len(gaps) else 0.0}


def run(set_dir: Path) -> pd.DataFrame:
    cfg = load_set_config(set_dir)
    set_id = cfg["set_id"]
    rows, unparsed = _parse_files(set_dir, cfg)

    # 모터 기록 (txt) — 기준 영상 값을 원점으로 사용
    motor = cfg.get("motor_log")
    if isinstance(motor, dict) and motor.get("type") == "txt_absolute":
        for r in rows:
            txt = _motor_txt(r["path"])
            r["motor_txt"] = str(txt.relative_to(set_dir)) if txt else ""
            r["motor_y_abs"], r["motor_t_abs"] = _read_motor(txt) if txt else (np.nan, np.nan)
        ref = [r for r in rows if r["kind"] == "reference" and np.isfinite(r["motor_y_abs"])]
        oy = float(np.median([r["motor_y_abs"] for r in ref])) if ref else 0.0
        ot = float(np.median([r["motor_t_abs"] for r in ref])) if ref else 0.0
        for r in rows:
            r["motor_y_mm"] = round(r["motor_y_abs"] - oy, 4)
            r["motor_t_deg"] = round(r["motor_t_abs"] - ot, 4)

    # 파일 정보 (해시는 캐시)
    for r in rows:
        p = r["path"]
        st = os.stat(p)
        r["birth_time"] = _ts(getattr(st, "st_birthtime", st.st_ctime))
        r["mtime"] = _ts(st.st_mtime)
        key = f"{p.stat().st_size}_{int(st.st_mtime)}_{p.name}"
        meta = cache_get(key.replace("/", "_").replace(" ", "_"), "stage0_meta")
        if meta is None:
            with Image.open(p) as im:
                meta = {"width": im.size[0], "height": im.size[1], "mode": im.mode, "sha1": file_sha1(p)}
            cache_put(key.replace("/", "_").replace(" ", "_"), "stage0_meta", meta)
        r.update(meta)
        r["size_mb"] = round(st.st_size / 2**20, 2)

    df = pd.DataFrame(rows)
    df["set_id"] = set_id
    df["file"] = df["path"].map(lambda p: str(p.relative_to(set_dir)).replace("\\", "/"))
    df["path"] = df["path"].map(str)

    # 촬영 시각 선택: 생성일시와 수정일시를 비교
    bq = _time_quality(pd.to_datetime(df["birth_time"]), df["file"])
    mq = _time_quality(pd.to_datetime(df["mtime"]), df["file"])
    pref = cfg.get("time_source", "auto")
    birth_bad = bq["same_second_files"] > 0.5 * bq["n"] or bq["span_min"] < 1.0
    chosen = "mtime" if (pref == "mtime" or birth_bad) else "birth_time"
    df["time_used"] = df[chosen]
    df = df.sort_values(["time_used", "file"]).reset_index(drop=True)
    df["shot_order"] = np.arange(1, len(df) + 1)

    # 접근 방향 : 직전 사진(촬영 순서) 대비 지령값이 커졌으면 +, 작아졌으면 −, 같으면 0
    for ax, col in (("y", "cmd_y_mm"), ("t", "cmd_t_deg")):
        d = df[col].diff()
        df[f"approach_{ax}"] = np.where(d > 1e-9, "+", np.where(d < -1e-9, "-", "0"))
        df.loc[0, f"approach_{ax}"] = "start"

    df["dup_sha1"] = df["sha1"].duplicated(keep=False)
    df.attrs.update(set_id=set_id, unparsed=unparsed, time_choice=chosen, birth_q=bq, mtime_q=mq)
    return df


# ---------------------------------------------------------------- 요약 · 그림
def _range_and_step(v: pd.Series) -> tuple[str, str]:
    u = np.unique(np.round(v.dropna().values, 4))
    if len(u) == 0:
        return "", ""
    steps = np.unique(np.round(np.diff(u), 4)) if len(u) > 1 else np.array([0.0])
    return f"{u.min():+.2f} ~ {u.max():+.2f} ({len(u)}값)", ", ".join(f"{s:g}" for s in steps[:4])


def summarize(df: pd.DataFrame) -> dict:
    cat = df
    s = {"set_id": df.attrs["set_id"], "n_images": len(df)}
    for k in ("reference", "y_sweep", "theta_sweep", "mixed"):
        s[f"n_{k}"] = int((df.kind == k).sum())
    s["y_range_mm"], s["y_steps_mm"] = _range_and_step(df.cmd_y_mm)
    s["t_range_deg"], s["t_steps_deg"] = _range_and_step(df.cmd_t_deg)
    s["resolutions"] = ", ".join(sorted({f"{w}x{h} {m}" for w, h, m in zip(df.width, df.height, df["mode"])}))
    s["duplicates_sha1"] = "; ".join(df.loc[df.dup_sha1, "file"]) or "없음"
    s["unparsed_files"] = "; ".join(df.attrs["unparsed"]) or "없음"
    s["time_source"] = df.attrs["time_choice"]
    s["capture_start"], s["capture_end"] = df.time_used.min(), df.time_used.max()
    bq, mq = df.attrs["birth_q"], df.attrs["mtime_q"]
    s["birth_same_second_files"], s["mtime_same_second_files"] = bq["same_second_files"], mq["same_second_files"]
    s["birth_span_min"], s["mtime_span_min"] = bq["span_min"], mq["span_min"]
    s["mtime_median_gap_s"], s["mtime_max_gap_min"] = mq["median_gap_s"], mq["max_gap_min"]
    if "motor_y_mm" in df:
        m = df.kind != "reference"
        s["motor_vs_name_max_dy_mm"] = float(np.nanmax(np.abs(df.loc[m, "motor_y_mm"] - df.loc[m, "cmd_y_mm"])))
        s["motor_vs_name_max_dt_deg"] = float(np.nanmax(np.abs(df.loc[m, "motor_t_deg"] - df.loc[m, "cmd_t_deg"])))
    for ax in ("y", "t"):
        vc = df[f"approach_{ax}"].value_counts()
        s[f"approach_{ax}_counts"] = ", ".join(f"{k}:{v}" for k, v in vc.items())
    return s


def plot(df: pd.DataFrame, tag: str) -> list[str]:
    plt = setup_matplotlib()
    out = []
    # 지령값 분포
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    if (df.kind == "mixed").any():
        m = df[df.kind != "reference"]
        axs[0].scatter(m.cmd_t_deg, m.cmd_y_mm, s=40, c="tab:green", alpha=.7)
        for (t, y), g in m.groupby(["cmd_t_deg", "cmd_y_mm"]):
            axs[0].annotate(f"{len(g)}", (t, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axs[0].set(xlabel="지령 Θ (°)", ylabel="지령 Y (mm)", title=f"{tag}: (Y, Θ) 조합과 장수")
        axs[1].hist(m.cmd_t_deg, bins=40, color="tab:orange", alpha=.7, label="Θ")
        axs[1].set(xlabel="지령 Θ (°)", ylabel="장수", title="Θ 분포")
    else:
        y = df[df.kind == "y_sweep"].cmd_y_mm
        t = df[df.kind == "theta_sweep"].cmd_t_deg
        axs[0].hist(y, bins=41, color="tab:blue"); axs[0].set(xlabel="지령 Y (mm)", ylabel="장수", title=f"{tag}: Y 스윕 ({len(y)}장)")
        axs[1].hist(t, bins=81, color="tab:orange"); axs[1].set(xlabel="지령 Θ (°)", ylabel="장수", title=f"{tag}: Θ 스윕 ({len(t)}장)")
    fig.tight_layout(); p = FIGURES / f"s0_command_dist_{tag}.png"; fig.savefig(p); plt.close(fig); out.append(str(p))

    # 촬영 순서 타임라인 (접근 방향 색)
    fig, axs = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    col = {"+": "tab:red", "-": "tab:blue", "0": "tab:gray", "start": "k"}
    for ax, c, ap, lab in ((axs[0], "cmd_y_mm", "approach_y", "지령 Y (mm)"), (axs[1], "cmd_t_deg", "approach_t", "지령 Θ (°)")):
        ax.plot(df.shot_order, df[c], color="0.75", lw=0.8, zorder=1)
        for k, g in df.groupby(ap):
            ax.scatter(g.shot_order, g[c], s=14, color=col.get(k, "k"), label=f"접근 {k} ({len(g)})", zorder=2)
        r = df[df.kind == "reference"]
        ax.scatter(r.shot_order, r[c], s=80, facecolors="none", edgecolors="k", label="기준 영상", zorder=3)
        ax.set_ylabel(lab); ax.legend(fontsize=8, ncol=5, loc="best")
    t0 = pd.to_datetime(df.time_used)
    axs[0].set_title(f"{tag}: 촬영 순서별 지령값 ({df.attrs['time_choice']} 기준, {t0.min():%m-%d %H:%M} ~ {t0.max():%H:%M})")
    axs[1].set_xlabel("촬영 순번")
    fig.tight_layout(); p = FIGURES / f"s0_timeline_{tag}.png"; fig.savefig(p); plt.close(fig); out.append(str(p))
    return out


def save(dfs: dict[str, pd.DataFrame]) -> None:
    cols = ["set_id", "file", "kind", "cmd_y_mm", "cmd_t_deg", "shot", "motor_y_mm", "motor_t_deg", "birth_time", "mtime",
            "time_used", "shot_order", "approach_y", "approach_t", "width", "height", "mode", "size_mb", "sha1", "dup_sha1", "path"]
    cat = pd.concat([d.reindex(columns=cols) for d in dfs.values()], ignore_index=True)
    write_csv(cat, TABLES / "catalog.csv")
    write_csv(pd.DataFrame([summarize(d) for d in dfs.values()]), TABLES / "inventory_summary.csv")
