# -*- coding: utf-8 -*-
"""Stage 5 — 누적 요약(summary_all_sets.csv) + 그림 위주 PDF 보고서(EDA_보고서.pdf).

수치는 모두 eda_results 의 표와 run_log.json 에서 읽어 자동으로 채운다 (새 세트가 들어오면 다시 실행).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from common import CENTER, FIGURES, RESULTS, TABLES, TOL_T_DEG, TOL_Y_UM, VIDEOS, rel, setup_matplotlib, write_csv
from figure_guides import GUIDES, INTRO, wrap


def summary_all_sets() -> pd.DataFrame:
    cat = pd.read_csv(TABLES / "catalog.csv", encoding="utf-8-sig")
    q = pd.read_csv(TABLES / "quality.csv", encoding="utf-8-sig")
    f = pd.read_csv(TABLES / "features.csv", encoding="utf-8-sig")
    C = pd.read_csv(CENTER / "center_by_set.csv", encoding="utf-8-sig") if (CENTER / "center_by_set.csv").exists() else pd.DataFrame()
    L = pd.read_csv(TABLES / "label_check.csv", encoding="utf-8-sig") if (TABLES / "label_check.csv").exists() else pd.DataFrame()
    rows = []
    for sid, g in cat.groupby("set_id", sort=False):
        qq, ff = q[q.set_id == sid], f[f.set_id == sid]
        r = {"set_id": sid, "n_images": len(g), "kinds": ", ".join(f"{k}:{v}" for k, v in g.kind.value_counts().items()),
             "capture_start": g.time_used.min(), "capture_end": g.time_used.max(),
             "cmd_y_range_mm": f"{g.cmd_y_mm.min():+.2f}~{g.cmd_y_mm.max():+.2f}", "cmd_t_range_deg": f"{g.cmd_t_deg.min():+.2f}~{g.cmd_t_deg.max():+.2f}",
             "img_mean_median": qq.img_mean.median(), "qrC_focus_median": qq.qrC_focus.median(),
             "noise_sigma_median": qq.img_noise_sigma.median(), "n_quality_warning": int((qq.quality_warning.fillna("") != "").sum()),
             "corner_ok": int(ff.corners_ok.sum())}
        if len(C) and sid in set(C.set_id):
            c = C[C.set_id == sid].iloc[0]
            r.update({"origin_cx_px": c.origin_cx_px, "origin_cy_px": c.origin_cy_px,
                      "origin_vs_prev_Y_um": c.get("vs_prev_as_Y_um", np.nan), "origin_vs_prev_scale_ppm": c.get("vs_prev_scale_ppm", np.nan)})
        if len(L):
            l = L[(L.set_id == sid) & L.dY_um.notna() & (L.kind != "reference")]
            if len(l):
                r.update({"label_vs_ref_mean_Y_um": l.dY_um.mean(), "label_vs_ref_rms_Y_um": float(np.sqrt((l.dY_um ** 2).mean())),
                          "label_vs_ref_mean_T_deg": l.dT_deg.mean(), "label_vs_ref_rms_T_deg": float(np.sqrt((l.dT_deg ** 2).mean()))})
        rows.append(r)
    S = pd.DataFrame(rows)
    write_csv(S, RESULTS / "summary_all_sets.csv")
    return S


def _static_noise() -> float:
    p = TABLES / "noise_repeats.csv"
    if not p.exists():
        return float("nan")
    n = pd.read_csv(p, encoding="utf-8-sig")
    st = n[(n.shift_dx_px.abs() < 0.03) & (n.shift_dy_px.abs() < 0.03)]
    return float(st.noise_wall_sigma.median()) if len(st) else float("nan")


def _key_lines(log: dict) -> list[str]:
    s2, s3, s4 = log.get("stage2", {}), log.get("stage3", {}), log.get("stage4", {})
    sep = pd.read_csv(TABLES / "separability.csv", encoding="utf-8-sig")
    top = sep.iloc[0]
    inv = pd.read_csv(TABLES / "inventory_summary.csv", encoding="utf-8-sig")
    q = pd.read_csv(TABLES / "quality.csv", encoding="utf-8-sig")
    unc = pd.read_csv(TABLES / "label_uncertainty.csv", encoding="utf-8-sig") if (TABLES / "label_uncertainty.csv").exists() else None
    lines = [
        f"Stage 0  {int(inv.n_images.sum())}장 해석, 중복 {sum(inv.duplicates_sha1 != '없음')}건, 촬영 시각 = 수정일시(생성일시는 복사 시각)",
        f"Stage 1  화질 경고 {int((q.quality_warning.fillna('') != '').sum())}장, 단일 영상 잡음 σ ~ {q.img_noise_sigma.median():.2f} (Immerkær), "
        f"연속 촬영 차영상 σ ~ {_static_noise():.2f} (밝기 단위)",
        f"Stage 2  최상위 단서 {top.feature}: Θ 분해능 {top.theta_res_static_deg:.4f}°(측정 잡음) / {top.theta_res_repeat_deg:.4f}°(재위치) / {top.theta_res_fit_deg:.4f}°(적합 잔차)",
    ]
    if s2.get("combined_top"):
        lines.append(f"         여러 특징 결합(대각 잡음, 낙관): Θ {s2['combined_top']['sigma_T_deg']:.4f}° (측정 잡음) / {s2['combined_top_repeat_noise']['sigma_T_deg']:.4f}° (재위치)")
    if (TABLES / "superposition.csv").exists():
        S = pd.read_csv(TABLES / "superposition.csv", encoding="utf-8-sig")
        w = S[S.feature == "w_diff_RL"].iloc[0]; c = S[S.feature == "curv"].iloc[0]
        # 2차 세트는 Θ ≥ 0 뿐이라 Θ·Θ² 계수가 섞임 → 참고값. 같은 선형 모델 비교는 보고서 4.2절
        lines.append(f"Stage 3  (참고, Θ² 포함 모델) 폭 차 Θ 감도 {w.set1_aT:.3f}→{w.set2_aT:.3f} px/°, 곡률 {c.set1_aT:.2f}→{c.set2_aT:.2f} px/° "
                     f"— 2차 세트 적합 불안정, 장비 조정 없음")
    if unc is not None:
        same = unc[unc.component.str.contains("통일")].iloc[0]; sess = unc[unc.component.str.contains("세션 간")].iloc[0]
        lines.append(f"Stage 4  (참고, 장비 오차라 평가 제외) 재위치 흩어짐 Θ {same.T_deg:.4f}° ({same.T_pct_of_tol:.0f} %), Y {same.Y_um:.0f} µm ({same.Y_pct_of_tol:.0f} %); "
                     f"세션 간 계통 Θ {sess.T_deg:+.4f}°, Y {sess.Y_um:+.0f} µm")
    return lines


def _text_page(pdf, plt, title: str, lines: list[str], size=13, png: Path | None = None):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.05, 0.93, title, fontsize=20, weight="bold")
    y = 0.85
    for l in lines:
        fig.text(0.05, y, l, fontsize=size, va="top", wrap=True)
        y -= 0.055 if size >= 13 else 0.045
    pdf.savefig(fig)
    if png is not None:
        fig.savefig(png, dpi=80)                 # 첫 장 미리보기 (PDF 뷰어 없이 확인용)
    plt.close(fig)


def _intro_page(pdf, plt):
    """그림 페이지 앞에 기호·특징 이름을 한 번 정리한다 (각 페이지 '보는 법'이 이를 전제로 함)."""
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.05, 0.93, "이 PDF 보는 법", fontsize=20, weight="bold")
    y = 0.86
    for head, items in INTRO:
        fig.text(0.05, y, head, fontsize=13, weight="bold", va="top"); y -= 0.04
        for it in items:
            for l in wrap("• " + it, 150):
                fig.text(0.07, y, l, fontsize=11, va="top"); y -= 0.032
        y -= 0.015
    pdf.savefig(fig); plt.close(fig)


def _img_page(pdf, plt, title: str, paths: list[Path], note: str = "", guide: str | None = None):
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        return
    n = len(paths)
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle(title, fontsize=18, weight="bold", x=0.05, ha="left", y=0.97)
    img_w = 0.64 if guide else 0.94           # 보는 법이 있으면 오른쪽 30 % 를 설명 칸으로 씀
    for i, p in enumerate(paths):
        ax = fig.add_axes([0.02, 0.05 + (n - 1 - i) * 0.88 / n, img_w, 0.86 / n])
        ax.imshow(plt.imread(str(p))); ax.axis("off")
    if guide:
        x0, y = 0.685, 0.90
        fig.text(x0, y, "보는 법", fontsize=13, weight="bold", va="top"); y -= 0.045
        for item in GUIDES[guide]:
            for l in wrap(item, 46):
                fig.text(x0, y, l, fontsize=9.5, va="top"); y -= 0.024
            y -= 0.010
    if note:
        fig.text(0.05, 0.015, note, fontsize=11)
    pdf.savefig(fig); plt.close(fig)


def _video_frames(paths: list[Path], times=(0.3, 0.6)) -> list[Path]:
    out = []
    for v in paths:
        if not v.exists():
            continue
        for t in times:
            fp = FIGURES / f"_frame_{v.stem}_{int(t * 100)}.png"
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)],
                                       capture_output=True, text=True).stdout.strip() or 0)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{dur * t:.2f}", "-i", str(v), "-frames:v", "1", str(fp)])
            out.append(fp)
    return out


def run_all(sets, no_video=False) -> dict:
    from matplotlib.backends.backend_pdf import PdfPages
    plt = setup_matplotlib()
    log = json.loads((RESULTS / "run_log.json").read_text(encoding="utf-8"))
    S = summary_all_sets()
    q = pd.read_csv(TABLES / "quality.csv", encoding="utf-8-sig")
    warn = q[q.quality_warning.fillna("") != ""]
    pdf_path = RESULTS / "EDA_보고서.pdf"
    tags = list(S.set_id)
    with PdfPages(pdf_path) as pdf:
        lines = _key_lines(log)
        lines += ["", "재촬영 필요 사진: " + ("없음" if warn.empty else ", ".join(warn.file.head(10)))]
        lines += [f"세트: " + ", ".join(f"{r.set_id} {r.n_images}장 ({r.capture_start[:10]})" for r in S.itertuples())]
        lines += [f"판정 기준: Θ ±{TOL_T_DEG}°, Y ±{TOL_Y_UM:.0f} µm (사진마다).  상세 해석은 EDA_보고서.md"]
        _text_page(pdf, plt, "QR Y·Θ 분리 데이터 EDA — 단계별 핵심 결론", lines, size=12, png=RESULTS / "EDA_보고서_요약.png")
        _intro_page(pdf, plt)
        _img_page(pdf, plt, "Stage 0  촬영 순서와 접근 방향", [FIGURES / f"s0_timeline_{t}.png" for t in tags], guide="s0_timeline")
        _img_page(pdf, plt, "Stage 1  화질 추이", [FIGURES / f"s1_quality_trend_{tags[0]}.png", FIGURES / "s1_quality_trend_compare.png"], guide="s1_trend")
        noise = [p for t in tags for p in (FIGURES / f"s1_noise_map_{t}.png", FIGURES / f"s1_reflection_{t}.png") if p.exists()]
        _img_page(pdf, plt, "Stage 1  촬영 잡음과 반사", noise[:3], guide="s1_noise")
        _img_page(pdf, plt, "Stage 2  Y·Θ 분리도 순위", [FIGURES / "s2_separability_rank.png"], guide="s2_sep")
        _img_page(pdf, plt, "Stage 2  특징 vs 지령값 (같은 화면 이동 기준)", [FIGURES / "s2_feature_vs_cmd_1.png"], guide="s2_feat")
        _img_page(pdf, plt, "Stage 2  변위장 — Θ 는 U 자(원근), Y 는 평평", [FIGURES / "s2_flow_profile.png", FIGURES / "s2_paired_diff_Y+2mm.png"], guide="s2_flow")
        _img_page(pdf, plt, "Stage 2  패치별 변위장", [FIGURES / "s2_flow_compare.png"], guide="s2_flow_map")
        for t in tags[1:]:
            _img_page(pdf, plt, f"Stage 3  {t}: 1차 모델로 예측한 잔차", [FIGURES / f"s3_superposition_residuals_{t}.png"], guide="s3_resid")
            _img_page(pdf, plt, f"Stage 3  {t}: 범위", [FIGURES / f"s3_out_of_range_{t}.png"], guide="s3_range")
        _img_page(pdf, plt, "Stage 3  세트별 원점 (C-2)", [FIGURES / "s3_center_compare.png"], guide="s3_center")
        _img_page(pdf, plt, "Stage 4  접근 방향과 재이동 (참고, 평가 제외)", [FIGURES / "s4_backlash.png", FIGURES / "s4_reapproach.png"], guide="s4_dev")
        _img_page(pdf, plt, "Stage 4  장비 재위치 흩어짐 (참고, 평가 제외)", [FIGURES / "s4_label_uncertainty.png"], guide="s4_unc")
        vids = sorted(VIDEOS.glob("*_sync*.mp4"))
        vids = [v for v in vids if "realtime" not in v.name]
        frames = _video_frames(vids[:3], times=(0.5,))
        _img_page(pdf, plt, "영상 대표 프레임 (A-2 동기화판)", frames[:3], guide="frames",
                  note=f"영상 파일: {rel(VIDEOS)}/ 의 MP4 {len(list(VIDEOS.glob('*.mp4')))}개 "
                       "(_shooting = 촬영 과정, _quality = 화질 경고판, _sync = 동기화판, _realtime = 실제 시간 비율)")
    return {"pdf": str(pdf_path), "summary": str(RESULTS / "summary_all_sets.csv"), "pdf_mb": round(pdf_path.stat().st_size / 2**20, 1)}
