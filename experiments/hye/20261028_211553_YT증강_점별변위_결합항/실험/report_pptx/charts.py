"""증강 보고서 그래프 (EDA 템플릿 그래프 모양에 맞춘다)"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

EXP = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(exist_ok=True)
INK, INK2, MUTED, LINE, BLUE, BG = "#111827", "#4B5563", "#9CA3AF", "#E5E7EB", "#0071E3", "#F7F7F7"
plt.rcParams.update({"font.family": "Noto Sans KR", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.unicode_minus": False})
def save(fig, name): fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor=BG); plt.close(fig)

S = json.load(open(EXP / "results/summary.json", encoding="utf-8"))
S2 = json.load(open(EXP / "results/module_eval_2noise/summary.json", encoding="utf-8"))
rows = list(csv.DictReader(open(EXP / "inputs/manifest.csv", encoding="utf-8-sig")))

# 1. 격자와 실측 자세
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.add_patch(plt.Rectangle((-2, -1), 4, 2, fc="#EAF2FD", ec=BLUE, lw=1.0))
ax.axhline(0, color=INK, lw=1.6); ax.axvline(0, color=INK, lw=1.6)
ax.add_patch(plt.Rectangle((-0.09, -0.09), 0.18, 0.18, fc="none", ec="#0F7B55", lw=1.2))
y14 = [(float(r["Y"]), float(r["T"])) for r in rows if not r["relative"].startswith("20261020")]
mix = sorted({(float(r["Y"]), float(r["T"])) for r in rows if r["relative"].startswith("20261020") and r["scope"] != "center"})
ax.scatter(*zip(*y14), s=6, color=INK2, label="1014 단일축 (281장)", zorder=3)
mi = [p for p in mix if p[1] <= 1]; mo = [p for p in mix if p[1] > 1]
ax.scatter(*zip(*mi), s=22, color=BLUE, label="1020 혼합 T ≤ 1 (15자세 × 3장)", zorder=4)
ax.scatter(*zip(*mo), s=22, facecolor="none", edgecolor=BLUE, label="1020 혼합 T = 2 (범위 밖, 4자세)", zorder=4)
ax.set_xlim(-2.3, 2.3); ax.set_ylim(-1.2, 2.25); ax.set_xlabel("Y (mm)"); ax.set_ylabel("T")
ax.text(-1.95, -0.93, "증강 격자 81 × 201 = 16,281점", color=BLUE, fontsize=8, weight="bold")
ax.legend(loc="upper left", fontsize=7, frameon=False)
ax.set_title("증강 격자와 실측 자세 (실측 혼합은 Y > 0, T > 0 한 사분면뿐)", loc="left")
save(fig, "c1_grid.png")

# 2. 반응식 교차검증
fam = ["Y1_T1", "Y1_T1s", "Y2_T2s", "Y3_T2s", "Y3_T3", "Y3_T3s"]
cv = S["shape_cv_1014"]
fig, ax = plt.subplots(figsize=(6.0, 3.9))
x = np.arange(len(fam)); w = 0.38
ax.bar(x - w / 2, [cv[f]["rms_Y"] for f in fam], w, color=MUTED, label="Y trial")
ax.bar(x + w / 2, [cv[f]["rms_T"] for f in fam], w, color=[BLUE if f == "Y2_T2s" else "#8DB9F0" for f in fam], label="T trial")
for i, f in enumerate(fam): ax.text(i + w / 2, cv[f]["rms_T"] + 0.0015, f"{cv[f]['rms_T']:.3f}", ha="center", fontsize=7, color=INK)
ax.axhline(S["noise_floor"]["rms"], color=INK, lw=0.9, ls="--"); ax.text(5.45, S["noise_floor"]["rms"] - 0.006, "1020 반복 잡음 0.120", ha="right", fontsize=7)
ax.set_xticks(x, ["Y 1차\nT 1차", "Y 1차\nT ± 따로", "Y 2차\nT ± 2차 (선택)", "Y 3차\nT ± 2차", "Y 3차\nT 3차", "Y 3차\nT ± 3차"], fontsize=7)
ax.set_ylim(0, 0.135); ax.set_ylabel("잔차 RMS (px, 선택 536점)"); ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_title("1014 단일축 반응식: trial 안 4겹 교차검증 잔차", loc="left")
save(fig, "c2_shape.png")

# 3. 영점 지도
orows = list(csv.DictReader(open(EXP / "results/origins.csv", encoding="utf-8-sig")))
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.axhline(0, color=INK, lw=0.8); ax.axvline(0, color=INK, lw=0.8)
col = {"실측 센터 사진": INK, "실측 trial 영점": BLUE, "실측 혼합 세션 영점": "#7C3AED"}
for r in orows:
    dx, dy = float(r["mean_dx_px"]), float(r["mean_dy_px"]); c = col.get(r["kind"], MUTED)
    ax.scatter(dx, dy, s=34, color=c, zorder=3, edgecolor="white", lw=0.6)
    ax.annotate(r["name"], (dx, dy), xytext=(4, 3), textcoords="offset points", fontsize=7, color=c if c != MUTED else INK2)
bp = np.mean([[float(r["mean_dx_px"]), float(r["mean_dy_px"])] for r in orows if r["name"] in ("S1", "S3", "S5")], 0)
ax.annotate("", xy=(-0.61, 0.0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#B45309", lw=1.4))
ax.annotate("", xy=(-0.17, -0.09), xytext=(0, -0.09), arrowprops=dict(arrowstyle="->", color="#B45309", lw=1.4))
ax.text(-1.02, -0.07, "+접근 백래시 −0.61", color="#B45309", fontsize=7); ax.text(-0.17, -0.16, "−접근 백래시 −0.17", color="#B45309", fontsize=7)
for lab, c in (("센터 사진", INK), ("trial 영점", BLUE), ("혼합 세션", "#7C3AED"), ("추가 후보 (주성분 ±2σ)", MUTED)):
    ax.scatter([], [], color=c, s=30, label=lab)
ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
ax.set_xlabel("판 평균 가로 이동 dx (px)"); ax.set_ylabel("판 평균 세로 이동 dy (px)")
ax.set_xlim(-1.2, 0.55); ax.set_ylim(-0.4, 0.75)
ax.set_title("영점 15개: 1014 센터 기준 판 평균 이동 (선택 536점)", loc="left")
save(fig, "c3_origins.png")

# 4. 결합항 검증 (1020 혼합, 자세 하나 빼기)
lab = ["1020 센터\n결합항 없음", "1020 센터\nYT", "세션 영점\n결합항 없음", "세션 영점\nYT", "세션 영점\nYT + YT² + Y²T"]
val = [0.486, 0.499, 0.135, 0.150, 0.201]
fig, ax = plt.subplots(figsize=(6.0, 3.9))
b = ax.bar(range(5), val, color=[MUTED, MUTED, BLUE, "#8DB9F0", "#8DB9F0"], width=0.6)
for i, v in enumerate(val): ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5, color=INK, weight="bold" if i == 2 else "normal")
ax.axhline(0.120, color=INK, lw=0.9, ls="--"); ax.text(4.4, 0.127, "반복 잡음 0.120", ha="right", fontsize=7)
ax.set_xticks(range(5), lab, fontsize=7); ax.set_ylim(0, 0.56); ax.set_ylabel("잔차 RMS (px, 선택 536점)")
ax.set_title("1020 혼합 T ≤ 1, 45장: 자세 하나 빼고 예측한 잔차", loc="left")
save(fig, "c4_coupling.png")

# 5. 반복 잡음 두 가지
nz = S2["noise_control_px"]
fig, ax = plt.subplots(figsize=(6.0, 3.9))
labs = ["1020 반복\n(T = 2 포함, 이전)", "1020 반복\n(T ≤ 1)", "1014 연속\n(반응식 잔차)"]
xs = [0.513, nz["1020반복"]["rms"], nz["1014연속"]["rms"]]; pdx = [0.174, nz["1020반복"]["plate_dx_sd"], nz["1014연속"]["plate_dx_sd"]]
x = np.arange(3); w = 0.36
ax.bar(x - w / 2, xs, w, color=[MUTED, BLUE, "#0F7B55"], label="점별 RMS")
ax.bar(x + w / 2, pdx, w, color=[MUTED, BLUE, "#0F7B55"], alpha=0.45, label="판 전체 dx 표준편차")
for i in range(3):
    ax.text(i - w / 2, xs[i] + 0.01, f"{xs[i]:.3f}", ha="center", fontsize=7.5); ax.text(i + w / 2, pdx[i] + 0.01, f"{pdx[i]:.3f}", ha="center", fontsize=7.5)
ax.set_xticks(x, labs, fontsize=7.5); ax.set_ylim(0, 0.6); ax.set_ylabel("px (control 측정, 선택 536점)")
ax.legend(frameon=False, fontsize=7)
ax.set_title("반복 촬영 잡음: control 측정 공간", loc="left")
save(fig, "c5_noise.png")

# 6. 쌍둥이 산점도 (joint Y 오차)
tw = list(csv.DictReader(open(EXP / "results/module_eval_2noise/twin_compare.csv", encoding="utf-8-sig")))
fig, ax = plt.subplots(figsize=(6.0, 4.0))
lim = 0.12
ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
ax.add_patch(plt.Rectangle((-0.09, -0.09), 0.18, 0.18, fc="none", ec="#0F7B55", lw=1.0))
for st, c, lb in (("1014 단일축", INK2, "1014 단일축 280장"), ("1020 혼합", BLUE, "1020 혼합 T ≤ 1, 45장")):
    L = [r for r in tw if r["set"] == st]
    ax.scatter([float(r["joint_real_y"]) - float(r["Y"]) for r in L], [float(r["joint_twin_y"]) - float(r["Y"]) for r in L], s=9, color=c, alpha=0.75, label=lb)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("실측 사진의 joint Y 오차 (mm)"); ax.set_ylabel("증강 쌍둥이의 joint Y 오차 (mm)")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_title("실측 사진 vs 증강 쌍둥이 (잡음 없는 쌍둥이)", loc="left")
save(fig, "c6_twin.png")

# 7. 오차 지도 (joint 통과율) 두 잡음
Ys, Ts = np.round(np.linspace(-2, 2, 81), 2), np.round(np.linspace(-1, 1, 201), 2)
YY, TT = np.meshgrid(Ys, Ts, indexing="ij"); Yf, Tf = YY.ravel(), TT.ravel()
cmap = LinearSegmentedColormap.from_list("p", [(0, "#C62F3A"), (0.8, "#F4C478"), (0.95, "#D6ECDE"), (1, "#0F7B55")])
for nzk, fn, ttl in (("1020반복", "c7_map_1020.png", "잡음 1020 반복 (T ≤ 1)"), ("1014연속", "c7_map_1014.png", "잡음 1014 연속")):
    r = np.load(EXP / f"results/module_eval_2noise/grid_predictions_{nzk}.npz")
    ok = ((np.abs(r["jy"] - Yf) <= 0.09) & (np.abs(r["jt"] - Tf) <= 0.09)).mean((0, 1)).reshape(81, 201)
    fig, ax = plt.subplots(figsize=(5.83, 2.45))
    im = ax.imshow(ok.T, origin="lower", extent=(-2.025, 2.025, -1.005, 1.005), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.grid(False); ax.axhline(0, color=INK, lw=1.0); ax.axvline(0, color=INK, lw=1.0)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="none", ec=INK, lw=1.0, ls="--"))
    ax.plot([-2, 2], [2 / 5.68, -2 / 5.68], color="#C62F3A", lw=0.8, ls=":")
    ax.set_xlabel("Y (mm)"); ax.set_ylabel("T")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 0.8, 0.95, 1]); cb.set_ticklabels(["0", "50", "80", "95", "100%"])
    ax.set_title(f"joint 두 축 0.09 통과율, {ttl}", loc="left")
    save(fig, fn)

# 8. control Y = 0 줄 통과율 (영점별)
names = [str(n) for n in np.load(EXP / "results/module_eval_2noise/grid_predictions_1014연속.npz")["names"]]
fig, ax = plt.subplots(figsize=(6.0, 3.9))
x = np.arange(len(names)); w = 0.38
for k, (nzk, c) in enumerate((("1020반복", BLUE), ("1014연속", "#0F7B55"))):
    r = np.load(EXP / f"results/module_eval_2noise/grid_predictions_{nzk}.npz")
    okc = (np.abs(r["cy"] - Yf) <= 0.09) & (np.abs(r["ct"] - Tf) <= 0.09)
    v = [okc[o][:, Yf == 0].mean() * 100 for o in range(len(names))]
    ax.bar(x + (k - 0.5) * w, v, w, color=c, label=f"잡음 {'1020 반복' if k == 0 else '1014 연속'}")
ax.set_xticks(x, names, rotation=45, ha="right", fontsize=7); ax.set_ylim(0, 115); ax.set_ylabel("통과율 (%)")
ax.axhline(95, color=MUTED, lw=0.8, ls="--")
ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
ax.set_title("control, Y = 0 줄 (T만 움직인 자세) 통과율: 영점별", loc="left")
save(fig, "c8_control.png")
print("charts ok")
