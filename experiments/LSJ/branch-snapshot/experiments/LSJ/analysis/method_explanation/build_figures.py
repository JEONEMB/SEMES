"""Explanatory figures from an existing same-label QR pair; no acceptance test.

Intermediate ECC display poses are illustrative, not an optimizer trace.
The sine waves illustrate the shift theorem, not a measured QR spectrum.
"""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 16,
                     "axes.unicode_minus": False, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
INK, A_COLOR, B_COLOR = "#172c3b", "#b62239", "#007d96"


def image_read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)


def load_pair():
    inventory = json.loads((BASE / "second_data_inspection/inventory.json").read_text(encoding="utf-8"))
    results = json.loads((BASE / "second_repeatability/repeatability_results.json").read_text(encoding="utf-8"))
    group = next(g for g in results["groups"] if g["label_y_mm"] == .1 and g["label_theta_degree"] == .1)
    root = Path(inventory["summary"]["extracted_root"])
    x0, y0, x1, y1 = group["rois"]["C"]["xyxy"]
    a, b = [image_read(root / f["file"])[y0:y1, x0:x1] for f in group["files"][:2]]
    return a, b, group["pairs"][0]["regions"]["C"]


def build_ecc(a, b, measurement):
    ecc = measurement["ecc_translation"]
    dx, dy = ecc["dx"], ecc["dy"]
    fig = plt.figure(figsize=(12, 9.6))
    gs = fig.add_gridspec(2, 6, left=.05, right=.95, bottom=.16, top=.83,
                          hspace=.62, wspace=.70, height_ratios=[1, 1])
    fig.text(.05, .96, "평행이동 ECC: 무늬를 겹쳐 보며 위치를 조정", fontsize=23, weight="bold", color=INK)
    fig.text(.05, .91, "실제 데이터 · Y=0.10mm, θ=0.10° · 가운데 QR · 사진 1 ↔ 사진 2", fontsize=16)
    zoom = (24, 24, 36, 36)
    for ax, im, name in [(fig.add_subplot(gs[0, :3]), a, "A: 사진 1"),
                         (fig.add_subplot(gs[0, 3:]), b, "B: 사진 2")]:
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.add_patch(Rectangle(zoom[:2], zoom[2], zoom[3], fill=False, edgecolor="#f29e25", lw=2))
        ax.set_title(name, fontsize=18, pad=8)
        ax.axis("off")
    fig.text(.5, .505, "표시 영역 확대 · 빨강=A / 청록=B · 두 밝기가 같으면 회색", ha="center", fontsize=16)
    yy, xx = np.indices(a.shape, dtype=np.float32)
    poses = [(0., 0., "시작", "B 위치 조정 없음"),
             (.30, 0., "중간 위치 예시", "B를 왼쪽 0.30px로 표시"),
             (dx, dy, "ECC가 찾은 위치", "B를 왼쪽 0.580px로 정렬")]
    zx, zy, zw, zh = zoom
    for i, (tx, ty, title, detail) in enumerate(poses):
        ax = fig.add_subplot(gs[1, i*2:i*2+2])
        aligned = cv2.remap(b.astype(np.float32), xx+tx, yy+ty, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        rgb = np.stack([a/255., aligned/255., aligned/255.], axis=-1)
        ax.imshow(rgb[zy:zy+zh, zx:zx+zw], interpolation="nearest")
        ax.set_title(title, fontsize=17, pad=8)
        ax.set_xlabel(detail, fontsize=13, labelpad=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(.05, .095, f"반환 변위: Δx = +{dx:.3f}px, Δy = {dy:+.3f}px  |  정합 후 ECC 점수 = {ecc['ecc']:.5f}", fontsize=17, color=INK)
    fig.text(.05, .045, "중간 위치는 설명용이며 실제 반복 계산 기록이 아닙니다. 정렬 방향은 반환 변위의 반대입니다.", fontsize=13)
    fig.savefig(HERE / "ecc-explanation.png", dpi=140)
    plt.close(fig)


def phase_maps(a, b):
    a = cv2.GaussianBlur(a.astype(np.float64), (5, 5), 0)
    b = cv2.GaussianBlur(b.astype(np.float64), (5, 5), 0)
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = [np.fft.fft2((im-im.mean())*window) for im in (a, b)]
    c = fb * np.conj(fa)
    amp = np.abs(c)
    c = np.divide(c, amp, out=np.zeros_like(c), where=amp > amp.max()*1e-12)
    response = np.fft.fftshift(np.fft.ifft2(c).real)
    cy, cx = np.array(response.shape)//2
    coarse = response[cy-3:cy+4, cx-3:cx+4]
    xs, ys = np.linspace(-.25, 1.25, 151), np.linspace(-.75, .75, 151)
    ey = np.exp(2j*np.pi*np.outer(ys, np.fft.fftfreq(a.shape[0])))
    ex = np.exp(2j*np.pi*np.outer(np.fft.fftfreq(a.shape[1]), xs))
    fine = (ey @ c @ ex).real / a.size
    return coarse, fine, xs, ys


def build_phase(a, b, measurement):
    phase = measurement["phase_correlation"]
    dx, dy = phase["dx"], phase["dy"]
    coarse, fine, xs, ys = phase_maps(a, b)
    fig = plt.figure(figsize=(12, 10.2))
    gs = fig.add_gridspec(2, 2, left=.10, right=.94, bottom=.23, top=.78,
                          height_ratios=[.72, 1.15], hspace=.65, wspace=.45)
    fig.text(.05, .96, "위상 상관: 파동 성분의 위상 차이로 변위를 찾음", fontsize=23, weight="bold", color=INK)
    fig.text(.05, .91, "A·B 영상 → FFT → 여러 성분의 위상 차이 결합 → 역변환 → 최고점 위치", fontsize=16)
    fig.text(.05, .86, "위 그림: 개념 설명용 파동 / 아래 두 맵: 같은 실제 QR 쌍으로 계산", fontsize=15)
    ax = fig.add_subplot(gs[0, :])
    xx = np.linspace(0, 12, 500)
    ax.plot(xx, np.cos(2*np.pi*(xx-3)/8), color=A_COLOR, lw=2.5, label="A의 파동 성분 (예시)")
    ax.plot(xx, np.cos(2*np.pi*(xx-3-dx)/8), color=B_COLOR, lw=2.5, ls="--", label="위치가 달라진 B 성분 (예시)")
    ax.set_xlim(0, 12); ax.set_ylim(-1.15, 1.55)
    ax.set_xticks([0, 3, 6, 9, 12]); ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("가로 위치 (px)", fontsize=14)
    ax.set_ylabel("파동값\n(설명용)", fontsize=14)
    ax.annotate("같은 주기, 다른 위상", xy=(3.27, 1.), xytext=(5.2, 1.30),
                arrowprops={"arrowstyle": "->", "color": INK}, fontsize=15)
    ax.legend(loc="lower right", fontsize=12, framealpha=.95)
    ax.grid(alpha=.2)
    for axis in ax.spines.values(): axis.set_color("#9ba6ad")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(coarse, origin="lower", extent=[-3.5, 3.5, -3.5, 3.5], cmap="magma", interpolation="nearest")
    ax.set_xticks([-3, -1, 0, 1, 3]); ax.set_yticks([-3, -1, 0, 1, 3])
    iy, ix = np.unravel_index(coarse.argmax(), coarse.shape)
    px, py = ix-3, iy-3
    assert (px, py) == (1, 0), (px, py)
    ax.plot(px, py, marker="+", ms=18, mew=2, color="#4edbff")
    ax.annotate("최고점 (1, 0)", xy=(px, py), xytext=(-2.9, 2.3), color="white", fontsize=14,
                arrowprops={"arrowstyle": "->", "color": "white"})
    ax.set_title("① 정수 픽셀 단위로 찾기", fontsize=17, pad=12)
    ax.set_xlabel("가로 변위 Δx (px)", fontsize=14)
    ax.set_ylabel("세로 변위 Δy (px)", fontsize=14)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(fine, origin="lower", extent=[xs[0],xs[-1],ys[0],ys[-1]], cmap="magma", interpolation="nearest", aspect="equal")
    ax.plot(dx, dy, marker="+", ms=18, mew=2, color="#4edbff")
    ax.annotate(f"최고점\n({dx:.3f}, {dy:.3f})", xy=(dx,dy), xytext=(-.17,.39), color="white", fontsize=14,
                arrowprops={"arrowstyle": "->", "color": "white"})
    ax.set_xticks([0,.5,1]); ax.set_yticks([-.5,0,.5])
    ax.set_title("② 주변을 소수점 단위로 계산", fontsize=17, pad=12)
    ax.set_xlabel("가로 변위 Δx (px)", fontsize=14)
    ax.set_ylabel("세로 변위 Δy (px)", fontsize=14)
    fig.text(.05, .105, f"반환 변위: Δx = +{dx:.3f}px, Δy = {dy:+.3f}px", fontsize=18, color=INK)
    fig.text(.05, .060, "맵의 한 칸 = 하나의 이동 후보에 대한 점수 · 밝을수록 높음 · QR 사진의 픽셀 위치가 아님", fontsize=13)
    fig.text(.05, .025, "위 파동은 개념도입니다. 실제 계산은 가로·세로의 여러 주파수 성분을 함께 사용합니다.", fontsize=13)
    fig.savefig(HERE / "phase-explanation.png", dpi=140)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a, b, measurement = load_pair()
    build_ecc(a,b,measurement)
    build_phase(a,b,measurement)
    (HERE / "figure_provenance.json").write_text(json.dumps({
        "condition": {"label_y_mm": .1, "label_theta_degree": .1},
        "pair": "photos 1 and 2", "marker": "C", "existing_measurements": measurement,
        "ecc_display": "0, illustrative +0.30 px sampling offset, stored final ECC dx/dy; not iteration history",
        "wave_display": "Illustrative sine waves only, not actual QR Fourier components",
        "phase_maps": "Recomputed from original pair with existing analysis preprocessing",
        "scope": "Method explanation; no new acceptance threshold or physical motion estimate"
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print("Created ecc-explanation.png, phase-explanation.png, figure_provenance.json")


if __name__ == "__main__":
    main()
