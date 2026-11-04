# -*- coding: utf-8 -*-
"""촬영 과정 영상 — A-1 기본판(+ 화질 경고), A-2 동기화판.

출력: GIF(공유용, 가로 960 px) + MP4(H.264, 균일 간격) + MP4(실제 촬영 시간 비율).
"""
from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from common import VIDEOS, read_gray, setup_matplotlib

FRAME_W, FRAME_H = 1600, 900
IMG_W, IMG_H = 1120, 630               # 왼쪽 사진 (16:9 축소)
PANEL_X = IMG_W + 10


def _font_img(text: str, size=22, color=(255, 255, 255), bg=(30, 30, 30), w=FRAME_W, h=40):
    """한글 글자를 PIL 로 그려 BGR 배열로 반환."""
    from PIL import ImageDraw, ImageFont
    im = Image.new("RGB", (w, h), bg[::-1])
    try:
        f = ImageFont.truetype("malgun.ttf", size)
    except OSError:
        f = ImageFont.load_default()
    ImageDraw.Draw(im).text((10, (h - size) // 2 - 2), text, font=f, fill=color[::-1])
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _timeline_background(df: pd.DataFrame, cols: list[tuple[str, str]], w: int, h: int):
    """오른쪽 타임라인 배경을 matplotlib 으로 그리고, 데이터→픽셀 변환 함수를 함께 반환."""
    plt = setup_matplotlib()
    fig, axs = plt.subplots(len(cols), 1, figsize=(w / 100, h / 100), dpi=100, sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (c, lab) in zip(axs, cols):
        ax.plot(df.shot_order, df[c], color="0.7", lw=0.8)
        ax.scatter(df.shot_order, df[c], s=6, c="0.4")
        ax.set_ylabel(lab, fontsize=9); ax.tick_params(labelsize=8)
    axs[-1].set_xlabel("촬영 순번", fontsize=9)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    bg = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(), cv2.COLOR_RGB2BGR)
    trans = []
    for ax in axs:
        T = ax.transData
        trans.append(lambda x, y, T=T: (int(T.transform((x, y))[0]), int(h - T.transform((x, y))[1])))
    plt.close(fig)
    return bg, trans


def _load_small(path: str, inset_rect):
    g = read_gray(path)
    small = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    ins = None
    if inset_rect is not None:
        x0, y0, x1, y1 = inset_rect
        ins = g[y0:y1, x0:x1].copy()
    return small, ins


def _write_mp4(frames: list[np.ndarray], path: Path, fps: float, repeats: list[int] | None = None) -> bool:
    ff = shutil.which("ffmpeg")
    if ff is None:
        return False
    h, w = frames[0].shape[:2]
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        for _ in range(repeats[i] if repeats else 1):
            pr.stdin.write(f.tobytes())
    pr.stdin.close()
    return pr.wait() == 0


def _write_gif(frames: list[np.ndarray], path: Path, ms: int, width: int = 960) -> float:
    ims = []
    for f in frames:
        h, w = f.shape[:2]
        f2 = cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ims.append(Image.fromarray(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)).quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)
    return path.stat().st_size / 2**20


def shooting_video(df: pd.DataFrame, tag: str, inset_rect=None, warn_col: str | None = None,
                   overlay=None, graph=None, suffix: str = "shooting", fps: float = 4.0) -> dict:
    """df : 촬영 순서로 정렬된 catalog (path, shot_order, cmd_y_mm, cmd_t_deg, file, time_used …)
    overlay(frame_small, row) : 사진 위에 그리기 (A-2),  graph : 오른쪽 아래 그래프 함수 (A-2)."""
    df = df.sort_values("shot_order").reset_index(drop=True)
    panel_w, panel_h = FRAME_W - PANEL_X - 10, FRAME_H - 60
    bg, tr = _timeline_background(df, [("cmd_y_mm", "지령 Y (mm)"), ("cmd_t_deg", "지령 Θ (°)")], panel_w, panel_h // 2 if graph else panel_h)
    with ThreadPoolExecutor(8) as ex:
        loaded = list(ex.map(lambda p: _load_small(p, inset_rect), df.path))

    frames = []
    for i, r in df.iterrows():
        small, ins = loaded[i]
        canvas = np.full((FRAME_H, FRAME_W, 3), 245, np.uint8)
        img = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        if overlay is not None:
            overlay(img, r)
        canvas[50:50 + IMG_H, :IMG_W] = img
        if ins is not None:                                           # QR 띠 확대 인셋
            ih = FRAME_H - 50 - IMG_H - 10
            iw = int(ins.shape[1] * ih / ins.shape[0])
            iw = min(iw, IMG_W)
            ins_c = cv2.cvtColor(cv2.resize(ins, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
            if overlay is not None and hasattr(overlay, "inset"):
                overlay.inset(ins_c, r, inset_rect, (iw, ih))
            canvas[60 + IMG_H:60 + IMG_H + ih, :iw] = ins_c
            cv2.putText(canvas, "QR band x" + f"{ih / ins.shape[0]:.2f}", (iw + 8, 60 + IMG_H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        tl = bg.copy()
        for k, c in enumerate(("cmd_y_mm", "cmd_t_deg")):
            x, y = tr[k](r.shot_order, r[c])
            cv2.circle(tl, (x, y), 7, (0, 0, 230), 2, cv2.LINE_AA)
        canvas[50:50 + tl.shape[0], PANEL_X:PANEL_X + tl.shape[1]] = tl
        if graph is not None:
            g = graph(i, panel_w, panel_h - tl.shape[0])
            canvas[50 + tl.shape[0]:50 + tl.shape[0] + g.shape[0], PANEL_X:PANEL_X + g.shape[1]] = g
        warn = ""
        if warn_col and isinstance(r.get(warn_col), str) and r[warn_col]:
            warn = f"  ⚠ 화질 경고: {r[warn_col]}"
            cv2.rectangle(canvas, (0, 50), (IMG_W - 1, 50 + IMG_H - 1), (0, 0, 255), 6)
        head = (f"[{tag}] #{r.shot_order:>3}  {r.file}  |  {str(r.time_used)[5:]}  |  지령 Y {r.cmd_y_mm:+.2f} mm, "
                f"Θ {r.cmd_t_deg:+.2f}°  |  접근 Y{r.approach_y} Θ{r.approach_t}{warn}")
        canvas[:40] = _font_img(head, color=(255, 255, 255) if not warn else (80, 80, 255))
        frames.append(canvas)

    out = {}
    stem = f"{tag}_{suffix}"
    gp = VIDEOS / f"{stem}.gif"
    out["gif_mb"] = round(_write_gif(frames, gp, ms=int(1000 / fps)), 1)
    if out["gif_mb"] > 20:                                            # 목표 20 MB 이하
        out["gif_mb"] = round(_write_gif(frames, gp, ms=int(1000 / fps), width=720), 1)
    out["gif"] = str(gp)
    mp = VIDEOS / f"{stem}.mp4"
    out["mp4"] = str(mp) if _write_mp4(frames, mp, fps) else ""
    t = (pd.to_datetime(df.time_used) - pd.Timestamp("1970-01-01")).dt.total_seconds()
    gaps = np.diff(t.values, append=t.values[-1])
    base = max(np.median(gaps[gaps > 0]) if (gaps > 0).any() else 1, 1)
    reps = [int(np.clip(round(g / base), 1, 12)) for g in gaps]
    mr = VIDEOS / f"{stem}_realtime.mp4"
    out["mp4_realtime"] = str(mr) if _write_mp4(frames, mr, fps, reps) else ""
    out["n_frames"] = len(frames)
    return out
