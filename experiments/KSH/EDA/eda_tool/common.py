# -*- coding: utf-8 -*-
"""공통 유틸 — 경로, 세트 설정, 영상 읽기, 캐시, 그림 글꼴, CSV 저장."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------- 경로
TOOL_DIR = Path(__file__).resolve().parent
EDA_DIR = TOOL_DIR.parent
REPO_DIR = EDA_DIR.parents[2]                     # src/KSH/EDA → 저장소 루트
DATA_ROOT = REPO_DIR / "data" / "img"             # 원본 (읽기 전용)
RESULTS = Path(os.environ["EDA_RESULTS"]).resolve() if os.environ.get("EDA_RESULTS") else EDA_DIR / "eda_results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
VIDEOS = RESULTS / "videos"
CENTER = RESULTS / "center"
CACHE = RESULTS / "cache"
YAML_DRAFT = EDA_DIR / "eda_results" / "set_yaml_draft"   # 설정 초안 — --results 로 결과 폴더를 바꿔도 같은 곳에서 읽음

# 판정 기준 · 고정 사실
TOL_T_DEG = 0.01
TOL_Y_UM = 90.0
IMG_W, IMG_H = 3840, 2160


def ensure_dirs() -> None:
    for d in (TABLES, FIGURES, VIDEOS, CENTER, CACHE, YAML_DRAFT):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 세트 설정
def load_set_config(set_dir: Path) -> dict:
    """세트 폴더의 set.yaml 을 읽는다. 없으면 eda_results/set_yaml_draft/<폴더명>.yaml 을 쓴다."""
    for p in (set_dir / "set.yaml", YAML_DRAFT / f"{set_dir.name}.yaml"):
        if p.exists():
            cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
            cfg["_config_path"] = str(p)
            cfg["_set_dir"] = str(set_dir)
            return cfg
    raise FileNotFoundError(f"set.yaml 없음: {set_dir} (초안은 {YAML_DRAFT})")


def list_set_dirs(root: Path = DATA_ROOT) -> list[Path]:
    """set.yaml(또는 초안)이 있는 세트 폴더 목록."""
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "set.yaml").exists() or (YAML_DRAFT / f"{d.name}.yaml").exists():
            out.append(d)
    return out


# ---------------------------------------------------------------- 영상
def read_gray(path: Path | str) -> np.ndarray:
    """한글 경로 대응 흑백 읽기 (원본은 24-bit RGB BMP, 세 채널이 같은 회색)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"영상 읽기 실패: {path}")
    return img


def file_sha1(path: Path | str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 캐시 (파일 해시 기준)
def cache_get(sha1: str, key: str):
    p = CACHE / key / f"{sha1}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def cache_put(sha1: str, key: str, value) -> None:
    d = CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sha1}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- 표 · 그림
def write_csv(df, path: Path) -> None:
    """엑셀에서 바로 열리도록 UTF-8 (BOM) 으로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 130
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    return plt


def rel(p: Path | str) -> str:
    """보고용 상대 경로 (EDA 폴더 기준)."""
    try:
        return str(Path(p).resolve().relative_to(EDA_DIR)).replace("\\", "/")
    except ValueError:
        return str(p)


def mad(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)
