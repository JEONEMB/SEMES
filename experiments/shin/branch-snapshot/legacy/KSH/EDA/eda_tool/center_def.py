# -*- coding: utf-8 -*-
"""C-1 Center(원점) 정의 — QR 중심과 플레이트 중심의 정의와 계산."""
from __future__ import annotations

import json

import numpy as np

from legacy.KSH.EDA.eda_tool.common import CENTER

VERSION = "1.0"
DEFINITION = {
    "version": VERSION,
    "corner_order": "TL, TR, BR, BL (영상 좌표, x 오른쪽 +, y 아래 +)",
    "corner_method": "QR 밝은 사각형의 각 변을 가로지르는 밝기 단면 120개에서 어두움→밝음 반값 경계를 서브픽셀로 찾고, "
                     "변마다 Huber 직선 맞춤 후 이웃 변 직선 교점 (모서리 15 % 구간 제외, 단면 폭 ±9 px)",
    "qr_center": {"primary": "diag — 대각선 TL–BR 과 TR–BL 의 교점 (원근에서도 사각형의 사영 중심)",
                  "alternative": "mean — 꼭짓점 4개 단순 평균"},
    "plate_center": {"primary": "mean3 — QR L·C·R 대각선 교점 중심의 평균",
                     "alternative": "C — 가운데 QR 의 대각선 교점 중심"},
    "units": "px (영상 좌표). µm·° 환산은 Stage 2 선형 적합 감도 사용",
}


def diag_center(c: np.ndarray) -> np.ndarray:
    """꼭짓점 4개 (TL, TR, BR, BL) 의 대각선 교점."""
    p, r = c[0], c[2] - c[0]
    q, s = c[1], c[3] - c[1]
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]            # numpy 2.x 는 2차원 np.cross 를 지원하지 않음
    t = cross(q - p, s) / cross(r, s)
    return p + t * r


def save_definition(extra: dict | None = None) -> None:
    d = dict(DEFINITION)
    if extra:
        d.update(extra)
    CENTER.mkdir(parents=True, exist_ok=True)
    (CENTER / "center_definition.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
