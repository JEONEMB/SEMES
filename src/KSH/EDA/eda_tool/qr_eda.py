# -*- coding: utf-8 -*-
"""QR Y·Θ 분리 EDA 툴 — 실행 진입점.

  python eda_tool/qr_eda.py --set data/img/1st --stages 0-4
  python eda_tool/qr_eda.py --all                       # 모든 세트 + 세트 간 비교 + 보고서
  python eda_tool/qr_eda.py --all --stages 2 --force    # 캐시 무시 재계산
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "--results" in sys.argv:                       # 결과 폴더 바꾸기 (common 을 읽기 전에 정해야 함)
    import os
    os.environ["EDA_RESULTS"] = sys.argv[sys.argv.index("--results") + 1]

import pandas as pd  # noqa: E402

from common import CACHE, DATA_ROOT, RESULTS, TABLES, ensure_dirs, list_set_dirs, load_set_config, rel  # noqa: E402


def _stages(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _tag(set_dir: Path) -> str:
    return load_set_config(set_dir)["set_id"]


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], help="세트 폴더 (여러 번 지정 가능)")
    ap.add_argument("--all", action="store_true", help=f"{DATA_ROOT} 아래 set.yaml(또는 초안)이 있는 모든 세트")
    ap.add_argument("--stages", default="0-5")
    ap.add_argument("--force", action="store_true", help="캐시 무시")
    ap.add_argument("--no-video", action="store_true", help="영상 생성 생략 (빠른 확인용)")
    ap.add_argument("--results", help="결과 폴더 (기본 eda_results) — 시험 실행용")
    a = ap.parse_args(argv)

    ensure_dirs()
    if a.force and CACHE.exists():
        for d in CACHE.iterdir():
            if d.name != "stage0_meta":
                shutil.rmtree(d)
    sets = [Path(s).resolve() for s in a.set] if a.set else list_set_dirs()
    if not sets:
        print("세트가 없습니다. --set 또는 set.yaml 을 확인하세요."); return 1
    stages = _stages(a.stages)
    log_path = RESULTS / "run_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}   # 다른 단계 결과는 유지

    if 0 in stages:
        import stage0_catalog as s0
        import media
        dfs = {}
        for d in sets:
            df = s0.run(d); dfs[d.name] = df
            figs = s0.plot(df, df.attrs["set_id"])
            print(f"[Stage 0] {df.attrs['set_id']}: {len(df)}장, 시각={df.attrs['time_choice']}, 해석불가={len(df.attrs['unparsed'])}")
            if not a.no_video:
                from qr_geom import find_qr_boxes
                from common import read_gray
                ref = df[df.kind == "reference"].iloc[0]
                boxes = find_qr_boxes(read_gray(ref.path))
                inset = None
                if boxes:
                    x0 = int(boxes[0]["x"] - 60); x1 = int(boxes[2]["x"] + boxes[2]["w"] + 60)
                    y0 = int(min(b["y"] for b in boxes) - 40); y1 = int(max(b["y"] + b["h"] for b in boxes) + 40)
                    inset = (x0, y0, x1, y1)
                v = media.shooting_video(df, df.attrs["set_id"], inset_rect=inset)
                log[f"s0_video_{df.attrs['set_id']}"] = v
                print(f"   영상: {rel(v['gif'])} ({v['gif_mb']} MB), {rel(v['mp4'])}")
        s0.save(dfs)
    for st in (1, 2, 3, 4, 5):
        if st in stages:
            mod = __import__({1: "stage1_quality", 2: "stage2_features", 3: "stage3_mixed", 4: "stage4_labels", 5: "report"}[st])
            log[f"stage{st}"] = mod.run_all(sets, no_video=a.no_video)
            log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")   # 단계마다 저장
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
