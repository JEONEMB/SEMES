"""Audit nominal sampling without guessing unknown filename conventions.

Usage: python tool/offaxis/inventory.py --root "data/SSAFY Test Image Set" --output docs/offaxis-eda
Unknown names deliberately retain empty coordinates; inspect them before analysis.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def coordinates(relative):
    stem = relative.stem
    if relative.parent.name == '기준 위치' and re.fullmatch(r'center\d+', stem):
        return 0., 0., 'new-reference'
    mixed = re.fullmatch(r'Y([+-]\d+(?:\.\d+)?) T([+-]\d+(?:\.\d+)?)', relative.parent.name)
    file_mixed = re.fullmatch(r'Y([+-]\d+(?:\.\d+)?) T([+-]\d+(?:\.\d+)?) \((\d+)\)', stem)
    if mixed and file_mixed and tuple(map(float, mixed.groups())) == tuple(map(float, file_mixed.groups()[:2])):
        return float(mixed[1]), float(mixed[2]), 'mixed'
    if stem.lower() == 'center' and len(relative.parts) == 1:
        return 0., 0., 'reference'
    match = re.fullmatch(r'([YT])([+-]\d+(?:\.\d+)?)', stem)
    if match and relative.parent.name == {'Y': 'Y', 'T': 'Theta'}[match[1]]:
        value = float(match[2])
        return (value, 0., 'Y-only') if match[1] == 'Y' else (0., value, 'theta-only')
    return None, None, 'unresolved'


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
            continue
        rel = path.relative_to(root)
        y, theta, category = coordinates(rel)
        sidecar = next((p for p in (path.with_suffix('.txt'), path.with_suffix('.txt.txt')) if p.exists()), None)
        actual = []
        if sidecar:
            actual = re.findall(r'[+-]?\d+(?:\.\d+)?', sidecar.read_text(encoding='utf-8-sig').splitlines()[-1])
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        rows.append(dict(path=rel.as_posix(), y_nominal_mm=y, theta_nominal_assumed_deg=theta,
                         category=category, width=width, height=height, mode=mode,
                         y_actual=float(actual[0]) if len(actual) == 2 else None,
                         theta_actual=float(actual[1]) if len(actual) == 2 else None,
                         bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                         sha256=digest.hexdigest()))
    if not rows:
        raise ValueError('No images found; check the input root')
    with (output / 'sampling.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    known = [r for r in rows if r['category'] != 'unresolved']
    y = np.array([r['y_nominal_mm'] for r in known])
    t = np.array([r['theta_nominal_assumed_deg'] for r in known])
    design = np.column_stack([np.ones(len(y)), y, t, y*y, t*t, y*t])
    summary = dict(images=len(rows), resolved=len(known), unresolved=len(rows)-len(known),
                   categories={c: sum(r['category'] == c for r in rows) for c in sorted({r['category'] for r in rows})},
                   offaxis=int(np.sum((y != 0) & (t != 0))),
                   quadratic_design_columns=['1', 'Y', 'theta', 'Y^2', 'theta^2', 'Y*theta'],
                   quadratic_design_rank=int(np.linalg.matrix_rank(design)) if known else 0,
                   coordinate_source='filename nominal command, not independently measured truth',
                   theta_unit='degrees assumed; acquisition log required to confirm',
                   time_source='filesystem mtime only; not a confirmed acquisition timestamp',
                   original_images_modified=False)
    (output / 'sampling_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 5))
    for category in sorted({r['category'] for r in known}):
        group = [r for r in known if r['category'] == category]
        ax.scatter([r['y_nominal_mm'] for r in group], [r['theta_nominal_assumed_deg'] for r in group], s=13, label=f'{category} (n={len(group)})')
    ax.set(xlabel='Nominal Y (mm)', ylabel='Nominal theta (degree assumed)',
           title=f'Available sampling: {len(rows)} images; unresolved {len(rows)-len(known)}')
    ax.grid(alpha=.25)
    if known:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'sampling.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
