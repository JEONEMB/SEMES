"""Track identical QR landmarks with one local reference per rig session."""
import csv
import io
import zipfile
from pathlib import Path

import numpy as np

from track import image, patch, qr_templates, track_qr

DATASETS = ('rig_1st', 'rig_grid', 'rig_grid_rev')
ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home()/'Desktop'


def main():
    output = ROOT/'evaluation'
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(DESKTOP/'rig_grid_rev.zip') as archive:
        canonical = qr_templates(image(archive, 'rig_grid_rev/center.bmp'))
    for name in DATASETS:
        prefix = name+'/'
        with zipfile.ZipFile(DESKTOP/(name+'.zip')) as archive:
            with io.TextIOWrapper(archive.open(prefix+'manifest.csv'), encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            ref = image(archive, prefix+'center.bmp')
            mapped = track_qr(ref, *canonical)
            points = np.rint(np.asarray(canonical[0])+mapped[:, :2]).astype(np.int32)
            assert len(points) == 42 and mapped[:, 2].min() > .9
            templates = [patch(ref, int(x), int(y), 17) for x, y in points]
            coarse = patch(ref, 1907, 1130, 75)
            shifts = []
            for i, row in enumerate(rows):
                shifts.append(track_qr(image(archive, prefix+row['file']+'.bmp'),
                                       points, templates, coarse))
                if i % 400 == 0:
                    print(name, i, len(rows), flush=True)
            shifts = np.asarray(shifts)
            assert np.isfinite(shifts).all() and shifts[:, :, 2].min() > .9
            np.savez_compressed(output/(name+'.npz'), points=points, shifts=shifts,
                                names=[r['file'] for r in rows],
                                labels=[[float(r[k]) for k in ('command_y_mm','command_t_deg')]
                                        for r in rows])


if __name__ == '__main__':
    main()
