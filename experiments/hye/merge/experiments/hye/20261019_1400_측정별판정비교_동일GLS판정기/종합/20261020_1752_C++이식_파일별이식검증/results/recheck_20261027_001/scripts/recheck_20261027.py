"""Read-only audit of originals; fresh C++ build and replay into a new directory.

Python orchestrates verification only. It is not part of C++ inference.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'recheck_20261027_001'
CM = Path('C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe')
BUILD = ROOT / 'build-recheck-20261027'
DATA = '../hw-analysis/data/hw/bmp/20261014'
records = []

def run(name, args):
    start = time.monotonic()
    with (OUT / (name + '.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run([str(x) for x in args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    records.append(dict(name=name, command=[str(x) for x in args], exit_code=result.returncode, seconds=time.monotonic()-start))
    (OUT / 'commands.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(name, result.returncode, flush=True)
    return result.returncode == 0

def exe(name):
    matches = list(BUILD.rglob(name + '.exe'))
    if len(matches) != 1:
        raise RuntimeError((name, matches))
    return matches[0]

def job(name, args):
    return run(name, [exe(name), *args])

def output(name):
    return str(OUT / name)

if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    if not run('configure', [CM, '-S', '.', '-B', BUILD, '-G', 'Visual Studio 18 2026', '-A', 'x64', '-DOpenCV_DIR=C:/opencv/build', '-DSEMES_ZLIB_SOURCE=C:/opencv/sources/3rdparty/zlib', f'-DSEMES_OPENCV4_DIR={ROOT / "build/dependencies/opencv411-build"}']):
        raise SystemExit(1)
    if not run('build', [CM, '--build', BUILD, '--config', 'Release', '--', '/m:1', '/nr:false']):
        raise SystemExit(1)
    run('ctest', [CM.with_name('ctest.exe'), '--test-dir', BUILD, '-C', 'Release', '--output-on-failure'])
    job('main', ['evaluate', 'models/edge630', DATA+'/center.bmp', DATA, 'config/folds4.json', output('full280')])
    # Fixed name archive; never silently choose a different experiment's cache.
    names = ROOT.parent/'codex_experiments_20261014/center_fixed_20261015/center_once_tracks.npz'
    job('evaluation', [output('full280'), names, 'config/folds4.json', output('oof4')])
    jobs = [
        ('historical_agent_evaluate', ['../analysis_agent/qr_measurements.csv', output('agent')]),
        ('historical_theta_only', ['../analysis_agent/qr_measurements.csv', output('theta')]),
        ('historical_measure_qr', [DATA, output('affine')]),
        ('historical_v2_estimate', ['..', output('v2_estimate')]),
        ('historical_v2_homography', ['../14_데이터_QR마커위치.csv', DATA, output('homography')]),
        ('historical_v2_judge', ['..', output('v2_judge')]),
        ('historical_v2_outputs', ['..', output('v2_judge'), output('v2_estimate'), output('v2_outputs')]),
        ('historical_v2_workbook_check', ['..', output('v2_judge'), '../23_YΘ구분표_v2.xlsx', output('v2_workbook')]),
        ('historical_v1_grid', ['..', output('v1_grid')]),
        ('historical_v1_track', ['..', DATA, output('v1_track'), 'frozen']),
        ('historical_v1_combos', ['..', output('v1_combos')]),
        ('historical_v1_stable', ['..', DATA, output('v1_combos')+'/combos.json', output('v1_stable')]),
        ('historical_v3_evaluate', ['..', output('v3_evaluate')]),
        ('historical_v3_outputs', ['..', output('v3_evaluate'), output('v3_outputs')]),
        ('historical_twin_lines', ['..', DATA, output('v2_judge')+'/decisions.csv', output('twin_lines')]),
        ('historical_twin_photos', ['..', DATA, output('v2_judge')+'/decisions.csv', output('twin_photos')]),
        ('historical_slide_data', ['..', output('v2_judge')+'/decisions.csv', output('v3_evaluate')+'/predictions.csv', output('slide_data')]),
        ('historical_slide_panels', ['..', DATA, output('slide_data')+'/methods.json', output('slide_panels')]),
        ('historical_zero_offset', ['..', DATA, output('zero_offset')]),
        ('historical_qr_only', ['..', DATA, output('qr_only')]),
        ('historical_ecc_tune', ['..', DATA, output('ecc_tune')]),
        ('historical_spline_checks', [output('spline')]),
        ('historical_pixel_fast', ['..', DATA, output('pixel_fast')]),
        ('historical_pixel_slow', ['..', DATA, output('pixel_slow')]),
        ('historical_run_g1', ['..', DATA, output('g1')]),
        ('historical_g1_compare', ['../analysis_v3', output('g1'), output('g1_parity')]),
        ('historical_robot_motion', [output('robot')]),
        ('historical_video_preview', [DATA, output('video')]),
    ]
    for name, args in jobs:
        job(name, args)
