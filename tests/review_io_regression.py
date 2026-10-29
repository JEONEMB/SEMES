"""Exercise native C++ config loading and CLI file-identity protection on temporary copies.
Usage: python tests/review_io_regression.py --bin BUILD/Release --work TEMP_DIR
No raw image is loaded or written by these tests.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--bin', required=True, type=Path)
parser.add_argument('--work', required=True, type=Path)
args = parser.parse_args()
payload = Path(__file__).resolve().parents[1]
binary = args.bin.resolve()
work = args.work.resolve()
work.mkdir(parents=True, exist_ok=True)
probe = binary / 'review_regression_test.exe'
cli = binary / 'main.exe'
outcomes = []


def config_probe(kind, data, description, valid=False):
    path = work / 'candidate.json'
    path.write_text(json.dumps(data, allow_nan=True), encoding='utf-8')
    run = subprocess.run([str(probe), '--config-probe', kind, path.name], cwd=work, capture_output=True)
    output = run.stdout.decode('utf-8', errors='replace').strip()
    assert run.returncode == (0 if valid else 3) and output == ('valid' if valid else 'invalid_configuration'), (description, run.returncode, output, run.stderr)
    outcomes.append({'case': description, 'passed': True})


def at(data, path):
    for item in path:
        data = data[item]
    return data


for kind in ('theta', 'y'):
    base = json.loads((payload / 'configs' / f'{kind}_alignment.json').read_text(encoding='utf-8-sig'))
    config_probe(kind, base, kind + ':original', valid=True)
    if kind == 'theta':
        root = ['schema_version', 'width', 'height', 'locator_roi_xywh', 'search_margin_px', 'tick_deg', 'tolerance_deg', 'max_step_deg', 'target_source', 'lines']
        nested = [('lines', i, key) for i in range(2) for key in ['name','x0','x1','expected_y','half_height','polarity','group4_slope','group5_slope']]
    else:
        root = ['schema_version','width','height','search_margin_px','tick_mm','target_profile','additional_tolerance_px','slope_groups','markers']
        nested = [('markers', i, key) for i in range(3) for key in ['name','roi_xywh','low_px','high_px','slopes_px_per_mm']]
    paths = [(k,) for k in root] + nested
    for path in paths:
        data = copy.deepcopy(base)
        del at(data, path[:-1])[path[-1]]
        config_probe(kind, data, f'{kind}:missing:{path}')
        original = at(base, path)
        wrong = 'not-a-number' if isinstance(original, (int,float)) else 123
        data = copy.deepcopy(base); at(data, path[:-1])[path[-1]] = wrong
        config_probe(kind, data, f'{kind}:wrong-type:{path}')
        if isinstance(original, (int,float)):
            for bad in (float('nan'), float('inf'), -float('inf')):
                data = copy.deepcopy(base); at(data, path[:-1])[path[-1]] = bad
                config_probe(kind, data, f'{kind}:nonfinite:{path}:{bad}')
        if isinstance(original, list):
            data = copy.deepcopy(base); at(data, path).pop()
            config_probe(kind, data, f'{kind}:short-array:{path}')
    if kind == 'y':
        extra = [(('additional_tolerance_px',), 0.01), (('slope_groups',0), 4),
                 (('markers',0,'slopes_px_per_mm',0), '14'), (('markers',0,'roi_xywh',0), 1545.5),
                 (('markers',0,'low_px'), 2000), (('markers',0,'slopes_px_per_mm',0), 0)]
    else:
        extra = [(('width',), 3840.5), (('locator_roi_xywh',0), 2147483647),
                 (('lines',0,'half_height'), 2147483647), (('lines',0,'group4_slope'), 0),
                 (('target_source',), 'unsupported'), (('tolerance_deg',), -0.005)]
    for path, bad in extra:
        data = copy.deepcopy(base); at(data, path[:-1])[path[-1]] = bad
        config_probe(kind, data, f'{kind}:invalid-value:{path}:{bad}')

# CLI rejects aliases before attempting to decode these deliberately fake input files.
# Check all protected inputs, including the separate Y config and image-list file.
alias_dir = work / 'aliases_한글'
alias_dir.mkdir(exist_ok=True)
inputs = {name: alias_dir / name for name in ['reference.bmp','theta.json','y.json','current.bmp','inputs.txt']}
for name,path in inputs.items():
    path.write_bytes(b'KEEP ORIGINAL CONTENT: ' + name.encode())
inputs['inputs.txt'].write_text(str(inputs['current.bmp'])+'\n', encoding='utf-8')
hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name,p in inputs.items()}
base_args = [str(cli), '--reference', str(inputs['reference.bmp']), '--config', str(inputs['theta.json']),
             '--y-config', str(inputs['y.json']), '--input-list', str(inputs['inputs.txt'])]
for name,path in inputs.items():
    variants = [str(path), os.path.relpath(path,work), str(path.parent / '.' / path.name), str(path).replace('\\','/')]
    if os.name == 'nt': variants.append(str(path).upper())
    hardlink = alias_dir / ('hardlink_' + name)
    if hardlink.exists(): hardlink.unlink()  # Only a known temporary test alias.
    os.link(path,hardlink)
    variants.append(str(hardlink))
    for i,output in enumerate(variants):
        run = subprocess.run(base_args+['--output',output],cwd=work,capture_output=True)
        assert run.returncode == 2 and b'output_aliases_input:' in run.stderr, (name,i,run.returncode,run.stderr)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in inputs.items()), (name,i,'input changed')
        outcomes.append({'case': f'alias:{name}:{i}', 'passed': True})

report = {'passed': len(outcomes), 'failed': 0, 'checks': outcomes, 'input_copies_unchanged': True}
(work / 'io_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS {len(outcomes)} native configuration and CLI alias checks; all input copies unchanged')
