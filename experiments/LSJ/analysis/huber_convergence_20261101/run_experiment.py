"""Ordinary Huber RLM: iteration-cap/tolerance comparisons only.

No implementation patches, point rejection, Gaussian fitting or theta calibration.
Run with .venv/Scripts/python.exe. Source data are read-only.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import sys
import time
import warnings

import numpy as np
import statsmodels
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
from statsmodels.robust import scale

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'adahuber_verification_20261101'
PROFILE = HERE.parent / 'gaussian_window_no_rejection_20261101/profiles.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(obj):
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def dump(name, obj):
    (HERE / name).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2,
                                       allow_nan=False), encoding='utf-8')


assert statsmodels.__version__ == '0.15.0'
signature = inspect.signature(RLM.fit)
assert signature.parameters['maxiter'].default == 50
assert signature.parameters['tol'].default == 1e-8
assert signature.parameters['scale_est'].default == 'mad'
assert signature.parameters['conv'].default == 'dev'
assert signature.parameters['update_scale'].default is True
assert HuberT().t == 1.345
assert 'center=0' in inspect.getsource(RLM._estimate_scale)

source_dir = HERE / 'sources'
source_dir.mkdir(exist_ok=True)
sources = []
for module_name in ['statsmodels.robust.robust_linear_model',
                    'statsmodels.robust.norms', 'statsmodels.robust.scale',
                    'statsmodels.regression._tools']:
    module = __import__(module_name, fromlist=['x'])
    p = Path(module.__file__)
    out = source_dir / (module_name.replace('.', '__') + '.py')
    out.write_bytes(p.read_bytes())
    sources.append({'module': module_name, 'installed_file': str(p),
                    'snapshot': str(out.relative_to(HERE)), 'sha256': sha(p)})

prior = json.loads((PRIOR / 'data_provenance.json').read_text(encoding='utf-8'))
assert sha(PROFILE) == prior['profile_sha256']
image_checks = []
for item in prior['images']:
    assert sha(item['path']) == item['sha256']
    image_checks.append({'id': item['id'], 'path': item['path'],
                         'sha256': item['sha256'], 'unchanged': True})

groups = defaultdict(list)
for line in PROFILE.read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    groups[row['ref'] + '_' + row['target']].append((row['x'], row['coarse_y']))
assert len(groups) == 8
assert sum(map(len, groups.values())) == 23056

# These comparison conditions were disclosed in the conversation before execution.
# Only A is the library default; B--E are sensitivity-test settings, not standards.
configs = [('A', 50, 1e-8), ('B', 100, 1e-8), ('C', 200, 1e-8),
           ('D', 200, 1e-10), ('E', 200, 1e-12),
           ('F', 400, 1e-10), ('G', 400, 1e-12),
           ('H', 800, 1e-10), ('I', 800, 1e-12)]
# F--I were added after D/E hit their caps, as disclosed in the progress update.
# 400/800 are cap-doubling diagnostic conditions, not new production settings.
comparison_pairs = [('A','B'),('B','C'),('C','D'),('D','E'),('A','E'),
                    ('D','F'),('F','H'),('E','G'),('G','I'),('A','I')]
settings = {'fixed': {'norm': 'HuberT(t=1.345)', 'residual': 'vertical',
                     'scale_est': 'mad', 'update_scale': True, 'conv': 'dev',
                     'start_params': 'None: original OLS initialization',
                     'cov': 'H1', 'design': '[ones, original image x]',
                     'input': 'all saved integer Sobel points; no point rejection',
                     'gaussian': False},
            'comparisons': [{'id': c, 'maxiter': n, 'tol': t} for c, n, t in configs],
            'extension_reason': 'At 200 iterations, tol=1e-10 failed in 2 inputs and tol=1e-12 in 7. Double caps to 400/800 to examine continuing behavior; no tolerance or fitting formula changes.',
            'acceptance': 'No invented geometric pass threshold; report termination and measured differences.'}
dump('settings.json', settings)

results = []
histories = []
coefficients = {}
point_data = {}
start = time.perf_counter()

def run_one(key, xy, config):
    label, cap, tol = config
    x, y = xy[:, 0], xy[:, 1]
    X = np.column_stack([np.ones(len(x)), x])
    # A fresh model per fit prevents cross-run result/model state reuse.
    model = RLM(y.copy(), X.copy(), M=HuberT(t=1.345), missing='raise')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tic = time.perf_counter()
        result = model.fit(maxiter=cap, tol=tol, scale_est='mad',
                           update_scale=True, conv='dev', cov='H1')
        elapsed = time.perf_counter() - tic
    h = result.fit_history
    iteration = int(h['iteration'])
    dev = np.asarray(h['deviance'][1:], float)
    params = np.asarray(h['params'][1:], float)
    scales = np.asarray(h['scale'], float)
    last_delta = float(abs(dev[-1] - dev[-2]))
    finite = bool(np.isfinite(params[-1]).all() and np.isfinite(dev[-2:]).all()
                  and np.isfinite(result.scale) and result.scale > 0)
    tol_met = finite and last_delta <= tol
    cap_met = iteration >= cap
    if tol_met and cap_met:
        reason = 'tolerance_and_cap'
    elif tol_met:
        reason = 'tolerance'
    elif cap_met:
        reason = 'iteration_cap'
    else:
        reason = 'other_or_nonfinite'
    b, m = map(float, result.params)
    resid = y - (b + m*x)
    weights_used = np.asarray(result.weights)
    # Observational calculation at the returned coefficients/scale; not refitted.
    weights_at_return = HuberT(t=1.345).weights(resid/result.scale)
    theoretical_scale = float(scale.mad(resid, center=0))
    scale_delta = abs(theoretical_scale - float(result.scale))
    row = {'input': key, 'config': label, 'n': len(x), 'maxiter': cap, 'tol': tol,
           'iteration': iteration, 'stop_reason': reason, 'tol_met': tol_met,
           'cap_met': cap_met, 'finite': finite, 'deviance_last': float(dev[-1]),
           'deviance_delta': last_delta, 'intercept_px': b, 'slope': m,
           'image_angle_deg': float(np.degrees(np.arctan(m))),
           'scale_px': float(result.scale), 'tau_px': 1.345*float(result.scale),
           'scale_formula_difference_px': scale_delta,
           'rmse_all_points_px': float(np.sqrt(np.mean(resid**2))),
           'max_abs_residual_px': float(np.max(abs(resid))),
           'used_downweighted_points': int(np.sum(weights_used < 1)),
           'returned_residual_downweighted_points': int(np.sum(weights_at_return < 1)),
           'min_used_weight': float(weights_used.min()),
           'weight_update_lag_max': float(np.max(abs(weights_used-weights_at_return))),
           'runtime_ms': elapsed*1000,
           'warnings': [str(w.message) for w in caught]}
    results.append(row)
    histories.append({'input': key, 'config': label, 'iteration': iteration,
                      'params_b_m': params, 'scale_px': scales, 'deviance': dev})
    coefficients[key, label] = np.array([b, m])
    print(key, label, 'iterations=', iteration, 'stop=', reason,
          'dev_delta=', format(last_delta, '.4g'), 'angle=', format(row['image_angle_deg'], '.12f'), flush=True)

for key, points in sorted(groups.items()):
    xy = np.array(points, dtype=np.float64)
    assert np.isfinite(xy).all() and np.unique(xy[:, 0]).size > 1
    point_data[key] = xy
    for config in configs:
        run_one(key, xy, config)

dump('results.json', results)
dump('histories.json', histories)
np.savez_compressed(HERE/'inputs.npz', **point_data)

comparisons = []
for key, xy in point_data.items():
    for left, right in comparison_pairs:
        db, dm = coefficients[key, right]-coefficients[key, left]
        differences = db+dm*xy[:, 0]
        ix = int(np.argmax(abs(differences)))
        a = coefficients[key, left]
        b = coefficients[key, right]
        comparisons.append({'input': key, 'from': left, 'to': right,
                            'intercept_delta_px': db, 'slope_delta': dm,
                            'angle_delta_deg': np.degrees(np.arctan(b[1])-np.arctan(a[1])),
                            'max_line_difference_px': float(abs(differences[ix])),
                            'at_x_px': float(xy[ix, 0]),
                            'coefficients_bitwise_equal': bool(np.array_equal(a, b))})
dump('comparisons.json', comparisons)

for filename, rows in [('results.csv', results), ('comparisons.csv', comparisons)]:
    with (HERE/filename).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

env = {name: importlib.metadata.version(name)
       for name in ['statsmodels','numpy','scipy','pandas','patsy','packaging','formulaic']}
dump('provenance.json', {'python': sys.version, 'executable': sys.executable,
     'packages': env, 'profile_file': str(PROFILE), 'profile_sha256': sha(PROFILE),
     'prior_point_verification': str(PRIOR/'data_provenance.json'),
     'image_checks': image_checks, 'input_counts': {k: len(v) for k,v in groups.items()},
     'source_snapshots': sources, 'elapsed_seconds': time.perf_counter()-start,
     'scope': 'Numerical convergence comparisons for ordinary Huber only. No theta calibration, no new filtering.'})
summary = {'runs': len(results), 'inputs': len(groups), 'points': sum(map(len, groups.values())),
           'all_finite': all(r['finite'] for r in results),
           'per_config': {}, 'max_comparison_differences': {}}
for label, cap, tol in configs:
    rows = [r for r in results if r['config']==label]
    summary['per_config'][label] = {'maxiter': cap, 'tol': tol,
       'tol_met_count': sum(r['tol_met'] for r in rows),
       'cap_only_count': sum(r['stop_reason']=='iteration_cap' for r in rows),
       'iteration_min': min(r['iteration'] for r in rows),
       'iteration_max': max(r['iteration'] for r in rows)}
for left, right in comparison_pairs:
    rows = [r for r in comparisons if r['from']==left and r['to']==right]
    worst = max(rows, key=lambda r:r['max_line_difference_px'])
    summary['max_comparison_differences'][left+'_'+right] = {
        'max_line_difference_px': worst['max_line_difference_px'], 'worst_input': worst['input'],
        'max_abs_angle_difference_deg': max(abs(r['angle_delta_deg']) for r in rows),
        'all_coefficients_bitwise_equal': all(r['coefficients_bitwise_equal'] for r in rows)}
dump('summary.json', summary)
print(json.dumps(clean(summary), ensure_ascii=False, indent=2), flush=True)
