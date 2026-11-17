"""Run one frozen deployed algorithm on every original motion image.

This is a full development-set acceptance check: the existing fitted model was
trained on these 280 labelled images. It is not an independent held-out score.
Inference sees only a reference path and an unlabelled current-image path.
"""
from pathlib import Path
import concurrent.futures
import csv
import hashlib
import json
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from infer import run, clean

OUT = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(slot, tasks, source_root, center):
    current = OUT / 'inputs' / f'worker_{slot}' / 'current.bmp'
    current.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, entry in tasks:
        source = source_root / ('Theta' if entry['axis'] == 'T' else 'Y') / entry['file']
        blob = source.read_bytes()
        current.write_bytes(blob)
        input_hash = hashlib.sha256(blob).hexdigest()
        # No truth label, source name, time, or session ID reaches this function.
        prediction = clean(run(center, current, mode='calibrated-runs'))
        assert sha(current) == input_hash
        # Ground truth is consulted only after prediction has returned.
        axis, value = entry['axis'], float(entry['value'])
        true_y, true_t = (value, 0.) if axis == 'Y' else (0., value)
        status = prediction.get('status')
        pred_y, pred_t = prediction.get('Y'), prediction.get('T')
        error_y = abs(pred_y - true_y) if pred_y is not None else None
        error_t = abs(pred_t - true_t) if pred_t is not None else None
        joint_correct = status == 'prediction' and prediction['axis'] == axis and prediction['sign'] == (1 if value > 0 else -1)
        passed = joint_correct and error_y <= .09 and error_t <= .09
        records.append(dict(
            image_id=index + 1, file=entry['file'], input_sha256=input_hash,
            true_axis=axis, true_value=value, true_Y=true_y, true_T=true_t,
            pred_axis=prediction.get('axis'), pred_sign=prediction.get('sign'),
            pred_value=prediction.get('value'), pred_Y=pred_y, pred_T=pred_t,
            abs_error_Y=error_y, abs_error_T=error_t,
            axis_sign_correct=joint_correct, passes_all_requirements=bool(passed),
            valid_points=prediction.get('valid_points'),
            normalized_cost_margin=prediction.get('normalized_cost_margin'),
            prediction=prediction))
        if len(records) % 20 == 0:
            print(f'worker {slot}: {len(records)}/{len(tasks)} completed', flush=True)
    return records

def main():
    start = time.perf_counter()
    reference = json.loads((BASE / 'reference.json').read_text(encoding='utf8'))
    source_root = Path(reference['source_path']).parent
    center = OUT / 'inputs' / 'center.bmp'
    center.parent.mkdir(parents=True, exist_ok=True)
    center.write_bytes(Path(reference['source_path']).read_bytes())
    assert sha(center) == reference['sha256']
    with (BASE / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        manifest = [r for r in csv.DictReader(f) if r['axis'] in ('Y', 'T')]
    assert len(manifest) == 280 and len({r['file'] for r in manifest}) == 280
    model_paths = [BASE / 'calibrated_runs/model.json', BASE / 'measurement/gn17_spread14_features.npz', BASE / 'infer.py']
    hashes_before = {str(p.relative_to(BASE)): sha(p) for p in model_paths}
    tasks = list(enumerate(manifest))
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, k, tasks[k::4], source_root, center) for k in range(4)]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda r: r['image_id'])
    assert len(rows) == 280 and len({r['file'] for r in rows}) == 280
    assert hashes_before == {str(p.relative_to(BASE)): sha(p) for p in model_paths}
    fields = [k for k in rows[0] if k != 'prediction']
    with (OUT / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    (OUT / 'predictions.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    failures = [r for r in rows if not r['passes_all_requirements']]
    (OUT / 'failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf8')
    summary = dict(
        evaluation='frozen_model_full_280_development_set_acceptance',
        data_was_used_to_fit_model=True,
        independent_holdout=False,
        inference_inputs='Only center.bmp and unlabelled worker/current.bmp; labels consulted after prediction.',
        model_hashes=hashes_before, model_unchanged_during_evaluation=True,
        total=280, correct=sum(r['axis_sign_correct'] for r in rows),
        passes_all_requirements=sum(r['passes_all_requirements'] for r in rows),
        failure_count=len(failures), elapsed_seconds=time.perf_counter()-start,
        max_abs_error_Y=max(r['abs_error_Y'] for r in rows),
        max_abs_error_T=max(r['abs_error_T'] for r in rows),
        all_original_metadata_unchanged=all(
            (source_root / ('Theta' if r['axis']=='T' else 'Y') / r['file']).stat().st_size == int(r['size']) and
            abs((source_root / ('Theta' if r['axis']=='T' else 'Y') / r['file']).stat().st_mtime-float(r['mtime'])) < 1e-5
            for r in manifest),
        per_axis={})
    for axis in ['Y','T']:
        subset=[r for r in rows if r['true_axis']==axis]
        worst=max(subset,key=lambda r:r[f'abs_error_{axis}'])
        summary['per_axis'][axis]=dict(
            n=len(subset),correct=sum(r['axis_sign_correct'] for r in subset),
            accuracy=sum(r['axis_sign_correct'] for r in subset)/len(subset),
            passes_all_requirements=sum(r['passes_all_requirements'] for r in subset),
            max_abs_error=max(r[f'abs_error_{axis}'] for r in subset),
            mae=sum(r[f'abs_error_{axis}'] for r in subset)/len(subset),
            worst_file=worst['file'], worst_true=worst['true_value'], worst_prediction=worst['pred_value'])
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 2

if __name__=='__main__':
    raise SystemExit(main())
