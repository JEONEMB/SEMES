"""Independent replay from raw bytes; no training or score recomputation.

Only this audit directory is written. Cached features are opened AFTER replay.
"""
from pathlib import Path
import ast
import builtins
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import time

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REBOOT = HERE.parents[1]
ROOT = REBOOT.parents[1]
SOURCE = REBOOT / 'parallax'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def records(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


@contextlib.contextmanager
def prohibit_python_reads():
    """Current bytes are already in memory. Any further Python file I/O fails."""
    def forbidden(*args, **kwargs):
        raise RuntimeError('Unexpected filesystem/cache read during byte extraction')
    old = (builtins.open, io.open, Path.read_bytes, Path.read_text, np.load)
    builtins.open = io.open = Path.read_bytes = Path.read_text = np.load = forbidden
    try:
        yield
    finally:
        builtins.open, io.open, Path.read_bytes, Path.read_text, np.load = old


def compare(a, b):
    if a.shape != b.shape:
        return {'shape_equal': False, 'new_shape': list(a.shape), 'cache_shape': list(b.shape), 'equal': False}
    if a.dtype.kind in 'OUS' or a.dtype.kind == 'b':
        bad = a != b
        return {'shape_equal': True, 'equal': bool(not bad.any()), 'mismatches': int(bad.sum())}
    finite = np.isfinite(a) & np.isfinite(b)
    finite_mismatch = int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
    nan_mismatch = int(np.count_nonzero(np.isnan(a) != np.isnan(b)))
    delta = np.abs(a[finite] - b[finite])
    max_abs = float(delta.max()) if len(delta) else 0.0
    return {'shape_equal': True, 'equal': bool(np.array_equal(a, b, equal_nan=True)),
            'finite_mask_mismatches': finite_mismatch, 'nan_mask_mismatches': nan_mismatch,
            'max_absolute_difference': max_abs, 'new_nonfinite_count': int((~np.isfinite(a)).sum())}


def main():
    started = time.perf_counter()
    cv2.setNumThreads(1)
    audited_files = [SOURCE/'extractor.py', SOURCE/'measure.py', SOURCE/'protocol.json',
                     SOURCE/'manifest.csv', SOURCE/'features.npz', ROOT/'ALGORITHM_RULES.md']
    initial_hashes = {str(p): digest(p) for p in audited_files}
    manifest = records(SOURCE/'manifest.csv')
    reference_path = ROOT/'codex_experiments_20261014/joint_yt_generalization_20261022/measurements/manifest.csv'
    reference = records(reference_path)
    assert len(manifest) == len(reference) == 341
    assert [int(x['index']) for x in manifest] == list(range(341))
    alignment = [Path(a['path']).resolve() == Path(b['path']).resolve() and a['sha256'] == b['sha256']
                 for a, b in zip(manifest, reference)]
    assert all(alignment), 'Measurement manifest differs from original source row order'

    spec = importlib.util.spec_from_file_location('raw_parallax_extractor_under_audit', SOURCE/'extractor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ex = module.ParallaxExtractor()
    names = [name for name, _ in module.CANDIDATES]
    assert list(inspect.signature(ex.extract_bytes).parameters) == ['image_bytes']
    protocol = json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    center_hash = digest(module.CENTER_PATH)
    assert center_hash == protocol['center_sha256']
    all_results = []
    rows = []
    failures = []
    for i, meta in enumerate(manifest):
        path = Path(meta['path'])
        image_bytes = path.read_bytes()
        sha = hashlib.sha256(image_bytes).hexdigest()
        assert sha == meta['sha256'] == meta['input_bytes_sha256'] == reference[i]['sha256']
        try:
            with prohibit_python_reads():
                result = ex.extract_bytes(image_bytes)
            assert result['image_sha256'] == sha
            all_results.append(result)
        except Exception as exc:
            failures.append({'index': i, 'path': str(path), 'exception': repr(exc)})
            raise
        row = {'index': i, 'path': str(path), 'bytes': len(image_bytes), 'sha256': sha,
               'raw_hash_matches_manifest': True, 'raw_hash_matches_source_manifest': True,
               'ncc_pad': result['ncc_pad'], 'elapsed_seconds': result['seconds']}
        for name in names:
            c = result['candidates'][name]
            row[name+'_success'] = c['success']
            row[name+'_roi_success'] = json.dumps(c['roi_success'].tolist())
            row[name+'_reasons'] = json.dumps(c['reasons'])
        rows.append(row)
        if (i+1) % 40 == 0:
            print(json.dumps({'processed': i+1, 'seconds': time.perf_counter()-started}), flush=True)

    # Only now open the original cached arrays to compare, after every raw image ran.
    actual = dict(index=np.arange(len(manifest)), points=ex.points, regions=ex.regions, ids=ex.ids,
                  boxes=np.array(module.BOXES), roi_names=np.array(module.NAMES), centroids=ex.centroids,
                  candidate_names=np.array(names), ncc_shifts=np.array([r['ncc_shifts'] for r in all_results]),
                  ncc_scores=np.array([r['ncc_scores'] for r in all_results]))
    fields = [('D','D'), ('centroidD','centroidD'), ('H','H'), ('success','success'),
              ('roi_success','roi_success'), ('ecc','ecc'), ('photometric_rmse','rmse'),
              ('gain','gain'), ('offset','offset')]
    for name in names:
        for field, target in fields:
            actual[target+'_'+name] = np.array([r['candidates'][name][field] for r in all_results])
    with np.load(SOURCE/'features.npz', allow_pickle=False) as cached:
        assert set(actual) == set(cached.files)
        comparisons = {name: compare(value, cached[name]) for name, value in actual.items()}
    manifest_success_equal = {}
    for name in names:
        manifest_success_equal[name] = all(
            bool(r[name+'_success']) == (m[name+'_success'].lower() == 'true') and
            json.loads(r[name+'_roi_success']) == json.loads(m[name+'_roi_success']) and
            json.loads(r[name+'_reasons']) == json.loads(m[name+'_reasons'])
            for r, m in zip(rows, manifest))

    tree = ast.parse((SOURCE/'extractor.py').read_text(encoding='utf-8'))
    imports = sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                     {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
    hash_groups = {}
    for row in rows:
        hash_groups.setdefault(row['sha256'], []).append(row['index'])
    final_hashes = {str(p): digest(p) for p in audited_files}
    audit = {
        'audit_scope': '341 raw image byte replays, extractor and cache integrity only; model training reviewed separately',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'n_requested': len(manifest), 'n_read_raw': len(rows), 'n_missing_or_exception': len(failures),
        'input_path_manifest': str(SOURCE/'manifest.csv'), 'independent_source_manifest': str(reference_path),
        'all_raw_hashes_match_both_manifests': all(r['raw_hash_matches_manifest'] and r['raw_hash_matches_source_manifest'] for r in rows),
        'manifest_row_order_matches_source': all(alignment),
        'full_input_paths_and_sha_file': str(HERE/'raw_inputs.csv'),
        'duplicate_raw_sha_groups': [v for v in hash_groups.values() if len(v)>1],
        'center_path': str(module.CENTER_PATH), 'center_sha256': center_hash,
        'center_protocol_hash_match': center_hash == protocol['center_sha256'],
        'source_and_cache_sha256_before': initial_hashes, 'source_and_cache_sha256_after': final_hashes,
        'audited_inputs_unchanged': initial_hashes == final_hashes,
        'audit_code_sha256': digest(__file__), 'python_image_extractor_imports': imports,
        'extractor_bound_signature': str(inspect.signature(ex.extract_bytes)),
        'extractor_read_guard': {'all_341_passed': True, 'forbidden_during_extract': ['builtins.open','io.open','Path.read_bytes','Path.read_text','numpy.load'],
                                 'fixed_center_and_protocol_loaded_before_guard': True},
        'code_review': {
            'current_input': 'Only bytes argument; fixed center/protocol read once, fixed ROI grid, NCC and ECC use decoded intensity arrays.',
            'no_filename_command_date_or_repeat_argument': True,
            'no_old_QR_displacement_import_or_read_in_extractor': True,
            'measure_loader_caveat': 'measure.py calls legacy data.load(), which opens old QR D to assemble/validate the shared manifest. It discards arrays and passes only Path(meta.path).read_bytes() to extract_bytes. This audit bypasses that loader and reproduces every feature directly.',
            'center_selected_ROI_claim': 'Protocol records selection from center only. Static code and byte replay cannot independently prove historical ROI selection decisions.',
            'virtual_grid': '45 coordinates are 3x3 projections of 5 fitted ROI warps, not 45 independently detected physical corners.'
        },
        'array_comparisons': comparisons,
        'all_cached_arrays_reproduced_exactly': all(x['equal'] for x in comparisons.values()),
        'manifest_success_roi_reasons_reproduced': manifest_success_equal,
        'successes': {name: int(actual['success_'+name].sum()) for name in names},
        'failed_images': {name: int((~actual['success_'+name]).sum()) for name in names},
        'failed_rois': {name: int((~actual['roi_success_'+name]).sum()) for name in names},
        'cv2_version': cv2.__version__, 'seconds': time.perf_counter()-started,
        'limits': ['Already exposed development data; no independent generalization claim.',
                   'Exact replay excludes stale or swapped cached measurements but does not prove every repeated bolt match is physically correct.',
                   'Unknown external depths, object rigidity and robot/camera geometry remain unverified.',
                   'This audit does not train a decoder or verify its folds, parameters or prediction scores.']
    }
    np.savez_compressed(HERE/'raw_reextracted_features.npz', **actual)
    with (HERE/'raw_inputs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (HERE/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['n_read_raw','all_raw_hashes_match_both_manifests','audited_inputs_unchanged',
                    'all_cached_arrays_reproduced_exactly','successes','failed_images','seconds']},ensure_ascii=False),flush=True)
    assert audit['all_cached_arrays_reproduced_exactly']
    assert audit['audited_inputs_unchanged']
    assert all(manifest_success_equal.values())


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
