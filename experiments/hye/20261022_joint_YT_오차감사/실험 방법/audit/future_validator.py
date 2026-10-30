"""Sealed future-image evaluation. No labels are accepted by the inference stage.

freeze -> infer -> score are separate commands. Outputs use exclusive creation.
This is an audit guard, not a cryptographic proof that an operator hid all labels.
Actual capture independence still requires documented human acquisition history.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, importlib.metadata, importlib.util, json, math, re, sys
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGES = ['numpy', 'scipy', 'opencv-python', 'scikit-learn', 'pandas', 'joblib', 'torch', 'threadpoolctl']
MANIFEST_COLUMNS = {'sample_id', 'image_path', 'capture_session_id', 'captured_at_utc'}

def now(): return datetime.now(timezone.utc).isoformat()

def date(value):
    x = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if x.tzinfo is None: raise ValueError('Timestamp must include its UTC offset.')
    return x.astimezone(timezone.utc)

def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def exclusive_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

def versions():
    result = {'python': sys.version.split()[0], 'cv2_runtime': cv2.__version__}
    for name in PACKAGES:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = None
    return result

def frozen_file(path, role):
    p = Path(path).resolve()
    return dict(role=role, path=str(p), sha256=sha(p))

def pixel_digest(raw):
    try: im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error: return None, None
    if im is None: return None, None
    return sha_bytes(str(im.shape).encode() + im.tobytes()), list(im.shape)

def freeze(args):
    if args.planned_count < 1: raise ValueError('Predeclare a positive planned sample count.')
    files = [frozen_file(args.model, 'model'), frozen_file(args.inference, 'inference'), frozen_file(args.reference, 'reference'), frozen_file(args.exposed, 'exposed_registry'), frozen_file(__file__, 'validator')]
    files += [frozen_file(path, 'dependency') for path in args.dependency]
    if getattr(args, 'dependency_manifest', None):
        snapshot = read_json(args.dependency_manifest)
        for record in snapshot['files']:
            if sha(record['path']) != record['sha256']: raise ValueError('Dependency snapshot is stale: ' + record['path'])
            if record['path'] not in {r['path'] for r in files}: files.append(frozen_file(record['path'], 'dependency'))
    protocol = dict(schema='yt-future-evaluation-v1', status='FROZEN; NO INDEPENDENT TEST RESULT YET', purpose=args.purpose, created_at_utc=now(), planned_count=args.planned_count, sampling_plan=args.sampling_plan, tolerance_Y=.09, tolerance_T=.09, threshold_numerical_epsilon=1e-12, expected_image_shape=[args.height, args.width], files=files, runtime=versions(), inference_contract='predict_bytes(raw_bytes, model_path=...) -> dict(status=ok, Y=float, T=float). No path, sample ID, date, command or truth is passed to inference.', score_contract='One sealed prediction batch, then a separate truth CSV. Measurement failures and abstentions remain in the total denominator. No labels are opened by infer.', limitations=['Hashes reject exact byte or decoded grayscale copies, not crops, augmentations, near-duplicates, or undisclosed prior exposure.', 'The complete imported inference dependency closure must be supplied with --dependency before freezing.', 'Capture history and label concealment attestations are required but cannot be proven by this program.', 'A finite batch passing both 0.09 limits is evidence for that batch, not a universal future guarantee.'])
    exclusive_json(args.output, protocol)
    return dict(protocol=str(Path(args.output).resolve()), sha256=sha(args.output), status=protocol['status'])

def checked_protocol(path):
    protocol = read_json(path)
    if protocol['schema'] != 'yt-future-evaluation-v1': raise ValueError('Unsupported protocol schema.')
    for record in protocol['files']:
        if sha(record['path']) != record['sha256']: raise ValueError('Frozen file changed: ' + record['path'])
    if versions() != protocol['runtime']: raise ValueError('Frozen runtime package versions changed.')
    return protocol

def role_file(protocol, role):
    matches = [x['path'] for x in protocol['files'] if x['role'] == role]
    if len(matches) != 1: raise ValueError('Expected one frozen ' + role)
    return matches[0]

def check_batch(protocol, manifest_path, attestation_path):
    columns, rows = read_csv(manifest_path)
    if set(columns or []) != MANIFEST_COLUMNS: raise ValueError('Manifest must contain exactly ' + ', '.join(sorted(MANIFEST_COLUMNS)) + '; labels belong only in truth.csv.')
    if len(rows) != protocol['planned_count']: raise ValueError('Manifest row count differs from the frozen planned count; no post-hoc omission or early stopping.')
    statement = read_json(attestation_path)
    for key in ['no_prior_development_use', 'labels_hidden_from_prediction_operator', 'complete_predeclared_batch', 'dependency_closure_reviewed']:
        if statement.get(key) is not True: raise ValueError('Required acquisition/operator attestation missing: ' + key)
    if not statement.get('responsible_person') or not statement.get('capture_record'):
        raise ValueError('Name a responsible person and the capture record.')
    real = protocol['purpose'] == 'future_real_capture'
    if real and statement.get('real_camera_captures') is not True: raise ValueError('Synthetic or transformed development images are not an independent real-capture test.')
    _, exposed = read_csv(role_file(protocol, 'exposed_registry'))
    old_bytes = {r['file_sha256'] for r in exposed}
    old_pixels = {r['grayscale_pixel_sha256'] for r in exposed}
    seen_ids, seen_bytes, seen_pixels, records = set(), set(), set(), []
    base = Path(manifest_path).resolve().parent
    for row in rows:
        sample = row['sample_id']
        if re.fullmatch('[0-9a-f]{32}', sample) is None or sample in seen_ids: raise ValueError('Use unique opaque 32-character hexadecimal sample IDs.')
        seen_ids.add(sample)
        captured = date(row['captured_at_utc'])
        if real and captured <= date(protocol['created_at_utc']): raise ValueError('This prospective protocol requires capture after the model freeze.')
        if real and captured > datetime.now(timezone.utc): raise ValueError('Capture timestamp lies in the future.')
        if not row['capture_session_id']: raise ValueError('Capture session is required.')
        path = Path(row['image_path'])
        path = path if path.is_absolute() else base / path
        raw = path.read_bytes()
        file_hash = sha_bytes(raw)
        pixels, shape = pixel_digest(raw)
        if file_hash in old_bytes or (pixels is not None and pixels in old_pixels): raise ValueError('Known development image or pixel-equivalent conversion: ' + sample)
        if file_hash in seen_bytes or (pixels is not None and pixels in seen_pixels): raise ValueError('Duplicate image within prospective batch: ' + sample)
        seen_bytes.add(file_hash)
        if pixels is not None: seen_pixels.add(pixels)
        preflight_failure = None
        if shape is None: preflight_failure = 'Image cannot be decoded.'
        elif shape != protocol['expected_image_shape']: preflight_failure = 'Unexpected image shape: ' + str(shape)
        records.append(dict(sample_id=sample, image_path=str(path.resolve()), capture_session_id=row['capture_session_id'], captured_at_utc=row['captured_at_utc'], file_sha256=file_hash, grayscale_pixel_sha256=pixels, shape=shape, preflight_failure=preflight_failure))
    return records, statement

def infer(args):
    protocol = checked_protocol(args.protocol)
    records, statement = check_batch(protocol, args.manifest, args.attestation)
    output = Path(args.output).resolve()
    # Claim the batch directory before importing or running the frozen predictor.
    output.mkdir(parents=True, exist_ok=False)
    exclusive_json(output / 'started.json', dict(started_at_utc=now(), protocol_path=str(Path(args.protocol).resolve()), protocol_sha256=sha(args.protocol), manifest_sha256=sha(args.manifest), attestation=statement, batch=records))
    module, load_error = None, None
    try:
        spec = importlib.util.spec_from_file_location('_frozen_future_predict', role_file(protocol, 'inference'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc: load_error = f'Frozen predictor could not load: {type(exc).__name__}: {exc}'
    predictions = []
    with (output / 'predictions.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'status', 'predY', 'predT', 'error'])
        writer.writeheader()
        for record in records:
            row = dict(sample_id=record['sample_id'], status='failed', predY='', predT='', error='')
            try:
                if load_error: raise RuntimeError(load_error)
                if record['preflight_failure']: raise ValueError(record['preflight_failure'])
                raw = Path(record['image_path']).read_bytes()
                if sha_bytes(raw) != record['file_sha256']: raise ValueError('Image changed after preflight.')
                result = module.predict_bytes(raw, model_path=role_file(protocol, 'model'))
                if result.get('status') != 'ok': raise ValueError(str(result.get('error', 'Predictor abstained or failed.')))
                y, t = float(result['Y']), float(result['T'])
                if not math.isfinite(y) or not math.isfinite(t): raise ValueError('Non-finite prediction.')
                row.update(status='ok', predY=y, predT=t)
            except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'
            predictions.append(row)
            writer.writerow(row)
            f.flush()
    checked_protocol(args.protocol)
    receipt = dict(schema='yt-sealed-predictions-v1', completed_at_utc=now(), protocol_path=str(Path(args.protocol).resolve()), protocol_sha256=sha(args.protocol), predictions_sha256=sha(output / 'predictions.csv'), started_sha256=sha(output / 'started.json'), n=len(predictions), failures=sum(r['status'] != 'ok' for r in predictions), truth_opened=False, purpose=protocol['purpose'])
    exclusive_json(output / 'receipt.json', receipt)
    return receipt

def score(args):
    batch = Path(args.batch).resolve()
    receipt = read_json(batch / 'receipt.json')
    protocol = checked_protocol(receipt['protocol_path'])
    if sha(receipt['protocol_path']) != receipt['protocol_sha256']: raise ValueError('Protocol was changed after prediction.')
    for filename, key in [('predictions.csv', 'predictions_sha256'), ('started.json', 'started_sha256')]:
        if sha(batch / filename) != receipt[key]: raise ValueError('Sealed batch file changed: ' + filename)
    truth = Path(args.truth).resolve()
    # One scoring attempt per sealed batch. Even an invalid truth file leaves the audit marker.
    exclusive_json(batch / 'scoring_started.json', dict(started_at_utc=now(), truth_path=str(truth), truth_sha256=sha(truth)))
    columns, labels = read_csv(truth)
    if set(columns or []) != {'sample_id', 'Y', 'T'}: raise ValueError('Truth CSV requires exactly sample_id,Y,T.')
    _, predictions = read_csv(batch / 'predictions.csv')
    pids = [r['sample_id'] for r in predictions]
    if len(labels) != len(predictions) or len({r['sample_id'] for r in labels}) != len(labels) or {r['sample_id'] for r in labels} != set(pids): raise ValueError('Truth must match every sealed prediction ID once, including failures.')
    actual = {r['sample_id']: (float(r['Y']), float(r['T'])) for r in labels}
    if not all(math.isfinite(x) for v in actual.values() for x in v): raise ValueError('Truth values must be finite.')
    scored, errors = [], []
    threshold = np.array([protocol['tolerance_Y'], protocol['tolerance_T']]) + protocol['threshold_numerical_epsilon']
    for row in predictions:
        y, t = actual[row['sample_id']]
        record = dict(row, Y=y, T=t, eY=None, eT=None, passed=False)
        if row['status'] == 'ok':
            error = np.abs(np.array([float(row['predY']), float(row['predT'])]) - np.array([y, t]))
            if not np.isfinite(error).all(): raise ValueError('Invalid sealed numeric prediction.')
            record.update(eY=float(error[0]), eT=float(error[1]), passed=bool((error <= threshold).all()))
            errors.append(error)
        scored.append(record)
    e = np.array(errors).reshape(-1, 2)
    passed = sum(row['passed'] for row in scored)
    result = dict(evaluation='prospective_real_capture_batch' if protocol['purpose'] == 'future_real_capture' else 'SYNTHETIC VALIDATOR SELF-TEST; NOT ROBOT PERFORMANCE', scored_at_utc=now(), n=len(scored), successful_predictions=len(e), failures=len(scored)-len(e), passed=passed, joint_tolerance_pass_rate=passed/len(scored), Y_MAE=float(e[:,0].mean()) if len(e) else None, Y_max=float(e[:,0].max()) if len(e) else None, T_MAE=float(e[:,1].mean()) if len(e) else None, T_max=float(e[:,1].max()) if len(e) else None, entire_batch_within_both_tolerances=passed == len(scored), metric_note='Errors summarize successful predictions; failures/abstentions remain in total n and count as not passed.', protocol_sha256=receipt['protocol_sha256'], prediction_sha256=receipt['predictions_sha256'], truth_sha256=sha(truth), limitations=protocol['limitations'], rows=scored)
    exclusive_json(batch / 'score.json', result)
    started = read_json(batch / 'started.json')
    with (batch / 'now_exposed_images.csv').open('x', encoding='utf-8', newline='') as f:
        fields = ['path', 'file_sha256', 'grayscale_pixel_sha256', 'development_status']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in started['batch']:
            writer.writerow(dict(path=r['image_path'], file_sha256=r['file_sha256'], grayscale_pixel_sha256=r['grayscale_pixel_sha256'], development_status='exposed after one evaluation; merge into cumulative registry before any future freeze'))
    return {k: v for k, v in result.items() if k not in ['rows', 'limitations']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze')
    for name in ['model', 'inference', 'reference', 'output', 'sampling-plan']: p.add_argument('--' + name, required=True)
    p.add_argument('--exposed', default=str(HERE / 'exposed_images.csv'))
    p.add_argument('--dependency', action='append', default=[])
    p.add_argument('--dependency-manifest', help='Dependency snapshot JSON with files[path,sha256].')
    p.add_argument('--planned-count', type=int, required=True)
    p.add_argument('--height', type=int, default=2160)
    p.add_argument('--width', type=int, default=3840)
    p.add_argument('--purpose', choices=['future_real_capture', 'synthetic_self_test'], default='future_real_capture')
    p = sub.add_parser('infer')
    for name in ['protocol', 'manifest', 'attestation', 'output']: p.add_argument('--' + name, required=True)
    p = sub.add_parser('score')
    p.add_argument('--batch', required=True)
    p.add_argument('--truth', required=True)
    args = parser.parse_args()
    result = {'freeze': freeze, 'infer': infer, 'score': score}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

if __name__ == '__main__': main()
