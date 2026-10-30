"""Small synthetic fixtures test leakage guards; no robot performance is claimed."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import csv, json, tempfile
import cv2, numpy as np
import future_validator as v

HERE = Path(__file__).resolve().parent

def csvfile(path, fields, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='synthetic_validator_', dir=HERE) as folder:
        h = Path(folder)
        assert h.resolve().is_relative_to(HERE.resolve())
        def picture(name, value):
            p = h / name
            p.write_bytes(cv2.imencode(p.suffix, np.full((8,8), value, np.uint8))[1].tobytes())
            return p
        known = picture('development.bmp', 10)
        conversion = picture('converted.png', 10)
        fresh = picture('synthetic_new.bmp', 20)
        corrupt = h / 'corrupt.bmp'; corrupt.write_bytes(b'not an image')
        model = h / 'model.bin'; model.write_bytes(b'mock model fixture')
        inference = h / 'mock_predict.py'
        inference.write_text("def predict_bytes(raw, model_path=None):\n    assert isinstance(raw, bytes)\n    return dict(status='ok', Y=0.0, T=0.0)\n", encoding='utf-8')
        registry = h / 'registry.csv'
        csvfile(registry, ['file_sha256','grayscale_pixel_sha256'], [dict(file_sha256=v.sha(known), grayscale_pixel_sha256=v.pixel_digest(known.read_bytes())[0])])
        args = SimpleNamespace(model=model, inference=inference, reference=known, exposed=registry, dependency=[], dependency_manifest=None, output=h/'protocol.json', planned_count=2, sampling_plan='Synthetic unit fixtures, never actual camera evidence', height=8, width=8, purpose='synthetic_self_test')
        v.freeze(args)
        protocol = v.checked_protocol(args.output)
        statement = dict(no_prior_development_use=True, labels_hidden_from_prediction_operator=True, complete_predeclared_batch=True, dependency_closure_reviewed=True, real_camera_captures=False, responsible_person='automated synthetic self-test', capture_record='generated constant 8x8 fixtures')
        attestation = h/'attestation.json'; v.exclusive_json(attestation, statement)
        manifest = h/'manifest.csv'
        def rows(paths):
            return [dict(sample_id=f'{i+1:032x}', image_path=str(p), capture_session_id='synthetic_fixture', captured_at_utc=datetime.now(timezone.utc).isoformat()) for i,p in enumerate(paths)]
        def rejected(name, func):
            try: func()
            except (ValueError, FileExistsError): checks.append(dict(check=name, passed=True)); return
            raise AssertionError('Expected rejection: '+name)
        for name, paths in [('known raw bytes rejected',[known,fresh]),('converted identical pixels rejected',[conversion,fresh]),('within batch duplicate rejected',[fresh,fresh]),('wrong sample count rejected',[fresh])]:
            csvfile(manifest, sorted(v.MANIFEST_COLUMNS), rows(paths))
            rejected(name, lambda: v.check_batch(protocol, manifest, attestation))
        csvfile(manifest, sorted(v.MANIFEST_COLUMNS)+['Y'], [dict(r,Y=0) for r in rows([fresh,corrupt])])
        rejected('labels in inference manifest rejected', lambda: v.check_batch(protocol, manifest, attestation))
        model.write_bytes(b'modified model')
        rejected('changed frozen model rejected', lambda: v.checked_protocol(args.output))
        model.write_bytes(b'mock model fixture')
        csvfile(manifest, sorted(v.MANIFEST_COLUMNS), rows([fresh,corrupt]))
        receipt = v.infer(SimpleNamespace(protocol=args.output, manifest=manifest, attestation=attestation, output=h/'batch'))
        assert receipt['n']==2 and receipt['failures']==1 and receipt['truth_opened'] is False
        checks.append(dict(check='corrupt image retained as prediction failure; labels absent during inference', passed=True))
        truth = h/'truth.csv'; csvfile(truth, ['sample_id','Y','T'], [dict(sample_id=f'{i+1:032x}', Y=0, T=0) for i in range(2)])
        result = v.score(SimpleNamespace(batch=h/'batch', truth=truth))
        assert result['n']==2 and result['failures']==1 and result['passed']==1 and result['joint_tolerance_pass_rate']==.5
        assert result['entire_batch_within_both_tolerances'] is False and 'SYNTHETIC' in result['evaluation']
        checks.append(dict(check='failure remains in denominator; 1/2 pass is 50 percent', passed=True))
        rejected('second scoring attempt rejected', lambda: v.score(SimpleNamespace(batch=h/'batch', truth=truth)))
        assert (h/'batch/now_exposed_images.csv').exists()
        checks.append(dict(check='evaluated photos exported to exposure registry', passed=True))
    report = dict(status='passed', scope='synthetic validator unit checks only; no fresh robot images evaluated', checks=checks)
    (HERE/'validator_selftest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__=='__main__': main()
