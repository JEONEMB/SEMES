"""Record the image inference dependency closure; does not run image inference.

This is a model snapshot, not a completed independent-test protocol. Before a
prospective test, freeze its sample count/sampling plan using future_validator.
"""
from pathlib import Path
import importlib.util, json, sys
import joblib
import future_validator as v

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = HERE.parents[2]

def main():
    inference = EXPERIMENT / 'predict.py'
    spec = importlib.util.spec_from_file_location('_snapshot_inference', inference)
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    model = Path(predictor.DEFAULT_MODEL).resolve()
    bundle = joblib.load(model)
    files = {inference.resolve(), model, Path(predictor.measurement.__file__).resolve(), Path(predictor.measurement.CENTER_PATH).resolve(), Path(__file__).resolve(), Path(v.__file__).resolve(), (ROOT/'ALGORITHM_RULES.md').resolve(), (HERE/'exposed_images.csv').resolve()}
    for module in list(sys.modules.values()):
        location = getattr(module, '__file__', None)
        if location:
            path = Path(location).resolve()
            if path.is_relative_to(ROOT) and path.is_file(): files.add(path)
    # Non-code files read at extractor module import are part of its fixed 630 IDs.
    files.update((ROOT/'joint_center_630_20261022/assets'/name).resolve() for name in ['corners.json','reference.npz'])
    records = [v.frozen_file(p, 'dependency_snapshot') for p in sorted(files, key=str)]
    report = dict(status='MODEL SNAPSHOT ONLY; independent test not run; sampling plan/count must be frozen before new capture', created_at_utc=v.now(), model=str(model), inference=str(inference), reference=str(predictor.measurement.CENTER_PATH), model_config=bundle['config'], model_train_count=len(bundle['train_indices']) if 'train_indices' in bundle else 341, default_model_is_adaptively_selected_development_candidate=True, files=records, runtime=v.versions(), closure_method='Import root predictor, load default model, enumerate imported project Python modules, add fixed corners/reference assets, model, original center, rules, exposure registry and validator.', limitations=['This snapshot is not independent performance evidence.', 'Image extraction is not run by this snapshot command.', 'Dynamically loaded future dependencies not reached by import/model loading require explicit review before freeze.'])
    (HERE/'dependency_snapshot.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(files=len(records), model=str(model), model_sha256=v.sha(model), status=report['status']), indent=2))

if __name__=='__main__': main()
