"""Image-only inference for the separately trained Y/T separation model.

Reuses the preserved fixed630 SciPy measurement implementation read-only.
The default model is this experiment's pass_first_model.joblib, not a prior model.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'joint_yt_generalization_20261022'
_spec=importlib.util.spec_from_file_location('_preserved_measurement_predict',PREVIOUS/'predict.py')
_previous=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_previous)
# The older measurement module adjusts sys.path; restore new model import priority.
if str(HERE) in sys.path:sys.path.remove(str(HERE))
sys.path.insert(0,str(HERE))
import learned_model  # noqa: F401; importable class identity for the new joblib bundle


def _load_bundle(model_path=None):
    path=Path(model_path) if model_path else HERE/'pass_first_model.joblib'
    bundle=joblib.load(path)
    if not {'model','config','ids','points'}.issubset(bundle):
        raise ValueError('Expected a joint separation model bundle.')
    if not isinstance(bundle['model'],learned_model.LearnedSeparation):
        raise ValueError('This entry point requires this experiment\'s LearnedSeparation model.')
    return bundle,hashlib.sha256(path.read_bytes()).hexdigest()


def predict_bytes(image_bytes,model_path=None):
    """Current image bytes -> Y/T estimate and opposite correction, no actuation."""
    started=time.perf_counter();bundle,digest=_load_bundle(model_path)
    measured=_previous._measure_bytes(image_bytes,bundle)
    with threadpool_limits(1):estimate=np.asarray(bundle['model'].predict(measured['D'][None]))[0]
    if estimate.shape!=(2,) or not np.isfinite(estimate).all():
        raise ValueError('Model did not return two finite estimates.')
    return dict(status='ok',validation_status='experimental_goal_not_met',Y=float(estimate[0]),T=float(estimate[1]),
                correction_Y=float(-estimate[0]),correction_T=float(-estimate[1]),actuation_executed=False,
                model='joint_yt_disentangle_pass_first',config=bundle['config'],model_sha256=digest,
                image_sha256=hashlib.sha256(image_bytes).hexdigest(),reference_sha256=_previous.CENTER_SHA256,
                measurement='fixed630_blurred_boundary_scipy_expanded_ncc',point_count=630,
                diagnostic_valid_count=int(np.sum(measured['valid'])),ncc_min=float(np.min(measured['coarse_score'])),
                measurement_seconds=float(measured['runtime_seconds']),total_seconds=time.perf_counter()-started,
                evaluation='No ground truth supplied; error or tolerance pass is not assessed. Saved model uses full calibration data.')


def predict_image(image_path,model_path=None):
    """The path is used only to read image bytes, never as a prediction feature."""
    return predict_bytes(Path(image_path).read_bytes(),model_path)


def main():
    parser=argparse.ArgumentParser(description='Experimental Y/T separation from an image and the original fixed center.')
    parser.add_argument('image');parser.add_argument('--model',help='Defaults to new pass_first_model.joblib')
    parser.add_argument('--output',help='Optional JSON output file')
    args=parser.parse_args()
    try:result=predict_image(args.image,args.model);code=0
    except Exception as exc:result=dict(status='error',error=str(exc),actuation_executed=False);code=1
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    print(text);raise SystemExit(code)


if __name__=='__main__':main()
