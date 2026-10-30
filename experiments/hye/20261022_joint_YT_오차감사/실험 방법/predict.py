"""Single-image inference for the preserved experimental robustness candidate.

No filename, command, capture date, or repeat number is a model input.
The default is the NEW development candidate; no production algorithm is changed.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
PREVIOUS=H.parent/'joint_yt_generalization_20261022'
spec=importlib.util.spec_from_file_location('_unchanged_fixed630_measurement',PREVIOUS/'predict.py')
measurement=importlib.util.module_from_spec(spec);spec.loader.exec_module(measurement)
for p in [H,H/'robust',H.parent/'joint_yt_disentangle_20261022']:sys.path.insert(0,str(p))
from robust_model import RobustBaggedForward
from learned_model import LearnedSeparation
from phase_model import PhaseNuisance

DEFAULT_MODEL=H/'robust/exploratory_nested_model.joblib'

def predict_bytes(image_bytes,model_path=None):
    started=time.perf_counter();path=Path(model_path) if model_path else DEFAULT_MODEL
    bundle=joblib.load(path)
    if not {'model','config','ids','points'}.issubset(bundle):raise ValueError('Expected fixed630 model bundle.')
    if not isinstance(bundle['model'],(RobustBaggedForward,LearnedSeparation,PhaseNuisance)):raise ValueError('Unexpected model class.')
    observed=measurement._measure_bytes(image_bytes,bundle)
    with threadpool_limits(1):q=np.asarray(bundle['model'].predict(observed['D'][None]))[0]
    if q.shape!=(2,) or not np.isfinite(q).all():raise ValueError('Expected two finite estimates.')
    return dict(status='ok',Y=float(q[0]),T=float(q[1]),correction_Y=float(-q[0]),correction_T=float(-q[1]),
                model_path=str(path),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config=bundle['config'],
                image_sha256=hashlib.sha256(image_bytes).hexdigest(),reference_sha256=measurement.CENTER_SHA256,
                point_count=630,diagnostic_valid_count=int(observed['valid'].sum()),ncc_min=float(observed['coarse_score'].min()),
                measurement='unchanged_fixed630_blurred_boundary_scipy_expanded_ncc',measurement_seconds=float(observed['runtime_seconds']),
                total_seconds=time.perf_counter()-started,actuation_executed=False,
                evaluation='No truth provided: error is not assessed. This is a development candidate, not independently validated for new captures.')

def predict_image(path,model_path=None):return predict_bytes(Path(path).read_bytes(),model_path)

def main():
    parser=argparse.ArgumentParser(description='Experimental Y/T estimation from image bytes. Does not operate the robot.')
    parser.add_argument('image');parser.add_argument('--model');parser.add_argument('--output');args=parser.parse_args()
    try:r=predict_image(args.image,args.model);code=0
    except Exception as e:r=dict(status='error',error=str(e),actuation_executed=False);code=1
    text=json.dumps(r,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(text,encoding='utf-8')
    print(text);raise SystemExit(code)

if __name__=='__main__':main()
