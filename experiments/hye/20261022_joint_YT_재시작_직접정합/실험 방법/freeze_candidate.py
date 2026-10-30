"""Record reproducible candidate and exposed-image hashes without changing models."""
from pathlib import Path
import sys,json,hashlib,platform,datetime
import numpy,cv2,scipy,sklearn,joblib
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path.insert(0,str(H.parent/'joint_yt_generalization_20261022'))
from data import load

def main():
    m,_=load()
    paths=[H/x for x in ['qr_external_model.joblib','external_model.joblib','predict.py','direct_models.py','direct/extractor.py','parallax/extractor.py','parallax/protocol.json']]
    paths += [H.parent/'joint_yt_generalization_20261022/forward_model.py',H.parent/'joint_yt_erroraudit_20261022/robust/robust_model.py',ROOT/'hw-analysis/data/hw/bmp/20261014/center.bmp',ROOT/'joint_center_630_20261022/assets/reference.npz']
    versions={k:v for k,v in [('python',platform.python_version()),('numpy',numpy.__version__),('opencv',cv2.__version__),('scipy',scipy.__version__),('sklearn',sklearn.__version__),('joblib',joblib.__version__)]}
    out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Candidate freeze only; all341 images are development-exposed; no independent acceptance test available',primary='qr_external',ablation='external',model_selection='affine_g5/linear in every outer and full-data internal selection',training_rows=341,inference_arguments='current image bytes; fixed original center and ROI protocol; no labels/date/repeat',files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},environment=versions,known_development_image_sha256=m.sha256.tolist(),future_evaluation='Keep this candidate fixed. Infer without labels on genuinely new nonduplicate capture data, then score Y/T errors and all-case dual0.09 tolerance; actual closed-loop return needs measurement after robot correction. Re-splitting these341 images is not an independent test.')
    (H/'frozen_manifest.json').write_text(json.dumps(out,indent=2));print('frozen',len(paths),'dependencies,341 known image hashes')

if __name__=='__main__':main()
