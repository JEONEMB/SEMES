"""Experimental improved joint model; bytes-only inference."""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
import cv2,joblib
from engine import extract,H,B
sys.path.insert(0,str(B.parent/'motion_web_20261022'))
from structural import StructuralExtractor

def predict(current_raw,center_raw,model_path=None):
    provenance=json.loads((B.parent/'motion_web_20261022/structural_assets/provenance.json').read_text())
    if hashlib.sha256(center_raw).hexdigest()!=provenance['center_sha256']:raise ValueError('Fixed calibrated center is required')
    image=cv2.imdecode(np.frombuffer(current_raw,np.uint8),0);center=cv2.imdecode(np.frombuffer(center_raw,np.uint8),0)
    if image is None or image.shape!=center.shape:raise ValueError('Invalid image or dimensions')
    bundle=joblib.load(model_path or H/'model.joblib');c=bundle['config']
    r=StructuralExtractor(center).extract(image)
    assert np.array_equal(r['ids'],bundle['ids'])
    q=r['quality'][:,:3]
    if 'measurement' in c:x=extract(r['points'],r['D'],q,c['measurement'],c['feature'])
    else:
        from fullfield import field_features
        x=field_features(r['points'],r['D'],q,c['feature'])
    y=bundle['model'].predict(x[None])[0]
    if np.array_equal(image,center):y=np.zeros(2)
    return dict(Y=float(y[0]),T=float(y[1]),model='joint_yt_improved_experimental',config=c,valid=int(r['valid'].sum()),total=630,feature=x.tolist(),mixed_photographs_validated=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--current',required=True);ap.add_argument('--center',required=True);ap.add_argument('--model');ap.add_argument('--out');a=ap.parse_args()
    result=predict(Path(a.current).read_bytes(),Path(a.center).read_bytes(),a.model);s=json.dumps(result,indent=2)
    if a.out:Path(a.out).write_text(s)
    print(s)
