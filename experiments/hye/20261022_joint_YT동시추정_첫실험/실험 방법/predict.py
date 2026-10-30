"""Image-byte inference; no filename, command, session or true axis input."""
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
import cv2
H=Path(__file__).resolve().parent; ROOT=H.parents[1]
sys.path.insert(0,str(ROOT/'motion_web_20261022'))
from structural import StructuralExtractor
from experiment import predict_joint

def predict(raw,center_raw):
    expected=json.loads((ROOT/'motion_web_20261022/structural_assets/provenance.json').read_text())['center_sha256']
    if hashlib.sha256(center_raw).hexdigest()!=expected:raise ValueError('fixed calibrated center required')
    center=cv2.imdecode(np.frombuffer(center_raw,np.uint8),0)
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
    if image is None or image.shape!=center.shape:raise ValueError('invalid image or dimensions')
    r=StructuralExtractor(center).extract(image)
    p=r['points'];u=(p[:,0]-1906)/486;v=(p[:,1]-1070)/122
    design=np.c_[np.ones(len(p)),u,v,u*u,u*v,v*v]
    x=np.linalg.lstsq(design,r['D'],rcond=None)[0].ravel()
    model=dict(np.load(H/'joint_model.npz'));model['quad']=bool(model['quad'])
    y=predict_joint(x[None],model)[0]
    if np.array_equal(image,center):y=np.zeros(2)
    return dict(Y=float(y[0]),T=float(y[1]),valid=int(r['valid'].sum()),total=len(p),feature=x.tolist(),model='joint_yt_experimental',mixed_pose_validated=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--current',required=True);ap.add_argument('--center',required=True);ap.add_argument('--out');a=ap.parse_args()
    result=predict(Path(a.current).read_bytes(),Path(a.center).read_bytes());s=json.dumps(result,indent=2)
    if a.out:Path(a.out).write_text(s)
    print(s)
