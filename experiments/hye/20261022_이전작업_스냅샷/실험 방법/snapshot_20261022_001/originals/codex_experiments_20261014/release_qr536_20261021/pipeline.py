"""Fixed-center QR-only536 interior structural points. No target label inputs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B/'measurement'),str(B)]
from spread import SpreadPairExtractor
from infer import solve_calibrated,clean
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def p12(points,d):
    u=(points[:,0]-1906)/486;v=(points[:,1]-1070)/122;A=np.c_[u*0+1,u,v,u*u,u*v,v*v]
    return np.linalg.lstsq(A,d,rcond=None)[0].ravel()
class Predictor:
    def __init__(self,center):
        ref=json.loads((H/'reference.json').read_text())
        if sha(center)!=ref['sha256']:raise ValueError('Different center requires new initialization and calibration validation.')
        self.q=SpreadPairExtractor(center,str(H/'features630.npz'));self.zero=self.q.extract(center)['D'];self.mask=np.load(H/'mask536.npy');self.model=json.loads((H/'model.json').read_text())
    def extract(self,current):
        q=self.q.extract(current);d=q['D'].astype(float)-self.zero
        return p12(q['points'][self.mask],d[self.mask]),int(q['valid'][self.mask].sum())
    def predict(self,current):
        X,valid=self.extract(current);r=clean(solve_calibrated(X,self.model));a=r['axis'];v=r['value']
        return dict(axis=a,sign=int(np.sign(v)),value=v,Y=v if a=='Y' else 0.,T=v if a=='T' else 0.,valid_points=valid,total_points=536,alternatives=r['alternatives'],actual_error=None,confidence_probability=None,scope='QR-only536 interior structural points; fixed camera/calibration; six learned origin vectors; post-selection development result, independent new capture not validated')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--center',required=True);ap.add_argument('--current',required=True);ap.add_argument('--out');a=ap.parse_args();r=Predictor(a.center).predict(a.current);txt=json.dumps(r,ensure_ascii=False,indent=2)
    if a.out:Path(a.out).write_text(txt,encoding='utf-8')
    print(txt)
