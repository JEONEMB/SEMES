"""Two-image fixed-center QR630 + screw39 experimental inference, no filename labels."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B/'measurement'),str(B)]
from spread import SpreadPairExtractor
from extract_pair import PairExtractor
from infer import solve_calibrated,clean
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def p12(points,d):
    u=(points[:,0]-1906)/486;v=(points[:,1]-1070)/122;A=np.c_[u*0+1,u,v,u*u,u*v,v*v]
    return np.linalg.lstsq(A,d,rcond=None)[0].ravel()
class Predictor:
    def __init__(self,center):
        meta=json.loads((H/'reference.json').read_text())
        if sha(center)!=meta['sha256']:raise ValueError('Calibration center changed; requires recalibration.')
        self.q=SpreadPairExtractor(center,str(H/'qr_features.npz'));self.s=PairExtractor(center,'gn17',H/'screw_features.npz',True)
        self.qzero=self.q.extract(center)['D'];self.szero=self.s.extract(center)['D'];self.model=json.loads((H/'model.json').read_text())
    def extract(self,current):
        q=self.q.extract(current);s=self.s.extract(current);qd=q['D'].astype(float)-self.qzero;sd=s['D'].astype(float)-self.szero
        X=np.r_[p12(q['points'],qd),np.mean(sd[s['regions']==3],axis=0),np.mean(sd[s['regions']==4],axis=0)]
        return X,int(q['valid'].sum()),int(s['valid'].sum())
    def predict(self,current):
        X,qv,sv=self.extract(current);r=clean(solve_calibrated(X,self.model));v=r['value'];a=r['axis']
        return dict(axis=a,sign=int(np.sign(v)),value=v,Y=v if a=='Y' else 0.,T=v if a=='T' else 0.,qr_valid=qv,screw_valid=sv,alternatives=r['alternatives'],actual_error=None,scope='fixed calibration, QR630 + 2 screws, six learned origins; NOT QR-only; independent capture not validated')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--center',required=True);ap.add_argument('--current',required=True);ap.add_argument('--out');a=ap.parse_args();r=Predictor(a.center).predict(a.current);txt=json.dumps(r,ensure_ascii=False,indent=2)
    if a.out:Path(a.out).write_text(txt,encoding='utf-8')
    print(txt)
