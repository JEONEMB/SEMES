"""Restored frozen six-origin model. Two image inputs only; labels are never read."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'measurement'))
from infer import solve_calibrated,clean
from spread import SpreadPairExtractor
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--center',required=True);ap.add_argument('--current',required=True);ap.add_argument('--out');args=ap.parse_args()
    if sha(args.center)!=json.loads((HERE/'reference.json').read_text(encoding='utf-8'))['sha256']:raise ValueError('Calibration center does not match')
    params=json.loads((HERE/'model.json').read_text(encoding='utf-8'))
    if sha(args.center)==sha(args.current):out=dict(axis='CENTER',value=0,Y=0,T=0,actual_error=None)
    else:
        r=SpreadPairExtractor(args.center,features=HERE/'features.npz').extract(args.current);ok=r['valid'];p=r['points'][ok];d=r['D'][ok]
        if ok.mean()<.95 or any(np.sum(ok&(r['regions']==q))<20 for q in range(3)):raise ValueError('Insufficient valid points')
        u=(p[:,0]-1906)/486;v=(p[:,1]-1070)/122;B=np.array([u*0+1,u,v,u*u,u*v,v*v]).T
        X=np.linalg.lstsq(B,d,rcond=None)[0].ravel()-params['zero_feature'];out=solve_calibrated(X,params)
        out.update(Y=out['value'] if out['axis']=='Y' else 0,T=out['value'] if out['axis']=='T' else 0,actual_error=None,confidence_probability=None,valid_points=int(ok.sum()))
    txt=json.dumps(clean(out),ensure_ascii=False,indent=2,allow_nan=False)
    if args.out:Path(args.out).write_text(txt,encoding='utf-8')
    print(txt)
if __name__=='__main__':main()
