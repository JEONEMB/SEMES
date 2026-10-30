"""Two-image inference for the seven recorded GLS comparison variants.

This does not replace the existing deployed infer.py. All experimental variants
require all 205 fixed tracks. Errors/accuracy need separately provided truth.
"""
from pathlib import Path
import argparse, json, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
from infer import solve_calibrated, sha, clean
from measurement.spread import SpreadPairExtractor

def run(center,current,method='spatial_gls_lw'):
    start=time.perf_counter()
    reference=json.loads((BASE/'reference.json').read_text(encoding='utf8'))
    if sha(center)!=reference['sha256']:
        raise ValueError('Use the calibration center; a new center needs recalibration.')
    params=json.loads((HERE/'models'/f'{method}.json').read_text(encoding='utf8'))
    ext=SpreadPairExtractor(center,features=BASE/'measurement/gn17_spread14_features.npz')
    r=ext.extract(current)
    if not r['valid'].all():
        return dict(status='insufficient_valid_points',valid_points=int(r['valid'].sum()),actual_error=None)
    # Only the fixed center's measured self-displacement is loaded; current-image
    # filename/truth and the training images' features are never looked up.
    center_d=np.load(HERE/'models'/'center_displacement.npy')
    d=r['D'].astype(float)-center_d
    if params['projection'] is None:
        x=(r['points'][:,0]-1906)/486;y=(r['points'][:,1]-1070)/122
        B=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
        X=np.linalg.lstsq(B,d,rcond=None)[0].ravel()
    else:
        X=np.asarray(params['projection'])@d.ravel()
    best=solve_calibrated(X,params)
    axis=best['axis'];v=float(best['value'])
    return clean(dict(status='prediction',method=method,axis=axis,sign=int(np.sign(v)),
        value=v,Y=v if axis=='Y' else 0.,T=v if axis=='T' else 0.,
        valid_points=int(r['valid'].sum()),runtime_seconds=time.perf_counter()-start,
        normalized_cost_margin=best['normalized_cost_margin'],confidence_probability=None,
        actual_error=None,training_scope='280 development images; six existing origin modes',
        alternatives=best['alternatives']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--center',required=True);p.add_argument('--current',required=True)
    p.add_argument('--method',default='spatial_gls_lw');p.add_argument('--out')
    a=p.parse_args();r=run(a.center,a.current,a.method)
    txt=json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)
    if a.out:Path(a.out).write_text(txt,encoding='utf8')
    print(txt)
