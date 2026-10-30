"""Two-image inference. Motion labels, filenames, times and run IDs are not inputs.

`calibrated-runs` models the six observed zero-origin modes. It is a development
model for that calibration distribution, not validated on a new acquisition run.
Per-image true error is only available if independent ground truth is supplied.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'measurement'))
sys.path.insert(0,str(HERE/'modeling'))
from spread import SpreadPairExtractor
from extract_pair import PairExtractor

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def solve_calibrated(X,params):
    W=np.array(params['W']);choices=[]
    for name,(axis,b,s,c) in params['branches'].items():
        b,s,c=map(np.array,[b,s,c]); z=W@(X-b);ss=W@s;cc=W@c
        v=float(z@ss/(ss@ss))
        for _ in range(8):
            j=ss+2*v*cc;r=z-v*ss-v*v*cc;v+=float(j@r/(j@j))
        cost=float(np.sum((z-v*ss-v*v*cc)**2))
        if v*(1 if name.endswith('+') else -1)<0:cost=float('inf')
        choices.append(dict(axis=axis,value=v,cost=cost,calibration_mode=name))
    choices.sort(key=lambda p:p['cost'])
    best=dict(choices[0]);other=min(p['cost'] for p in choices if p['axis']!=best['axis'])
    best['cost_other_axis']=other
    best['normalized_cost_margin']=(other-best['cost'])/(other+best['cost']+1e-12)
    best['alternatives']=choices
    return best

def run(center,current,mode='calibrated-runs'):
    start=time.perf_counter()
    ref=HERE/'reference.json'
    if ref.exists() and sha(center)!=json.loads(ref.read_text(encoding='utf8'))['sha256']:
        raise ValueError('This model requires the same calibration center image. Renaming/copying it is allowed; a different reference requires recalibration.')
    if sha(center)==sha(current):
        return dict(status='identical_reference',axis='CENTER',sign=0,value=0.,Y=0.,T=0.,mode=mode,actual_error=None,confidence_probability=None)
    if mode=='calibrated-runs':
        params=json.loads((HERE/'calibrated_runs/model.json').read_text(encoding='utf8'))
        ext=SpreadPairExtractor(center,features=HERE/params['point_feature_path'])
        r=ext.extract(current); p=r['points'];d=r['D'];v=r['valid'];reg=r['regions']
        if v.mean()<.95 or any(np.sum(v&(reg==j))<20 for j in range(3)):
            return dict(status='insufficient_valid_points',valid_fraction=float(v.mean()),actual_error=None)
        x=(p[v,0]-1906)/486;y=(p[v,1]-1070)/122
        B=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
        X=np.linalg.lstsq(B,d[v],rcond=None)[0].ravel()-np.array(params['zero_feature'])
        best=solve_calibrated(X,params)
        scope=params['scope']; validation='calibrated_runs/stress_metrics.csv'
    else:
        import pickle
        with (HERE/'modeling/deploy_model.pkl').open('rb') as f: bundle=pickle.load(f)
        measurement=bundle['measurement'];features=bundle.get('features_file')
        if measurement.startswith('gn17_spread'):
            feature_path=HERE/(features or f'measurement/{measurement}_features.npz')
            ext=SpreadPairExtractor(center,features=feature_path)
        else: ext=PairExtractor(center,measurement,features=HERE/(features or 'measurement/features.npz'))
        r=ext.extract(current);v=r['valid']
        if not v.all():return dict(status='insufficient_valid_points',valid_fraction=float(v.mean()),actual_error=None)
        model=bundle['model']
        # Training retains the original point order, including unused screw rows.
        # The QR-only extractor returns the initial QR rows of that same order.
        refdisp=np.load(HERE/'modeling/deploy_reference.npz')['D0']
        D=(r['D']-refdisp[:len(r['D'])])[None]
        result=model.predict(D)
        axis=str(result['pred_axis'][0]);cy=float(result['cost_Y'][0]);ct=float(result['cost_T'][0])
        chosen,other=(cy,ct) if axis=='Y' else (ct,cy)
        best=dict(axis=axis,value=float(result['pred_value'][0]),normalized_cost_margin=(other-chosen)/(other+chosen+1e-12),alternatives=[dict(axis='Y',value=float(result['hyp_Y'][0]),cost=cy),dict(axis='T',value=float(result['hyp_T'][0]),cost=ct)])
        scope='Global axis curves; single-axis motion, same camera and calibration reference.'
        validation='modeling/unified_isolated_stratified_metrics.json'
    value=float(best['value']);axis=best['axis']
    out=dict(status='prediction',mode=mode,axis=axis,sign=int(np.sign(value)),value=value,Y=value if axis=='Y' else 0.,T=value if axis=='T' else 0.,
        normalized_cost_margin=float(best['normalized_cost_margin']),confidence_probability=None,actual_error=None,
        valid_points=int(v.sum()),total_points=len(v),runtime_seconds=time.perf_counter()-start,scope=scope,validation_file=validation,
        confidence_note='Cost margin is not a calibrated probability. Actual pose error cannot be determined from two images without independent ground truth.',
        alternatives=best.get('alternatives'))
    return out

def clean(v):
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    return v

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--center',required=True);p.add_argument('--current',required=True);p.add_argument('--mode',choices=['calibrated-runs','general'],default='calibrated-runs');p.add_argument('--out');p.add_argument('--truth-axis',choices=['Y','T']);p.add_argument('--truth-value',type=float);args=p.parse_args()
    result=run(args.center,args.current,args.mode)
    if args.truth_axis is not None and args.truth_value is not None and result['status']=='prediction':
        truth=np.array([args.truth_value if args.truth_axis=='Y' else 0.,args.truth_value if args.truth_axis=='T' else 0.])
        err=np.abs(np.array([result['Y'],result['T']])-truth)
        correct=result['axis']==args.truth_axis and result['sign']==int(np.sign(args.truth_value))
        result['actual_error']=dict(Y=float(err[0]),T=float(err[1]),axis_and_sign_correct=correct,passes_009=bool(correct and (err<=.09).all()))
    # Infinite costs for disallowed branch signs become JSON null.
    txt=json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)
    if args.out:Path(args.out).write_text(txt,encoding='utf8')
    print(txt)

if __name__=='__main__':main()
