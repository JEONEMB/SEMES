"""Audit frozen630 adapter against preserved measurements and baseline; no training."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import server
OUT=HERE

def evaluate(name,pred_axis,pred):
    axis=name[0];v=float(name[1:-4]);Y=v if axis=='Y' else 0;T=v if axis=='T' else 0
    py=pred if pred_axis=='Y' else 0;pt=pred if pred_axis=='T' else 0
    ay=abs(py-Y);at=abs(pt-T);axis_ok=axis==pred_axis;sign_ok=axis_ok and np.sign(v)==np.sign(pred)
    return dict(file=name,axis=axis,value=v,pred_axis=pred_axis,pred=pred,eY=ay,eT=at,axis_match=bool(axis_ok),sign_match=bool(sign_ok),passed=bool(sign_ok and ay<=.09 and at<=.09))

def solve(X):
    choices=[];W=np.array(server.params['W'])
    for key,(axis,b,s,c) in server.params['branches'].items():
        z=W@(X-np.array(b));s=W@np.array(s);c=W@np.array(c);v=float(z@s/(s@s))
        for _ in range(8):
            J=s+2*v*c;v+=float(J@(z-v*s-v*v*c)/max(J@J,1e-20))
        choices.append((float(np.sum((z-v*s-v*v*c)**2)),axis,v))
    return min(choices)

if __name__=='__main__':
    frame=pd.read_csv(server.HERE.parent/'analysis_grid_corners_20261019/out/corners_280.csv')
    baseline=pd.read_csv(server.MODEL/'predictions.csv');baseline=baseline[(baseline.method=='v1_edge630')&(baseline.protocol=='full280')].set_index('file')
    ref=server.extractor.points;ids=server.extractor.ids
    u,v=(ref[:,0]-1906)/486,(ref[:,1]-1070)/122;B=np.c_[np.ones(len(u)),u,v,u*u,u*v,v*v]
    rows=[]
    for name,g in frame[frame.file!='center.bmp'].groupby('file'):
        g=g.set_index('id').loc[ids];d=g[['x','y']].to_numpy()-ref
        X=np.linalg.lstsq(B,d,rcond=None)[0].ravel();cost,axis,pred=solve(X)
        old=baseline.loc[name];assert axis==old.pred_axis and abs(pred-old.pred)<1e-7
        row=evaluate(name,axis,pred);assert abs(row['eY']-old.eY)<1e-7 and abs(row['eT']-old.eT)<1e-7
        row.update(valid=int(g.valid.sum()),total=630,source='preserved630_measurements_recomputed_model');rows.append(row)
    result=pd.DataFrame(rows);assert len(result)==280
    result.to_csv(OUT/'recomputed280.csv',index=False);result[~result.passed].to_csv(OUT/'failures.csv',index=False)
    summary=dict(protocol='full280 resubstitution; preserved measurements, no new OOF evaluation',n=280,axis_correct=int(result.axis_match.sum()),axis_sign_correct=int(result.sign_match.sum()),passed=int(result.passed.sum()),Y_max=float(result.eY.max()),T_max=float(result.eT.max()),reference_ids=630,excluded_diagonal=125)
    print(json.dumps(summary),flush=True)
    live=[]
    for name in ['Y+0.05.bmp','Y+0.50.bmp','T-0.05.bmp','center.bmp']:
        path=server.ROOT/('center.bmp' if name=='center.bmp' else ('Y/' if name.startswith('Y') else 'Theta/')+name)
        r=server.infer(path.read_bytes());t=r['trace']
        expected=frame[frame.file==name].set_index('id').loc[ids]
        difference=float(np.max(abs(np.array(t['points'])+np.array(t['displacements'])-expected[['x','y']].to_numpy())))
        assert difference<.003,(name,difference)
        assert t['ids']==ids.tolist() and len(t['points'])==630
        selected=next(c for c in t['candidates'] if c['branch']==t['selected_branch'])
        assert abs(sum(x*x for x in selected['whitened_residual'])-selected['cost'])<1e-7
        row=dict(file=name,axis=r['axis'],value=r['value'],valid=r['valid'],seconds=r['seconds'],max_coordinate_difference_px=difference)
        if name!='center.bmp':
            old=baseline.loc[name];assert r['axis']==old.pred_axis and abs(r['value']-old.pred)<1e-4
            row['evaluation']=evaluate(name,r['axis'],r['value'])
        else:assert r['axis']=='CENTER' and r['Y']==0 and r['T']==0
        live.append(row);print(json.dumps(row),flush=True)
        if name=='Y+0.50.bmp':(OUT/'ui_sample.json').write_text(json.dumps(r))
    for raw in [b'invalid',__import__('cv2').imencode('.png',np.zeros((40,40),np.uint8))[1].tobytes()]:
        try:server.infer(raw)
        except ValueError:pass
        else:raise AssertionError('Invalid image accepted')
    summary['live_cases']=live
    (OUT/'checks.json').write_text(json.dumps(summary,indent=2))
