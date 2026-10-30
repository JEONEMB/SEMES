"""Train-fold separation diagnostics and date-held-out development stress tests."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'joint_yt_generalization_20261022'))
from data import load
from metrics import metric
from evaluate_parallax import qr_grid
from direct_models import DirectDecoder

def main():
    m,a=load();ext=np.load(H/'parallax/features.npz');q=np.load(H/'direct/features.npz');qd,_=qr_grid(q['H_homography_g5']);y=m[['Y','T']].to_numpy()
    ds={'old_630':a['D'],'dense_qr':qd,'external':ext['D_affine_g5'],'qr_external':np.concatenate([qd,ext['D_affine_g5']],axis=1)};diag=[];stress=[];pred=[]
    for fold in range(4):
        it=np.flatnonzero((m.fold.to_numpy()!=fold)&(m.dataset.to_numpy()==20261020));keys=[tuple(v) for v in y[it]];groups=sorted(set(keys));gy=np.array(groups)
        for name,D in ds.items():
            gx=np.array([D[it][np.array([k==v for k in keys])].mean(axis=0).ravel() for v in groups]);coef=np.linalg.lstsq(np.c_[np.ones(len(gy)),gy],gx,rcond=None)[0];J=coef[1:];cos=float(J[0]@J[1]/(np.linalg.norm(J[0])*np.linalg.norm(J[1])));tcomp=-.09*(J[0]@J[1])/(J[1]@J[1]);ambiguity=.09*J[0]+tcomp*J[1];normalized=J/np.linalg.norm(J,axis=1,keepdims=True)
            diag.append(dict(fold=fold,observation=name,training_images=len(it),training_conditions=len(groups),cosine=cos,normalized_jacobian_condition=float(np.linalg.cond(normalized)),Y_009_compensating_T=float(tcomp),remaining_coordinate_RMS_px=float(np.sqrt(np.mean(ambiguity**2)))))
    for family in ['external','qr_external']:
        D=ds[family]
        for train_date,test_date in [(20261014,20261020),(20261020,20261014)]:
            tr=np.flatnonzero(m.dataset.to_numpy()==train_date);te=np.flatnonzero(m.dataset.to_numpy()==test_date);cfg=dict(measurement='affine_g5',head='linear');model=DirectDecoder(cfg).fit(D[tr],y[tr],m.iloc[tr]);p=model.predict(D[te]);stress.append(dict(method=family,train_date=train_date,test_date=test_date,**metric(y[te],p)));r=m.iloc[te].copy();r['predY']=p[:,0];r['predT']=p[:,1];r['method']=family;r['train_date']=train_date;r['eY']=abs(r.predY-r.Y);r['eT']=abs(r.predT-r['T']);r['passed']=(r.eY<=.09+1e-12)&(r.eT<=.09+1e-12);pred.append(r)
    pd.DataFrame(diag).to_csv(H/'spatial_response_diagnostic.csv',index=False);pd.DataFrame(stress).to_csv(H/'date_stress_summary.csv',index=False);pd.concat(pred).to_csv(H/'date_stress_predictions.csv',index=False)
    (H/'date_stress_protocol.json').write_text(json.dumps(dict(scope='Retrospective development stress test on exposed dates; NOT independent new-photo validation',choice='affine_g5 linear frozen after main nested result; no hyperparameter search in this test',reference='original fixed center from20261014 also used for reverse-date measurement; reverse test is calibration-domain transfer under this known reference, not unseen-date reference transfer',purpose='check whether mixed-label training is required and reveal direction-dependent extrapolation',reverse_domain='20261020 has nonnegative commands and narrower Y;20261014 includes negative/larger commands'),indent=2))
    print(pd.DataFrame(stress).to_string(index=False));print(pd.DataFrame(diag).groupby('observation')[['cosine','normalized_jacobian_condition','remaining_coordinate_RMS_px']].mean().to_string())

if __name__=='__main__':
    with threadpool_limits(1):main()
