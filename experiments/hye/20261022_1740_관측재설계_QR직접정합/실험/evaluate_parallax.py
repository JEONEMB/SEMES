"""Prespecified spatial-diversity experiment, including an external-only ablation.

Outer folds identical to prior work. Inner choice balances both dates, then
maximum error, then mean. All observations remain in validation denominators.
The full image corpus is exposed development data, not independent validation.
"""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'joint_yt_generalization_20261022'))
from data import load
from metrics import metric
from direct_models import DirectDecoder
from evaluate_direct import make_rows

def qr_grid(Hs):
    boxes=[(1552,969,1736,1167),(1809,971,1993,1169),(2067,973,2252,1171)]
    points=[];ds=[]
    for q,(x0,y0,x1,y1) in enumerate(boxes):
        p=np.array([(x0+a*(x1-x0),y0+b*(y1-y0)) for b in [.2,.5,.8] for a in [.2,.5,.8]])
        z=np.c_[p,np.ones(9)]@Hs[:,q].transpose(0,2,1)
        ds.append(z[:,:,:2]/z[:,:,2:]-p);points.extend(p)
    return np.concatenate(ds,axis=1),np.array(points)

def main():
    m,_=load();a=np.load(H/'parallax/features.npz');q=np.load(H/'direct/features.npz');qd,qp=qr_grid(q['H_homography_g5']);y=m[['Y','T']].to_numpy()
    measurements=['translation_g5','affine_g5'];configs=[dict(measurement=x,head=h) for x in measurements for h in ['linear','quadratic','balanced']]
    protocol=dict(scope='ADAPTIVE DEVELOPMENT only',observation='5 external bolt/bracket fixed reference patches; external-only and 3QR+external separate families',configs=configs,selection='inner3fold: average of date-specific failure fractions, then pooled maximum, then pooled mean; BOTH dates counted equally',features='9 warp-projected grid anchors per patch; QR shared observation parameters homography_g5 fixed before this experiment; projected anchors are NOT independent physical point measurements',inference='image bytes only; no filename, label, date or repeat',failure_policy='all failed observations retained in evaluation denominator',new_measurements=a.files)
    (H/'parallax_protocol.json').write_text(json.dumps(protocol,indent=2))
    splits=json.loads((H.parent/'joint_yt_disentangle_20261022/selection.json').read_text());out=[];audit=[];fixed=[]
    for family in ['external','qr_external']:
        Ds={c:a['D_'+c] if family=='external' else np.concatenate([qd,a['D_'+c]],axis=1) for c in measurements}
        goods={c:a['success_'+c]&(q['success_homography_g5'] if family=='qr_external' else True) for c in measurements}
        for fold in range(5):
            start=time.perf_counter();inner=splits[fold]['inner_splits'] if fold<4 else splits[-1]['full_selection']['inner_splits'];tr=np.flatnonzero(m.fold.to_numpy()!=fold) if fold<4 else np.arange(len(m));te=np.flatnonzero(m.fold.to_numpy()==fold) if fold<4 else np.arange(len(m));scores=[]
            for k,c in enumerate(configs):
                D=Ds[c['measurement']];good=goods[c['measurement']];errs=[];dates=[]
                for split in inner:
                    it=np.array(split['train']);it=it[good[it]];iv=np.array(split['validation'])
                    assert not set(m.iloc[it].cv_group)&set(m.iloc[iv].cv_group)
                    model=DirectDecoder(c).fit(D[it],y[it],m.iloc[it]);p=np.full((len(iv),2),np.nan);p[good[iv]]=model.predict(D[iv[good[iv]]]);errs.extend(abs(p-y[iv]));dates.extend(m.iloc[iv].dataset)
                e=np.asarray(errs);dates=np.array(dates);finite=np.isfinite(e).all(axis=1);bad=(~finite)|np.any(e>.09+1e-12,axis=1);ev=e[finite]
                scores.append(dict(candidate=k,config=c,date_balanced_failure_rate=float(np.mean([bad[dates==d].mean() for d in np.unique(dates)])),failures=int(bad.sum()),maximum=float(ev.max()) if len(ev) else 1e100,mean=float(ev.mean()) if len(ev) else 1e100))
                if fold<4:
                    train=tr[good[tr]];model=DirectDecoder(c).fit(D[train],y[train],m.iloc[train]);p=np.full((len(te),2),np.nan);p[good[te]]=model.predict(D[te[good[te]]]);fixed.append(make_rows(m,te,p,family,'development_fixed_oof',k))
            chosen=min(scores,key=lambda r:(r['date_balanced_failure_rate'],r['maximum'],r['mean']))['candidate'];c=configs[chosen];D=Ds[c['measurement']];good=goods[c['measurement']];train=tr[good[tr]]
            model=DirectDecoder(c).fit(D[train],y[train],m.iloc[train]);p=np.full((len(te),2),np.nan);p[good[te]]=model.predict(D[te[good[te]]]);out.append(make_rows(m,te,p,family,'development_nested_oof' if fold<4 else 'training_reapplication',chosen))
            joblib.dump(dict(model=model,config=c,train_indices=train,family=family,protocol=protocol),H/(f'{family}_fold{fold}.joblib' if fold<4 else f'{family}_model.joblib'))
            audit.append(dict(family=family,fold=fold,chosen=chosen,inner_splits=inner,scores=scores))
            n=m.iloc[te].dataset.to_numpy()==20261020;print(family,fold,c,metric(y[te][n],p[n]),'seconds',time.perf_counter()-start,flush=True)
            pd.concat(out).to_csv(H/'parallax_progress.csv',index=False)
    result=pd.concat(out);result.to_csv(H/'parallax_predictions.csv',index=False);pd.concat(fixed).to_csv(H/'parallax_fixed_predictions.csv',index=False);(H/'parallax_selection.json').write_text(json.dumps(audit,indent=2))
    rows=[]
    for (family,protocol),g in result.groupby(['method','protocol']):
        for ds in ['all',20261014,20261020]:
            z=g if ds=='all' else g[g.dataset==ds];rows.append(dict(method=family,protocol=protocol,dataset=ds,**metric(z[['Y','T']],z[['predY','predT']])))
    pd.DataFrame(rows).to_csv(H/'parallax_summary.csv',index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
