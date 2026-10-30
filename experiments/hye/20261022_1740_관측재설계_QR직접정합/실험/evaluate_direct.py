"""Nested development evaluation of four dense measurements and three decoders."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;GENERAL=H.parent/'joint_yt_generalization_20261022';OLD=H.parent/'joint_yt_disentangle_20261022'
sys.path.insert(0,str(GENERAL))
from data import load
from metrics import metric
from direct_models import DirectDecoder,CONFIGS

def make_rows(m,ix,p,method,protocol,candidate):
    r=m.iloc[ix].copy();r['predY']=p[:,0];r['predT']=p[:,1];r['eY']=abs(r.predY-r.Y);r['eT']=abs(r.predT-r['T']);r['passed']=(r.eY<=.09+1e-12)&(r.eT<=.09+1e-12);r['method']=method;r['protocol']=protocol;r['candidate']=candidate;return r

def main():
    m,old_a=load();a=np.load(H/'direct/features.npz');y=m[['Y','T']].to_numpy();splits=json.loads((OLD/'selection.json').read_text());out=[];audit=[];fixed=[]
    protocol=dict(scope='ADAPTIVE DEVELOPMENT; no independent new photos',new_observation='raw QR intensities -> joint affine/projective warp perQR -> smooth displacements; historical630 boundary estimates not used',configs=CONFIGS,choice='3fold inner grouped selection; failures then max then mean; exact historical outer/inner splits',failure_policy='Failed measurement/prediction is retained in denominator, not old-measurement fallback',prior_splits_sha256=hashlib.sha256((OLD/'selection.json').read_bytes()).hexdigest())
    (H/'direct_protocol.json').write_text(json.dumps(protocol,indent=2))
    for fold in range(5):
        start=time.perf_counter();original=splits[fold] if fold<4 else splits[-1]['full_selection'];tr=np.flatnonzero(m.fold.to_numpy()!=fold) if fold<4 else np.arange(len(m));te=np.flatnonzero(m.fold.to_numpy()==fold) if fold<4 else np.arange(len(m));scores=[]
        for k,c in enumerate(CONFIGS):
            D=a['D_'+c['measurement']];good=a['success_'+c['measurement']];errs=[]
            for split in original['inner_splits']:
                it=np.array(split['train']);it=it[good[it]];iv=np.array(split['validation']);iv=iv[m.iloc[iv].dataset.to_numpy()==20261020]
                assert not set(m.iloc[it].cv_group)&set(m.iloc[iv].cv_group)
                p=np.full((len(iv),2),np.nan);model=DirectDecoder(c).fit(D[it],y[it],m.iloc[it]);p[good[iv]]=model.predict(D[iv[good[iv]]]);errs.extend(abs(p-y[iv]))
            e=np.asarray(errs);finite=np.isfinite(e).all(axis=1);ev=e[finite];scores.append(dict(candidate=k,config=c,failures=int((~finite).sum()+np.any(ev>.09+1e-12,axis=1).sum()),maximum=float(ev.max()) if len(ev) else 1e100,mean=float(ev.mean()) if len(ev) else 1e100))
            if fold<4:
                train=tr[good[tr]];p=np.full((len(te),2),np.nan);model=DirectDecoder(c).fit(D[train],y[train],m.iloc[train]);p[good[te]]=model.predict(D[te[good[te]]]);fixed.append(make_rows(m,te,p,'fixed_candidate','development_fixed_oof',k))
        chosen=min(scores,key=lambda r:(r['failures'],r['maximum'],r['mean']))['candidate'];c=CONFIGS[chosen];D=a['D_'+c['measurement']];good=a['success_'+c['measurement']];train=tr[good[tr]]
        model=DirectDecoder(c).fit(D[train],y[train],m.iloc[train]);p=np.full((len(te),2),np.nan);p[good[te]]=model.predict(D[te[good[te]]]);out.append(make_rows(m,te,p,'direct_selected','development_nested_oof' if fold<4 else 'training_reapplication',chosen))
        audit.append(dict(fold=fold,chosen=chosen,inner_splits=original['inner_splits'],scores=scores))
        joblib.dump(dict(model=model,config=c,train_indices=train,ids=old_a['ids'],points=old_a['points']),H/(f'direct_fold{fold}.joblib' if fold<4 else 'direct_model.joblib'))
        n=m.iloc[te].dataset.to_numpy()==20261020;print('fold',fold,c,metric(y[te][n],p[n]),'seconds',time.perf_counter()-start,flush=True)
        pd.concat(out).to_csv(H/'direct_progress.csv',index=False)
    result=pd.concat(out);result.to_csv(H/'direct_predictions.csv',index=False);pd.concat(fixed).to_csv(H/'direct_fixed_predictions.csv',index=False);(H/'direct_selection.json').write_text(json.dumps(audit,indent=2))
    summary=[]
    for protocol,g in result.groupby('protocol'):
        for ds in ['all',20261014,20261020]:
            z=g if ds=='all' else g[g.dataset==ds];summary.append(dict(protocol=protocol,dataset=ds,**metric(z[['Y','T']],z[['predY','predT']])))
    pd.DataFrame(summary).to_csv(H/'direct_summary.csv',index=False);print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
