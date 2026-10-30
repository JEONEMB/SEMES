"""Exploratory nested development evaluation, candidate list fixed before run."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;OLD=H.parent/'joint_yt_disentangle_20261022';GENERAL=H.parent/'joint_yt_generalization_20261022'
sys.path.insert(0,str(GENERAL))
from data import load,metric
from phase_model import PhaseNuisance,CONFIGS

def main():
    protocol=H/'phase_protocol.json'
    if protocol.exists():raise RuntimeError('Do not overwrite phase experiment.')
    protocol.write_text(json.dumps(dict(scope='Exploratory ADAPTIVE DEVELOPMENT comparison; not independent test',reason='10-failure analysis suggests estimator sensitivity to small shape changes; subpixel-periodic measurement bias is only a hypothesis',sources=['https://opg.optica.org/oe/abstract.cfm?uri=oe-23-15-19242','https://doi.org/10.1016/j.optlaseng.2016.11.014'],method='Estimate periodic shape nuisance using ONLY within-condition TRAIN repeats; never average test repeats or read test Y/T. No pixel alteration, no synthetic accuracy claims.',configs=CONFIGS,selection='same inner grouped splits as prior; rank failures,max,mean',previous_selection_sha256=hashlib.sha256((OLD/'selection.json').read_bytes()).hexdigest()),indent=2))
    m,a=load();D=a['D'];y=m[['Y','T']].to_numpy();splits=json.loads((OLD/'selection.json').read_text());out=[];audit=[];fixed=[]
    for fold in range(5):
        original=splits[fold] if fold<4 else splits[-1]['full_selection'];tr=np.flatnonzero(m.fold.to_numpy()!=fold) if fold<4 else np.arange(len(m));te=np.flatnonzero(m.fold.to_numpy()==fold) if fold<4 else np.arange(len(m));scores=[]
        for k,c in enumerate(CONFIGS):
            pred=[];true=[]
            for split in original['inner_splits']:
                it=np.array(split['train']);iv=np.array(split['validation']);iv=iv[m.iloc[iv].dataset.to_numpy()==20261020]
                assert not set(m.iloc[it].cv_group)&set(m.iloc[iv].cv_group)
                model=PhaseNuisance(c).fit(D[it],y[it],m.iloc[it]);pred.append(model.predict(D[iv]));true.append(y[iv])
            err=abs(np.vstack(pred)-np.vstack(true));scores.append(dict(candidate=k,config=c,failures=int(np.any(err>.09+1e-12,axis=1).sum()),maximum=float(err.max()),mean=float(err.mean())))
            if fold<4:
                p=PhaseNuisance(c).fit(D[tr],y[tr],m.iloc[tr]).predict(D[te]);z=m.iloc[te].copy();z['predY']=p[:,0];z['predT']=p[:,1];z['candidate']=k;fixed.append(z)
        chosen=min(scores,key=lambda r:(r['failures'],r['maximum'],r['mean']))['candidate'];model=PhaseNuisance(CONFIGS[chosen]).fit(D[tr],y[tr],m.iloc[tr]);p=model.predict(D[te]);z=m.iloc[te].copy();z['predY']=p[:,0];z['predT']=p[:,1];z['eY']=abs(z.predY-z.Y);z['eT']=abs(z.predT-z['T']);z['passed']=(z.eY<=.09+1e-12)&(z.eT<=.09+1e-12);z['protocol']='nested_development_oof' if fold<4 else 'training_reapplication';out.append(z)
        audit.append(dict(fold=fold,chosen=chosen,scores=scores,inner_splits=original['inner_splits']))
        joblib.dump(dict(model=model,config=CONFIGS[chosen],train_indices=tr,ids=a['ids'],points=a['points']),H/(f'phase_fold{fold}.joblib' if fold<4 else 'phase_model.joblib'))
        nn=m.iloc[te].dataset.to_numpy()==20261020;print(fold,CONFIGS[chosen],metric(y[te][nn],p[nn]),flush=True)
    allp=pd.concat(out);allp.to_csv(H/'phase_predictions.csv',index=False);pd.concat(fixed).to_csv(H/'phase_fixed_predictions.csv',index=False);(H/'phase_selection.json').write_text(json.dumps(audit,indent=2))
    summary=[]
    for protocol,g in allp.groupby('protocol'):
        for ds in ['all',20261014,20261020]:
            z=g if ds=='all' else g[g.dataset==ds];summary.append(dict(protocol=protocol,dataset=ds,**metric(z[['Y','T']],z[['predY','predT']])))
    pd.DataFrame(summary).to_csv(H/'phase_summary.csv',index=False);print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
