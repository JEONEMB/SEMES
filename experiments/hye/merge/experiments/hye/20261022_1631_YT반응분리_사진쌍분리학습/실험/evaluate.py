"""Nested grouped comparison of research-derived separation training methods."""
from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;PREV=H.parent/'joint_yt_generalization_20261022'
sys.path.insert(0,str(PREV))
from data import load,metric
from learned_model import LearnedSeparation

CONFIGS=[dict(method='baseline',params=dict(family='global',feature='raw_shift',shrink=s)) for s in [.01,.1]]
CONFIGS += [dict(method='baseline',params=dict(family='local',feature='raw_shift',shrink=s,width=w)) for s,w in [(.01,20.),(.1,100.)]]
CONFIGS += [dict(method='repeat',feature='raw_shift',params=dict(fit_mode=f,repeat_mix=mix,shrink=s)) for f in ['new_condition_means','axis_pairs'] for mix in [0.,.5] for s in [.01,.1]]
CONFIGS += [dict(method='paired',feature='P12',params=dict(alpha=1.,pair_weight=1.,repeat_weight=10.))]
CONFIGS += [dict(method='conditional',feature='P12',params=dict(base_shrink=.1,mode='remove_quadratic',family='gls',shrink=.5))]
CONFIGS += [dict(method='neural',feature='raw_shift',params=dict(components=64,pair_weight=1.,crossfit=False))]

def select(m,a,tr,seed):
    D=a['D'];y=m[['Y','T']].to_numpy();new=m.dataset.to_numpy()==20261020;groups=m.cv_group.to_numpy();strata=np.where(new,'new_'+m.scope.astype(str),'old')
    inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(tr,strata[tr],groups[tr]));records=[]
    for k,c in enumerate(CONFIGS):
        pp=[];yy=[]
        for ti,vi in inner:
            it=tr[ti];iv=tr[vi];iv=iv[new[iv]]
            assert not set(groups[it])&set(groups[iv])
            model=LearnedSeparation(c,a['points'],a['regions']).fit(D[it],y[it],m.iloc[it]);pp.append(model.predict(D[iv]));yy.append(y[iv])
        errors=abs(np.vstack(pp)-np.vstack(yy));records.append(dict(candidate=k,config=c,failures=int(np.any(errors>.09+1e-12,axis=1).sum()),maximum=float(errors.max()),mean=float(errors.mean())))
    pass_first=min(records,key=lambda r:(r['failures'],r['maximum'],r['mean']))['candidate']
    max_first=min(records,key=lambda r:(r['maximum'],r['failures'],r['mean']))['candidate']
    return dict(pass_first=pass_first,max_first=max_first),dict(seed=seed,inner_splits=[dict(train=tr[x].tolist(),validation=tr[z].tolist()) for x,z in inner],scores=records,selected=dict(pass_first=pass_first,max_first=max_first))

def rows(m,ix,p,method,fold,protocol):
    z=m.iloc[ix][['index','dataset','file','relative','Y','T','scope','condition','cv_group']].copy();z['predY']=p[:,0];z['predT']=p[:,1];z['eY']=abs(z.predY-z.Y);z['eT']=abs(z.predT-z['T']);z['correctionY']=-p[:,0];z['correctionT']=-p[:,1];z['passed']=(z.eY<=.09+1e-12)&(z.eT<=.09+1e-12);z['method']=method;z['fold']=fold;z['protocol']=protocol;return z

def main():
    m,a=load();D=a['D'];y=m[['Y','T']].to_numpy();new=m.dataset.to_numpy()==20261020
    protocol=dict(configs=CONFIGS,folds=m.fold.tolist(),source=str(PREV/'measurements/features.npz'),primary='pass_first: same priority as previous experiment; fewest failures then maximum error',secondary='max_first: minimize worst axis error, then failure count; separate diagnostic objective',training='all pair graphs, repeat noise, nuisance fits, PCA/neural weights from training only',input='only D from image and fixedreference at predict',limits='family development used this corpus; nested validation is not unseen future-date validation')
    (H/'protocol.json').write_text(json.dumps(protocol,indent=2))
    predictions=[];selection=[]
    for f in range(4):
        tr=np.flatnonzero(m.fold.to_numpy()!=f);te=np.flatnonzero(m.fold.to_numpy()==f);start=time.monotonic()
        chosen,audit=select(m,a,tr,20261022+f);selection.append(dict(fold=f,**audit))
        for key,k in chosen.items():
            model=LearnedSeparation(CONFIGS[k],a['points'],a['regions']).fit(D[tr],y[tr],m.iloc[tr]);p=model.predict(D[te]);predictions.append(rows(m,te,p,key,f,'nested_group_oof'))
            joblib.dump(dict(model=model,config=CONFIGS[k],ids=a['ids'],points=a['points'],train_indices=tr),H/f'{key}_fold{f}.joblib')
            print('fold',f,key,CONFIGS[k],metric(y[te][new[te]],p[new[te]]),flush=True)
        print('seconds',round(time.monotonic()-start,1),flush=True)
        pd.concat(predictions).to_csv(H/'progress_predictions.csv',index=False);(H/'selection.json').write_text(json.dumps(selection,indent=2))
    chosen,audit=select(m,a,np.arange(len(m)),20261101);selection.append(dict(full_selection=audit))
    for key,k in chosen.items():
        model=LearnedSeparation(CONFIGS[k],a['points'],a['regions']).fit(D,y,m);p=model.predict(D);predictions.append(rows(m,np.arange(len(m)),p,key,-1,'resub341'))
        joblib.dump(dict(model=model,config=CONFIGS[k],ids=a['ids'],points=a['points'],train_indices=np.arange(len(m))),H/f'{key}_model.joblib')
    p=pd.concat(predictions,ignore_index=True);p.to_csv(H/'predictions.csv',index=False,encoding='utf-8-sig');p[~p.passed].to_csv(H/'failures.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for (method,protocol),g in p.groupby(['method','protocol']):
        for dataset in ['all',20261014,20261020]:
            z=g if dataset=='all' else g[g.dataset==dataset];summary.append(dict(method=method,protocol=protocol,dataset=dataset,**metric(z[['Y','T']],z[['predY','predT']])))
    pd.DataFrame(summary).to_csv(H/'summary.csv',index=False,encoding='utf-8-sig');(H/'selection.json').write_text(json.dumps(selection,indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
