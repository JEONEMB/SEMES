"""Nested grouped validation. All new-date repeats, including center, stay together."""
import json,time,hashlib,sys
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from data import H,ROOT,load,metric,save_splits
from joint_model import JointRegressor

CONFIGS=[dict(family='global',feature=f,shrink=s) for f in ['raw1260','raw_shift'] for s in [.001,.01,.1,.3]]
CONFIGS += [dict(family='local',feature='raw_shift',shrink=s,width=w) for w in [20.,100.] for s in [.01,.1]]

def rank(y,p):
    e=abs(y-p);return (int(np.sum(np.any(e>.09+1e-12,axis=1))),float(e.max()),float(e.mean()))

def select(m,D,y,tr,seed):
    tr=np.asarray(tr);new=m.dataset.to_numpy()==20261020
    strat=np.where(new,'new_'+m.scope.astype(str),'old')
    inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(tr,strat[tr],m.cv_group.to_numpy()[tr]))
    candidates=[];audit=[]
    for c in CONFIGS:
        pp=[];yy=[]
        for ti,vi in inner:
            it=tr[ti];iv=tr[vi];iv=iv[new[iv]]
            assert not set(m.cv_group.iloc[it])&set(m.cv_group.iloc[iv])
            model=JointRegressor(c).fit(D[it],y[it]);pp.append(model.predict(D[iv]));yy.append(y[iv])
        candidates.append(rank(np.vstack(yy),np.vstack(pp)))
    for ti,vi in inner:audit.append(dict(train=tr[ti].tolist(),validation=tr[vi].tolist()))
    best=min(range(len(CONFIGS)),key=lambda i:candidates[i])
    return CONFIGS[best],dict(seed=seed,inner_splits=audit,scores=[dict(config=c,score=s) for c,s in zip(CONFIGS,candidates)],selected=CONFIGS[best])

def frame(m,ix,p,method,protocol,fold):
    z=m.iloc[ix][['index','dataset','file','relative','Y','T','scope','condition','cv_group']].copy();z['predY']=p[:,0];z['predT']=p[:,1];z['correctionY']=-p[:,0];z['correctionT']=-p[:,1];z['eY']=abs(z.predY-z.Y);z['eT']=abs(z.predT-z['T']);z['both_within09']=(z.eY<=.09+1e-12)&(z.eT<=.09+1e-12);z['method']=method;z['protocol']=protocol;z['fold']=fold
    return z

def main():
    m,a=load();D=a['D'];y=m[['Y','T']].to_numpy();splits=save_splits(m);new=m.dataset.to_numpy()==20261020;old=~new
    protocol=dict(outer='original280 fixed4fold plus grouped new mixed commands; new center3 all in fold0',selection='inner3fold: fewest new-date images outside both0.09, then worst axis error, then mean error',configs=CONFIGS,seed=20261022,measurement='original strict329 plus same SciPy solver with expanded NCC12',no_label_at_inference=True,limitations=['Development family was informed by this corpus; nested result is not an untouched future-date test.','20261020 training conditions are used in calibration; this is not old-only date transfer.','Correction is negative prediction; robot actuation not executed.'])
    (H/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[];selections=[]
    for f,s in enumerate(splits):
        tr=np.array(s['train_indices']);te=np.array(s['test_indices']);started=time.monotonic()
        assert not set(m.cv_group.iloc[tr])&set(m.cv_group.iloc[te])
        c,audit=select(m,D,y,tr,20261022+f);audit['outer_fold']=f;selections.append(audit)
        model=JointRegressor(c).fit(D[tr],y[tr]);joblib.dump(dict(model=model,config=c,ids=a['ids'],points=a['points'],train_indices=tr),H/f'fold{f}_model.joblib')
        p=model.predict(D[te]);records.append(frame(m,te,p,'selected_joint','nested_group_oof',f))
        for name,control in [('raw_global',dict(family='global',feature='raw1260',shrink=.1)),('interaction_global',dict(family='global',feature='raw_shift',shrink=.01)),('interaction_local',dict(family='local',feature='raw_shift',shrink=.01,width=20.))]:
            pp=JointRegressor(control).fit(D[tr],y[tr]).predict(D[te]);records.append(frame(m,te,pp,name,'fixed_config_group_oof',f))
        print('outer',f,'selected',c,'new',metric(y[te][new[te]],p[new[te]]),'seconds',round(time.monotonic()-started,1),flush=True)
        pd.concat(records).to_csv(H/'progress_predictions.csv',index=False)
        (H/'selection.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    c,audit=select(m,D,y,np.arange(len(m)),20261101);selections.append(dict(final_selection=audit))
    model=JointRegressor(c).fit(D,y);joblib.dump(dict(model=model,config=c,ids=a['ids'],points=a['points'],train_indices=np.arange(len(m))),H/'model.joblib')
    p=model.predict(D);records.append(frame(m,np.arange(len(m)),p,'selected_joint','resub341',-1))
    # New model family trained on the old date only: descriptive transfer controls.
    for feature in ['raw1260','raw_shift']:
        c0=dict(family='global',feature=feature,shrink=.01)
        p0=JointRegressor(c0).fit(D[old],y[old]).predict(D[new]);records.append(frame(m,np.flatnonzero(new),p0,feature,'old_only_transfer',-1))
    p=pd.concat(records,ignore_index=True);p.to_csv(H/'predictions.csv',index=False,encoding='utf-8-sig');p[~p.both_within09].to_csv(H/'failures.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for (method,protocol),g in p.groupby(['method','protocol']):
        for dataset in ['all',20261014,20261020]:
            z=g if dataset=='all' else g[g.dataset==dataset]
            if len(z):rows.append(dict(method=method,protocol=protocol,dataset=dataset,**metric(z[['Y','T']],z[['predY','predT']])))
    summary=pd.DataFrame(rows);summary.to_csv(H/'summary.csv',index=False,encoding='utf-8-sig')
    (H/'selection.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    (H/'final_config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
