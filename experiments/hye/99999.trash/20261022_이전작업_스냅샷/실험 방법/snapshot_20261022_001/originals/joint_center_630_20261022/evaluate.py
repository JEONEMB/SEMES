"""Fixed train-only origin fitting; nested CV on the expanded 341-image corpus."""
from pathlib import Path
import json,time,copy
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from model import JointOrigins
from features import FeatureMaps
from local_model import LocalOrigins
HERE=Path(__file__).resolve().parent;DATA=HERE/'results';OUT=DATA/'local_surfaces';OUT.mkdir(exist_ok=True)
CONFIGS=[dict(feature='P12',degree=d,shrink=s,origins=o) for d in [1,2] for s in [.2,1.] for o in ['learned','learned_actual']]+[dict(feature=f,degree=d,shrink=s,origins='learned_actual') for f in ['QR18','QR36','raw1260'] for d in [1,2] for s in ([1.] if f=='raw1260' else [.2,1.])]
CONFIGS += [dict(feature=f,degree=2,shrink=.2,origins='learned_actual',local=True) for f in ['P12','QR36']]
SCORE_RULE='inner heldout failure count at both |errors|<=0.09; then maximum axis error; then sum axis MAE'

def metrics(p):
    finite=p.predY.notna()&p.predT.notna();errors=np.abs(p.loc[finite,['predY','predT']].to_numpy()-p.loc[finite,['Y','T']].to_numpy())
    return dict(n=len(p),inference_failures=int((~finite).sum()),Y_MAE=float(errors[:,0].mean()) if len(errors) else None,Y_max=float(errors[:,0].max()) if len(errors) else None,T_MAE=float(errors[:,1].mean()) if len(errors) else None,T_max=float(errors[:,1].max()) if len(errors) else None,passed=int(np.sum(np.all(errors<=.09+1e-12,axis=1))),pass_rate=float(np.sum(np.all(errors<=.09+1e-12,axis=1))/len(p)))

def make_rows(m,indices,pred,method,protocol,fold):
    rows=m.iloc[indices][['index','file','relative','Y','T','scope','group']].copy();rows['predY']=pred[:,0];rows['predT']=pred[:,1];rows['eY']=abs(rows.predY-rows.Y);rows['eT']=abs(rows.predT-rows['T']);rows['passed']=(rows.eY<=.09+1e-12)&(rows.eT<=.09+1e-12);rows['method']=method;rows['protocol']=protocol;rows['fold']=fold;return rows

def train(config,X,m,tr):
    params=dict(config);feature=params.pop('feature','P12');cls=LocalOrigins if params.pop('local',False) else JointOrigins;model=cls(**params).fit(BANK[feature],m[['Y','T']].to_numpy(),m.group.to_numpy(),m.scope.to_numpy(),tr,m.relative.to_numpy());model.feature_name=feature;return model

def infer(model,ix):return model.predict(BANK[model.feature_name][ix])

def rank(m,ix,yt):
    e=abs(yt-m[['Y','T']].to_numpy()[ix]);return (int(np.sum(np.any(e>.09+1e-12,axis=1))),float(e.max()),float(e.mean(axis=0).sum()))

def select(X,m,tr,seed):
    strata=m.scope.to_numpy().astype(str);groups=np.where(m.scope.to_numpy()=='mixed',m.condition.to_numpy(),m.relative.to_numpy())
    inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(tr,strata[tr],groups[tr]));scores=[]
    for config in CONFIGS:
        ix=[];ys=[]
        for a,b in inner:
            model=train(config,X,m,tr[a]);out=infer(model,tr[b]);ix.extend(tr[b]);ys.extend(out['YT'])
        scores.append(rank(m,np.array(ix),np.array(ys)))
    best=min(range(len(scores)),key=lambda i:scores[i]);return CONFIGS[best],dict(selected=CONFIGS[best],scores=[dict(config=c,score=s) for c,s in zip(CONFIGS,scores)])

def frozen(X):
    params=json.loads((HERE/'assets/model.json').read_text());W=np.array(params['W']);choices=[]
    for key,(axis,b,s,c) in params['branches'].items():
        z=(X-np.array(b))@W.T;s=W@np.array(s);c=W@np.array(c);v=z@s/(s@s)
        for _ in range(8):
            j=s+2*v[:,None]*c;v+=np.sum(j*(z-v[:,None]*s-v[:,None]**2*c),axis=1)/np.maximum(np.sum(j*j,axis=1),1e-20)
        cost=np.sum((z-v[:,None]*s-v[:,None]**2*c)**2,axis=1);p=np.c_[v,np.zeros(len(v))] if axis=='Y' else np.c_[np.zeros(len(v)),v];choices.append((p,cost))
    best=np.argmin(np.stack([a[1] for a in choices],axis=1),axis=1);return np.array([choices[k][0][i] for i,k in enumerate(best)])

if __name__=='__main__':
    m=pd.read_csv(DATA/'manifest.csv');a=np.load(DATA/'measurements.npz');X=a['X'];assert m.success.all() and X.shape==(341,12)
    maps=FeatureMaps(a['points'],a['regions']);BANK={name:maps.transform(a['D'],name) for name in ['P12','QR18','QR36','raw1260']}
    # Preserve the agreed 280-image outer split exactly, then append grouped mixed inputs and center samples.
    original=json.loads((HERE.parent/'codex_experiments_20261014/multiref_4fold_20261021/splits.json').read_text());fold=np.full(len(m),-1)
    for f,s in enumerate(original):fold[(m.scope=='single')&m.file.isin(s['test'])]=f
    mixed=m[m.scope=='mixed'];condition_table=mixed.drop_duplicates('condition').sort_values(['T','Y']);rng=np.random.default_rng(20261022)
    # Balance large-T conditions across folds; all three repetitions of a condition stay together.
    counter=0
    for t,g in condition_table.groupby('T',sort=True):
        for j,condition in enumerate(rng.permutation(g.condition.to_numpy())):fold[(m.scope=='mixed')&(m.condition==condition)]=(counter+j)%4
        counter+=len(g)
    for j,index in enumerate(m.index[m.scope=='center']):fold[index]=j%4
    assert np.all(fold>=0)
    (OUT/'protocol.json').write_text(json.dumps(dict(selection=SCORE_RULE,configs=CONFIGS,original280_split='multiref_4fold_20261021/splits.json',mixed_repetitions_grouped=True,scope='expanded train split contains 210 old single-axis inputs plus training-only mixed and center images; not identical training corpus to earlier old-only OOF scores'),indent=2))
    splits=[dict(fold=f,train=m.relative[fold!=f].tolist(),test=m.relative[fold==f].tolist()) for f in range(4)];(OUT/'splits.json').write_text(json.dumps(splits,indent=2))
    records=[];selections=[];audits=[]
    with threadpool_limits(1):
        for f in range(4):
            tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);config,selection=select(X,m,tr,20261022+f);selection['fold']=f;selections.append(selection)
            model=train(config,X,m,tr);model.save(OUT/f'fold{f}_model.npz');pred=infer(model,te);rows=make_rows(m,te,pred['YT'],'multi_center_joint','oof4',f)
            rows['center']=pred['center_id'];rows['cost']=pred['cost'];rows['margin']=pred['margin'];rows['alternativeY']=pred['alternative_YT'][:,0];rows['alternativeT']=pred['alternative_YT'][:,1];records.append(rows)
            for name,control in [('single_center_joint',train(dict(degree=2,shrink=.5,origins='fixed'),X,m,tr)),('shared_surface_center0',copy.deepcopy(model))]:
                if name=='shared_surface_center0':control.centers=control.centers[:1];control.origin_ids=control.origin_ids[:1];control.cw=control.cw[:1]
                records.append(make_rows(m,te,infer(control,te)['YT'],name,'oof4',f))
            audits.append(dict(fold=f,training_rows=tr.tolist(),test_rows=te.tolist(),candidate_ids=model.origin_ids,candidate_count=len(model.centers),no_test_rows_used=not bool(set(tr)&set(te))))
            print('fold',f,config,'candidates',len(model.centers),metrics(rows),flush=True)
        tr=np.arange(len(m));config,selection=select(X,m,tr,20261030);selections.append(dict(full_selection=selection));model=train(config,X,m,tr);model.save(HERE/'model.npz');pred=infer(model,tr)
        rows=make_rows(m,tr,pred['YT'],'multi_center_joint','resub341',-1);rows['center']=pred['center_id'];rows['cost']=pred['cost'];rows['margin']=pred['margin'];rows['alternativeY']=pred['alternative_YT'][:,0];rows['alternativeT']=pred['alternative_YT'][:,1];records.append(rows)
        for name,control in [('single_center_joint',train(dict(degree=2,shrink=.5,origins='fixed'),X,m,tr)),('shared_surface_center0',copy.deepcopy(model))]:
            if name=='shared_surface_center0':control.centers=control.centers[:1];control.origin_ids=control.origin_ids[:1];control.cw=control.cw[:1]
            records.append(make_rows(m,tr,infer(control,tr)['YT'],name,'resub341',-1))
        old=frozen(X);records.append(make_rows(m,tr,old,'frozen_Y_or_T_wide_init','resub341',-1));strict=old.copy();strict[~m.strict_success.to_numpy()]=np.nan;records.append(make_rows(m,tr,strict,'frozen_Y_or_T_strict_init','resub341',-1))
    p=pd.concat(records,ignore_index=True);p.to_csv(OUT/'predictions.csv',index=False);p[~p.passed].to_csv(OUT/'failures.csv',index=False)
    stats=[]
    for (name,protocol),g in p.groupby(['method','protocol']):
        for scope in ['all','single','mixed','center']:
            sub=g if scope=='all' else g[g.scope==scope];stats.append(dict(method=name,protocol=protocol,subset=scope,**metrics(sub)))
    summary=pd.DataFrame(stats);summary.to_csv(OUT/'summary.csv',index=False)
    (OUT/'selection.json').write_text(json.dumps(selections,indent=2));(OUT/'audit.json').write_text(json.dumps(audits,indent=2));(OUT/'final_centers.json').write_text(json.dumps(dict(config=config,candidate_ids=model.origin_ids,centers=model.centers.tolist(),source='training-only estimated intercepts plus available actual center feature vectors'),indent=2))
    print(summary[summary.subset=='all'].to_string(index=False),flush=True)
