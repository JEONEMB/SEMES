"""Known actuator-step prior: a separately scoped experiment, not a free gain.

Valid only for nonzero single-axis command-grid inputs, with the same trained
zero origin. Off-grid/mixed motions and stationary frames need separate models.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import zero_experiment as Z

OUT=Z.HERE/'lattice'

def predict(m,X,te,model,mode,grid):
    if not grid: return Z.predict(m,X,te,model,mode)
    branches,W=model; out=[]
    for name,(axis,b,s,c) in branches.items():
        delta=.05 if axis=='Y' else .01
        Q=(X[te]-b)@W.T; sw=s@W.T; cw=c@W.T
        v=Q@sw/(sw@sw)
        for _ in range(8):
            J=sw+2*v[:,None]*cw; r=Q-v[:,None]*sw-(v*v)[:,None]*cw
            v+=np.sum(J*r,1)/np.maximum(np.sum(J*J,1),1e-20)
        vs=(np.round(v/delta)[:,None]+np.arange(-3,4)[None,:])*delta
        r=Q[:,None,:]-vs[:,:,None]*sw-vs[:,:,None]**2*cw
        costs=np.sum(r*r,axis=-1)
        if grid=='nonzero': costs=np.where(np.abs(vs)<delta*.1,np.inf,costs)
        if mode=='inferred_session': costs=np.where(vs*(1 if name.endswith('+') else -1)>=0,costs,np.inf)
        j=costs.argmin(1); out.append((name,axis,vs[np.arange(len(te)),j],costs[np.arange(len(te)),j]))
    costs=np.stack([p[3] for p in out],1); j=costs.argmin(1)
    return pd.DataFrame([dict(file=m.file.iloc[i],axis=m.axis.iloc[i],value=m.value.iloc[i],session=m.session.iloc[i],pred_axis=out[j[k]][1],pred=out[j[k]][2][k],branch=out[j[k]][0],cost=costs[k,j[k]]) for k,i in enumerate(te)])

def run(m,X,x,U,tr,te,c):
    lab=m.value.to_numpy().copy()
    if c['correction']: lab,_=Z.correct_labels(m,x,tr,'session')
    mdl=Z.fit(m,X,tr,lab,c['mode'],c['quadratic'],U if c['nuisance'] else None)
    return predict(m,X,te,mdl,c['mode'],c['grid'])

def key(df):
    sc=Z.score(df)
    return (1-sc['axis_sign_accuracy'],1-sc['strict_pass09'],sc['vector_Y_max']+sc['vector_T_max'],sc['Y_matched_MAE']+sc['T_matched_MAE'])

def main():
    import argparse
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--measurements',nargs='+');parser.add_argument('--tag',default='lattice');args=parser.parse_args()
    OUT=Z.HERE/args.tag
    OUT.mkdir(exist_ok=True)
    configs=[dict(mode=mode,correction=False,quadratic=q,nuisance=n,grid=grid) for mode in ['center','free','inferred_session'] for q in [False,True] for n in [False,True] for grid in [None,'nonzero']]
    (OUT/'configs.json').write_text(json.dumps(configs,indent=2),encoding='utf8')
    all_rows=[];all_pred=[];nested_rows=[];nested_pred=[];selection=[]
    for meas in (args.measurements or [None,'gn17','direct_p12']):
        tag=meas or 'legacy_gn17'; m,X,x,U=Z.load(meas)
        for cv in ['interleaved','stratified5','session_out']:
            for j,c in enumerate(configs):
                dfs=[run(m,X,x,U,tr,te,c).assign(fold=f) for f,tr,te in Z.splits(m,cv)]
                df=pd.concat(dfs);meta=dict(measurement=tag,cv=cv,candidate=j,**c)
                all_rows.append(dict(**meta,**Z.score(df)));all_pred.append(df.assign(**meta))
            pd.DataFrame(all_rows).to_csv(OUT/'fixed_metrics.csv',index=False)
            print(tag,cv,'fixed done',flush=True)
        # Nested 3-fold: candidate selection sees only the outer train images.
        ids=np.flatnonzero(m.axis!='C')
        collected=[]
        for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=3014).split(ids,m.session.iloc[ids])):
            tr,te=ids[a],ids[b]; trials=[]
            inner=list(StratifiedKFold(3,shuffle=True,random_state=4014+fold).split(tr,m.session.iloc[tr]))
            for j,c in enumerate(configs):
                p=pd.concat([run(m,X,x,U,tr[ii],tr[jj],c) for ii,jj in inner])
                trials.append((key(p),j))
            _,best=min(trials)
            df=run(m,X,x,U,tr,te,configs[best]).assign(fold=fold,candidate=best,measurement=tag)
            collected.append(df);selection.append(dict(measurement=tag,fold=fold,candidate=best,**configs[best]))
            print(tag,'nested fold',fold,'candidate',best,'errors',int((df.axis!=df.pred_axis).sum()),flush=True)
        df=pd.concat(collected);nested_pred.append(df);nested_rows.append(dict(measurement=tag,**Z.score(df)))
        pd.DataFrame(nested_rows).to_csv(OUT/'nested_metrics.csv',index=False)
        pd.concat(nested_pred).to_csv(OUT/'nested_predictions.csv',index=False)
        pd.DataFrame(selection).to_csv(OUT/'nested_selections.csv',index=False)
    pd.concat(all_pred).to_csv(OUT/'fixed_predictions.csv',index=False)
    print(pd.DataFrame(nested_rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
