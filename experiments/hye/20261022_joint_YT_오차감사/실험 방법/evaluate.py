"""Frozen extension of prior nested development CV, never an independent test.

Prior 15 candidates' inner scores are reused with EXACTLY the same splits.
Four new robustness candidates were specified before their results were viewed.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;OLD=H.parent/'joint_yt_disentangle_20261022';DATA=H.parent/'joint_yt_generalization_20261022'
for p in [DATA,OLD,H/'robust']:sys.path.insert(0,str(p))
from data import load,metric
from learned_model import LearnedSeparation
from robust_model import RobustBaggedForward

NEW=[dict(fit_mode='new_condition_means',shrink=.1,seed=20261022,huber=h,bags=b) for h,b in [(None,1),(1.5,1),(None,12),(1.5,12)]]
OLD_CONFIGS=json.loads((OLD/'protocol.json').read_text())['configs']
OLD_SELECTION=json.loads((OLD/'selection.json').read_text())

def get_model(k,a):
    return LearnedSeparation(OLD_CONFIGS[k],a['points'],a['regions']) if k<len(OLD_CONFIGS) else RobustBaggedForward(NEW[k-len(OLD_CONFIGS)])

def select(m,a,audit):
    D=a['D'];y=m[['Y','T']].to_numpy();scores=list(audit['scores']);records=[]
    for j,c in enumerate(NEW):
        pp=[];yy=[]
        for split in audit['inner_splits']:
            it=np.array(split['train']);iv=np.array(split['validation']);iv=iv[m.iloc[iv].dataset.to_numpy()==20261020]
            assert not set(m.iloc[it].cv_group)&set(m.iloc[iv].cv_group)
            model=RobustBaggedForward(c).fit(D[it],y[it],m.iloc[it]);p=model.predict(D[iv]);pp.append(p);yy.append(y[iv])
        error=abs(np.vstack(pp)-np.vstack(yy));k=len(OLD_CONFIGS)+j
        records.append(dict(candidate=k,config=c,failures=int(np.any(error>.09+1e-12,axis=1).sum()),maximum=float(error.max()),mean=float(error.mean())))
    scores.extend(records);rank=lambda r:(r['failures'],r['maximum'],r['mean'])
    return dict(extended=min(scores,key=rank)['candidate'],robust_only=min(records,key=rank)['candidate']),dict(inner_splits=audit['inner_splits'],scores=scores)

def prediction_rows(m,ix,p,method,fold,protocol):
    z=m.iloc[ix].copy();z['predY']=p[:,0];z['predT']=p[:,1];z['eY']=abs(z.predY-z.Y);z['eT']=abs(z.predT-z['T']);z['passed']=(z.eY<=.09+1e-12)&(z.eT<=.09+1e-12);z['method']=method;z['fold']=fold;z['protocol']=protocol;z['correctionY']=-p[:,0];z['correctionT']=-p[:,1];return z

def main():
    m,a=load();D=a['D'];y=m[['Y','T']].to_numpy();predictions=[];selection=[]
    for f in range(4):
        tr=np.flatnonzero(m.fold.to_numpy()!=f);te=np.flatnonzero(m.fold.to_numpy()==f);start=time.monotonic()
        chosen,audit=select(m,a,OLD_SELECTION[f]);selection.append(dict(fold=f,chosen=chosen,**audit))
        for method,k in chosen.items():
            model=get_model(k,a).fit(D[tr],y[tr],m.iloc[tr]);p=model.predict(D[te]);predictions.append(prediction_rows(m,te,p,method,f,'nested_development_oof'))
            joblib.dump(dict(model=model,config=OLD_CONFIGS[k] if k<len(OLD_CONFIGS) else NEW[k-len(OLD_CONFIGS)],candidate=k,train_indices=tr,ids=a['ids'],points=a['points']),H/f'{method}_fold{f}.joblib')
            n=m.iloc[te].dataset.to_numpy()==20261020;print('fold',f,method,'candidate',k,metric(y[te][n],p[n]),flush=True)
        print('seconds',time.monotonic()-start,flush=True)
        pd.concat(predictions).to_csv(H/'progress_predictions.csv',index=False);(H/'selection.json').write_text(json.dumps(selection,indent=2))
    chosen,audit=select(m,a,OLD_SELECTION[-1]['full_selection']);selection.append(dict(full_selection=dict(chosen=chosen,**audit)))
    for method,k in chosen.items():
        model=get_model(k,a).fit(D,y,m);p=model.predict(D);predictions.append(prediction_rows(m,np.arange(len(m)),p,method,-1,'training_reapplication'))
        joblib.dump(dict(model=model,config=OLD_CONFIGS[k] if k<len(OLD_CONFIGS) else NEW[k-len(OLD_CONFIGS)],candidate=k,train_indices=np.arange(len(m)),ids=a['ids'],points=a['points']),H/f'{method}_model.joblib')
    result=pd.concat(predictions,ignore_index=True);result.to_csv(H/'predictions.csv',index=False,encoding='utf-8-sig');(H/'selection.json').write_text(json.dumps(selection,indent=2))
    summary=[]
    for (method,protocol),g in result.groupby(['method','protocol']):
        for ds in ['all',20261014,20261020]:
            z=g if ds=='all' else g[g.dataset==ds];summary.append(dict(method=method,protocol=protocol,dataset=ds,**metric(z[['Y','T']],z[['predY','predT']])))
    pd.DataFrame(summary).to_csv(H/'summary.csv',index=False);print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    protocol=H/'protocol.json'
    if not protocol.exists():
        protocol.write_text(json.dumps(dict(evaluation_scope='ADAPTIVE DEVELOPMENT DATA; NOT INDEPENDENT NEW-PHOTO PERFORMANCE',primary='extended: old15 + preregistered robustness4, same inner group folds, rank failures/max/mean',secondary='robust_only: 4 robustness candidates, prespecified separate family comparison',new_configs=NEW,old_selection_sha256=hashlib.sha256((OLD/'selection.json').read_bytes()).hexdigest(),old_protocol_sha256=hashlib.sha256((OLD/'protocol.json').read_bytes()).hexdigest(),rules='Same-command repeats grouped; 3 new centers grouped fold0; no test labels in fitting or filename at inference; full341 accounted for; no threshold changes',independent_test='No existing341 photo can be a clean test after development exposure'),indent=2),encoding='utf-8')
    with threadpool_limits(1):main()
