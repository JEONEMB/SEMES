"""Stress tests for the promising QR-cropped 205-point, six-mode calibration.
Settings chosen after development experiments: results are not an untouched test.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import zero_experiment as Z

OUT=Z.HERE/'calibrated_runs'

def main():
    OUT.mkdir(exist_ok=True); m,X,x,U=Z.load('gn17_spread14')
    ids=np.flatnonzero(m.axis!='C'); rows=[]; pred=[]
    cfg=dict(mode='inferred_session',quadratic=True,nuisance=False)
    folds=[]
    for cv in ['interleaved','stratified5']:
        folds.append((cv,list(Z.splits(m,cv))))
    block=[]
    for k in range(5):
        te=[]
        for s in m.session[m.axis!='C'].unique():
            i=m.index[m.session==s].to_numpy(); i=i[np.argsort(m.value.iloc[i])]
            te.extend(np.array_split(i,5)[k])
        te=np.sort(te);block.append((str(k),np.setdiff1d(ids,te),te))
    folds.append(('blocked_within_session5',block))
    for seed in range(20):
        fs=[(str(k),ids[a],ids[b]) for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=7000+seed).split(ids,m.session.iloc[ids]))]
        folds.append((f'repeated5_seed{seed}',fs))
    for cv,fs in folds:
        ps=[]
        for fold,tr,te in fs:
            mdl=Z.fit(m,X,tr,m.value.to_numpy(),cfg['mode'],cfg['quadratic'])
            d=Z.predict(m,X,te,mdl,cfg['mode']).assign(fold=fold,cv=cv)
            ps.append(d)
        d=pd.concat(ps);rows.append(dict(cv=cv,**Z.score(d)));pred.append(d)
    pd.DataFrame(rows).to_csv(OUT/'stress_metrics.csv',index=False)
    pd.concat(pred).to_csv(OUT/'stress_predictions.csv',index=False)
    # Full fit is for use, never scored as validation accuracy.
    model=Z.fit(m,X,ids,m.value.to_numpy(),cfg['mode'],cfg['quadratic'])
    params=dict(W=model[1].tolist(),branches={k:[a,b.tolist(),s.tolist(),c.tolist()] for k,(a,b,s,c) in model[0].items()},measurement='gn17_spread14',config=cfg,zero_feature=X[m.axis=='C'][0].tolist(),point_feature_path='measurement/gn17_spread14_features.npz',scope='Same six calibration-run origin modes, nonzero single-axis inputs. New runs/zero modes and mixed motion unvalidated.')
    (OUT/'model.json').write_text(json.dumps(params,indent=2),encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
