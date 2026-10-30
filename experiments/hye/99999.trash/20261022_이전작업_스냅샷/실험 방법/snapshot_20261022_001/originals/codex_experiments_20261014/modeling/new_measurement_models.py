"""Independent new measurements through the same model/CV implementation."""
import json, time
import numpy as np
import pandas as pd
from experiment import HERE, ROOT, MotionModel, load_data, configs, split_folds, evaluate_config, records, metrics, objective

def load_new(name):
    base=load_data()
    source=ROOT/'codex_experiments_20261014'/'measurement'/f'{name}.npz'
    if name=='direct_p12':
        source=ROOT/'codex_experiments_20261014'/'direct_alignment'/'direct_p12.npz'
    z=np.load(source)
    names=z['names'].astype(str); ix=np.array([list(names).index(n) for n in base['names']]); center=list(names).index('center.bmp')
    if name=='direct_p12':
        F=base['F']; x=(F.x.to_numpy()-1906)/486; y=(F.y.to_numpy()-1070)/122
        B=np.column_stack([np.ones(len(F)),x,y,x*x,x*y,y*y])
        X=z['X']-z['X'][center]
        base['D']=np.einsum('pk,nkd->npd',B,X[ix].reshape(len(ix),6,2))
        base['quality']=np.ones(base['D'].shape[:2])
    else:
        F=pd.DataFrame(z['points'],columns=['x','y'])
        F['region']=np.array(['QR좌','QR중','QR우','판나사','판나사'])[z['regions'].astype(int)]
        F['strength']=z['strengths']
        base['F']=F; base['D']=(z['D']-z['D'][center])[ix]
        # Individual track confidence conventions differ (LK eigenvalues / GN
        # residual), so these runs do not apply legacy NCC confidence weights.
        base['quality']=np.ones(base['D'].shape[:2])
    return base

def nested_run(data,cs,prefix,protocol):
    indices=np.arange(len(data['D'])); records_out=[]; chosen=[]; innerm=[]
    outer=split_folds(data,indices) if protocol=='stratified' else [(indices[data['sessions']!=s],indices[data['sessions']==s]) for s in np.unique(data['sessions'])]
    start=time.time()
    for ofold,(tr,te) in enumerate(outer):
        inner=split_folds(data,tr,seed=20261015+ofold); best=None
        for c in cs:
            f=evaluate_config(data,c,inner); obj=objective(f)
            innerm.append(dict(outer_fold=ofold,config=c['id'],**metrics(f)))
            if best is None or obj<best[0]: best=(obj,c)
        c=best[1]; model=MotionModel(c).fit(data['D'][tr],data['F'],data['quality'][tr],data['axes'][tr],data['values'][tr])
        f=records(data,te,model.predict(data['D'][te])); f['outer_fold']=ofold; f['config']=c['id']; records_out.append(f)
        chosen.append(dict(outer_fold=ofold,config=c['id'],test_sessions=','.join(np.unique(data['sessions'][te])),inner_objective=str(best[0]),**metrics(f)))
        pd.concat(records_out,ignore_index=True).to_csv(HERE/f'{prefix}_nested_{protocol}_predictions.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(chosen).to_csv(HERE/f'{prefix}_nested_{protocol}_selections.csv',index=False,encoding='utf-8-sig')
        print(prefix,protocol,ofold,c['id'],'err',(~f.axis_correct).sum(),'elapsed',round(time.time()-start),flush=True)
    (HERE/f'{prefix}_nested_{protocol}_metrics.json').write_text(json.dumps(metrics(pd.concat(records_out)),indent=2),encoding='utf-8')
    pd.DataFrame(innerm).to_csv(HERE/f'{prefix}_nested_{protocol}_inner.csv',index=False,encoding='utf-8-sig')

def main():
    # Full legacy search remains described in experiment.py. New independent
    # measurements use all QR points, and fixed spatial representations.
    cs=[c for c in configs() if c['subset']=='all' and c['kind']!='raw' and not any(k in c for k in ['quality','topk','classifier'])]
    (HERE/'new_candidate_configs.json').write_text(json.dumps(cs,indent=2),encoding='utf-8')
    allmetrics=[]; allpred=[]
    for name in ['gn17','direct_p12','lk31_blur','lk31','lk51','lk15']:
        data=load_new(name)
        cset=[c for c in cs if c['kind']=='p12'] if name=='direct_p12' else cs
        for c in cset:
            f=evaluate_config(data,c,split_folds(data,np.arange(len(data['D'])))); f['measurement']=name; f['config']=c['id']
            allpred.append(f); row=dict(measurement=name,config=c['id'],**metrics(f)); allmetrics.append(row)
            print('newbroad',name,c['id'],row['axis_errors'],'ymax',row['Y_matched_max_error'],'tmax',row['T_matched_max_error'],flush=True)
        pd.DataFrame(allmetrics).to_csv(HERE/'new_broad_metrics.csv',index=False,encoding='utf-8-sig')
        pd.concat(allpred,ignore_index=True).to_csv(HERE/'new_broad_predictions.csv',index=False,encoding='utf-8-sig')
    # QR-only nested selection of the original 71 candidates, then independently
    # extracted GN and direct-pixel measurements. A test set never picks its model.
    oldcs=[c for c in configs() if c['subset']!='screws']
    for protocol in ['stratified','session_out']:
        nested_run(load_data(),oldcs,'legacy_qr_only',protocol)
        nested_run(load_new('gn17'),cs,'gn17',protocol)
        nested_run(load_new('direct_p12'),[c for c in cs if c['kind']=='p12'],'direct_p12',protocol)

if __name__=='__main__': main()
