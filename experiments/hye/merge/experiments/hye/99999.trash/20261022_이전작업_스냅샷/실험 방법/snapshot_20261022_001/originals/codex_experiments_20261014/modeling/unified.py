"""Unified nested selection of new extractors and models (raw command labels).

Inner folds choose measurement, motion model and optional command grid.
The isolated subset permits only finite-support LK or per-QR-cropped GN.
"""
import json,pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import HERE,ROOT,MotionModel,split_folds,evaluate_config,records,metrics,objective

def independent_data(name):
    folder=ROOT/'codex_experiments_20261014'
    source=(folder/'direct_alignment' if name.startswith('direct') else folder/'measurement')/f'{name}.npz'
    z=np.load(source); names=z['names'].astype(str)
    target=sorted([n for n in names if n!='center.bmp'],key=lambda n:(n[0],float(n[1:-4])))
    ix=np.array([list(names).index(n) for n in target]); center=list(names).index('center.bmp')
    axes=np.array([n[0] for n in target]);values=np.array([float(n[1:-4]) for n in target])
    sessions=np.array([('T' if a=='T' else ('Y2' if abs(round(v/.05))%2 else 'Y1'))+('+' if v>0 else '-') for a,v in zip(axes,values)])
    if name.startswith('direct'):
        rows=[]
        for r,(a,b,c,d) in zip(['QR좌','QR중','QR우'],[(1563,982,1721,1153),(1821,983,1980,1154),(2079,984,2239,1157)]):
            rows.extend((x,y,r) for x in np.linspace(a+30,c-30,5) for y in np.linspace(b+30,d-30,5))
        F=pd.DataFrame(rows,columns=['x','y','region']);x=(F.x.to_numpy()-1906)/486;y=(F.y.to_numpy()-1070)/122
        B=np.column_stack([np.ones(len(F)),x,y,x*x,x*y,y*y])
        D=np.einsum('pk,nkd->npd',B,(z['X'][ix]-z['X'][center]).reshape(len(ix),6,2))
    else:
        F=pd.DataFrame(z['points'],columns=['x','y']);F['region']=np.array(['QR좌','QR중','QR우','판나사','판나사'])[z['regions'].astype(int)]
        F['strength']=z['strengths'];D=(z['D']-z['D'][center])[ix]
    return dict(D=D,quality=np.ones(D.shape[:2]),F=F,names=np.array(target),axes=axes,values=values,sessions=sessions)

def candidate_list():
    candidates=[]
    for name in ['lk31','lk51','gn17','gn31','direct_p12_newinit','gn17_spread','gn17_spread14']:
        settings=[(1,.1,'none',False),(2,.1,'none',False),(2,.1,'dy',False),(2,.1,'both',False),(2,1.,'none',False),(2,.1,'dy',True)]
        for degree,shrink,nuisance,grid in settings:
            candidates.append(dict(measurement=name,id=f'{name}_{len(candidates):02d}',kind='p12',degree=degree,shrink=shrink,nuisance=nuisance,grid=grid,anchor=False,subset='all'))
    for name in ['lk51','gn17']:
        candidates.append(dict(measurement=name,id=name+'_qr18',kind='qr18',degree=1,shrink=.001,nuisance='none',grid=False,anchor=False,subset='all'))
    return candidates

def is_isolated(c): return c['measurement'] in ['lk31','lk51','gn17_spread','gn17_spread14']

def save_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def run_nested(datasets,cs,protocol):
    data=next(iter(datasets.values()));indices=np.arange(len(data['D']));start=time.time()
    outer=split_folds(data,indices) if protocol=='stratified' else [(indices[data['sessions']!=s],indices[data['sessions']==s]) for s in np.unique(data['sessions'])]
    allrec={'strict':[],'isolated':[]};selections={'strict':[],'isolated':[]};innerrows=[]
    for fold,(tr,te) in enumerate(outer):
        inner=split_folds(data,tr,seed=20261015+fold); best={'strict':None,'isolated':None}
        for c in cs:
            f=evaluate_config(datasets[c['measurement']],c,inner);obj=objective(f)
            innerrows.append(dict(outer_fold=fold,candidate=c['id'],measurement=c['measurement'],objective=list(obj),**metrics(f)))
            for group in ['strict','isolated']:
                if group=='isolated' and not is_isolated(c):continue
                if best[group] is None or obj<best[group][0]:best[group]=(obj,c)
        for group in ['strict','isolated']:
            obj,c=best[group]; d=datasets[c['measurement']]
            m=MotionModel(c).fit(d['D'][tr],d['F'],d['quality'][tr],d['axes'][tr],d['values'][tr])
            f=records(d,te,m.predict(d['D'][te]));f['outer_fold']=fold;f['candidate']=c['id'];f['measurement']=c['measurement'];allrec[group].append(f)
            selections[group].append(dict(outer_fold=fold,candidate=c['id'],measurement=c['measurement'],test_sessions=','.join(np.unique(data['sessions'][te])),inner_objective=list(obj),**metrics(f)))
            pd.concat(allrec[group],ignore_index=True).to_csv(HERE/f'unified_{group}_{protocol}_predictions.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(selections[group]).to_csv(HERE/f'unified_{group}_{protocol}_selections.csv',index=False,encoding='utf-8-sig')
            print('unified',group,protocol,fold,c['id'],'errors',(~f.axis_correct).sum(),'seconds',round(time.time()-start),flush=True)
    for group in allrec:save_json(HERE/f'unified_{group}_{protocol}_metrics.json',metrics(pd.concat(allrec[group],ignore_index=True)))
    pd.DataFrame(innerrows).to_csv(HERE/f'unified_{protocol}_inner_metrics.csv',index=False,encoding='utf-8-sig')

def final_models(datasets,cs):
    data=next(iter(datasets.values()));folds=split_folds(data,np.arange(len(data['D'])))
    best={'strict':None,'isolated':None};rows=[];pred=[]
    for c in cs:
        f=evaluate_config(datasets[c['measurement']],c,folds);obj=objective(f);f['candidate']=c['id'];f['measurement']=c['measurement'];pred.append(f)
        rows.append(dict(candidate=c['id'],measurement=c['measurement'],**metrics(f)))
        for group in best:
            if group=='isolated' and not is_isolated(c):continue
            if best[group] is None or obj<best[group][0]:best[group]=(obj,c)
    pd.DataFrame(rows).to_csv(HERE/'unified_full_cv_selection_metrics.csv',index=False,encoding='utf-8-sig')
    pd.concat(pred,ignore_index=True).to_csv(HERE/'unified_full_cv_selection_predictions.csv',index=False,encoding='utf-8-sig')
    for group,(_,c) in best.items():
        d=datasets[c['measurement']];m=MotionModel(c).fit(d['D'],d['F'],d['quality'],d['axes'],d['values'])
        features_file=('measurement/'+c['measurement']+'_features.npz') if c['measurement'].startswith('gn17_spread') else ('measurement/features.npz' if not c['measurement'].startswith('direct') else None)
        payload=dict(model=m,config=c,measurement=c['measurement'],features_file=features_file,reference_points=d['F'][['x','y']].to_numpy(),reference_regions=d['F'].region.to_numpy(),n_train=len(d['names']),label_basis='filename command labels',scope=group)
        with open(HERE/f'deployed_{group}.pkl','wb') as f:pickle.dump(payload,f)
        save_json(HERE/f'deployed_{group}_metadata.json',{k:v for k,v in payload.items() if k not in ['model','reference_points','reference_regions']})
        if group=='isolated':
            with open(HERE/'deploy_model.pkl','wb') as f:pickle.dump(payload,f)
            metadata={k:v for k,v in payload.items() if k not in ['model','reference_points','reference_regions']}
            metadata.update(reference_points=payload['reference_points'].tolist(),reference_regions=payload['reference_regions'].tolist(),selection_criterion='full-data 3-fold CV lexicographic axis/sign errors, strict failures, max vector error, mean squared vector error',unbiased_cv=['unified_isolated_stratified_metrics.json','unified_isolated_session_out_metrics.json'],runtime_input='two images only; no filename/session metadata')
            save_json(HERE/'deployment.json',metadata)
        print('deploy',group,c['id'],flush=True)

def screws_fallback(datasets,cs):
    rows=[];pred=[]
    for name in ['lk31','gn17']:
        d=datasets[name];ix=np.arange(len(d['D']));outer_session=[(ix[d['sessions']!=s],ix[d['sessions']==s]) for s in np.unique(d['sessions'])]
        for c0 in [c for c in cs if c['measurement']==name and c['kind']=='p12' and not c['grid'] and c['shrink']==.1 and c['nuisance'] in ['none','dy']]:
            for subset in ['all','screws']:
                c=dict(c0);c.update(subset=subset,id=c0['id']+'_'+subset)
                for cv,folds in [('stratified',split_folds(d,ix)),('session_out',outer_session)]:
                    f=evaluate_config(d,c,folds);f['measurement']=name;f['candidate']=c['id'];f['protocol']=cv;f['subset']=subset;pred.append(f)
                    rows.append(dict(measurement=name,candidate=c['id'],protocol=cv,subset=subset,**metrics(f)))
    pd.DataFrame(rows).to_csv(HERE/'new_screw_fallback_metrics.csv',index=False,encoding='utf-8-sig')
    pd.concat(pred,ignore_index=True).to_csv(HERE/'new_screw_fallback_predictions.csv',index=False,encoding='utf-8-sig')

def main():
    cs=candidate_list();save_json(HERE/'unified_candidate_configs.json',cs)
    datasets={n:independent_data(n) for n in sorted({c['measurement'] for c in cs})}
    for n,d in datasets.items():assert np.array_equal(d['names'],next(iter(datasets.values()))['names']),n
    # The isolated subset includes the QR18 model on finite-support LK51.
    for protocol in ['stratified','session_out']:run_nested(datasets,cs,protocol)
    final_models(datasets,cs);screws_fallback(datasets,cs)
    save_json(HERE/'unified_metadata.json',dict(candidates=len(cs),isolated_candidates=sum(is_isolated(c) for c in cs),outer_protocols=['stratified 3-fold','6 leave-acquisition-session-out folds'],inner_protocol='session-stratified 3-fold; seed 20261015 + outer fold',ground_truth='unadjusted filename command labels',runtime_inputs='center image and current image only; no filename/session hint',minimum_command_steps={'Y':.05,'T':.01},tolerance={'Y':.09,'T':.09},selection='minimize joint axis/sign mistakes, then tolerance failures, then worst vector error, then squared vector error',support_scope={'strict':'QR objective windows; global cubic prefilter may have negligible IIR tails','isolated':'finite-support LK windows or complete per-QR cropping before filtering'},deployment_selection='same fixed candidate list chosen using all-data 3-fold CV; not an unbiased final test score'))

if __name__=='__main__':main()
