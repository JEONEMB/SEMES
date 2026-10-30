"""Explicit point-count and commanded-step lattice ablations."""
import json
import numpy as np
import pandas as pd
from experiment import HERE, configs, load_data, split_folds, evaluate_config, metrics
from new_measurement_models import load_new, nested_run

def main():
    rows=[]; rec=[]; cs=[]
    for k in [25,50,100,232,400,600,0]:
        for degree in [1,2]:
            for selection in ['information','strength'] if k else ['information']:
                cs.append(dict(kind='p12',degree=degree,shrink=.1,anchor=False,subset='all',nuisance='none',topk=k,selection=selection,id=f'point_{k}_{degree}_{selection}'))
    (HERE/'point_candidate_configs.json').write_text(json.dumps(cs,indent=2),encoding='utf-8')
    for name in ['gn17','gn31']:
        try: data=load_new(name)
        except FileNotFoundError: continue
        for c in cs:
            f=evaluate_config(data,c,split_folds(data,np.arange(len(data['D'])))); f['measurement']=name; f['config']=c['id']; rec.append(f)
            row=dict(measurement=name,config=c['id'],**metrics(f)); rows.append(row)
            print('points',name,c['id'],row['axis_errors'],flush=True)
        pd.DataFrame(rows).to_csv(HERE/'point_selection_metrics.csv',index=False,encoding='utf-8-sig')
        pd.concat(rec,ignore_index=True).to_csv(HERE/'point_selection_predictions.csv',index=False,encoding='utf-8-sig')
        nested_run(data,cs,f'{name}_points','stratified')
        nested_run(data,cs,f'{name}_points','session_out')
    gc=[]
    for c in configs():
        if c['kind']!='p12' or c['subset']!='all' or any(k in c for k in ['quality','topk','classifier','robust']): continue
        gc.append(c)
        cg=dict(c); cg.update(grid=True,id=c['id']+'_grid');gc.append(cg)
    (HERE/'grid_candidate_configs.json').write_text(json.dumps(gc,indent=2),encoding='utf-8')
    rows=[];rec=[]
    for name in ['legacy','gn17','direct_p12']:
        data=load_data() if name=='legacy' else load_new(name)
        for c in gc:
            f=evaluate_config(data,c,split_folds(data,np.arange(len(data['D'])))); f['measurement']=name; f['config']=c['id'];rec.append(f)
            row=dict(measurement=name,config=c['id'],**metrics(f));rows.append(row)
            print('grid',name,c['id'],row['axis_errors'],row['joint_errors'],flush=True)
        pd.DataFrame(rows).to_csv(HERE/'grid_metrics.csv',index=False,encoding='utf-8-sig')
        pd.concat(rec,ignore_index=True).to_csv(HERE/'grid_predictions.csv',index=False,encoding='utf-8-sig')
        nested_run(data,gc,f'{name}_grid','stratified')
        nested_run(data,gc,f'{name}_grid','session_out')

if __name__=='__main__': main()
