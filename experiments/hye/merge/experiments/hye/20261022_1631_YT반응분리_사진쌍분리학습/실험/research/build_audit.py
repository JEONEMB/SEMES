"""Descriptive factorial coverage and training-only contrast availability audit."""
from pathlib import Path
import json,itertools
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent
OLD=H.parents[1]/'joint_yt_generalization_20261022'

def contrasts(c):
    out=[]
    for varying,fixed in [('Y','T'),('T','Y')]:
        for level,g in c.groupby(fixed):
            for (_,a),(_,b) in itertools.combinations(g.sort_values(varying).iterrows(),2):
                out.append(dict(varying=varying,fixed=fixed,fixed_level=float(level),Y1=float(a['Y']),T1=float(a['T']),Y2=float(b['Y']),T2=float(b['T']),delta=float(b[varying]-a[varying]),count1=int(a['count']),count2=int(b['count'])))
    return out

def main():
    m=pd.read_csv(OLD/'manifest_with_folds.csv');new=m[m.dataset==20261020];mixed=new[new.scope=='mixed']
    c=mixed.groupby(['Y','T']).size().reset_index(name='count');pairs=contrasts(c)
    c.to_csv(H/'conditions.csv',index=False);pd.DataFrame(pairs).to_csv(H/'axis_contrast_pairs.csv',index=False)
    ys=sorted(c['Y'].unique().tolist());ts=sorted(c['T'].unique().tolist());seen=set(map(tuple,c[['Y','T']].to_numpy()))
    missing=[dict(Y=float(y),T=float(t)) for y,t in itertools.product(ys,ts) if (y,t) not in seen]
    folds=[]
    for f in range(4):
        tr=new[new.fold!=f];cc=tr[tr.scope=='mixed'].groupby(['Y','T']).size().reset_index(name='count');pc=contrasts(cc)
        folds.append(dict(fold=f,training_new_images=len(tr),training_mixed_conditions=len(cc),fixed_T_varying_Y_pairs=sum(p['varying']=='Y' for p in pc),fixed_Y_varying_T_pairs=sum(p['varying']=='T' for p in pc),within_condition_noise_degrees_of_freedom=sum(len(g)-1 for _,g in tr.groupby('condition')),new_center_training_images=int((tr.scope=='center').sum())))
    result=dict(images=len(new),mixed_images=len(mixed),center_images=int((new.scope=='center').sum()),mixed_conditions=len(c),replicates_per_condition=c['count'].tolist(),Y_levels=ys,T_levels=ts,positive_mixed_only=bool(((mixed['Y']>0)&(mixed['T']>0)).all()),negative_mixed_images=int(((mixed['Y']<0)|(mixed['T']<0)).sum()),new_date_Y_only_images=int(((new['Y']!=0)&(new['T']==0)).sum()),new_date_T_only_images=int(((new['T']!=0)&(new['Y']==0)).sum()),same_T_different_Y_condition_pairs=sum(p['varying']=='Y' for p in pairs),same_Y_different_T_condition_pairs=sum(p['varying']=='T' for p in pairs),observed_factorial_cells=len(c),all_observed_levels_cartesian_cells=len(ys)*len(ts),missing_cells=missing,outer_training_availability=folds,within_condition_noise_degrees_of_freedom_all60=sum(len(g)-1 for _,g in new.groupby('condition')),limitations=['Descriptive coverage only: do not use all-condition means or covariances inside a held-out evaluation.','All three repeats of an unseen condition must remain outside training and pair generation.','Positive-only support limits claims about signed mixtures; it does not by itself explain errors inside the observed positive quadrant.','Repeat filenames1/2/3 are not evidence of known session or actual zero changes.'])
    (H/'coverage.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    a=np.load(OLD/'measurements/features.npz');D=a['D'].reshape(len(m),-1);rows=[]
    for target,fixed in [('Y','T'),('T','Y')]:
        for value,g in mixed.groupby(fixed):
            if g[target].nunique()<3:continue
            cc=g.groupby(target).apply(lambda v:np.mean(D[v.index],axis=0),include_groups=False)
            levels=np.array(cc.index,dtype=float);X=np.stack(cc.to_list());design=np.c_[np.ones(len(levels)),levels]
            coef=np.linalg.lstsq(design,X,rcond=None)[0];res=X-design@coef
            rows.append(dict(target=target,fixed=fixed,fixed_level=float(value),distinct_levels=len(levels),sensitivity_L2=float(np.linalg.norm(coef[1])),sensitivity_RMS_px_per_command=float(np.sqrt(np.mean(coef[1]**2))),condition_mean_linear_fit_residual_RMS_px=float(np.sqrt(np.mean(res**2)))))
    pd.DataFrame(rows).to_csv(H/'conditional_linear_sensitivity.csv',index=False)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
