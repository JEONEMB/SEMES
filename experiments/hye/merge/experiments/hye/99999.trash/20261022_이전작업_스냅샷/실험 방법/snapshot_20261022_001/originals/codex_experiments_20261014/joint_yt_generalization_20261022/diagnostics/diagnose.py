"""Read-only diagnosis of the preserved joint model; counterfactuals are not validation."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
OLD=ROOT/'codex_experiments_20261014/joint_yt_improved_20261022'
sys.path.insert(0,str(OLD))
from engine import extract

def metric(y,p):
    e=abs(y-p)
    return dict(n=len(y),Y_MAE=float(e[:,0].mean()),Y_max=float(e[:,0].max()),T_MAE=float(e[:,1].mean()),T_max=float(e[:,1].max()),both_pass=int(np.all(e<=.09,axis=1).sum()))

def main():
    src=ROOT/'codex_experiments_20261014/evaluate_archives_20261022'
    manifest=pd.read_csv(src/'source_manifest.csv');good=manifest[manifest.cache_status=='measured'].copy()
    ds=[];xs=[];quals=[]
    bundle=joblib.load(OLD/'fixedfeature_model.joblib');coord=joblib.load(OLD/'coordinate_model.joblib')
    for row in good.itertuples():
        z=json.loads(Path(row.cache_path).read_text(encoding='utf-8'))['trace'];points=np.array(z['points']);d=np.array(z['displacements']);q=np.array(z['quality'])[:,:3]
        assert np.array_equal(z['ids'],bundle['ids'])
        ds.append(d);xs.append(extract(points,d,q,'uniform','global2'));quals.append(q)
    D=np.stack(ds);X=np.stack(xs);Q=np.stack(quals);y=good[['Y','T']].to_numpy()
    train=good.original_training_overlap.to_numpy();new=(good.dataset==20261020).to_numpy();mix=new&(good.scope=='mixed').to_numpy();cent=new&(good.scope=='center').to_numpy()
    model=bundle['model'];pred=model.predict(X);pred[good.reference_center_overlap.to_numpy()]=0
    pipe=model.members[0];xt=pipe[0].transform(X);trees=pipe[-1]
    votes=np.stack([t.predict(xt) for t in trees.estimators_])
    report=dict(configuration=bundle['config'],training=dict(n=int(train.sum()),single_Y=int((train&(y[:,0]!=0)).sum()),single_T=int((train&(y[:,1]!=0)).sum()),mixed=int((train&(y[:,0]!=0)&(y[:,1]!=0)).sum())),new_measured=metric(y[new],pred[new]),new_mixed=metric(y[mix],pred[mix]))
    # Each multioutput tree leaf and ensemble prediction is a convex average of training targets.
    # Noise augmentation changes X only, so the original single-axis target convex hull is unchanged.
    hull=[]
    for row in manifest[manifest.dataset==20261020].itertuples():
        lo=0.;hi=max(abs(row.Y),abs(row.T))
        for _ in range(80):
            mid=(lo+hi)/2
            if max(abs(row.Y)-mid,0)/2+max(abs(row.T)-mid,0)<=1:hi=mid
            else:lo=mid
        hull.append(dict(file=row.name,Y=row.Y,T=row.T,minimum_possible_max_axis_error_in_original_target_convex_hull=hi,tolerance09_achievable_in_hull=hi<=.09+1e-12))
    hull=pd.DataFrame(hull);hull.to_csv(H/'tree_convex_hull_lower_bounds.csv',index=False,encoding='utf-8-sig')
    assert np.max(abs(pred[:,0])/2+abs(pred[:,1]))<=1+1e-10
    report['old_tree_architecture_limit']=dict(training_hull='abs(pred_Y)/2 + abs(pred_T) <= 1',unavoidable_tolerance09_failures_of60=int((~hull.tolerance09_achievable_in_hull).sum()),minimum_possible_max_axis_error_Y1_T1=1/3,explanation='Squared-error multioutput tree leaves average original targets; noise augmentation never adds mixed targets. This is a bound for this preserved architecture and training set only, not an imaging impossibility bound.',source='https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html')
    details=good[['index','dataset','name','Y','T','scope']].copy()
    details['pred_Y']=pred[:,0];details['pred_T']=pred[:,1];details['eY']=abs(pred[:,0]-y[:,0]);details['eT']=abs(pred[:,1]-y[:,1]);details['tree_fraction_Y_zero']=(abs(votes[:,:,0])<1e-10).mean(0);details['tree_fraction_T_zero']=(abs(votes[:,:,1])<1e-10).mean(0)
    details['mean_dx']=D[:,:,0].mean(1);details['mean_dy']=D[:,:,1].mean(1);details['rms_displacement_px']=np.sqrt(np.mean(D*D,axis=(1,2)))
    details['median_boundary_rmse']=np.median(Q[:,:,0],axis=1);details['median_window_stability_px']=np.median(Q[:,:,1],axis=1)
    details.to_csv(H/'diagnostic_per_image.csv',index=False,encoding='utf-8-sig')
    names=['constant','v','u','v2','uv','u2'];pd.DataFrame({'feature':[f'{n}_{axis}' for n in names for axis in ['dx','dy']],'importance':trees.feature_importances_}).to_csv(H/'tree_feature_importance.csv',index=False)
    report['feature_importance_translation']=float(trees.feature_importances_[:2].sum())
    report['mixed_zero_Y_trees_by_T']={str(t):dict(n=int(sel.sum()),fraction_Y_zero=float((abs(votes[:,sel,0])<1e-10).mean()),Y_MAE=float(abs(y[sel,0]-pred[sel,0]).mean())) for t in np.unique(y[mix,1]) for sel in [mix&(y[:,1]==t)]}
    # Image-derived center vectors are diagnostic here; new centers are not refit or silently inserted.
    cp=coord['model'].predict(D[cent].reshape(cent.sum(),-1));c=details[cent].copy();c['coordinate_pred_Y']=cp[:,0];c['coordinate_pred_T']=cp[:,1];c.to_csv(H/'new_center_drift.csv',index=False,encoding='utf-8-sig')
    shift=X[cent].mean(0);rec=model.predict(X[mix]-shift)
    report['counterfactual_recenter_from_all_three_new_centers']=dict(metric=metric(y[mix],rec),warning='Diagnostic only: uses all new center images; not independent test performance or approved inference change.')
    for tag,features in [('P12',X),('raw1260',D.reshape(len(D),-1))]:
        A=np.c_[np.ones(train.sum()),y[train]];co=np.linalg.lstsq(A,features[train],rcond=None)[0];r=features[train]-A@co
        vY,vT=co[1:3];std=np.maximum(np.sqrt(np.mean(r*r,axis=0)),1e-9)
        cos=float(np.dot(vY,vT)/(np.linalg.norm(vY)*np.linalg.norm(vT)))
        standard_cos=float(np.dot(vY/std,vT/std)/(np.linalg.norm(vY/std)*np.linalg.norm(vT/std)))
        a=features[mix]-np.c_[np.ones(mix.sum()),y[mix]]@co
        orthY=vY-vT*np.dot(vY,vT)/np.dot(vT,vT); wY=orthY/np.dot(orthY,vY)
        conds={tuple(yy):mix&np.all(y==yy,axis=1) for yy in np.unique(y[mix],axis=0)}
        repeat_res=np.concatenate([features[ix]-features[ix].mean(0) for ix in conds.values()])
        repeat_df=sum(int(ix.sum())-1 for ix in conds.values())
        report[tag+'_forward_diagnostic']=dict(sensitivity_Y_norm=float(np.linalg.norm(vY)),sensitivity_T_norm=float(np.linalg.norm(vT)),cosine_Y_T=cos,standardized_cosine_Y_T=standard_cos,training_residual_rms=float(np.sqrt(np.mean(r*r))),mixed_residual_rms=float(np.sqrt(np.mean(a*a))),Y_orthogonal_to_T_norm=float(np.linalg.norm(orthY)),Y_orthogonal_to_T_rms_per_coordinate=float(np.sqrt(np.mean(orthY**2))),Y_tolerance09_distinct_rms_per_coordinate=float(.09*np.sqrt(np.mean(orthY**2))),fraction_Y_norm_remaining_after_T=float(np.linalg.norm(orthY)/np.linalg.norm(vY)),Euclidean_projection_Y_training_residual_std=float(np.std(r@wY,ddof=1)),Euclidean_projection_Y_new_repeat_std=float(np.std(repeat_res@wY,ddof=1)),new_repeat_feature_rms=float(np.sqrt(np.mean(repeat_res**2))),warning='Linear forward approximation fitted to original single-axis training only. Euclidean projection is a diagnostic, not an optimal noise covariance estimator or an impossibility bound.')
        report[tag+'_forward_diagnostic']['Euclidean_projection_Y_new_repeat_std']=float(np.sqrt(np.sum((repeat_res@wY)**2)/repeat_df))
        report[tag+'_forward_diagnostic']['pooled_repeat_degrees_of_freedom']=repeat_df
        report[tag+'_forward_diagnostic']['Y_tolerance09_over_pooled_repeat_std']=float(.09/np.sqrt(np.sum((repeat_res@wY)**2)/repeat_df))
        if tag=='P12':
            # Build synthetic sums without mixed training: evidence for architecture's inability to add motions.
            combinations=[]
            for yy in [.05,.1,.5,1.]:
                for tt in [.01,.1,.5,1.]:
                    a=train&(y[:,0]==yy)&(y[:,1]==0);b=train&(y[:,0]==0)&(y[:,1]==tt)
                    if a.any() and b.any():
                        synthetic=X[a].mean(0)+X[b].mean(0)-co[0];out=model.predict(synthetic[None])[0]
                        combinations.append(dict(Y=yy,T=tt,pred_Y=out[0],pred_T=out[1],eY=abs(out[0]-yy),eT=abs(out[1]-tt)))
            pd.DataFrame(combinations).to_csv(H/'synthetic_additivity_diagnostic.csv',index=False)
    # Same-condition repeats quantify observed noise without assuming independent pixels.
    raw=D.reshape(len(D),-1);conditions=[];means=[];ids=[];repeat_rms=[]
    for yy,ix in conds.items():
        rr=raw[ix]-raw[ix].mean(0);rms=float(np.sqrt(np.mean(rr*rr)))
        row=dict(Y=yy[0],T=yy[1],n=int(ix.sum()),mean_dx=float(D[ix,:,0].mean()),mean_dy=float(D[ix,:,1].mean()),raw_repeat_rms_px=rms,raw_repeat_max_pair_rms_px=max(float(np.sqrt(np.mean((aa-bb)**2))) for aa in raw[ix] for bb in raw[ix]),Y_pred_mean=float(pred[ix,0].mean()),Y_pred_std=float(pred[ix,0].std(ddof=1)),T_pred_mean=float(pred[ix,1].mean()),T_pred_std=float(pred[ix,1].std(ddof=1)))
        conditions.append(row);means.append(raw[ix].mean(0));ids.append(yy);repeat_rms.append(rms)
    pd.DataFrame(conditions).to_csv(H/'condition_repeat_noise.csv',index=False)
    pairs=[]
    for i in range(len(ids)):
        for j in range(i):
            if abs(ids[i][0]-ids[j][0])<=.18:continue
            dist=float(np.sqrt(np.mean((means[i]-means[j])**2)));noise=float(np.sqrt(repeat_rms[i]**2+repeat_rms[j]**2))
            pairs.append(dict(a_Y=ids[i][0],a_T=ids[i][1],b_Y=ids[j][0],b_T=ids[j][1],delta_Y=abs(ids[i][0]-ids[j][0]),mean_field_distance_rms_px=dist,pooled_within_repeat_rms_px=noise,mean_distance_over_pooled_repeat=dist/max(noise,1e-12)))
    pd.DataFrame(pairs).sort_values('mean_field_distance_rms_px').to_csv(H/'different_Y_nearest_condition_pairs.csv',index=False)
    # Closest actual image pairs with materially differentY; no threshold used for model selection.
    image_pairs=[];ii=np.flatnonzero(mix)
    for a in range(len(ii)):
        i=ii[a]
        for j in ii[:a]:
            if abs(y[i,0]-y[j,0])<=.18:continue
            image_pairs.append(dict(a_name=good.iloc[i]['name'],b_name=good.iloc[j]['name'],a_Y=y[i,0],a_T=y[i,1],b_Y=y[j,0],b_T=y[j,1],field_distance_rms_px=float(np.sqrt(np.mean((raw[i]-raw[j])**2)))))
    pd.DataFrame(image_pairs).sort_values('field_distance_rms_px').head(50).to_csv(H/'different_Y_nearest_image_pairs.csv',index=False,encoding='utf-8-sig')
    report['closest_condition_pairs_different_Y']=sorted(pairs,key=lambda z:z['mean_field_distance_rms_px'])[:3]
    report['limitations']=['File-name commands are evaluation targets, not measured physical robot pose.','P12 and raw geometry comparisons hold same630 boundary coordinates fixed.','Center subtraction and synthetic sums are diagnostic counterfactuals, not test success.','12 unmeasured T=2 photos are excluded from numeric metrics but remain inference failures.']
    (H/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
