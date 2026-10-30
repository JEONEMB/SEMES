"""Direct 410-coordinate GLS cost comparison; existing measurements unchanged.

Each axis has shared linear/quadratic coefficients and observed training-run
intercepts. All covariance estimates use only training residuals. Test-session
labels are recorded for audit, never supplied as a known inference branch.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib,json,sys,time
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent.parent
sys.path.insert(0,str(EXPERIMENT))
import zero_experiment as Z

def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_inputs():
    source=EXPERIMENT/'measurement'/'gn17_spread14.npz'
    z=np.load(source);qr=z['regions']<3;names=z['names'].astype(str)
    assert int(qr.sum())==205
    assert z['valid'][:,qr].all()
    D=z['D'][:,qr].astype(np.float64)
    D-=D[names=='center.bmp']
    X=D.reshape(len(D),410)
    m,B,_,_=Z.load('gn17_spread14')
    assert np.array_equal(m.file.to_numpy(),names)
    assert np.isfinite(X).all() and np.isfinite(B).all()
    pts=z['points'][qr].astype(float);x=(pts[:,0]-1906)/486;y=(pts[:,1]-1070)/122
    design=np.column_stack([np.ones(len(x)),x,y,x*x,x*y,y*y])
    reconstructed=np.linalg.lstsq(design,D.transpose(1,0,2).reshape(205,-1),rcond=None)[0].reshape(6,len(D),2).transpose(1,0,2).reshape(len(D),12)
    assert np.allclose(B,reconstructed,atol=1e-12,rtol=0)
    return m,X,B,dict(source=str(source),source_sha256=sha256(source),zero_experiment_sha256=sha256(EXPERIMENT/'zero_experiment.py'),names_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),D_sha256=hashlib.sha256(D.tobytes()).hexdigest(),baseline12_sha256=hashlib.sha256(B.tobytes()).hexdigest(),images=len(D),noncenter_images=int((m.axis!='C').sum()),qr_points=205,direct_dimensions=410,baseline_dimensions=12,all_qr_tracks_valid=True,baseline_reconstruction_max_difference=float(np.max(abs(B-reconstructed))))

def fit_coefficients(m,X,tr):
    """Same regression design as Z.fit inferred_session quadratic=True."""
    branches={};residual=[];rank=0;axis_info=[]
    for a in ['Y','T']:
        ii=tr[m.axis.to_numpy()[tr]==a]
        v=m.value.to_numpy()[ii];ss=m.session.to_numpy()[ii];sessions=np.unique(ss)
        A=np.column_stack([v,v*v,(ss[:,None]==sessions).astype(float)])
        coef=np.linalg.lstsq(A,X[ii],rcond=None)[0]
        rank_a=int(np.linalg.matrix_rank(A));rank+=rank_a
        residual.append(X[ii]-A@coef)
        for k,s in enumerate(sessions):branches[str(s)]=(a,coef[2+k],coef[0],coef[1])
        axis_info.append(dict(axis=a,n=len(ii),rank=rank_a,columns=A.shape[1],condition=float(np.linalg.cond(A)),observed_sessions=sessions.tolist()))
    return branches,np.vstack(residual),rank,axis_info

def make_metric(R,parameter_rank,kind):
    dim=R.shape[1];dof=max(len(R)-parameter_rank,1)
    std=np.sqrt(np.maximum(np.sum(R*R,axis=0)/dof,1e-14))
    info=dict(residual_rows=len(R),residual_dimension=dim,regression_rank=parameter_rank,residual_dof=dof,residual_std_min=float(std.min()),residual_std_max=float(std.max()),covariance_train_only=True)
    if kind=='identity':
        W=np.eye(dim);info.update(shrinkage=None,metric='Euclidean coordinate cost, all pixel coordinates equally weighted')
    elif kind=='diagonal':
        W=np.diag(1/std);info.update(shrinkage=1.,metric='Inverse training residual variance per pixel coordinate')
    elif kind=='ledoit_wolf':
        # Standardize each residual coordinate to unit RMS. Since the OLS design
        # contains run intercepts, residual means are numerically zero. Estimate
        # the correlation matrix with LW, then return to physical coordinates.
        rms=np.sqrt(np.maximum(np.mean(R*R,axis=0),1e-14))
        estimator=LedoitWolf(assume_centered=True).fit(R/rms)
        C=estimator.covariance_
        L=np.linalg.cholesky(C)
        # W X(column) has squared norm X' diag(1/std) C^-1 diag(1/std) X.
        W=solve_triangular(L,np.eye(dim),lower=True)/std[None,:]
        info.update(shrinkage=float(estimator.shrinkage_),metric='Ledoit-Wolf standardized residual correlation, rescaled by residual std',correlation_trace=float(np.trace(C)),correlation_condition=float(np.linalg.cond(C)))
    else:raise ValueError(kind)
    return W,info

def enrich(df):
    df=df.copy()
    df['axis_correct']=df.axis==df.pred_axis
    df['sign_correct']=np.sign(df.value)==np.sign(df.pred)
    df['joint_correct']=df.axis_correct&df.sign_correct
    df['true_Y']=np.where(df.axis=='Y',df.value,0.)
    df['true_T']=np.where(df.axis=='T',df.value,0.)
    df['pred_Y']=np.where(df.pred_axis=='Y',df.pred,0.)
    df['pred_T']=np.where(df.pred_axis=='T',df.pred,0.)
    df['error_Y_abs']=np.abs(df.pred_Y-df.true_Y)
    df['error_T_abs']=np.abs(df.pred_T-df.true_T)
    df['strict_pass09']=df.joint_correct&(df.error_Y_abs<=.09)&(df.error_T_abs<=.09)
    return df

def complete_score(df,expected_names):
    assert len(df)==280 and df.file.nunique()==280
    assert set(df.file)==set(expected_names)
    out=Z.score(df)
    out.update(axis_correct=int(df.axis_correct.sum()),axis_sign_correct=int(df.joint_correct.sum()),axis_errors=int((~df.axis_correct).sum()),joint_errors=int((~df.joint_correct).sum()),strict_pass09_count=int(df.strict_pass09.sum()),all_images_pass=bool(df.strict_pass09.all()),prediction_rows_complete=True)
    return out

def protocols(m):
    ids=np.flatnonzero(m.axis!='C')
    yield 'resubstitution',-1,[('full',ids,ids)]
    for seed in range(7000,7020):
        folds=[(str(k),ids[tr],ids[te]) for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(ids,m.session.iloc[ids]))]
        yield 'repeated_stratified5',seed,folds
    yield 'session_out',-1,list(Z.splits(m,'session_out'))

def checks(m,X,B):
    ids=np.flatnonzero(m.axis!='C');split=StratifiedKFold(5,shuffle=True,random_state=7000)
    ti,vi=next(split.split(ids,m.session.iloc[ids]));tr,te=ids[ti],ids[vi]
    br,R,rank,_=fit_coefficients(m,X,tr);W,_=make_metric(R,rank,'ledoit_wolf')
    perturbed=m.copy();perturbed.loc[te,'value']+=1000.;perturbed.loc[te,'session']='unseen';perturbed.loc[te,'axis']='Q'
    X2=X.copy();X2[te]+=10000.
    br2,R2,rank2,_=fit_coefficients(perturbed,X2,tr);W2,_=make_metric(R2,rank2,'ledoit_wolf')
    assert np.array_equal(R,R2) and np.array_equal(W,W2)
    for key in br:assert all(np.array_equal(a,b) for a,b in zip(br[key][1:],br2[key][1:]))
    original=Z.predict(m,X,te,(br,W),'inferred_session')
    hidden=Z.predict(perturbed,X,te,(br,W),'inferred_session')
    assert np.array_equal(original.pred_axis,hidden.pred_axis) and np.array_equal(original.pred,hidden.pred)
    bb,_,_,_=fit_coefficients(m,B,tr);reference=Z.fit(m,B,tr,m.value.to_numpy(),'inferred_session',True)
    maxcoef=max(float(np.max(abs(x-y))) for key in bb for x,y in zip(bb[key][1:],reference[0][key][1:]))
    assert maxcoef<1e-12
    # A common response covariance cannot alter multivariate OLS estimates when
    # all output coordinates have exactly the same regression design.
    rng=np.random.default_rng(41);A=rng.normal(size=(50,5));Y=rng.normal(size=(50,12));K=rng.normal(size=(12,12));C=K@K.T+np.eye(12);chol=np.linalg.cholesky(C);Wy=solve_triangular(chol,np.eye(12),lower=True)
    ols=np.linalg.lstsq(A,Y,rcond=None)[0];gls_white=np.linalg.lstsq(A,Y@Wy.T,rcond=None)[0]@np.linalg.inv(Wy.T)
    coef_invariance=float(np.max(abs(ols-gls_white)));assert coef_invariance<1e-12
    return dict(heldout_features_labels_groups_do_not_change_coefficients_or_covariance=True,test_metadata_does_not_choose_inference_branch=True,independent_regression_max_coefficient_difference_vs_existing=maxcoef,multioutput_OLS_GLS_same_design_coefficient_difference=coef_invariance,minimum_training_residual_rank_warning='410 coordinates exceed 224 training images; full unregularized covariance cannot be inverted',test_seed=7000)

def summarize(metrics):
    f=metrics[metrics.protocol=='repeated_stratified5'];rows=[]
    for method,g in f.groupby('method'):
        rows.append(dict(method=method,seeds=len(g),predictions_total=int(g.n.sum()),axis_errors_mean=float(g.axis_errors.mean()),axis_errors_min=int(g.axis_errors.min()),axis_errors_max=int(g.axis_errors.max()),joint_accuracy_mean=float(g.axis_sign_accuracy.mean()),joint_accuracy_min=float(g.axis_sign_accuracy.min()),joint_accuracy_max=float(g.axis_sign_accuracy.max()),seeds_axis_sign_100=int((g.axis_sign_accuracy==1).sum()),seeds_all_images_pass=int(g.all_images_pass.sum()),strict_pass09_mean=float(g.strict_pass09.mean()),worst_Y_vector_error=float(g.vector_Y_max.max()),worst_T_vector_error=float(g.vector_T_max.max())))
    summary=pd.DataFrame(rows);summary.to_csv(HERE/'repeated_seed_statistics.csv',index=False)
    pivot=f.pivot(index='seed',columns='method',values='joint_errors');ref=pivot['baseline12_fixed001'];paired=[]
    for method in pivot.columns:
        delta=pivot[method]-ref
        paired.append(dict(method=method,comparison='method minus baseline12_fixed001 joint errors on the identical seed',mean_error_difference=float(delta.mean()),min_error_difference=int(delta.min()),max_error_difference=int(delta.max()),seeds_better=int((delta<0).sum()),seeds_same=int((delta==0).sum()),seeds_worse=int((delta>0).sum())))
    pd.DataFrame(paired).to_csv(HERE/'paired_seed_comparison.csv',index=False)
    prior_path=EXPERIMENT/'calibrated_runs'/'stress_metrics.csv'
    prior_match=None
    if prior_path.exists():
        prior=pd.read_csv(prior_path);differences=[]
        for row in f[f.method=='baseline12_fixed001'].itertuples():
            old=prior[prior.cv=='repeated5_seed'+str(int(row.seed)-7000)]
            if len(old)==1:differences.append(abs(float(old.axis_sign_accuracy.iloc[0])-row.axis_sign_accuracy))
        prior_match=dict(compared_seeds=len(differences),max_axis_sign_accuracy_difference=max(differences) if differences else None)
    output=dict(repeated_seed_statistics=rows,paired_seed_comparison=paired,resubstitution=metrics[metrics.protocol=='resubstitution'].to_dict('records'),session_out=metrics[metrics.protocol=='session_out'].to_dict('records'),existing_baseline_reproduction=prior_match,interpretation=['Direct 410D LW improves resubstitution residuals, but has more held-out mistakes than the existing 12D baseline in all 20 identical seeds.','The 20 seeds reuse the same 280 images; they are split-sensitivity measurements, not 20 independent datasets.','Leaving an entire T sign/session out removes that sign branch from training. This tests unsupported new origin modes and has a structural support limitation.'])
    (HERE/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    return summary

def main():
    HERE.mkdir(parents=True,exist_ok=True);start=time.time();m,X,B,input_info=load_inputs()
    (HERE/'input_identity.json').write_text(json.dumps(input_info,indent=2,ensure_ascii=False),encoding='utf-8')
    (HERE/'checks.json').write_text(json.dumps(checks(m,X,B),indent=2),encoding='utf-8')
    methods=['direct410_identity','direct410_diagonal','direct410_ledoit_wolf','baseline12_fixed001']
    metrics=[];allpred=[];diagnostics=[];splitmanifest=[]
    expected=m.file[m.axis!='C'].tolist()
    for protocol,seed,folds in protocols(m):
        records={method:[] for method in methods}
        for fold,tr,te in folds:
            assert not (set(tr)&set(te)) or protocol=='resubstitution'
            splitmanifest.append(dict(protocol=protocol,seed=seed,fold=fold,train_count=len(tr),test_count=len(te),train_names=m.file.iloc[tr].tolist(),test_names=m.file.iloc[te].tolist()))
            branches,R,rank,axisinfo=fit_coefficients(m,X,tr)
            for kind in ['identity','diagonal','ledoit_wolf']:
                method='direct410_'+kind;W,info=make_metric(R,rank,kind)
                df=enrich(Z.predict(m,X,te,(branches,W),'inferred_session'))
                df=df.assign(method=method,protocol=protocol,seed=seed,fold=fold)
                records[method].append(df)
                diagnostics.append(dict(method=method,protocol=protocol,seed=seed,fold=fold,train_count=len(tr),test_count=len(te),axis_regressions=json.dumps(axisinfo),**info))
            model=Z.fit(m,B,tr,m.value.to_numpy(),'inferred_session',True)
            df=enrich(Z.predict(m,B,te,model,'inferred_session')).assign(method='baseline12_fixed001',protocol=protocol,seed=seed,fold=fold)
            records['baseline12_fixed001'].append(df)
        for method in methods:
            df=pd.concat(records[method],ignore_index=True)
            row=dict(method=method,protocol=protocol,seed=seed,**complete_score(df,expected));metrics.append(row);allpred.append(df)
            print(protocol,seed,method,'errors',row['joint_errors'],'maxY',round(row['vector_Y_max'],8),'maxT',round(row['vector_T_max'],8),'elapsed',round(time.time()-start,1),flush=True)
        pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(HERE/'fold_diagnostics.csv',index=False)
        # Each protocol is persisted separately immediately; final combined table
        # is written once so partial completion cannot masquerade as 20 seeds.
        pd.concat([df for df in allpred if df.protocol.iloc[0]==protocol and int(df.seed.iloc[0])==seed],ignore_index=True).to_csv(HERE/f'predictions_{protocol}_{seed}.csv',index=False)
    pd.concat(allpred,ignore_index=True).to_csv(HERE/'all_predictions.csv',index=False)
    summary=summarize(pd.DataFrame(metrics))
    (HERE/'split_manifest.json').write_text(json.dumps(splitmanifest,indent=2),encoding='utf-8')
    metadata=dict(seconds=time.time()-start,methods=methods,protocols=dict(resubstitution='fit and predict the same 280; not validation',repeated_stratified5='20 identical existing seeds 7000 through 7019; 5 folds each; 280 unique OOF predictions per seed',session_out='6 acquisition sessions held out in turn; inference cannot use unknown branch offsets'),regression='Y/T separate shared v and v² coefficients plus available training-session intercepts; same design per output',covariance='all estimates training residuals only; identity / residual diagonal / Ledoit-Wolf standardized correlation',ground_truth='unmodified raw filename commands, physical pose unobserved',infer_group_policy='no known test axis/sign/session at inference; minimize over learned sign-constrained branches',units='Y and T remain dataset numeric units; 0.09 is checked separately on each coordinate',scope='same six calibrated origin modes for repeated stratified CV; unseen zero modes not guaranteed',write_scope=str(HERE),measurement_cache_only=True,existing_deployment_unchanged=True,total_prediction_rows=sum(len(df) for df in allpred),completed_seeds=list(range(7000,7020)))
    (HERE/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
