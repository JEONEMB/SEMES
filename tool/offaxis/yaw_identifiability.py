"""Tick-scale Y/yaw distinguishability using frozen L630 measurements.

Primary: independently calibrated two-column forward Jacobian, unweighted
projection, mixed-coordinate repeat jitter with common motion retained.
Secondary: cross-fitted shrinkage covariance and a central-axis calibration.
No extraction changes and no actuator commands.
"""
import sys
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.analysis_packages'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.analysis_packages/mplconfig'))
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ppt_analysis import old_group

BASE=ROOT/'docs/offaxis-eda/ppt-method-v2'
OUT=ROOT/'docs/offaxis-eda/yaw-separation/identifiability'
TICKS=np.array([.05,.01])


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load_data():
    meta=pd.read_csv(BASE/'frames.csv').set_index('path')
    points=pd.read_csv(BASE/'points.csv.gz',usecols=['path','qr','id','diagonal','x','y','accepted'])
    points=points[~points.diagonal]
    ids=[]
    for qr in 'LCR':
        ids.extend((qr,int(i)) for i in sorted(points[points.qr.eq(qr)].id.unique()))
    assert len(ids)==630
    wide=points.pivot(index='path',columns=['qr','id'],values=['x','y'])
    order=[(d,qr,i) for qr,i in ids for d in ['x','y']]
    X=wide[order].loc[meta.index].to_numpy()
    assert X.shape==(341,1260) and np.isfinite(X).all()
    return meta,X,ids


def transform(X,kind):
    if kind=='L630_full':return X.copy()
    # First three QR have 212 / 210 / 208 L corners; preserve identities.
    pts=X.reshape(len(X),630,2)
    if kind=='L630_centered':return (pts-pts.mean(1,keepdims=True)).reshape(len(X),-1)
    if kind=='frame_mean':return pts.mean(1)
    return np.concatenate([pts[:,:212].mean(1),pts[:,212:422].mean(1),pts[:,422:].mean(1)],axis=1)


def fit_axis(X,meta,mask):
    ix=np.flatnonzero(mask)
    groups=np.array([old_group(p) for p in meta.index[ix]])
    names=sorted(set(groups));G=np.column_stack([groups==s for s in names]).astype(float)
    q=meta.iloc[ix][['y_mm','theta_deg']].to_numpy()/TICKS
    design=np.column_stack([G,q])
    assert np.linalg.matrix_rank(design)==design.shape[1]
    beta=np.linalg.lstsq(design,X[ix],rcond=None)[0]
    return beta[-2:].T,{s:beta[i] for i,s in enumerate(names)},dict(frames=len(ix),groups=names)


def estimator(J,E=None,df=None):
    if E is None:return np.linalg.pinv(J,rcond=1e-12)
    # C=.5*S+.5*diag(S). Woodbury avoids inverting a 1260x1260 matrix
    # from only 38 repeat degrees of freedom. The .5 choice is fixed a priori.
    variance=np.sum(E**2,axis=0)/df
    diagonal=.5*np.maximum(variance,1e-8)
    U=E*np.sqrt(.5/df)
    invJ=J/diagonal[:,None]
    core=np.eye(len(E))+(U/diagonal)@U.T
    weighted=invJ-(U.T/diagonal[:,None])@np.linalg.solve(core,U@invJ)
    return np.linalg.solve(J.T@weighted,weighted.T)


def repeat_residuals(X,meta):
    ix=np.flatnonzero(meta.category.eq('mixed'))
    q=meta.iloc[ix][['y_mm','theta_deg']].to_numpy()
    coords=np.unique(q,axis=0)
    E=np.zeros((len(ix),X.shape[1]));group=np.zeros(len(ix),int)
    for k,c in enumerate(coords):
        keep=np.all(q==c,axis=1);assert keep.sum()==3
        E[keep]=X[ix[keep]]-X[ix[keep]].mean(0);group[keep]=k
    return ix,E,group,coords


def measurement_scores(J,E):
    norms=np.linalg.norm(J,axis=0)
    cosine=float(np.clip(J[:,0]@J[:,1]/np.prod(norms),-1,1))
    singular=np.linalg.svd(J,compute_uv=False)
    A=estimator(J)
    error=E@A.T
    sigma=np.sqrt(np.sum(error**2,axis=0)/38)
    one_axis=np.column_stack([E@J[:,k]/(J[:,k]@J[:,k]) for k in [0,1]])
    known_sigma=np.sqrt(np.sum(one_axis**2,axis=0)/38)
    unique=[]
    for target,other in [(0,1),(1,0)]:
        vector=J[:,target]-J[:,other]*(J[:,other]@J[:,target])/(J[:,other]@J[:,other])
        amplitude=np.linalg.norm(vector)
        noise=E@(vector/amplitude)
        noise_sd=np.sqrt(np.sum(noise**2)/38)
        unique.append(dict(axis=['Y','yaw'][target],signal_l2_px=float(amplitude),
            signal_rms_per_component_px=float(amplitude/np.sqrt(len(vector))),
            noise_sd_on_unit_direction_px=float(noise_sd),one_tick_snr=float(amplitude/noise_sd)))
    return A,error,dict(pattern_angle_degrees=float(np.degrees(np.arccos(abs(cosine)))),
        signed_cosine=cosine,singular_values=singular.tolist(),condition_number=float(singular[0]/singular[-1]),
        column_l2_px_per_tick=norms.tolist(),repeat_sigma_ticks=sigma.tolist(),
        known_other_axis_repeat_sigma_ticks=known_sigma.tolist(),one_tick_snr=(1/sigma).tolist(),unique=unique)


def noise_bootstrap(error,group):
    # Resample complete coordinate triplets, not individual points/images.
    # This describes sensitivity across this observed pose set, not future-run CI.
    squares=np.array([np.sum(error[group==k]**2,axis=0)/2 for k in range(19)])
    rng=np.random.default_rng(261026)
    draws=rng.integers(0,19,(2000,19))
    sigmas=np.sqrt(squares[draws].mean(1))
    return dict(sigma_ticks_pose_bootstrap_05_95=np.quantile(sigmas,[.05,.95],axis=0).tolist(),
        snr_pose_bootstrap_05_95=np.quantile(1/sigmas,[.05,.95],axis=0).tolist())


def unique_column(J,target):
    other=1-target
    return J[:,target]-J[:,other]*(J[:,other]@J[:,target])/(J[:,other]@J[:,other])


def axis_validation(raw,meta,axis,local):
    scores=[];rows=[];stability=[]
    for calibration,scope in [('full_axis_range',axis),('central_axis_range',local)]:
        source=np.flatnonzero(scope)
        groups=np.array([old_group(p) if scope[i] else None for i,p in enumerate(meta.index)])
        q=meta[['y_mm','theta_deg']].to_numpy()/TICKS
        masks={}
        for split in ['interleaved','middle_block']:
            test=np.zeros(len(meta),bool)
            for name in sorted(set(groups[source])):
                indices=np.flatnonzero(scope&(groups==name))
                indices=indices[np.argsort(np.max(abs(q[indices]),axis=1))]
                if split=='interleaved':test[indices[2::3]]=True
                else:
                    n=len(indices);test[indices[n//3:2*n//3]]=True
            masks[split]=test
        for feature in ['L630_full','L630_centered','three_QR_means','frame_mean']:
            X=transform(raw,feature)
            for split,test in masks.items():
                J,offsets,_=fit_axis(X,meta,scope&~test)
                A=estimator(J)
                ix=np.flatnonzero(test)
                predictions=np.array([A@(X[i]-offsets[groups[i]]) for i in ix])
                err=predictions-q[ix]
                scores.append(dict(calibration=calibration,feature=feature,split=split,test_images=len(ix),
                    Y_rmse_ticks=np.sqrt(np.mean(err[:,0]**2)),yaw_rmse_ticks=np.sqrt(np.mean(err[:,1]**2)),
                    both_within_one_tick=float(np.mean(np.max(abs(err),axis=1)<=1))))
                for i,p,e in zip(ix,predictions,err):
                    rows.append(dict(calibration=calibration,feature=feature,split=split,path=meta.index[i],
                        Y_pred_ticks=p[0],yaw_pred_ticks=p[1],Y_error_ticks=e[0],yaw_error_ticks=e[1]))
            # Disjoint old-axis coordinates with similar range in every group.
            half=np.zeros(len(meta),bool)
            for name in sorted(set(groups[source])):
                indices=np.flatnonzero(scope&(groups==name))
                indices=indices[np.argsort(np.max(abs(q[indices]),axis=1))]
                half[indices[::2]]=True
            j1,_,_=fit_axis(X,meta,scope&half)
            j2,_,_=fit_axis(X,meta,scope&~half)
            for target in [0,1]:
                u,v=unique_column(j1,target),unique_column(j2,target)
                stability.append(dict(calibration=calibration,feature=feature,axis=['Y','yaw'][target],
                    unique_pattern_cosine=float(u@v/np.linalg.norm(u)/np.linalg.norm(v)),
                    half_1_unique_signal_l2_px=np.linalg.norm(u),half_2_unique_signal_l2_px=np.linalg.norm(v)))
    pd.DataFrame(scores).to_csv(OUT/'axis_coordinate_holdout.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'axis_holdout_predictions.csv',index=False)
    pd.DataFrame(stability).to_csv(OUT/'pattern_split_stability.csv',index=False)


def figures(models):
    summary=pd.read_csv(OUT/'summary.csv')
    patterns=pd.read_csv(OUT/'point_patterns.csv')
    p=patterns[patterns.calibration.eq('full_axis_range')]
    fig,axes=plt.subplots(2,2,figsize=(13,6),layout='constrained')
    for ax,x,y,title,magnification in [
        (axes[0,0],'Y_dx_per_tick','Y_dy_per_tick','Y one tick: 0.05 mm',90),
        (axes[0,1],'yaw_dx_per_tick','yaw_dy_per_tick','Yaw one tick: 0.01 degree',90),
        (axes[1,0],'unique_Y_dx','unique_Y_dy','Y component after removing yaw',15000),
        (axes[1,1],'unique_yaw_dx','unique_yaw_dy','Yaw component after removing Y',15000)]:
        ax.quiver(p.x,p.y,p[x]*magnification,p[y]*magnification,angles='xy',scale_units='xy',scale=1,width=.0025)
        ax.scatter(p.x,p.y,s=1,c='#bbbbbb');ax.invert_yaxis();ax.set_aspect('equal')
        ax.set_title(f'{title}\nArrows magnified x{magnification:,}; positions are original pixels',fontsize=10)
        ax.set_xlabel('image x (px)');ax.set_ylabel('image y (px)')
    fig.suptitle('Full-range calibration: large common motion, very small distinguishing motion')
    fig.savefig(OUT/'01_motion_patterns.png',dpi=170);plt.close(fig)
    jitter=pd.read_csv(OUT/'repeat_projection.csv')
    primary=jitter[jitter.calibration.eq('full_axis_range')&jitter.feature.eq('L630_full')&jitter.weighting.eq('unweighted')]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,axis,col in [(axes[0],'Y','Y_repeat_error_ticks'),(axes[1],'yaw','yaw_repeat_error_ticks')]:
        ax.hist(primary[col],bins=16,color='#3a82b3',alpha=.8)
        ax.axvspan(-1,1,color='#3eaf72',alpha=.2,label='+/-1 tick')
        ax.axvline(0,c='#444444',lw=1);ax.set_xlabel('Repeat deviation (ticks)');ax.set_ylabel('Images')
        ax.set_title(axis);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same-pose repeat jitter projected through a fixed axis-only estimator; 19 triplets')
    fig.savefig(OUT/'02_repeat_jitter.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,calib in zip(axes,['full_axis_range','central_axis_range']):
        g=summary[summary.calibration.eq(calib)&summary.feature.eq('L630_full')]
        xx=np.arange(len(g));w=.32
        ax.bar(xx-w/2,g.Y_one_tick_snr,w,label='Y');ax.bar(xx+w/2,g.yaw_one_tick_snr,w,label='yaw')
        ax.axhline(1,c='#aa4444',ls='--',label='1 tick equals repeat sigma')
        ax.axhline(3,c='#579f65',ls=':',label='3-sigma descriptive guide')
        ax.set_xticks(xx,['Unweighted','Noise-weighted\nleave-one-pose-out']);ax.set_ylabel('One-tick signal / repeat sigma')
        ax.set_title(calib);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('A central noise-weighted variant looks better for jitter; check bias and pattern stability separately')
    fig.savefig(OUT/'03_tick_snr.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    prediction=pd.read_csv(OUT/'mixed_predictions.csv')
    for ax,axis,truecol,predcol in [(axes[0],'Y','Y_true_ticks','Y_pred_ticks'),(axes[1],'yaw','yaw_true_ticks','yaw_pred_ticks')]:
        for calib,color in [('full_axis_range','#387eb4'),('central_axis_range','#e47d2f')]:
            g=prediction[prediction.calibration.eq(calib)&prediction.feature.eq('L630_full')&prediction.weighting.eq('unweighted')]
            ax.scatter(g[truecol],g[predcol],s=18,alpha=.65,c=color,label=calib)
        lo=min(0,prediction[predcol].quantile(.05));hi=prediction[truecol].max()
        ax.plot([lo,hi],[lo,hi],c='#666666',ls='--');ax.set_title(axis);ax.set_xlabel('Command truth (ticks)');ax.set_ylabel('Predicted (ticks)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Mixed predictions: patterns learned on old axis session; new center1 reference only')
    fig.savefig(OUT/'04_mixed_predictions.png',dpi=170);plt.close(fig)


def checks():
    rng=np.random.default_rng(82);J=rng.normal(size=(40,2));A=estimator(J)
    q=rng.normal(size=(20,2));assert np.max(abs((q@J.T)@A.T-q))<1e-12
    noise=rng.normal(size=(57,40));noise=noise.reshape(19,3,40);noise-=noise.mean(1,keepdims=True);E=noise.reshape(57,40)
    W=estimator(J,E,38);assert np.max(abs(W@J-np.eye(2)))<1e-10
    for k in [0,1]:
        u=unique_column(J,k);AA=u/(u@J[:,k]);assert np.max(abs(AA-A[k]))<1e-12
    # Compare Woodbury implementation to a dense shrinkage covariance solve.
    S=E.T@E/38;C=.5*S+.5*np.diag(np.maximum(np.diag(S),1e-8))
    weighted=np.linalg.solve(C,J)
    dense=np.linalg.solve(J.T@weighted,weighted.T)
    err=float(abs(dense-W).max());assert err<1e-10
    return dict(noiseless_two_axis_recovery=True,nuisance_projection_equivalence=True,woodbury_dense_max_difference=err)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    numerical=checks();meta,raw,ids=load_data()
    axis=meta.category.isin(['Y-only','theta-only']).to_numpy()
    local=axis&(((meta.category.eq('Y-only'))&meta.y_mm.abs().le(.5))|
                ((meta.category.eq('theta-only'))&meta.theta_deg.abs().le(.2))).to_numpy()
    results=[];predictions=[];jitter=[];patterns=[];models={};reference_checks=[]
    for calibration,mask in [('full_axis_range',axis),('central_axis_range',local)]:
        for feature in ['L630_full','L630_centered','three_QR_means','frame_mean']:
            X=transform(raw,feature)
            J,offsets,info=fit_axis(X,meta,mask)
            ix,E,group,coords=repeat_residuals(X,meta)
            A,error,score=measurement_scores(J,E)
            score.update(noise_bootstrap(error,group))
            # No mixed images enter the primary forward-pattern calibration.
            for weighting in ['unweighted','shrinkage_50_crossfit']:
                held_noise=np.zeros((57,2));held_pred=np.zeros((57,2))
                ref=X[meta.index.get_loc('기준 위치/center1.bmp')]
                for k,c in enumerate(coords):
                    test=group==k
                    AA=A if weighting=='unweighted' else estimator(J,E[~test],36)
                    assert np.max(abs(AA@J-np.eye(2)))<1e-5
                    held_noise[test]=E[test]@AA.T
                    held_pred[test]=(X[ix[test]]-ref)@AA.T
                sigma=np.sqrt(np.sum(held_noise**2,axis=0)/38)
                truth=meta.iloc[ix][['y_mm','theta_deg']].to_numpy()/TICKS
                err=held_pred-truth
                results.append(dict(calibration=calibration,feature=feature,weighting=weighting,
                    axis_training_frames=info['frames'],components=X.shape[1],angle_deg=score['pattern_angle_degrees'],
                    condition_number=score['condition_number'],Y_repeat_sigma_ticks=sigma[0],yaw_repeat_sigma_ticks=sigma[1],
                    Y_one_tick_snr=1/sigma[0],yaw_one_tick_snr=1/sigma[1],
                    Y_mixed_rmse_ticks=np.sqrt(np.mean(err[:,0]**2)),yaw_mixed_rmse_ticks=np.sqrt(np.mean(err[:,1]**2)),
                    mixed_both_within_tick=float(np.mean(np.max(abs(err),axis=1)<=1))))
                for n,path in enumerate(meta.index[ix]):
                    predictions.append(dict(calibration=calibration,feature=feature,weighting=weighting,path=path,
                        Y_true_ticks=truth[n,0],yaw_true_ticks=truth[n,1],Y_pred_ticks=held_pred[n,0],yaw_pred_ticks=held_pred[n,1],
                        Y_error_ticks=err[n,0],yaw_error_ticks=err[n,1],theta_axis_extrapolation=bool(truth[n,1]>100)))
                    jitter.append(dict(calibration=calibration,feature=feature,weighting=weighting,path=path,
                        coordinate_group=int(group[n]),Y_repeat_error_ticks=held_noise[n,0],yaw_repeat_error_ticks=held_noise[n,1]))
            refs=meta.category.eq('new-reference').to_numpy()
            rp=(X[refs]-ref)@A.T
            for path,p in zip(meta.index[refs],rp):
                reference_checks.append(dict(calibration=calibration,feature=feature,path=path,Y_pred_ticks=p[0],yaw_pred_ticks=p[1]))
            key=calibration+'__'+feature
            models[key]=dict(J_tick=J.tolist(),unweighted_estimator=A.tolist(),diagnostics=score,calibration=info)
            if feature=='L630_full':
                pos=raw[meta.index.get_loc('center.bmp')].reshape(630,2)
                yaw_unique=J[:,1]-J[:,0]*(J[:,0]@J[:,1])/(J[:,0]@J[:,0])
                y_unique=J[:,0]-J[:,1]*(J[:,1]@J[:,0])/(J[:,1]@J[:,1])
                for n,(qr,i) in enumerate(ids):
                    patterns.append(dict(calibration=calibration,qr=qr,id=i,x=pos[n,0],y=pos[n,1],
                        Y_dx_per_tick=J[2*n,0],Y_dy_per_tick=J[2*n+1,0],
                        yaw_dx_per_tick=J[2*n,1],yaw_dy_per_tick=J[2*n+1,1],
                        unique_yaw_dx= yaw_unique[2*n],unique_yaw_dy=yaw_unique[2*n+1],
                        unique_Y_dx=y_unique[2*n],unique_Y_dy=y_unique[2*n+1]))
    pd.DataFrame(results).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'mixed_predictions.csv',index=False)
    pd.DataFrame(jitter).to_csv(OUT/'repeat_projection.csv',index=False)
    pd.DataFrame(patterns).to_csv(OUT/'point_patterns.csv',index=False)
    pd.DataFrame(reference_checks).to_csv(OUT/'reference_checks.csv',index=False)
    (OUT/'models_and_diagnostics.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    pred=pd.DataFrame(predictions);byrange=[]
    for keys,g in pred.groupby(['calibration','feature','weighting']):
        for name,mask in [('all_57',np.ones(len(g),bool)),('theta_within_old_axis',g.yaw_true_ticks.le(100).to_numpy()),
                          ('within_central_calibration',((g.Y_true_ticks.abs()<=10)&(g.yaw_true_ticks.abs()<=20)).to_numpy())]:
            h=g.loc[mask];e=h[['Y_error_ticks','yaw_error_ticks']].to_numpy()
            byrange.append(dict(calibration=keys[0],feature=keys[1],weighting=keys[2],subset=name,images=len(h),
                Y_rmse_ticks=float(np.sqrt(np.mean(e[:,0]**2))),yaw_rmse_ticks=float(np.sqrt(np.mean(e[:,1]**2))),
                Y_bias_ticks=float(e[:,0].mean()),yaw_bias_ticks=float(e[:,1].mean()),
                both_within_tick=float(np.mean(np.max(abs(e),axis=1)<=1))))
    pd.DataFrame(byrange).to_csv(OUT/'mixed_by_range.csv',index=False)
    nj=pd.DataFrame(jitter).merge(pred[['calibration','feature','weighting','path','Y_true_ticks','yaw_true_ticks']],
        on=['calibration','feature','weighting','path'],validate='one_to_one')
    noise_by_range=[]
    for keys,g in nj.groupby(['calibration','feature','weighting']):
        for name,mask in [('all_57',np.ones(len(g),bool)),('theta_within_old_axis',g.yaw_true_ticks.le(100).to_numpy()),
                          ('within_central_calibration',((g.Y_true_ticks.abs()<=10)&(g.yaw_true_ticks.abs()<=20)).to_numpy())]:
            h=g.loc[mask];n=h.coordinate_group.nunique();e=h[['Y_repeat_error_ticks','yaw_repeat_error_ticks']].to_numpy()
            sd=np.sqrt(np.sum(e**2,axis=0)/(2*n))
            noise_by_range.append(dict(calibration=keys[0],feature=keys[1],weighting=keys[2],subset=name,
                coordinates=n,images=len(h),Y_repeat_sigma_ticks=sd[0],yaw_repeat_sigma_ticks=sd[1],
                Y_one_tick_snr=1/sd[0],yaw_one_tick_snr=1/sd[1]))
    pd.DataFrame(noise_by_range).to_csv(OUT/'repeat_sigma_by_range.csv',index=False)
    axis_validation(raw,meta,axis,local)
    figures(models)
    sources=[BASE/'frames.csv',BASE/'points.csv.gz',Path(__file__),Path(__file__).with_name('ppt_analysis.py')]
    manifest=dict(date='2026-10-26',numerical_checks=numerical,features='Fixed L630, all finite fits; no extraction changes',
        tick_Y_mm=.05,tick_yaw_deg=.01,axis_frames=280,central_axis_frames=int(local.sum()),
        repeat_frames=57,repeat_coordinates=19,repeat_degrees_of_freedom=38,
        common_frame_motion_removed=False,covariance_shrinkage=.5,
        noise_covariance_crossfit='held coordinate and all three repeats excluded',
        primary_pattern_uses_mixed=False,reference='new center1 only; no mixed-label bias calibration',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    print('PRIMARY',json.dumps(models['full_axis_range__L630_full']['diagnostics'],indent=2),flush=True)
    return meta,raw,models


if __name__=='__main__':main()
