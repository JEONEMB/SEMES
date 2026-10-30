"""Read-only, all60 image diagnostics for the saved pass_first OOF predictions."""
from pathlib import Path
import sys,json,hashlib,re
import numpy as np,pandas as pd,joblib,cv2
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent;BASE=H.parents[1];PRIOR=BASE/'joint_yt_disentangle_20261022';GENERAL=BASE/'joint_yt_generalization_20261022'
sys.path.insert(0,str(PRIOR));sys.path.insert(0,str(GENERAL))
import learned_model
from data import load
from features import FeatureMaps

def raw_shift_gradient(d,beta):
    raw=d.ravel();n=len(d);p=len(raw);h=d[:,0].mean()/100
    g=beta[:p]+h*beta[p:]
    g[::2]+=(raw@beta[p:])/(n*100)
    return g.reshape(n,2)

def model_gradient(wrapper,d,maps):
    model=wrapper.model
    if wrapper.config['method']=='repeat':
        return np.stack([raw_shift_gradient(d,model.beta[:,j].copy()) for j in range(2)],axis=-1)
    assert wrapper.config['method']=='conditional' and wrapper.config['feature']=='P12'
    assert model.config['mode']=='remove_quadratic' and model.config['family']=='gls'
    raw=d.ravel();h=d[:,0].mean()/100;xx=np.r_[raw,raw*h]
    pt=model.base.predict(xx)[0,1]
    beta=model.base.projector@model.base.initial@np.diag(model.base.scale)
    gt=raw_shift_gradient(d,beta[:,1].copy())
    gp=maps.P12.T@model.head.beta.reshape(6,2)
    gy=gp-(model.co[2]@model.head.beta+2*pt*(model.co[5]@model.head.beta))*gt
    return np.stack([gy,gt],axis=-1)

def standardized_forward_residual(wrapper,d,truth,pred,Dtr,ytr,maps):
    model=wrapper.model;bank=wrapper.bank(d[None]);x=bank[wrapper.config['feature']][0]
    if wrapper.config['method']=='repeat':
        rpred=(x-model.origin-pred@model.slope)/model.std
        rtrue=(x-model.origin-truth@model.slope)/model.std
    else:
        Xtr=wrapper.bank(Dtr)['P12'];yy,tt=ytr.T;P=np.c_[np.ones(len(ytr)),yy,tt,yy*yy,yy*tt,tt*tt]
        scale=np.maximum(np.sqrt(np.mean((Xtr-P@model.co)**2,axis=0)),1e-6)
        def phi(y):a,b=y;return np.array([1,a,b,a*a,a*b,b*b])
        rpred=(x-phi(pred)@model.co)/scale;rtrue=(x-phi(truth)@model.co)/scale
    return float(np.sqrt(np.mean(rpred*rpred))),float(np.sqrt(np.mean(rtrue*rtrue)))

def main():
    m,a=load();D=a['D'];Q=a['quality'];V=a['valid'];regions=a['regions'];points=a['points'];maps=FeatureMaps(points,regions);y=m[['Y','T']].to_numpy();new=m.dataset.to_numpy()==20261020
    p=pd.read_csv(PRIOR/'new60_group_oof_corrections.csv').set_index('index');assert len(p)==60
    sources=pd.read_csv(GENERAL/'measurements/manifest.csv').set_index('index')
    models={f:joblib.load(PRIOR/f'pass_first_fold{f}.joblib') for f in range(4)}
    old_hashes={str(PRIOR/f'pass_first_fold{f}.joblib'):hashlib.sha256((PRIOR/f'pass_first_fold{f}.joblib').read_bytes()).hexdigest() for f in range(4)}
    raw=D.reshape(len(D),-1);shape=(D-D.mean(axis=1)[:,None,:]).reshape(len(D),-1);records=[];qrrows=[];neighbors=[];gradcheck=[]
    for i in np.flatnonzero(new):
        row=p.loc[i];f=int(row.fold);wrapper=models[f]['model'];tr=np.asarray(models[f]['train_indices']);condition=np.flatnonzero(new&(m.condition.to_numpy()==m.condition[i]));peers=condition[condition!=i]
        pred=np.array([row.predY,row.predT]);true=y[i];mean_pred=p.loc[condition,['predY','predT']].to_numpy().mean(0);meanD=D[condition].mean(0);delta=D[i]-meanD
        mean_shift=delta.mean(0);delta_shape=delta-mean_shift;G=model_gradient(wrapper,D[i],maps)
        if i in [291,305,336]:
            rng=np.random.default_rng(100+i);direction=rng.normal(size=D[i].shape);eps=1e-5
            finite=(wrapper.predict((D[i]+eps*direction)[None])[0]-wrapper.predict((D[i]-eps*direction)[None])[0])/(2*eps)
            analytic=np.sum(G*direction[:,:,None],axis=(0,1));gradcheck.append(dict(index=int(i),max_abs_error=float(np.max(abs(finite-analytic)))))
            assert np.allclose(finite,analytic,rtol=1e-6,atol=1e-6)
        repeat_res=[]
        for _,gr in m.iloc[tr].groupby('cv_group'):
            if len(gr)>1:repeat_res.extend(D[gr.index]-D[gr.index].mean(0))
        rr=np.array(repeat_res);projected=np.einsum('npj,pjk->nk',rr,G)
        fp,ft=standardized_forward_residual(wrapper,D[i],true,pred,D[tr],y[tr],maps)
        rec=dict(index=int(i),file=row['file'],source_path=sources.loc[i,'path'],Y=true[0],T=true[1],predY=pred[0],predT=pred[1],signed_error_Y=pred[0]-true[0],signed_error_T=pred[1]-true[1],eY=abs(pred[0]-true[0]),eT=abs(pred[1]-true[1]),passed=bool(row.passed),fold=f,model_method=wrapper.config['method'],model_config=json.dumps(wrapper.config),condition=m.condition[i],repeat=int(re.search(r'\((\d+)\)',row['file']).group(1)) if '(' in row['file'] else None,
                 condition_mean_predY=mean_pred[0],condition_mean_predT=mean_pred[1],condition_bias_Y=mean_pred[0]-true[0],condition_bias_T=mean_pred[1]-true[1],individual_pred_deviation_Y=pred[0]-mean_pred[0],individual_pred_deviation_T=pred[1]-mean_pred[1],same_condition_passes=int(p.loc[condition,'passed'].sum()),condition_n=len(condition),peer_predY_min=float(p.loc[peers,'predY'].min()),peer_predY_max=float(p.loc[peers,'predY'].max()),
                 valid_points=int(V[i].sum()),ncc_min=float(sources.loc[i,'ncc_min']),pad=int(sources.loc[i,'pad']),rmse_median=float(np.median(Q[i,:,0])),rmse_p95=float(np.quantile(Q[i,:,0],.95)),window_stability_median=float(np.median(Q[i,:,1])),window_stability_p95=float(np.quantile(Q[i,:,1],.95)),contrast_median=float(np.median(Q[i,:,2])),sigma_median=float(np.median(Q[i,:,3:5])),sigma_p95=float(np.quantile(Q[i,:,3:5],.95)),
                 repeat_D_rms=float(np.sqrt(np.mean(delta*delta))),repeat_centered_D_rms=float(np.sqrt(np.mean(delta_shape*delta_shape))),repeat_mean_dx=float(mean_shift[0]),repeat_mean_dy=float(mean_shift[1]),
                 gradient_Y_norm=float(np.linalg.norm(G[:,:,0])),gradient_T_norm=float(np.linalg.norm(G[:,:,1])),linearized_repeat_delta_Y=float(np.sum(G[:,:,0]*delta)),linearized_repeat_delta_T=float(np.sum(G[:,:,1]*delta)),translation_component_Y=float(np.sum(G[:,:,0]*mean_shift)),shape_component_Y=float(np.sum(G[:,:,0]*delta_shape)),training_repeat_projected_Y_std=float(projected[:,0].std(ddof=1)),training_repeat_projected_T_std=float(projected[:,1].std(ddof=1)),forward_residual_at_prediction=fp,forward_residual_at_truth=ft)
        trnew=tr[new[tr]]
        sameT=trnew[np.isclose(y[trnew,1],true[1])];sameY=trnew[np.isclose(y[trnew,0],true[0])]
        rec.update(train_same_T_Y_values=json.dumps(sorted(set(y[sameT,0]))),train_same_Y_T_values=json.dumps(sorted(set(y[sameY,1]))),Y_outside_sameT_training_range=bool(len(sameT) and (true[0]<y[sameT,0].min() or true[0]>y[sameT,0].max())),T_outside_sameY_training_range=bool(len(sameY) and (true[1]<y[sameY,1].min() or true[1]>y[sameY,1].max())))
        for metric_name,xx in [('raw',raw),('shape',shape)]:
            distance=np.sqrt(np.mean((xx[tr]-xx[i])**2,axis=1));order=np.argsort(distance)[:3]
            for rank,k in enumerate(order):
                j=tr[k];neighbors.append(dict(index=int(i),file=row['file'],metric=metric_name,rank=rank+1,training_index=int(j),training_file=m.file[j],training_Y=y[j,0],training_T=y[j,1],rms_distance=float(distance[k])))
                if rank==0:rec.update({f'nearest_{metric_name}_file':m.file[j],f'nearest_{metric_name}_Y':y[j,0],f'nearest_{metric_name}_T':y[j,1],f'nearest_{metric_name}_distance':float(distance[k])})
        pixel=cv2.imdecode(np.fromfile(str(sources.loc[i,'path']),dtype=np.uint8),cv2.IMREAD_GRAYSCALE).astype(np.float64)
        qr_differential_translation_Y=0.;within_qr_shape_Y=0.;pixel_laplacian=[]
        for k in range(3):
            rr=regions==k;local=D[i,rr]-D[i,rr].mean(0);local_mean=(D[condition][:,rr]-D[condition][:,rr].mean(1)[:,None,:]).mean(0)
            dm=delta[rr].mean(0);dt=float(np.sum(G[rr,:,0]*(dm-mean_shift)));ds=float(np.sum(G[rr,:,0]*(delta[rr]-dm)));qr_differential_translation_Y+=dt;within_qr_shape_Y+=ds
            cur=points[rr]+D[i,rr];lo=np.maximum(np.floor(cur.min(0)-4).astype(int),0);hi=np.minimum(np.ceil(cur.max(0)+4).astype(int),np.array(pixel.shape)[::-1]-1);roi=pixel[lo[1]:hi[1]+1,lo[0]:hi[0]+1];lap=cv2.Laplacian(roi,cv2.CV_64F);focus=float(np.var(lap)/max(np.var(roi),1e-12));pixel_laplacian.append(focus)
            qrrows.append(dict(index=int(i),file=row['file'],qr=k+1,passed=bool(row.passed),valid=int(V[i,rr].sum()),count=int(rr.sum()),rmse_median=float(np.median(Q[i,rr,0])),stability_p95=float(np.quantile(Q[i,rr,1],.95)),sigma_median=float(np.median(Q[i,rr,3:5])),sigma_p95=float(np.quantile(Q[i,rr,3:5],.95)),sigma_lower_bound_fraction=float(np.mean(Q[i,rr,3:5]<.350001)),contrast_median=float(np.median(Q[i,rr,2])),repeat_shape_rms=float(np.sqrt(np.mean((local-local_mean)**2))),first_order_Y_contribution=float(np.sum(G[rr,:,0]*delta[rr])),global_translation_Y=float(np.sum(G[rr,:,0]*mean_shift)),differential_translation_Y=dt,within_qr_shape_Y=ds,mean_dx_vs_condition=float(dm[0]),mean_dy_vs_condition=float(dm[1]),gradient_Y_norm=float(np.linalg.norm(G[rr,:,0])),pixel_laplacian_normalized=focus,roi_variance=float(np.var(roi)),roi_bounds=json.dumps([int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])])))
        rec.update(qr_differential_translation_Y=qr_differential_translation_Y,within_qr_shape_Y=within_qr_shape_Y,pixel_laplacian_normalized_mean=float(np.mean(pixel_laplacian)),sigma_lower_bound_fraction=float(np.mean(Q[i,:,3:5]<.350001)))
        records.append(rec)
    df=pd.DataFrame(records);qdf=pd.DataFrame(qrrows)
    for col in ['rmse_median','rmse_p95','window_stability_p95','sigma_p95','pixel_laplacian_normalized_mean','repeat_D_rms','repeat_centered_D_rms','gradient_Y_norm','forward_residual_at_prediction']:
        df[col+'_percentile_all60']=df[col].rank(pct=True)*100
    for c in ['condition_bias_Y','individual_pred_deviation_Y']:
        assert np.allclose(df.signed_error_Y,df.condition_bias_Y+df.individual_pred_deviation_Y)
    df.to_csv(H/'all60_diagnostics.csv',index=False,encoding='utf-8-sig');df[~df.passed].to_csv(H/'ten_failure_diagnostics.csv',index=False,encoding='utf-8-sig');qdf.to_csv(H/'all60_qr_diagnostics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(neighbors).to_csv(H/'training_neighbors.csv',index=False,encoding='utf-8-sig')
    # Observational comparisons on all60, not a trained failure detector.
    quality_cols=['valid_points','ncc_min','rmse_median','rmse_p95','window_stability_p95','sigma_p95','sigma_lower_bound_fraction','pixel_laplacian_normalized_mean','contrast_median','repeat_D_rms','repeat_centered_D_rms','gradient_Y_norm','forward_residual_at_prediction']
    comparison=df.groupby('passed')[quality_cols].agg(['mean','median','min','max']);comparison.to_csv(H/'passed_failed_comparison.csv',encoding='utf-8-sig')
    correlations={c:float(df[[c,'eY']].corr(method='spearman').iloc[0,1]) for c in quality_cols}
    info=dict(images=60,failures=int((~df.passed).sum()),all_failures_Y_only=bool((df.loc[~df.passed,'eT']<=.09).all()),measurement_failed=0,source_predictions=str(PRIOR/'new60_group_oof_corrections.csv'),model_hashes=old_hashes,gradient_checks=gradcheck,descriptive_spearman_with_Y_error=correlations,invalid_diagnostic={'sigma_median':'All60 medians are at fitted lower bound0.35 (between-image std <2e-12). Do not interpret floating-point rank as blur.'},limits=['File commands are targets, not independent measured physical pose.','Same-condition means and truth-conditioned residuals are retrospective diagnostics, never deployment inputs.','Model gradients show sensitivity of this saved estimator, not a physical causal attribution.','Percentiles and correlations use all60 images; small dependent repeated corpus is not an independent significance test.','Normalized pixel Laplacian is a focus proxy affected by QR geometry and pixel phase; it is not calibrated physical blur.','Forward residuals use different saved model feature families; between-fold ranks are descriptive, not calibrated uncertainty.','No refit, inference gating, filename input, or changed prediction occurred.'])
    for fn,digest in old_hashes.items():assert hashlib.sha256(Path(fn).read_bytes()).hexdigest()==digest
    (H/'audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    bad=df[~df.passed].sort_values('eY');fig,ax=plt.subplots(figsize=(13,6),constrained_layout=True)
    ix=np.arange(len(bad));ax.barh(ix,bad.condition_bias_Y,label='조건 3장 평균의 Y 편향',color='#78a5b4');ax.scatter(bad.signed_error_Y,ix,color='#b7382d',s=45,label='해당 사진의 명령 기준 Y 오차');ax.hlines(ix,bad.condition_bias_Y,bad.signed_error_Y,color='#b7382d',lw=2)
    ax.axvline(.09,color='#888',ls='--');ax.axvline(-.09,color='#888',ls='--');ax.axvline(0,color='#444',lw=.6);ax.set_yticks(ix,bad.file);ax.set_xlabel('추정 Y - 파일명 명령 Y');ax.set_title('실패10장: 조건 전체 편향과 개별 반복 차이 분리');ax.legend();ax.grid(axis='x',alpha=.15);fig.savefig(H/'ten_error_decomposition.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,c,label in zip(axs,['rmse_p95','repeat_centered_D_rms','forward_residual_at_prediction'],['경계 맞춤 RMSE 95%점','같은 조건 대비 형태 좌표 RMS','추정 자세의 표준화 순방향 잔차']):
        ax.scatter(df[c],df.eY,c=np.where(df.passed,'#3b899b','#d04a37'),alpha=.8,s=25);ax.axhline(.09,color='#888',ls='--');ax.set(xlabel=label,ylabel='Y 절대오차');ax.grid(alpha=.15)
    fig.suptitle('전체60장 진단: 청록=통과, 주황=실패');fig.savefig(H/'all60_quality_vs_error.png',dpi=170);plt.close(fig)
    print(df[~df.passed][['file','predY','predT','eY','condition_bias_Y','individual_pred_deviation_Y','same_condition_passes','valid_points','rmse_p95_percentile_all60','repeat_centered_D_rms_percentile_all60']].to_string(index=False))
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
