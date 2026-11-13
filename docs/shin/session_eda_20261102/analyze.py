"""Session EDA. Run with bundled Python and workspace .analysis_packages on PYTHONPATH.
No source image or original estimator is modified. All random generators are seeded.
"""
from pathlib import Path
import json, hashlib, re, struct, html, itertools
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SRC=ROOT/'qrwithbar-lanczos-src/experiments/hye/20261102_173432_t_six_methods'
RAW=Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'figure.dpi':130})
RNG=np.random.default_rng(507)

def write_json(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def prepare():
    o=pd.read_csv(SRC/'observations.csv')
    fs=pd.read_csv(HERE/'filesystem_metadata.csv')
    m=fs[fs.extension=='.bmp'].copy()
    assert len(m)==575 and set(m.path)==set(o.path)
    for col in ['creation_utc','modified_utc']:
        m[col]=pd.to_datetime(m[col],format='ISO8601',utc=True)
    m=m.sort_values(['modified_utc','path']).reset_index(drop=True)
    gap=m.modified_utc.diff().dt.total_seconds().fillna(0)
    m['gap_s']=gap
    m['session']=(gap>120).cumsum()+1
    m['date']=m.modified_utc.dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d')
    m['local_time']=m.modified_utc.dt.tz_convert('Asia/Seoul').astype(str)
    m['capture_timestamp_available']=False
    m['time_basis']='LastWriteTimeUtc proxy, NOT capture time'
    m['creation_session_120']=(m.creation_utc.diff().dt.total_seconds().fillna(0)>120).cumsum()+1
    # Read all headers / EXIF and sidecars. Actual coordinates are audit data only.
    extra=[]
    for p in m.path:
        f=RAW/p
        with f.open('rb') as h: header=h.read(54)
        with Image.open(f) as im:
            exif=dict(im.getexif()); size=im.size
            info={k:str(v) for k,v in im.info.items()}
        sidecars=list(f.parent.glob(f.stem+'.txt*'))
        vals=[]
        if sidecars:
            vals=re.findall(r'[-+]?\d+(?:\.\d+)?',sidecars[0].read_text(errors='replace'))
        extra.append({'path':p,'header_size':struct.unpack('<I',header[14:18])[0],
                      'width':size[0],'height':size[1],'exif_tags':len(exif),
                      'image_info':json.dumps(info),'actual_file':str(sidecars[0]) if sidecars else '',
                      'actual_Y':float(vals[0]) if len(vals)>=2 else np.nan,
                      'actual_T':float(vals[1]) if len(vals)>=2 else np.nan})
    m=m.merge(pd.DataFrame(extra),on='path',validate='one_to_one')
    d=o.merge(m,on='path',validate='one_to_one').copy()
    assert (d.session==d.session_proxy+1).all()
    d['pose']=d.Y.round(6).astype(str)+':'+d['T'].round(6).astype(str)
    d['actual_origin_Y']=d.actual_Y-d.Y
    d['actual_origin_T']=d.actual_T-d['T']
    d.drop(columns=[c for c in d if re.fullmatch(r'f\d+',c)]).to_csv(HERE/'image_inventory.csv',index=False)
    sens=[]
    for threshold in [60,90,120,180,300,600,900,1800,3600]:
        labels=(gap>threshold).cumsum()+1
        sens.append({'gap_threshold_s':threshold,'all_segments':labels.nunique(),
                     'nonreference_segments':labels[m.path.isin(o[o.cohort!='reference'].path)].nunique()})
    pd.DataFrame(sens).to_csv(HERE/'session_threshold_sensitivity.csv',index=False)
    write_json('metadata_audit.json',{'images':len(m),'creation_min':m.creation_utc.min(),
        'creation_max':m.creation_utc.max(),'creation_span_s':(m.creation_utc.max()-m.creation_utc.min()).total_seconds(),
        'modification_dates':list(m.date.unique()),'internal_capture_tags':int(m.exif_tags.sum()),
        'bmp_header_sizes':sorted(m.header_size.unique().tolist()),'sidecar_images':int(m.actual_Y.notna().sum()),
        'interpretation':'CreationTime is a copy/import artifact; sessions use modified-time proxies only.'})
    return d

def verify_hashes():
    audit=pd.read_csv(ROOT/'qrwithbar-lanczos-src/modules/t_estimation/results-raw575/audit.csv')
    rows=[]
    for r in audit.itertuples():
        digest=hashlib.file_digest((RAW/r.path).open('rb'),'sha256').hexdigest()
        rows.append({'path':r.path,'sha256':digest,'matches_prior':digest==r.sha256})
    check=pd.DataFrame(rows);check.to_csv(HERE/'hash_audit.csv',index=False)
    assert check.matches_prior.all()
    print('HASHES VERIFIED',len(check),flush=True)

def feature_audit(d):
    # Cached extraction is used during the new extraction; final report asserts remeasurement identity.
    fresh=HERE/'remeasured/observations.csv'
    if fresh.exists() and len(pd.read_csv(fresh))==575:
        f=pd.read_csv(fresh)
        cols=[f'f{i}' for i in range(104)]
        joined=d[['path']+cols].merge(f[['path','status']+cols],on='path',suffixes=('_old','_new'),validate='one_to_one')
        err=np.max(np.abs(joined[[c+'_old' for c in cols]].to_numpy()-joined[[c+'_new' for c in cols]].to_numpy()))
        assert len(joined)==575 and not joined.status.any() and err<1e-8
        write_json('remeasurement_validation.json',{'images':len(joined),'failures':int(joined.status.sum()),'max_feature_difference':float(err),'features':104})
        print('REMEASURED IDENTICAL',err,flush=True)

def summarize(d):
    pred=pd.read_csv(SRC/'results/predictions_method7.csv')
    e=d.merge(pred[['path','fold','T_pred','error_ticks']],on='path',validate='one_to_one').copy()
    assert len(e)==571 and np.allclose((e.T_pred-e['T'])*100,e.error_ticks)
    e['abs_ticks']=e.error_ticks.abs()
    e['time_min']=e.groupby('session').modified_utc.transform(lambda s:(s-s.min()).dt.total_seconds()/60)
    e.to_csv(HERE/'samples_with_errors.csv',index=False)
    summary=[]
    for s,q in d.groupby('session'):
        z=e[e.session==s]; poses=z.groupby('pose').error_ticks.mean().to_numpy()
        ci=np.quantile(RNG.choice(poses,(3000,len(poses)),replace=True).mean(1),[.025,.975]) if len(poses)>=3 else [np.nan,np.nan]
        ts=stats.spearmanr(z.time_min,z.error_ticks) if len(z)>=8 else None
        summary.append({'session':int(s),'start_KST':q.local_time.min(),'end_KST':q.local_time.max(),
            'images':len(q),'evaluated':len(z),'poses':len(poses),'Y_min':q.Y.min(),'Y_max':q.Y.max(),'T_min':q['T'].min(),'T_max':q['T'].max(),
            'bias_ticks':z.error_ticks.mean(),'bias_ci_lo':ci[0],'bias_ci_hi':ci[1],
            'MAE_ticks':z.abs_ticks.mean(),'RMSE_ticks':np.sqrt(np.mean(z.error_ticks**2)),
            'max_abs_ticks':z.abs_ticks.max(),'pass_1tick':int((z.abs_ticks<=1).sum()),
            'time_error_spearman':ts.statistic if ts else np.nan,'time_error_p_uncorrected':ts.pvalue if ts else np.nan})
    summary=pd.DataFrame(summary);summary.to_csv(HERE/'session_summary.csv',index=False)
    # Pairwise exact-pose overlap is essential to origin identifiability.
    overlap=np.zeros((14,14),dtype=int)
    for a,b in itertools.product(range(1,15),repeat=2):
        overlap[a-1,b-1]=len(set(d[d.session==a].pose)&set(d[d.session==b].pose))
    pd.DataFrame(overlap,index=range(1,15),columns=range(1,15)).to_csv(HERE/'session_pose_overlap.csv')
    groups=e.groupby('pose').error_ticks
    within=sum(((q-q.mean())**2).sum() for _,q in groups)
    df=len(e)-groups.ngroups
    total=((e.error_ticks-e.error_ticks.mean())**2).sum()
    rep=groups.agg(['size','mean','std','min','max']).reset_index()
    rep['range_ticks']=rep['max']-rep['min'];rep.to_csv(HERE/'repeat_pose_errors.csv',index=False)
    centered=e.error_ticks-e.groupby('session').error_ticks.transform('mean')
    write_json('error_decomposition.json',{'images':len(e),'poses':groups.ngroups,'repeated_poses':int((rep['size']>1).sum()),
        'pooled_within_pose_sd_ticks':np.sqrt(within/df),'within_pose_SSE_fraction':within/total,
        'MAE_ticks':e.abs_ticks.mean(),'RMSE_ticks':np.sqrt(np.mean(e.error_ticks**2)),
        'max_ticks':e.abs_ticks.max(),'pass_1tick':int((e.abs_ticks<=1).sum()),
        'oracle_session_centered_RMSE_ticks':np.sqrt(np.mean(centered**2)),
        'oracle_session_centered_max_ticks':centered.abs().max(),
        'oracle_session_centering_note':'In-sample descriptive removal of session means. Not predictive evaluation or deployable correction.',
        'nonzero_pose_cross_session_overlap':int(d[d['T'].ne(0)|d.Y.ne(0)].groupby('pose').session.nunique().gt(1).sum())})
    return e,summary,overlap

def origins(d):
    # A local measured response; this is an equivalent Y/T displacement, not calibrated 3D pose.
    cols=[f'f{i}' for i in range(10)]
    base=d[(d.date=='2026-10-04')&(d.Y.abs()<=.200001)&(d['T'].abs()<=.200001)].copy()
    y=base.Y.to_numpy();t=base['T'].to_numpy();B=np.c_[np.ones(len(base)),y,t,y*y,t*t]
    local_rank=int(np.linalg.matrix_rank(B))
    coef=np.linalg.lstsq(B,base[cols].to_numpy(),rcond=None)[0]
    J=coef[1:3].T
    # Estimate noise from actual repeated images; a point count is not a replicate count.
    rep=d[d.cohort!='reference'];X=rep[cols].to_numpy();means=rep.groupby('pose')[cols].transform('mean').to_numpy()
    residual=X-means;df=len(rep)-rep.pose.nunique()
    cov=residual.T@residual/df
    cov=.8*cov+.2*np.diag(np.diag(cov))
    W=np.linalg.pinv(cov);P=np.linalg.solve(J.T@W@J,J.T@W)
    centers=d[d.cohort=='reference'].sort_values('modified_utc').copy()
    baseline=centers[centers.path=='center.bmp'][cols].to_numpy()[0]
    changes=centers[cols].to_numpy()-baseline
    equivalent=changes@P.T
    unexplained=changes-equivalent@J.T
    rows=[]
    for k,(_,r) in enumerate(centers.iterrows()):
        row={'path':r.path,'session':r.session,'time_KST':r.local_time,
             'equivalent_Y_mm':equivalent[k,0],'equivalent_T_deg':equivalent[k,1],
             'equivalent_T_ticks':equivalent[k,1]*100,
             'unexplained_QR_RMS_px':np.sqrt(np.mean(unexplained[k,:6]**2)),
             'unexplained_line_RMS_px':np.sqrt(np.mean(unexplained[k,6:]**2))}
        row.update({f'QR{q}_dx_px':changes[k,q*2] for q in range(3)})
        row.update({f'QR{q}_dy_px':changes[k,q*2+1] for q in range(3)})
        rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(HERE/'reference_origin_comparison.csv',index=False)
    pd.DataFrame(J,index=cols,columns=['d_feature_dY','d_feature_dT']).to_csv(HERE/'local_motion_jacobian.csv')
    # Sensitivity to local calibration range and sensor block. No false independent-point CI.
    sensitivity=[]
    for radius in [.10,.15,.20,.30,.50]:
        b=d[(d.date=='2026-10-04')&(d.Y.abs()<=radius+1e-6)&(d['T'].abs()<=radius+1e-6)]
        y=b.Y.to_numpy();t=b['T'].to_numpy();B=np.c_[np.ones(len(b)),y,t,y*y,t*t]
        c=np.linalg.lstsq(B,b[cols].to_numpy(),rcond=None)[0]
        for mode,ix in [('QR+line',np.arange(10)),('QR only',np.arange(6)),('line only',np.arange(6,10))]:
            j=c[1:3,ix].T;w=np.linalg.pinv(cov[np.ix_(ix,ix)])
            p=np.linalg.pinv(j.T@w@j)@j.T@w
            eq=changes[:,ix]@p.T
            for k,(_,r) in enumerate(centers.iterrows()):
                sensitivity.append({'path':r.path,'radius':radius,'features':mode,'Y_mm':eq[k,0],'T_ticks':eq[k,1]*100,
                                    'weighted_jacobian_condition':np.linalg.cond(np.linalg.cholesky(w).T@j)})
    pd.DataFrame(sensitivity).to_csv(HERE/'origin_sensitivity.csv',index=False)
    # Nominal-origin consistency of sidecar coordinates.
    side=d[d.actual_Y.notna()].copy()
    side.groupby('session').agg(n=('path','size'),origin_Y_min=('actual_origin_Y','min'),origin_Y_max=('actual_origin_Y','max'),
        origin_T_min=('actual_origin_T','min'),origin_T_max=('actual_origin_T','max')).to_csv(HERE/'sidecar_origin_audit.csv')
    # Same-date repeated center1->center2->center3 comparisons have minimal extrapolation.
    write_json('origin_model.json',{'local_calibration_images':len(base),'local_range':.2,'polynomial_rank':local_rank,
        'jacobian_rank':int(np.linalg.matrix_rank(J)),'noise_repeat_dof':df,
        'warning':'Equivalent displacement is conditional on local Oct-4 response. Camera/fixture changes and coordinate zero changes cannot be separated by these data.'})
    return centers,result,changes,J

def diagnostics(d,e):
    # Detailed point/line data are re-extracted. Read old cache only during development.
    import cv2
    cache=HERE/'remeasured/cache' if (HERE/'remeasured/cache/frame574.yml.gz').exists() else SRC/'cache'
    metrics=[];points=[];edges=[]
    for r in d.itertuples():
        fs=cv2.FileStorage(str(cache/f'frame{r.index}.yml.gz'),cv2.FILE_STORAGE_READ)
        p=fs.getNode('points').mat();v=fs.getNode('edges').mat();fs.release()
        metrics.append({'path':r.path,'line_invalid':int((v[:,5]==0).sum()),'line_rmse_median':np.median(v[:,4]),
            'line_contrast_median':np.median(v[:,3]),'line_sigma_median':np.median(v[:,2])})
        points.append(p[:,:2]);edges.append(v)
    met=pd.DataFrame(metrics).merge(e[['path','session','abs_ticks','error_ticks']],on='path',how='left')
    met.to_csv(HERE/'feature_quality.csv',index=False)
    m=met.dropna(subset=['abs_ticks'])
    corr=[]
    for c in ['line_invalid','line_rmse_median','line_contrast_median','line_sigma_median']:
        s=stats.spearmanr(m[c],m.abs_ticks)
        corr.append({'feature':c,'spearman_abs_error':s.statistic,'p_uncorrected':s.pvalue})
    pd.DataFrame(corr).to_csv(HERE/'quality_correlations.csv',index=False)
    # Full point cloud repeatability measured by QR block; do not divide noise by sqrt(630).
    a=np.asarray(points);ed=np.asarray(edges);noise=[]
    for pose,sub in d[d.cohort!='reference'].groupby('pose'):
        if len(sub)<2:continue
        ix=sub.index.to_numpy();z=a[ix];scatter=z-z.mean(0)
        row={'pose':pose,'n':len(sub),'point_RMS_px':np.sqrt(np.sum(scatter**2)/((len(sub)-1)*630)),
             'line_repeat_y_SD_px':np.sqrt(np.sum((ed[ix,:,1]-ed[ix,:,1].mean(0))**2)/((len(sub)-1)*120))}
        for q in range(3):
            mask=p[:,2]==q
            zq=z[:,mask].mean(1)
            row[f'QR{q}_centroid_SD_px']=np.sqrt(np.var(zq,axis=0,ddof=1).sum())
        noise.append(row)
    pd.DataFrame(noise).to_csv(HERE/'feature_repeatability.csv',index=False)
    # Direct center-to-center displacement; motion-mode conversion is intentionally separate.
    ci=d.index[d.cohort=='reference'].to_numpy();anchor=ci[d.loc[ci,'path'].to_numpy()=='center.bmp'][0]
    center_rows=[]
    for ix in ci:
        delta=a[ix]-a[anchor]
        mu=delta.mean(0);nontranslation=delta-mu
        center_rows.append({'path':d.loc[ix,'path'],'dx_px':mu[0],'dy_px':mu[1],
            'translation_norm_px':np.linalg.norm(mu),'nontranslation_point_RMS_px':np.sqrt(np.mean(np.sum(nontranslation**2,axis=1)))})
    pd.DataFrame(center_rows).to_csv(HERE/'reference_pixel_displacements.csv',index=False)
    # Revisit historical extraction comparison as a repeatability check, not absolute truth.
    fixture=ROOT/'qrwithbar-lanczos-src/modules/qrwithbar_lanczos/tests/fixtures/comparison_features.csv'
    comp=pd.read_csv(fixture);comp['pose']=comp.path.str.split('/').str[0]
    check=[]
    for name,g in comp.groupby('method'):
        values=g[[f'f{i}' for i in range(6)]].to_numpy()
        resid=values-g.groupby('pose')[[f'f{i}' for i in range(6)]].transform('mean').to_numpy()
        dof=len(g)-g.pose.nunique()
        check.append({'method':name,'images':len(g),'failed_measurements':int(g.failures.sum()),
                      'QR_centroid_pooled_component_SD_px':np.sqrt(np.sum(resid**2)/(dof*6))})
    pd.DataFrame(check).to_csv(HERE/'historical_extraction_repeatability.csv',index=False)
    return met

def label_audit(d,e):
    rows=[];audit=d[['path','Y','T','pose']].copy();audit['audit_T']=audit['T']
    for i,r in d.iterrows():
        parent=Path(r.path).parent.name
        val=re.search(r'Y([+-]\d+(?:\.\d+)?)\s+T([+-]\d+(?:\.\d+)?)',parent)
        if val and abs(float(val[2])-r['T'])>.00001:
            audit.loc[i,'audit_T']=float(val[2])
            rows.append({'path':r.path,'issue':'filename_folder_disagreement','filename_T':r['T'],
                         'folder_T':float(val[2]),'actual_T':r.actual_T,'assumption':'folder + matching sidecar pairing; source unchanged'})
        if pd.notna(r.actual_T) and abs(r.actual_origin_T+1.67)>.0001:
            rows.append({'path':r.path,'issue':'actual_minus_nominal_origin_disagreement',
                         'filename_T':r['T'],'folder_T':float(val[2]) if val else np.nan,'actual_T':r.actual_T,
                         'assumption':'could be execution/rounding/log/label issue; do not silently relabel'})
    pd.DataFrame(rows).to_csv(HERE/'label_anomalies.csv',index=False)
    audit['audit_pose']=audit.Y.round(6).astype(str)+':'+audit.audit_T.round(6).astype(str)
    q=e[['path','error_ticks','T_pred']].merge(audit[['path','audit_T','audit_pose']],on='path')
    old_df=len(e)-e.pose.nunique();new_df=len(q)-q.audit_pose.nunique()
    old_sse=sum(((g-g.mean())**2).sum() for _,g in e.groupby('pose').error_ticks)
    q['audit_error_ticks']=(q.T_pred-q.audit_T)*100
    new_sse=sum(((g-g.mean())**2).sum() for _,g in q.groupby('audit_pose').audit_error_ticks)
    # Existing predictions are fixed: a relabelled score is diagnostic, not corrected OOF.
    write_json('label_sensitivity.json',{'anomaly_images':len(set(r['path'] for r in rows)),
        'filename_folder_mismatches':sum(r['issue']=='filename_folder_disagreement' for r in rows),
        'sidecar_origin_mismatches':sum(r['issue']=='actual_minus_nominal_origin_disagreement' for r in rows),
        'original_within_pose_sd_ticks':np.sqrt(old_sse/old_df),
        'folder_audited_within_pose_sd_ticks':np.sqrt(new_sse/new_df),
        'fixed_prediction_audited_MAE_ticks':q.audit_error_ticks.abs().mean(),
        'note':'A corrected calibration model and all folds must be rerun before claiming corrected evaluation performance.'})
    # Show the misplaced repeat in feature space, with original and audited labels.
    near=d[(d.Y==.05)&d['T'].between(.30,.34)].copy()
    near=near.merge(audit[['path','audit_T']],on='path')
    fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for a,col,title in [(ax[0],'T','파일명 기준'),(ax[1],'audit_T','폴더·대응 TXT를 대조한 분류')]:
        for t,g in near.groupby(col):a.scatter(g.f0,g.f1,s=50,label=f'T={t:.2f}')
        suspect=near[near['T']!=near.audit_T]
        a.scatter(suspect.f0,suspect.f1,s=180,facecolors='none',edgecolors='red',linewidths=2)
        a.set(xlabel='좌측 QR 평균 dx (px)',ylabel='좌측 QR 평균 dy (px)',title=title);a.legend()
    fig.suptitle('빨간 원: T=.32 폴더에 들어 있는 T=.33 파일명 이미지')
    fig.savefig(HERE/'05_label_audit.png');plt.close(fig)

def session_features(d,e):
    rows=[]
    for sid,g in d.groupby('session'):
        time=(g.modified_utc-g.modified_utc.min()).dt.total_seconds().to_numpy()/60
        yy=g.Y.to_numpy();tt=g['T'].to_numpy()
        for i in range(104):
            v=g[f'f{i}'].to_numpy()
            X=np.c_[np.ones(len(g)),yy,tt]
            rank=np.linalg.matrix_rank(X)
            coef=np.linalg.lstsq(X,v,rcond=None)[0]
            fit=X@coef
            corr=stats.spearmanr(time,v).statistic if len(g)>2 and np.ptp(v)>0 else np.nan
            rows.append({'session':sid,'feature':f'f{i}','n':len(g),'mean':v.mean(),
                         'sd':v.std(ddof=1) if len(v)>1 else np.nan,'min':v.min(),'max':v.max(),
                         'time_spearman':corr,'intercept_descriptive':coef[0],
                         'Y_slope':coef[1] if np.ptp(yy)>0 else np.nan,
                         'T_slope':coef[2] if np.ptp(tt)>0 else np.nan,
                         'design_rank':rank,'residual_SD':np.sqrt(np.sum((v-fit)**2)/(len(g)-rank)) if len(g)>rank else np.nan,
                         'note':'within-session descriptive slopes; intercepts are not independent origin estimates'})
    pd.DataFrame(rows).to_csv(HERE/'session_feature_statistics.csv',index=False)
    fig,axs=plt.subplots(3,4,figsize=(17,11),layout='constrained',sharey=True)
    for ax,(sid,g) in zip(axs.flat,e.groupby('session')):
        field='T' if g['T'].nunique()>1 else 'Y'
        ax.scatter(g[field],g.error_ticks,s=15,alpha=.75)
        means=g.groupby(field).error_ticks.mean();ax.plot(means.index,means.values,color='#db8142',linewidth=1)
        ax.axhline(0,color='#888',lw=.7);ax.axhline(1,color='#c88',ls='--',lw=.7);ax.axhline(-1,color='#c88',ls='--',lw=.7)
        ax.set(xlabel='T (°)' if field=='T' else 'Y (mm)',title=f'S{sid:02} · n={len(g)}, MAE={g.abs_ticks.mean():.3f}')
        ax.set_ylim(-3.2,3.2);ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.supylabel('T 부호 오차 (tick) · 주황선은 같은 명령 좌표의 평균')
    fig.suptitle('기준 전용 S06·S07 제외: 12개 세션 후보별 오차 경향')
    fig.savefig(HERE/'06_each_session.png');plt.close(fig)

def charts(d,e,s,overlap,centers,origin,changes):
    colors=plt.colormaps['tab20'](np.linspace(0,1,14))
    fig,ax=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    creation=(d.creation_utc-d.creation_utc.min()).dt.total_seconds()
    modification=(d.modified_utc-d.modified_utc.min()).dt.total_seconds()/3600
    ax[0,0].scatter(modification,creation,c=[colors[i-1] for i in d.session],s=12)
    ax[0,0].set(xlabel='수정 시각: 최초 기록 이후 시간 (h)',ylabel='생성 시각: 복사 시작 이후 (s)',title='생성 시각은 촬영 시각을 보존하지 않음')
    for sid,q in d.groupby('session'):
        day=q.date.iloc[0];tm=q.modified_utc.dt.tz_convert('Asia/Seoul');x=tm.dt.hour+tm.dt.minute/60+tm.dt.second/3600
        ax[0,1].scatter(x,np.full(len(q),sid),s=10,color=colors[sid-1])
    ax[0,1].set(xlabel='기록 시각 (KST 시 단위; 날짜는 세션 표 참조)',ylabel='수정 시각 세션 후보',yticks=range(1,15),title='120초 초과 공백 → 14개 후보 (촬영 로그 아님)')
    sens=pd.read_csv(HERE/'session_threshold_sensitivity.csv')
    ax[1,0].plot(sens.gap_threshold_s,sens.all_segments,'o-');ax[1,0].set(xscale='log',xlabel='구분 공백 (초, 로그 축)',ylabel='세션 후보 수',title='시간 경계에 대한 민감도')
    for sid,q in d.groupby('session'):
        ax[1,1].scatter(q.Y,q['T'],s=14,color=colors[sid-1],label=f'S{sid:02}')
    ax[1,1].set(xlabel='Y 명령 (mm)',ylabel='T 명령 (°)',title='촬영 구간과 자세가 강하게 얽혀 있음');ax[1,1].legend(ncol=4,fontsize=8)
    fig.savefig(HERE/'01_metadata_sessions.png');plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    valid=s[s.evaluated>0]
    ax[0,0].bar(valid.session,valid.MAE_ticks,color=[colors[i-1] for i in valid.session]);ax[0,0].set(ylabel='MAE (tick)',xlabel='세션 후보',title='기존 method 7 좌표 보류 OOF 오차')
    ax[0,1].axhline(0,color='#777');ax[0,1].errorbar(valid.session,valid.bias_ticks,yerr=np.maximum(0,np.array([valid.bias_ticks-valid.bias_ci_lo,valid.bias_ci_hi-valid.bias_ticks])),fmt='o',capsize=4)
    ax[0,1].set(xlabel='세션 후보',ylabel='평균 부호 오차 (tick)',title='자세 단위 bootstrap 95% CI (기술 통계)')
    for sid,q in e.groupby('session'):
        ax[1,0].scatter(q['T'],q.error_ticks,s=12,alpha=.6,color=colors[sid-1])
    ax[1,0].axhline(0,color='#777');ax[1,0].axhline(1,ls='--',color='#c66');ax[1,0].axhline(-1,ls='--',color='#c66')
    ax[1,0].set(xlabel='T 명령 (°)',ylabel='T 예측 - 명령 (tick)',title='오차는 원점의 일정한 평행 이동만으로 설명되지 않음')
    q=e[e.session==8].sort_values('modified_utc');ax[1,1].scatter(q.time_min,q.error_ticks,c=q['T'],cmap='viridis',s=14)
    ax[1,1].axhline(0,color='#777');ax[1,1].set(xlabel='S08 시작 이후 분',ylabel='부호 오차 (tick)',title='S08 시간 경향: T 증가와 시간 증가가 함께 진행')
    for a in ax.flat:a.grid(alpha=.2);a.set_axisbelow(True)
    fig.savefig(HERE/'02_session_errors.png');plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    labels=['10/4 center','10/15 center1','10/15 center2','10/15 center3']
    for k in range(1,4):
        ax[0,0].plot(range(6),changes[k,:6],'o-',label=labels[k])
    ax[0,0].set(xticks=range(6),xticklabels=['左 dx','左 dy','中 dx','中 dy','右 dx','右 dy'],ylabel='10/4 center 대비 변위 (px)',title='명령 (Y,T)=(0,0)의 실제 영상 특징 차이');ax[0,0].legend()
    ss=pd.read_csv(HERE/'origin_sensitivity.csv');q=ss[ss.features=='QR+line']
    for path,g in q.groupby('path'):
        if path=='center.bmp':continue
        ax[0,1].plot(g.radius,g.T_ticks,'o-',label=Path(path).name)
    ax[0,1].set(xlabel='국소 보정 반경: |Y|≤r, |T|≤r',ylabel='영상 차이에 해당하는 T (tick)',title='원점의 tick 환산이 불안정 → 확정값으로 채택 불가');ax[0,1].legend()
    im=ax[1,0].imshow(overlap,cmap='Blues');fig.colorbar(im,ax=ax[1,0],label='두 구간에 공통인 명령 자세 수')
    ax[1,0].set(xticks=range(14),xticklabels=range(1,15),yticks=range(14),yticklabels=range(1,15),xlabel='세션 후보',ylabel='세션 후보',title='정확히 같은 자세를 비교 가능한 구간은 기준 사진뿐')
    side=d[d.actual_Y.notna()];ax[1,1].scatter(side.session,side.actual_origin_T*100,s=12,label='Actual T − 명령 T')
    ax[1,1].set(xlabel='세션 후보',ylabel='명령 원점의 Actual T (tick)',title='동봉 로그: 10/15의 원점 좌표 일관성')
    fig.savefig(HERE/'03_origin_evidence.png');plt.close(fig)
    rep=pd.read_csv(HERE/'repeat_pose_errors.csv');f=pd.read_csv(HERE/'feature_repeatability.csv');met=pd.read_csv(HERE/'feature_quality.csv')
    fig,ax=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    ax[0].hist(rep.loc[rep['size']>1,'range_ticks'],bins=20,color='#4588a2');ax[0].set(xlabel='동일 자세 최대 - 최소 오차 (tick)',ylabel='자세 수',title='반복 촬영에서 남는 변동')
    ax[1].hist(f.point_RMS_px,bins=20,color='#6b9c75');ax[1].set(xlabel='동일 자세 QR 점 위치 RMS (px)',ylabel='자세 수',title='Lanczos 추출 후에도 남는 영상 측정 변동')
    q=met.dropna(subset=['abs_ticks']);ax[2].scatter(q.line_invalid,q.abs_ticks,s=12,alpha=.5);ax[2].set(xlabel='저품질 선 표본 수 / 120',ylabel='절대오차 (tick)',title='선 품질 단독으로 실패를 설명하는가')
    fig.savefig(HERE/'04_repeatability_quality.png');plt.close(fig)

def main():
    d=prepare();print('METADATA PREPARED',flush=True)
    verify_hashes()
    feature_audit(d)
    e,s,overlap=summarize(d)
    centers,origin,changes,J=origins(d)
    diagnostics(d,e);label_audit(d,e);session_features(d,e);charts(d,e,s,overlap,centers,origin,changes)
    print(s.to_string(index=False));print('ORIGINS');print(origin.to_string(index=False))
    print('SIDECARS');print(pd.read_csv(HERE/'sidecar_origin_audit.csv').to_string(index=False))

if __name__=='__main__':main()
