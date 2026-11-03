from refine_origin import *
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize._numdiff import approx_derivative
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})

def main():
    original,f,a=load();summary=pd.read_csv(P/'refinement_summary.csv');cv=pd.read_csv(P/'refinement_CV.csv');post=pd.read_csv(P/'refinement_zero_holdouts.csv')
    baseline=summary[summary.config=='audited_time'].set_index('scheme')
    choices=[]
    for name,q in summary.groupby('config'):
        q=q.set_index('scheme')
        if (q.QR_RMSE<=baseline.QR_RMSE*1.05).all():choices.append((float(q.loc['contiguous','right_RMSE']),name))
    _,selected=min(choices);cfg=next(c for c in CONFIGS if c[0]==selected)
    _,split,rr,core,y5,bg=cfg;d=audited_data(original,split)
    m=RefinedModel(d,f,a,rr,core,y5).fit(maxiter=650)
    probe=np.random.default_rng(507).normal(0,.1,m.npar)
    analytic=m.objective_scene(probe,np.ones_like(m.wq))[1]
    numeric=approx_derivative(lambda z:m.objective_scene(z,np.ones_like(m.wq))[0],probe,method='3-point').ravel()
    grad=np.linalg.norm(analytic-numeric)/max(1,np.linalg.norm(numeric));assert grad<1e-5
    rows=[];origin_QR=None
    for i,s in enumerate(m.specs):
        dim=m.dims[i];value=m.Cblocks[i,0,:dim]@np.linalg.inv(s['L'])
        if s['name']=='QR':origin_QR=value
        rows.append({'name':s['name'],'feature_id':s['fid'],'importance':float(m.importance[i]),'dimension':dim,
                     'origin_values':value.tolist(),'whitener':s['L'].tolist()})
    save('standard_origin_v3.json',{'schema':'audited_shared_origin.v3','status':'development_revalidated_statistical_reference',
        'selected_config':selected,'selection':'Minimize independent-right contiguous-block forward RMSE among tested candidates without >5% QR RMSE deterioration in either split scheme.',
        'reference_gauge':'Mean latent coordinates of the four nominal-zero reference images equals zero. Zero-reference exposures have separate bias groups from motion sweeps.',
        'image_frame_convention':'Oct15 registered into Oct4 frame using nominal-zero QR reference means; not causal separation of camera and stage motion.',
        'day_translation_xy_px':m.D,'origin_QR_features_6':origin_QR,'origin_blocks':rows,'polynomial_powers':m.powers,
        'coefficients_whitened_padded':m.Cblocks,'scan_bias_YT':m.B,'bias_prior_sigma_Y_mm':m.sy,'bias_prior_sigma_T_deg':m.st,
        'gradient_relative_error':grad,'optimizer_rounds':m.optimization,'raw_bound_hits':int(np.sum(np.abs(m.u)>2.999)),
        'data_policy':'One filename-derived T is corrected to folder/local-sequence .32 in derived tables only; .57 images restored, Actual-log conflict retained; coherent repeat deviations soft-weighted, no source image deleted.',
        'physical_origin_certified':False,'T_accuracy_verified':False,'evaluation_note':'Same existing development data, adaptively explored. No new-session external validation.'})
    weightmap,rep=training_repeat_weights(d,f,a);rep.to_csv(P/'final_repeat_diagnostics.csv',index=False)
    out=d[['row_id','path','cohort','Y','T_original','T','session','group','date','correction_reason']].copy()
    ids=d.group.to_numpy(int)-1;out['Y_bias_mm']=m.B[ids,0];out['T_bias_deg']=m.B[ids,1]
    out['Y_standard']=out.Y+out.Y_bias_mm;out['T_standard']=out['T']+out.T_bias_deg
    out['training_image_weight']=[weightmap[int(i)] for i in d.row_id]
    out['Actual_log_conflict']=d.path.str.contains('Y+0.05 T+0.57/',regex=False)
    out.to_csv(P/'coordinates_standard_v3.csv',index=False)
    pd.DataFrame({'group':np.arange(1,18),'Y_bias_mm':m.B[:,0],'T_bias_deg':m.B[:,1],'T_bias_ticks':100*m.B[:,1]}).to_csv(P/'biases_v3.csv',index=False)
    mapping=d.groupby(['group','session','cohort']).agg(images=('path','size'),Y_min=('Y','min'),Y_max=('Y','max'),T_min=('T','min'),T_max=('T','max'))
    mapping.to_csv(P/'grouping_v3.csv')
    # Build a genuine 630-point origin with exactly the QR block's row weights.
    pointdata=[]
    for r in d.itertuples():
        fs=cv2.FileStorage(str(ROOT/'analysis/session_eda_20261102/remeasured/cache'/f'frame{r.index}.yml.gz'),cv2.FILE_STORAGE_READ)
        points=fs.getNode('points').mat();fs.release();pointdata.append(points[:,:2]-(m.D if r.date=='2026-10-15' else 0))
    X=np.asarray(pointdata).reshape(len(d),-1);w=m.weights[0];H=m.A.T@(w[:,None]*m.A)+np.diag(m.reg)
    rhs=m.A.T@(w[:,None]*X);rhs[0]+=m.reg[0]*(X[m.refs]*m.wq[0,m.refs,None]).sum(0)/m.wq[0,m.refs].sum()
    cloud=np.linalg.solve(H,rhs)[0].reshape(630,2)
    pd.DataFrame({'point_id':np.arange(630),'qr_id':points[:,2].astype(int),'x_standard_px':cloud[:,0],'y_standard_px':cloud[:,1]}).to_csv(P/'standard_origin_v3_QR630.csv',index=False)
    natural=[]
    for r in rows:
        if r['feature_id'] is not None and r['importance']>0:
            z=f.iloc[r['feature_id']];natural.append({'feature_id':r['feature_id'],'family':z.family,'x_standard_px':z.reference_x+r['origin_values'][0],
                'y_standard_px':z.reference_y+r['origin_values'][1]})
    pd.DataFrame(natural).to_csv(P/'standard_origin_v3_natural.csv',index=False)

    # Stability under resampling of complete pose groups; fixed origin anchors imply conditional intervals.
    rng=np.random.default_rng(507);bootstrap=[]
    for i in range(40):
        chunks=[d[d.cohort=='reference']]
        for _,g in d[d.cohort!='reference'].groupby('group'):
            keys=g.pose.unique()
            for key in rng.choice(keys,len(keys),replace=True):chunks.append(g[g.pose==key])
        sample=pd.concat(chunks,ignore_index=True)
        fit=RefinedModel(sample,f,a,rr,core,y5).fit(start=m.u,maxiter=300)
        oq=fit.Cblocks[0,0,:6]@np.linalg.inv(fit.specs[0]['L'])
        bootstrap.append({'replicate':i,**{f'f{k}':oq[k] for k in range(6)}})
        if (i+1)%10==0:print('ORIGIN BOOTSTRAP',i+1,flush=True)
    boot=pd.DataFrame(bootstrap);boot.to_csv(P/'conditional_bootstrap_v3.csv',index=False)
    pd.DataFrame({'feature':[f'f{i}' for i in range(6)],'estimate':origin_QR,'conditional_lo':np.quantile(boot.iloc[:,1:],.025,axis=0),
                  'conditional_hi':np.quantile(boot.iloc[:,1:],.975,axis=0)}).to_csv(P/'conditional_origin_intervals_v3.csv',index=False)
    chosen=summary[summary.config==selected].set_index('scheme');cp=post[post.config==selected];bp=post[post.config=='audited_time']
    changes={scheme:{'QR_RMSE_before':float(baseline.loc[scheme,'QR_RMSE']),'QR_RMSE_after':float(chosen.loc[scheme,'QR_RMSE']),
             'right_RMSE_before':float(baseline.loc[scheme,'right_RMSE']),'right_RMSE_after':float(chosen.loc[scheme,'right_RMSE'])}for scheme in ['pose','contiguous']}
    center_before=bp[bp.path.str.endswith('center3.bmp')].iloc[0];center_after=cp[cp.path.str.endswith('center3.bmp')].iloc[0]
    cv95=float(cv[(cv.config==selected)&(cv.scheme=='pose')].right_RMS_px.quantile(.95))
    save('result_summary_v3.json',{'selected':selected,'source_images':len(d),'evaluation_images_per_scheme':555,'label_corrections':int((out.T_original!=out['T']).sum()),
        'restored_Actual_conflict_images':int(out.Actual_log_conflict.sum()),'hard_removed_images':0,'repeat_soft_downweighted_images':int((out.training_image_weight<.999).sum()),
        'changes':changes,'center3_QR_before_after':[center_before.QR_RMS_px,center_after.QR_RMS_px],
        'center3_right_before_after':[center_before.right_RMS_px,center_after.right_RMS_px],'new_right_pose_95pct':cv95,
        'origin_features':origin_QR,'gradient_relative_error':grad,'conditioning_note':'Conditional bootstrap excludes anchor/image-frame/model-selection uncertainty. Physical T truth remains unknown.'})
    figures(d,f,a,summary,cv,post,selected,rep)
    print('EXPORTED',selected,origin_QR,flush=True)

def figures(d,f,a,summary,cv,post,selected,rep):
    old=pd.read_csv(P/'v2_region_OOF_residuals.csv')
    grid=old.groupby(['session','family']).RMS.median().unstack()
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained');im=ax.imshow(grid.to_numpy(),aspect='auto',cmap='YlOrRd',vmin=0,vmax=.7)
    ax.set(xticks=range(len(grid.columns)),xticklabels=grid.columns,yticks=range(len(grid)),yticklabels=[f'S{v:02}' for v in grid.index],
           title='기존 모델: 관측 부위별 보류 잔차 중앙값',xlabel='관측군',ylabel='수정시각 구간 후보')
    for i in range(len(grid)):
        for k in range(len(grid.columns)):
            val=grid.iloc[i,k]
            if pd.notna(val):ax.text(k,i,f'{val:.2f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='좌표 성분 RMS (px)');fig.savefig(P/'01_connection_break_heatmap.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,lo,hi,why in [(axs[0],.30,.34,'.32 폴더의 .33 파일명'),(axs[1],.53,.60,'.57 영상은 주변 궤적과 연속적')]:
        q=d[(d.Y==.05)&d['T'].between(lo,hi)]
        ax.scatter(q['T'],q.f0,s=35,color='#3a7a9b',label='감사 후 라벨')
        if lo==.30:
            z=q[q.T_original!=q['T']];ax.scatter(z.T_original,z.f0,facecolors='none',edgecolors='red',s=130,label='원래 파일명 .33')
            for r in z.itertuples():ax.annotate('',xy=(r.T,r.f0),xytext=(r.T_original,r.f0),arrowprops={'arrowstyle':'->','color':'red'})
        else:
            z=q[np.isclose(q['T'],.57)];ax.scatter(np.full(len(z),.56),z.f0,marker='x',color='red',s=80,label='Actual 로그로 환산한 .56')
        ax.set(xlabel='국소 순서의 T 라벨 (°)',ylabel='좌측 QR 평균 dx (px)',title=why);ax.legend(fontsize=9)
    fig.savefig(P/'02_label_evidence.png',dpi=160);plt.close(fig)
    inv=pd.read_csv(ROOT/'analysis/session_eda_20261102/image_inventory.csv');q=inv[inv.session==11].sort_values('modified_utc')
    times=pd.to_datetime(q.modified_utc,format='ISO8601');elapsed=(times-times.min()).dt.total_seconds()
    fig,ax=plt.subplots(figsize=(13,4.5),layout='constrained')
    ax.plot(elapsed,q.Y,'o-',color='#648aa2')
    for k,(x,(_,r)) in enumerate(zip(elapsed,q.iterrows())):
        label=('center3 원점 복귀' if r.Y==0 else f'Y=.1, T=.01 반복 {k+1}')+'\n'+str(pd.to_datetime(r.modified_utc).tz_convert('Asia/Seoul').time())
        offset=[(8,18),(-42,45),(38,18),(-5,25)][k]
        ax.annotate(label,(x,r.Y),xytext=offset,textcoords='offset points',ha='left' if k==0 else 'center',fontsize=9,
                    arrowprops={'arrowstyle':'-','color':'#607080','linewidth':.7})
    ax.set_xlim(-8,145)
    ax.axvline(elapsed.iloc[-1]-23,ls='--',color='#d89436');ax.set(xlabel='해당 구간 시작 이후 초 (수정 시각 대리)',ylabel='Y 명령 라벨 (mm)',ylim=(-.02,.16),
        title='S11: 마지막 반복 사진 후 46초에 원점 복귀 사진 · 120초 공백 기준은 이를 분리하지 못함')
    fig.savefig(P/'03_zero_reset_timeline.png',dpi=160);plt.close(fig)
    row=rep.nlargest(1,'common_norm').iloc[0];ri=int(row.row_id);meta=d[d.row_id==ri].iloc[0];other=d[(d.pose==meta.pose)&(d.row_id!=ri)].row_id.to_numpy(int)
    raw=Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set')/meta.path
    image=cv2.imdecode(np.fromfile(str(raw),np.uint8),cv2.IMREAD_GRAYSCALE)
    fig,ax=plt.subplots(figsize=(15,8),layout='constrained');ax.imshow(image,cmap='gray',vmin=0,vmax=255)
    for family,sub in f.groupby('family'):
        xy=[];motion=[]
        for fid in sub.feature_id:
            vo=a['valid'][other,fid]
            if not a['valid'][ri,fid] or vo.sum()<1:continue
            xy.append(a['v'][ri,fid,:2]);motion.append(a['v'][ri,fid,:2]-np.median(a['v'][other[vo],fid,:2],axis=0))
        if len(motion)>=2:
            pos=np.median(xy,axis=0);vec=np.median(motion,axis=0)
            ax.arrow(pos[0],pos[1],vec[0]*200,vec[1]*200,width=3,head_width=20,color='#eb475b',length_includes_head=True)
            ax.text(pos[0]+15,pos[1]-25,f'{family}\n({vec[0]:+.3f}, {vec[1]:+.3f})px',color='#ffcd43',fontsize=9,bbox={'facecolor':'#16252c','alpha':.8,'pad':3})
    ax.set(title=f'반복 편차 최대 후보: {Path(meta.path).name}\n여러 부위가 함께 이동 · 화살표는 변위 200배 확대, 정합 불량 사진으로 단정하지 않음',xlim=(0,3840),ylim=(2160,0))
    fig.savefig(P/'04_repeat_deviation_photo.png',dpi=150);plt.close(fig)
    configs=[c[0] for c in CONFIGS];fig,axs=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for ax,metric in zip(axs,['QR_RMSE','right_RMSE']):
        for scheme,color,offset in [('pose','#407b97',-.18),('contiguous','#d39d48',.18)]:
            vals=[summary[(summary.config==name)&(summary.scheme==scheme)][metric].iloc[0] for name in configs]
            ax.bar(np.arange(len(configs))+offset,vals,width=.35,color=color,label=scheme)
        ax.set(xticks=range(len(configs)),xticklabels=['기존 구간','원점 분리','+반복 강건','본체 위주','본체+배경 Y5','전체+Y5'],ylabel='좌표 성분 RMSE (px)',title='QR' if metric=='QR_RMSE' else '원점 생성 미사용 오른쪽 특징');ax.legend()
    fig.savefig(P/'05_refinement_validation.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,field in zip(axs,['QR_RMS_px','right_RMS_px']):
        for name,color,offset in [('audited_time','#b2bbc1',-.18),(selected,'#288395',.18)]:
            q=post[post.config==name].sort_values('path');ax.bar(np.arange(3)+offset,q[field],width=.35,color=color,label='반영 전' if name=='audited_time' else 'v3')
        ax.set(xticks=range(3),xticklabels=['center1','center2','center3'],ylabel='보류 원점 사진 RMS (px)',title='QR' if field.startswith('QR') else '보류 오른쪽 특징');ax.legend()
    fig.savefig(P/'06_zero_holdout_before_after.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
