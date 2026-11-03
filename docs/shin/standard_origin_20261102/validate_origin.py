from fit_origin import *
from scipy.optimize._numdiff import approx_derivative

full,L=data();d=full[~full.suspect].copy();j=json.loads((P/'standard_origin.json').read_text())
m=Model(d,L,j['degree'],True,j['sigma_session_label_Y_mm'],j['sigma_session_label_T_deg'])
B=np.asarray(j['scan_group_bias_YT'])
u=((B[m.free]-B[m.fixed])/[m.sy,m.st]).ravel()
w=pd.read_csv(P/'fit_residuals.csv').robust_weight.to_numpy()
m.u=u;m.objective(u,w,True)
assert np.max(np.abs(m.origin()-np.asarray(j['origin_features_10'])))<1e-7
probe=RNG.normal(0,.2,len(u))
v,grad=m.objective(probe,np.ones(len(d)))
num=approx_derivative(lambda q:m.objective(q,np.ones(len(d)))[0],probe,method='3-point').ravel()
rel=np.linalg.norm(grad-num)/max(1,np.linalg.norm(num))
assert rel<1e-5,rel
assert np.max(np.abs(m.anchor@m.B))<1e-12
assert np.isfinite(m.C).all()

# Repeated photos are resampled together. The four coordinate anchors are held fixed.
boots=[]
for b in range(60):
    chunks=[d[d.cohort=='reference']]
    for gid,q in d[d.cohort!='reference'].groupby('group'):
        keys=q.pose.unique()
        for key in RNG.choice(keys,len(keys),replace=True):chunks.append(q[q.pose==key])
    sample=pd.concat(chunks,ignore_index=True)
    k=Model(sample,L,m.degree,True,m.sy,m.st).fit(start=u,maxiter=250)
    origin=k.origin()
    boots.append({'replicate':b,**{f'f{i}':origin[i] for i in range(10)}})
    if (b+1)%20==0:print('BOOTSTRAP',b+1,flush=True)
boot=pd.DataFrame(boots);boot.to_csv(P/'conditional_bootstrap_origins.csv',index=False)
interval=pd.DataFrame({'feature':[f'f{i}' for i in range(10)],'estimate':m.origin(),
        'conditional_lo':np.quantile(boot.iloc[:,1:],.025,axis=0),'conditional_hi':np.quantile(boot.iloc[:,1:],.975,axis=0)})
interval.to_csv(P/'conditional_origin_intervals.csv',index=False)

frame=[]
for dx,dy in [(-.08,0),(.08,0),(0,-.08),(0,.08)]:
    k=Model(d,L,m.degree,True,m.sy,m.st);k.D0=k.D0+[dx,dy]
    k.fit(start=u,maxiter=350)
    frame.append({'frame_dx_perturbation_px':dx,'frame_dy_perturbation_px':dy,
                  **{f'f{i}':k.origin()[i] for i in range(10)}})
pd.DataFrame(frame).to_csv(P/'anchor_frame_sensitivity.csv',index=False)

# Tangent ambiguity at the origin: diagonal scaling/whitening matters for this diagnostic.
_,dy,dt=m.basis(np.zeros((1,2)));J=np.c_[(dy@m.C).ravel(),(dt@m.C).ravel()]
cosine=float(J[:,0]@J[:,1]/np.linalg.norm(J[:,0])/np.linalg.norm(J[:,1]))
save('validation.json',{'analytic_gradient_relative_error':rel,'anchor_mean_bias':m.anchor@m.B,
    'bootstrap_replicates':60,'bootstrap_unit':'same-pose group within scan, reference anchors and model family fixed',
    'bootstrap_is_physical_origin_confidence_interval':False,
    'whitened_Y_T_tangent_cosine_at_origin':cosine,
    'Y_T_tangent_angle_degrees':np.degrees(np.arccos(np.clip(cosine,-1,1))),
    'raw_parameter_bound_hits':int(np.sum(np.abs(u)>2.999)),
    'bound_note':'Bounds apply to gauge-reduced raw parameters; exported bias can exceed three prior sigma after recentering. Boundary contact flags unresolved bias decomposition.',
    'all_optimizer_rounds_successful':all(q['success'] for q in j['optimization'])})

starts=[]
for k in range(5):
    initial=np.clip(u+RNG.normal(0,.1,len(u)),-3,3)
    q=Model(d,L,m.degree,True,m.sy,m.st).fit(start=initial,maxiter=450)
    starts.append({'start':k,'origin_f0':q.origin()[0],'origin_f1':q.origin()[1],
                   'success':all(v['success'] for v in q.optimization)})
pd.DataFrame(starts).to_csv(P/'multistart_stability.csv',index=False)

import cv2
fs=cv2.FileStorage(str(SRC/'cache/reference.yml.gz'),cv2.FILE_STORAGE_READ)
refpoints=fs.getNode('points').mat();fs.release()
cloud=pd.read_csv(P/'standard_origin_630points.csv')
qr=[]
for k in range(3):
    mask=cloud.qr_id.to_numpy()==k
    pos=cloud.loc[mask,['x_standard_px','y_standard_px']].mean().to_numpy()
    baseline=refpoints[refpoints[:,2]==k,:2].mean(0)
    shift=pos-baseline
    assert np.max(np.abs(shift-m.origin()[k*2:k*2+2]))<1e-8
    qr.append({'qr_id':k,'x_standard_px':pos[0],'y_standard_px':pos[1],'dx_from_asset_reference_px':shift[0],'dy_from_asset_reference_px':shift[1]})
pd.DataFrame(qr).to_csv(P/'standard_origin_qr_centers.csv',index=False)

fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
axs[0].scatter(boot.f0,boot.f1,s=20,alpha=.6,label='기준 사진을 고정한 자세 묶음 bootstrap')
axs[0].scatter([m.origin()[0]],[m.origin()[1]],s=180,marker='*',color='red',label='채택 표준 원점')
axs[0].set(xlabel='표준 원점의 좌측 QR dx (px)',ylabel='좌측 QR dy (px)',title='같은 가정 안에서의 표본 변동');axs[0].legend(fontsize=8)
sen=pd.read_csv(P/'prior_and_shape_sensitivity.csv').query('session==3')
for deg,q in sen.groupby('degree'):axs[1].plot(q.sigma_T_deg*100,q.origin_f0,'o-',label=f'{deg}차 공통 반응')
axs[1].set(xscale='log',xlabel='스캔 T 바이어스 사전 표준편차 (tick)',ylabel='표준 원점의 좌측 QR dx (px)',title='모델 가정 변화에 대한 원점 위치 민감도');axs[1].legend()
axs[1].set_xticks([.25,.5,1,2,4],['0.25','0.5','1','2','4']);axs[1].minorticks_off()
fig.savefig(P/'06_origin_stability.png',dpi=160);plt.close(fig)
print('VALIDATED',rel,cosine,flush=True)
