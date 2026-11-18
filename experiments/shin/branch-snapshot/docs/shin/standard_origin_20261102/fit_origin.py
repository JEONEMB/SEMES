"""One statistical coordinate origin with session-constant label biases.
Sources remain read-only. No claim of physically calibrated ground truth.
"""
from pathlib import Path
import json,itertools,time,copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

P=Path(__file__).resolve().parent;ROOT=P.parents[1]
OLD=ROOT/'analysis/session_eda_20261102'
SRC=OLD/'remeasured'
P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
RNG=np.random.default_rng(507)
def save(name,o): (P/name).write_text(json.dumps(o,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if isinstance(o,np.ndarray) else str(o)),encoding='utf-8')

def noise_whitener(g):
    g=g[g.cohort!='reference'];cols=[f'f{i}' for i in range(10)]
    r=g[cols].to_numpy()-g.groupby('pose')[cols].transform('mean').to_numpy()
    C=r.T@r/(len(g)-g.pose.nunique());C=.75*C+.25*np.diag(np.diag(C))
    eig,V=np.linalg.eigh(C);eig=np.maximum(eig,1e-5)
    return (V/np.sqrt(eig))@V.T

def data():
    x=pd.read_csv(SRC/'observations.csv')
    inv=pd.read_csv(OLD/'image_inventory.csv')
    d=x.merge(inv[['path','session','date','modified_utc']],on='path',validate='one_to_one').copy()
    d['modified_utc']=pd.to_datetime(d.modified_utc,format='ISO8601')
    bad=set(pd.read_csv(OLD/'label_anomalies.csv').path)
    d['suspect']=d.path.isin(bad)
    d['group']=d.session
    d.loc[(d.session==3)&(d['T']<0),'group']=15
    d['pose']=d.Y.round(6).astype(str)+':'+d['T'].round(6).astype(str)
    # Real repeated photographs set the measurement covariance, not 630 independent points.
    g=d[(~d.suspect)&(d.cohort!='reference')]
    cols=[f'f{i}' for i in range(10)]
    r=g[cols].to_numpy()-g.groupby('pose')[cols].transform('mean').to_numpy()
    C=r.T@r/(len(g)-g.pose.nunique());C=.75*C+.25*np.diag(np.diag(C))
    eig,V=np.linalg.eigh(C);eig=np.maximum(eig,1e-5)
    L=(V/np.sqrt(eig))@V.T
    save('measurement_covariance.json',{'covariance':C,'whitener':L,'definition':'pooled observed repeat covariance; 25% diagonal shrinkage; floor 1e-5 px^2'})
    return d,L

class Model:
    def __init__(self,d,L,degree=2,bias=True,sigma_y=.05,sigma_t=.01,image_shift=True):
        self.L=L;self.degree=degree;self.bias=bias;self.image_shift=image_shift
        self.sy=sigma_y;self.st=sigma_t
        self.sid=d.group.to_numpy(int)-1;self.nsession=15
        self.q=d[['Y','T']].to_numpy(float)
        self.X=d[[f'f{i}' for i in range(10)]].to_numpy()@L
        self.day=(d.date.to_numpy()=='2026-10-15').astype(float)
        self.refs=(d.cohort.to_numpy()=='reference')
        assert self.refs.any()
        self.anchor=np.bincount(self.sid[self.refs],minlength=15)/self.refs.sum()
        # Standard coordinate convention: mean latent position of training zero references = (0,0).
        # A missing session is never assigned a fitted correction at prediction time.
        self.present=np.bincount(self.sid,minlength=15)>0
        self.fixed=np.flatnonzero(self.anchor)[0]
        self.free=np.array([s for s in range(15) if self.present[s] and s!=self.fixed],int)
        self.M=np.array([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[0,1],[0,0],[0,1],[0,0]],float).T@L
        self.powers=[(a,b) for k in range(degree+1) for a in range(k,-1,-1) for b in [k-a]]
        self.reg=np.array([4 if a+b==0 else 0 if a+b==1 else .02 for a,b in self.powers])
        reference=d[self.refs]
        xy=np.c_[reference[['f0','f2','f4']].mean(axis=1),reference[['f1','f3','f5']].mean(axis=1)]
        dates=reference.date.to_numpy()
        self.D0=xy[dates=='2026-10-15'].mean(0)-xy[dates=='2026-10-04'].mean(0)
        # Register acquisition-day image frames using only the zero-reference images.
        # Do not let a free translation absorb the desired stage-coordinate origin.
        self.npar=2*len(self.free) if bias else 0
        self.u=np.zeros(self.npar)

    def unpack(self,u):
        B=np.zeros((15,2));off=0
        if self.bias:
            off=2*len(self.free);B[self.free]=u[:off].reshape(-1,2)*[self.sy,self.st]
            B-=self.anchor@B
            B[~self.present]=0
        D=self.D0 if self.image_shift else np.zeros(2)
        return B,D

    def basis(self,z):
        y=z[:,0]/2;t=z[:,1]/2
        A=np.column_stack([y**a*t**b for a,b in self.powers])
        Ay=np.column_stack([a/2*y**(a-1)*t**b if a else np.zeros(len(z)) for a,b in self.powers])
        At=np.column_stack([b/2*y**a*t**(b-1) if b else np.zeros(len(z)) for a,b in self.powers])
        return A,Ay,At

    def objective(self,u,weights,store=False):
        B,D=self.unpack(u);z=self.q+B[self.sid]
        A,Ay,At=self.basis(z);V=self.X-self.day[:,None]*(D@self.M)
        H=A.T@(weights[:,None]*A)+np.diag(self.reg)
        origin_prior=V[self.refs].mean(0)
        rhs=A.T@(weights[:,None]*V);rhs[0]+=self.reg[0]*origin_prior
        C=np.linalg.solve(H,rhs)
        R=A@C-V;wr=weights[:,None]*R
        val=float(np.sum(wr*R)+np.sum(self.reg[1:,None]*C[1:]*C[1:])+self.reg[0]*np.sum((C[0]-origin_prior)**2))
        grad=np.zeros_like(u);off=0
        if self.bias:
            # Envelope theorem: exact gradient after eliminating linear coefficients.
            gz=np.c_[2*np.sum(wr*(Ay@C),axis=1),2*np.sum(wr*(At@C),axis=1)]
            gb=np.zeros((15,2));np.add.at(gb,self.sid,gz)
            prior=B/np.array([self.sy,self.st]);val+=np.sum(prior[self.present]**2)
            gb+=2*B/np.array([self.sy**2,self.st**2])
            gb=gb-self.anchor[:,None]*gb.sum(0)
            off=2*len(self.free)
            grad[:off]=(gb[self.free]*[self.sy,self.st]).ravel()
        if store:self.C=C;self.weights=weights;self.B=B;self.D=D;self.residual=R;self.A=A
        return val,grad

    def fit(self,start=None,maxiter=350,robust=True):
        u=np.zeros(self.npar) if start is None else start.copy()
        weights=np.ones(len(self.q));iterations=[]
        for outer in range(4 if robust else 1):
            result=minimize(lambda v:self.objective(v,weights),u,jac=True,method='L-BFGS-B',
                bounds=[(-3,3)]*self.npar,
                options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-5,'maxls':40}) if self.npar else None
            if result is not None:u=result.x;iterations.append({'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit)})
            self.objective(u,weights,True)
            radius=np.sqrt(np.mean(self.residual**2,axis=1))
            weights=np.minimum(1,2.5/np.maximum(radius,1e-9))
        self.u=u;self.objective(u,weights,True);self.optimization=iterations
        return self

    def predict(self,d):
        q=d[['Y','T']].to_numpy()+self.B[d.group.to_numpy(int)-1]
        A,_,_=self.basis(q)
        return (A@self.C+(d.date.to_numpy()=='2026-10-15')[:,None]*(self.D@self.M))@np.linalg.inv(self.L)

    def origin(self):return self.C[0]@np.linalg.inv(self.L)

def tracks(d):
    rows=[]
    # Test each direction separately: a single long series can contain a return or a branch.
    for sid in [1,2,3,4,5,8,9,10,12,13,14]:
        for branch in (['negative','positive'] if sid==3 else ['all']):
            q=d[(d.session==sid)&(~d.suspect)].copy()
            if branch=='negative':q=q[q['T']<0]
            if branch=='positive':q=q[q['T']>0]
            field='T' if q['T'].nunique()>1 else 'Y';v=q[field].to_numpy()
            if q[field].nunique()<3:continue
            times=(q.modified_utc-q.modified_utc.min()).dt.total_seconds().to_numpy()/60
            A=np.c_[np.ones(len(q)),v]
            for feature in [f'f{i}' for i in range(10)]:
                x=q[feature].to_numpy();c=np.linalg.lstsq(A,x,rcond=None)[0];r=x-A@c
                # Bias consistency is checked through residuals. No causal inference from monotone time.
                ac=stats.spearmanr(times,r).statistic if np.ptp(r)>0 else np.nan
                rows.append({'session':sid,'branch':branch,'feature':feature,'axis':field,'n':len(q),
                    'intercept':c[0],'slope':c[1],'linear_R2':1-np.sum(r*r)/np.sum((x-x.mean())**2) if np.ptp(x)>0 else np.nan,
                    'residual_SD_px':r.std(ddof=2),'residual_95pct_abs_px':np.quantile(np.abs(r),.95),'time_residual_spearman':ac})
    t=pd.DataFrame(rows);t.to_csv(P/'within_session_tracks.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for col,sid in enumerate([3,8]):
        q=d[(d.session==sid)&(~d.suspect)];v=q['T'].to_numpy();x=q.f0.to_numpy()
        c=np.linalg.lstsq(np.c_[np.ones(len(q)),v],x,rcond=None)[0];r=x-np.c_[np.ones(len(q)),v]@c
        axs[0,col].scatter(v,x,s=12);order=np.argsort(v);axs[0,col].plot(v[order],(c[0]+c[1]*v)[order],color='#d77c39')
        axs[0,col].set(xlabel='T 라벨 (°)',ylabel='좌측 QR dx (px)',title=f'S{sid:02}: 미세 틱 궤적과 직선 적합')
        axs[1,col].scatter(v,r,s=12);axs[1,col].axhline(0,color='#777');axs[1,col].set(xlabel='T 라벨 (°)',ylabel='직선 잔차 (px)',title='좁은 띠는 일정 바이어스와 양립하나 이를 증명하지는 않음')
    fig.savefig(P/'01_track_hypothesis.png',dpi=160);plt.close(fig)
    return t

def eval_cv(d,L):
    # Standard-anchor images are calibration anchors and never counted as held-out test data.
    # Exact coordinate repeats stay together, independent of image numbering.
    from hashlib import sha256
    folds=np.array([int(sha256(k.encode()).hexdigest()[:8],16)%5 for k in d.pose])
    refs=d.cohort.to_numpy()=='reference'
    configs=[('no_bias_deg2',2,False,.05,.01),('no_bias_deg3',3,False,.05,.01),
             ('bias_deg2_t005',2,True,.025,.005),('bias_deg2_t010',2,True,.05,.01),
             ('bias_deg3_t005',3,True,.025,.005),('bias_deg3_t010',3,True,.05,.01)]
    results=[];preds=[]
    for name,deg,bias,sy,st in configs:
        for fold in range(5):
            train=d[(folds!=fold)|refs].copy();test=d[(folds==fold)&~refs].copy()
            local_L=noise_whitener(train)
            m=Model(train,local_L,deg,bias,sy,st).fit(maxiter=220)
            predicted=m.predict(test);target=test[[f'f{i}' for i in range(10)]].to_numpy()
            r=predicted-target;z=r@local_L;rad=np.sqrt(np.mean(z*z,axis=1))
            for ix,(_,row) in enumerate(test.iterrows()):preds.append({'config':name,'fold':fold,'path':row.path,'session':row.session,
                'whitened_RMS':rad[ix],'QR_RMS_px':np.sqrt(np.mean(r[ix,:6]**2)),'line_RMS_px':np.sqrt(np.mean(r[ix,6:]**2))})
            print('CV',name,fold,'median normalized error',round(float(np.median(rad)),3),flush=True)
        q=pd.DataFrame(preds);q=q[q.config==name]
        results.append({'config':name,'degree':deg,'bias':bias,'sigma_Y':sy,'sigma_T':st,'n_test':len(q),
            'median_normalized_RMS':q.whitened_RMS.median(),'mean_normalized_RMS':q.whitened_RMS.mean(),
            'p95_normalized_RMS':q.whitened_RMS.quantile(.95),'QR_RMSE_px':np.sqrt(np.mean(q.QR_RMS_px**2)),
            'line_RMSE_px':np.sqrt(np.mean(q.line_RMS_px**2))})
        pd.DataFrame(results).to_csv(P/'cv_summary.csv',index=False);pd.DataFrame(preds).to_csv(P/'cv_predictions.csv',index=False)
    return pd.DataFrame(results)

def export_origin(m,d,full,L,cv):
    cols=[f'f{i}' for i in range(10)]
    save('standard_origin.json',{'schema':'statistical_standard_origin.v1','status':'provisional_statistical_reference',
        'session_biases_status':'model-dependent; inspect prior sensitivity and boundary contact before treating as corrected truth',
        'coordinate_origin_Y_mm':0,'coordinate_origin_T_deg':0,
        'gauge':'Mean latent Y/T of the four nominal-zero reference images equals zero; image coordinates anchored to Oct-4 frame.',
        'origin_features_10':m.origin(),'shared_polynomial_powers':m.powers,'shared_coefficients_whitened':m.C,'whitener':L,
        'scan_group_bias_YT':m.B,'scan_group_definition':'1-14 original time segments; group3 positive T sweep, group15 negative T sweep of original session3',
        'oct15_common_image_translation_xy_px':m.D,'image_shift_reference_prior_xy_px':m.D0,
        'sigma_session_label_Y_mm':m.sy,'sigma_session_label_T_deg':m.st,'degree':m.degree,
        'origin_is_physical_ground_truth':False,'optimization':m.optimization,
        'evaluation':'5-fold held-coordinate FORWARD-feature validation; calibration anchors always retained. Not T truth accuracy; not future-session validation.'})
    corrected=full[['path','cohort','Y','T','session','group','date','suspect']].copy()
    corrected['estimated_session_Y_bias_mm']=m.B[full.group.to_numpy(int)-1,0]
    corrected['estimated_session_T_bias_deg']=m.B[full.group.to_numpy(int)-1,1]
    corrected['estimated_session_T_bias_ticks']=corrected.estimated_session_T_bias_deg*100
    corrected['estimated_Y']=corrected.Y+corrected.estimated_session_Y_bias_mm
    corrected['estimated_T']=corrected['T']+corrected.estimated_session_T_bias_deg
    corrected['used_for_fit']=~corrected.suspect
    corrected.to_csv(P/'coordinates_in_standard_frame.csv',index=False)
    biases=pd.DataFrame({'group':np.arange(1,16),'Y_bias_mm':m.B[:,0],'T_bias_deg':m.B[:,1],'T_bias_ticks':m.B[:,1]*100})
    biases.to_csv(P/'session_biases.csv',index=False)
    # Build one virtual 630-point reference, not a blurred average raster image.
    import cv2
    points=[];edges=[]
    for row in d.itertuples():
        fs=cv2.FileStorage(str(SRC/'cache'/f'frame{row.index}.yml.gz'),cv2.FILE_STORAGE_READ)
        p=fs.getNode('points').mat();e=fs.getNode('edges').mat();fs.release()
        day=row.date=='2026-10-15'
        points.append(p[:,:2]-(m.D if day else 0))
        edges.append(e[:,1]-(m.D[1] if day else 0))
    target=np.asarray(points).reshape(len(d),-1)
    H=m.A.T@(m.weights[:,None]*m.A)+np.diag(m.reg)
    rhs=m.A.T@(m.weights[:,None]*target);rhs[0]+=m.reg[0]*target[m.refs].mean(0)
    C=np.linalg.solve(H,rhs);origin_points=C[0].reshape(630,2)
    ed=np.asarray(edges);rhs=m.A.T@(m.weights[:,None]*ed);rhs[0]+=m.reg[0]*ed[m.refs].mean(0)
    Cedge=np.linalg.solve(H,rhs)
    pd.DataFrame({'point_id':np.arange(630),'qr_id':p[:,2].astype(int),'x_standard_px':origin_points[:,0],'y_standard_px':origin_points[:,1]}).to_csv(P/'standard_origin_630points.csv',index=False)
    pd.DataFrame({'sample_id':np.arange(120),'x_reference_px':e[:,0],'y_standard_px':Cedge[0]}).to_csv(P/'standard_origin_120edges.csv',index=False)
    # Prior sensitivity measures model dependence. It is not a frequentist confidence interval.
    sensitivity=[]
    for st in [.0025,.005,.01,.02,.04]:
        for deg in [2,3]:
            q=Model(d,L,deg,True,st*5,st).fit(maxiter=350)
            for s in range(15):sensitivity.append({'sigma_T_deg':st,'degree':deg,'session':s+1,'Y_bias_mm':q.B[s,0],'T_bias_ticks':q.B[s,1]*100,
                    'origin_f0':q.origin()[0],'origin_f1':q.origin()[1]})
            print('SENSITIVITY',deg,st,flush=True)
    pd.DataFrame(sensitivity).to_csv(P/'prior_and_shape_sensitivity.csv',index=False)
    pred=m.predict(d);r=pred-d[cols].to_numpy()
    residual=d[['path','session','cohort','Y','T']].copy()
    residual['normalized_RMS']=np.sqrt(np.mean((r@L)**2,axis=1));residual['robust_weight']=m.weights
    residual.to_csv(P/'fit_residuals.csv',index=False)
    save('fit_summary.json',{'fit_images':len(d),'source_images':len(full),'suspect_images_not_fit':int(full.suspect.sum()),
        'origin_features':m.origin(),'image_translation':m.D,'anchor_mean_bias':m.anchor@m.B,
        'hypothesis':'session label bias is constant; common smooth forward response + day common image translation',
        'uncertainty_note':'Label bias and image shift remain conditional on this model and regularizers.'})
    figures(m,d,cv,pd.DataFrame(sensitivity),origin_points)

def figures(m,d,cv,sens,points):
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,feature,label in zip(axs,['f0','f7'],['좌측 QR dx (px)','상부 선 기울기 특징 (px)']):
        for gid,title in [(3,'S03 양의 T'),(15,'S03 음의 T'),(8,'S08 미세 T')]:
            q=d[d.group==gid];z=q[['Y','T']].to_numpy()+m.B[gid-1]
            zeroY=z.copy();zeroY[:,0]=0
            A=m.basis(z)[0];A0=m.basis(zeroY)[0]
            y_effect=((A-A0)@m.C)@np.linalg.inv(m.L)
            day=(q.date.to_numpy()=='2026-10-15')[:,None]
            values=q[[f'f{i}' for i in range(10)]].to_numpy()-day*(m.D@m.M)@np.linalg.inv(m.L)-y_effect
            ax.scatter(z[:,1],values[:,int(feature[1:])],s=13,label=title)
        ts=np.linspace(-1.05,1.08,150);line=(m.basis(np.c_[np.zeros(len(ts)),ts])[0]@m.C)@np.linalg.inv(m.L)
        ax.plot(ts,line[:,int(feature[1:])],color='#777',linewidth=1,label='공통 반응의 Y=0 단면')
        ax.scatter([0],[m.origin()[int(feature[1:])]],marker='*',s=220,color='red',label='표준 원점의 특징값')
        ax.set(xlabel='모델의 표준 T (°)',ylabel=label,title='모델로 Y 영향·날짜 영상 이동을 보정한 표현');ax.legend(fontsize=8)
    fig.savefig(P/'02_standard_origin_tracks.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for deg,style in [(2,'-'),(3,'--')]:
        q=sens[(sens.sigma_T_deg==.01)&(sens.degree==deg)]
        axs[0].plot(q.session,q.T_bias_ticks,'o'+style,label=f'{deg}차 모델')
    axs[0].axhline(0,color='#777');axs[0].set(xlabel='수정 시각 세션 후보',ylabel='라벨에 더하는 T 바이어스 (tick)',title='표준 좌표계의 세션별 보정량');axs[0].legend()
    for sid in [3,8,14]:
        q=sens[(sens.session==sid)&(sens.degree==m.degree)]
        axs[1].plot(q.sigma_T_deg*100,q.T_bias_ticks,'o-',label=f'S{sid:02}')
    axs[1].set(xscale='log',xlabel='바이어스 사전 표준편차 (tick)',ylabel='추정 T 바이어스 (tick)',title='사전 허용폭에 따른 추정 민감도');axs[1].legend()
    axs[1].set_xticks([.25,.5,1,2,4],['0.25','0.5','1','2','4']);axs[1].minorticks_off()
    fig.savefig(P/'03_bias_sensitivity.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    pos=np.arange(len(cv));ax.bar(pos,cv.median_normalized_RMS,color=['#8398a7' if not v else '#3c8d9b' for v in cv.bias])
    ax.set(xticks=pos,xticklabels=cv.config, ylabel='보류 사진 특징의 정규화 RMS 중앙값',title='원점 보정의 보류 자세 검증 · T 참값 오차가 아님');ax.tick_params(axis='x',rotation=20)
    fig.savefig(P/'04_forward_validation.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,4),layout='constrained')
    for q in range(3):
        a=points[q*210:(q+1)*210];ax.scatter(a[:,0],a[:,1],s=8,label=f'QR {q}')
    ax.set_aspect('equal');ax.invert_yaxis();ax.set(xlabel='기준 영상 x (px)',ylabel='기준 영상 y (px)',title='구성한 단일 가상 표준 원점: QR 630점 좌표');ax.legend()
    fig.savefig(P/'05_virtual_reference.png',dpi=160);plt.close(fig)

def main():
    full,L=data();tracks(full)
    d=full[~full.suspect].copy()
    cv=eval_cv(d,L)
    # Choose among session-bias models by held-out geometric prediction, with prior strength visible.
    best=cv[cv.bias].sort_values(['median_normalized_RMS','p95_normalized_RMS']).iloc[0]
    print('SELECTED',best.to_dict(),flush=True)
    m=Model(d,L,int(best.degree),True,float(best.sigma_Y),float(best.sigma_T)).fit(maxiter=700)
    export_origin(m,d,full,L,cv)
    print('DONE',m.origin(),flush=True)

if __name__=='__main__':main()
