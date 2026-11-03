from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
P=Path(__file__).resolve().parent;ROOT=P.parents[1];OLD=ROOT/'analysis/session_eda_20261102'
sys.path.insert(0,str(ROOT/'analysis/standard_origin_20261102'))
from fit_origin import Model as OldModel

def save(name,x):
    (P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8')
def load():
    d=pd.read_csv(OLD/'remeasured/observations.csv')
    inv=pd.read_csv(OLD/'image_inventory.csv')
    d=d.merge(inv[['path','session','date']],on='path',validate='one_to_one').copy();d['row_id']=np.arange(len(d))
    d['group']=d.session;d.loc[(d.session==3)&(d['T']<0),'group']=15
    bad=set(pd.read_csv(OLD/'label_anomalies.csv').path);d['suspect']=d.path.isin(bad)
    d['pose']=d.Y.round(6).astype(str)+':'+d['T'].round(6).astype(str)
    features=pd.read_csv(P/'feature_definitions.csv')
    features['family']=np.where(features.reference_y<900,'upper_fixture',np.where(features.reference_y>1600,'lower_structure',np.where(features.reference_x<1400,'left_panel',np.where(features.reference_x>2500,'right_panel','body_fastener'))))
    features['partition']=np.where(features.reference_x<2000,'A_left','B_right')
    features.to_csv(P/'feature_partitions.csv',index=False)
    m=np.load(P/'measurements.npz');q=np.load(P/'quality.npz')
    array={'v':m['values'],'valid':q['valid'],'quality':q['quality'],'line_valid':np.load(P/'line_valid.npy')}
    return d,features,array

def whitening(x,mask,d,extra_cov=None):
    # Empirical repeated-image scatter, with conditional fitting covariance added as a floor.
    chunks=[]
    for _,g in d.groupby('pose'):
        ix=g.index.to_numpy();z=x[ix][mask[ix]]
        if len(z)>1:chunks.append(z-z.mean(0))
    if chunks:
        residual=np.concatenate(chunks);C=residual.T@residual/max(len(residual)-len(chunks),1)
    else:C=np.eye(x.shape[1])*.02**2
    C=.75*C+.25*np.diag(np.diag(C))
    if extra_cov is not None:C+=extra_cov
    ev,U=np.linalg.eigh(C);ev=np.maximum(ev,.01**2)
    return (U/np.sqrt(ev))@U.T

def blocks(train,features,array,partition=None,natural=True):
    # All selection and covariance fitting use TRAINING rows only.
    tr=train.reset_index(drop=True);ri=tr.row_id.to_numpy(int)
    n=len(tr);day=(tr.date.to_numpy()=='2026-10-15')[:,None]
    refs=tr.cohort.to_numpy()=='reference'
    xy=np.c_[tr[['f0','f2','f4']].mean(axis=1),tr[['f1','f3','f5']].mean(axis=1)]
    D=xy[refs&(day[:,0]>0)].mean(0)-xy[refs&(day[:,0]==0)].mean(0)
    out=[]
    def add(name,raw,valid,importance,shift,fid=None,extra=None,quality_weight=None):
        z=raw-day*shift
        L=whitening(z,valid,tr,extra)
        out.append({'name':name,'raw':raw,'registered':z,'mask':valid,'L':L,'weight':importance,'shift':shift,
                    'fid':fid,'quality_weight':np.ones(n) if quality_weight is None else quality_weight})
    add('QR',tr[[f'f{i}' for i in range(6)]].to_numpy(),np.ones(n,bool),1,np.tile(D,3))
    for k in range(2):
        valid=array['line_valid'][ri,k]
        if valid.mean()>=.65:add(f'line{k}',tr[[f'f{6+k*2}',f'f{7+k*2}']].to_numpy(),valid,1,np.array([D[1],0]))
    if natural:
        candidates=[]
        for f in features.itertuples():
            if partition and f.partition!=partition:continue
            valid=array['valid'][ri,f.feature_id]
            if valid.mean()<.85 or (valid&refs).sum()<2:continue
            candidates.append(f)
        counts=pd.Series([f.family for f in candidates]).value_counts()
        for f in candidates:
            fid=f.feature_id;v=array['v'][ri,fid];q=array['quality'][ri,fid];valid=array['valid'][ri,fid]
            C=np.array([[np.nanmedian(q[valid,0]),np.nanmedian(q[valid,1])],[np.nanmedian(q[valid,1]),np.nanmedian(q[valid,2])]])
            sigma=np.maximum(q[:,5],.005);median=np.nanmedian(sigma[valid])
            qw=np.minimum(1,(median/sigma)**2);qw=np.where(valid,qw,0)
            raw=v[:,:2]-np.array([f.reference_x,f.reference_y])
            add(f'natural_{fid}',raw,valid,1/counts[f.family],D,fid,C,qw)
    return out,D

def block_data(spec,d,array,features):
    ri=d.row_id.to_numpy(int);day=(d.date.to_numpy()=='2026-10-15')[:,None]
    if spec['name']=='QR':raw=d[[f'f{i}' for i in range(6)]].to_numpy();valid=np.ones(len(d),bool)
    elif spec['name'].startswith('line'):
        k=int(spec['name'][-1]);raw=d[[f'f{6+k*2}',f'f{7+k*2}']].to_numpy();valid=array['line_valid'][ri,k]
    else:
        fid=spec['fid'];f=features.iloc[fid];raw=array['v'][ri,fid,:2]-np.array([f.reference_x,f.reference_y]);valid=array['valid'][ri,fid]
    z=(raw-day*spec['shift'])@spec['L']
    return np.nan_to_num(z),valid,raw

class SceneModel(OldModel):
    def __init__(self,train,features,array,bias=True,gain=False,natural=True,partition='A_left'):
        train=train.reset_index(drop=True)
        super().__init__(train,np.eye(10),3,bias,.05,.01,True)
        self.train=train;self.features=features;self.array=array
        self.specs,self.D0=blocks(train,features,array,partition,natural)
        self.natural=natural;self.partition=partition;self.gain=gain
        self.base_npar=self.npar;self.npar+=2 if gain else 0
        self.u=np.zeros(self.npar);self.sg=.005
        B=len(self.specs);N=len(train)
        self.Xblocks=np.zeros((B,N,6));self.wq=np.zeros((B,N));self.dims=[];self.importance=[]
        for k,s in enumerate(self.specs):
            z=s['registered']@s['L'];dim=z.shape[1];self.Xblocks[k,:,:dim]=np.nan_to_num(z)
            self.wq[k]=s['mask']*s['quality_weight'];self.dims.append(dim);self.importance.append(s['weight'])
        self.importance=np.array(self.importance)

    def unpack_all(self,u):
        B,D=super().unpack(u[:self.base_npar]);scales=np.zeros(15)
        if self.gain:
            scales[[2,7]]=u[self.base_npar:]*self.sg
            scales[[2,7,14]]-=np.mean(scales[[2,7,14]])
        return B,D,scales

    def latent(self,d,u=None):
        B,D,S=self.unpack_all(self.u if u is None else u);sid=d.group.to_numpy(int)-1
        z=d[['Y','T']].to_numpy()+B[sid];z[:,1]+=d['T'].to_numpy()*S[sid]
        return z

    def objective_scene(self,u,robust,store=False):
        B,D,S=self.unpack_all(u);z=self.q+B[self.sid];z[:,1]+=self.q[:,1]*S[self.sid]
        A,Ay,At=self.basis(z)
        weights=self.wq*robust
        H=np.einsum('np,bn,nq->bpq',A,weights,A,optimize=True)+np.diag(self.reg)[None,:,:]
        O=np.sum(self.Xblocks[:,self.refs,:]*self.wq[:,self.refs,None],axis=1)/np.maximum(self.wq[:,self.refs].sum(1)[:,None],1e-9)
        rhs=np.einsum('np,bn,bnd->bpd',A,weights,self.Xblocks,optimize=True);rhs[:,0,:]+=self.reg[0]*O
        C=np.linalg.solve(H,rhs)
        R=np.einsum('np,bpd->bnd',A,C,optimize=True)-self.Xblocks
        wr=weights[:,:,None]*R*self.importance[:,None,None]
        penalty=np.sum(self.reg[None,1:,None]*C[:,1:,:]**2,axis=(1,2))+self.reg[0]*np.sum((C[:,0,:]-O)**2,axis=1)
        value=np.sum(wr*R)+np.sum(penalty*self.importance)
        grad=np.zeros_like(u)
        predy=np.einsum('np,bpd->bnd',Ay,C,optimize=True);predt=np.einsum('np,bpd->bnd',At,C,optimize=True)
        gz=2*np.c_[np.sum(wr*predy,axis=(0,2)),np.sum(wr*predt,axis=(0,2))]
        if self.bias:
            gb=np.zeros((15,2));np.add.at(gb,self.sid,gz)
            value+=np.sum((B[self.present]/[self.sy,self.st])**2)
            gb+=2*B/np.array([self.sy**2,self.st**2]);gb-=self.anchor[:,None]*gb.sum(0)
            grad[:self.base_npar]=(gb[self.free]*[self.sy,self.st]).ravel()
        if self.gain:
            gs=np.zeros(15);np.add.at(gs,self.sid,gz[:,1]*self.q[:,1])
            value+=np.sum((S/self.sg)**2);gs+=2*S/self.sg**2
            gs[[2,7,14]]-=np.mean(gs[[2,7,14]])
            grad[self.base_npar:]=gs[[2,7]]*self.sg
        if store:self.Cblocks=C;self.B=B;self.D=D;self.S=S;self.A=A;self.weights=weights;self.residual_blocks=R
        return float(value),grad

    def fit(self,start=None,maxiter=250):
        u=np.zeros(self.npar) if start is None else start.copy();robust=np.ones_like(self.wq);logs=[]
        for _ in range(4):
            r=minimize(lambda v:self.objective_scene(v,robust),u,jac=True,method='L-BFGS-B',bounds=[(-3,3)]*self.npar,
                options={'maxiter':maxiter,'ftol':1e-9,'gtol':1e-5,'maxls':35}) if self.npar else None
            if r is not None:u=r.x;logs.append({'success':bool(r.success),'iterations':int(r.nit),'message':str(r.message)})
            self.objective_scene(u,robust,True)
            radius=np.sqrt(np.sum(self.residual_blocks**2,axis=2)/np.array(self.dims)[:,None])
            robust=np.minimum(1,2.5/np.maximum(radius,1e-9))
        self.u=u;self.objective_scene(u,robust,True);self.optimization=logs
        return self

    def score_baseline(self,test):
        A=self.basis(self.latent(test))[0];qr_spec=self.specs[0]
        predicted=(A@self.Cblocks[0,:,:6])@np.linalg.inv(qr_spec['L'])
        target=(test[[f'f{i}' for i in range(6)]].to_numpy()-(test.date.to_numpy()=='2026-10-15')[:,None]*qr_spec['shift'])
        residual=predicted-target
        return np.sqrt(np.mean(residual**2,axis=1))

    def independent_feature_score(self,test):
        # B-right features never enter the origin/bias objective, even on test photographs.
        specs,_=blocks(self.train,self.features,self.array,'B_right',True)
        specs=[s for s in specs if s['fid'] is not None]
        A=self.basis(self.latent(self.train))[0];Atest=self.basis(self.latent(test))[0]
        errors=[];valids=[];weights=[];norms=[]
        for s in specs:
            Z,valid,_=block_data(s,self.train,self.array,self.features)
            w=valid*s['quality_weight'];H=A.T@(w[:,None]*A)+np.diag(self.reg)
            rhs=A.T@(w[:,None]*Z);O=Z[self.refs&valid].mean(0);rhs[0]+=self.reg[0]*O
            C=np.linalg.solve(H,rhs)
            T,vmask,_=block_data(s,test,self.array,self.features)
            R=Atest@C-T;raw=R@np.linalg.inv(s['L'])
            errors.append(np.sum(raw**2,axis=1));norms.append(np.sum(R**2,axis=1));valids.append(vmask);weights.append(s['weight'])
        if not errors:return np.full(len(test),np.nan),np.zeros(len(test),int),np.full(len(test),np.nan)
        errors=np.array(errors);valids=np.array(valids);W=valids*np.array(weights)[:,None]
        denominator=np.maximum(W.sum(0),1e-12)
        counts=valids.sum(0);score=np.sqrt(np.sum(W*errors,axis=0)/(2*denominator));norm=np.sqrt(np.sum(W*np.array(norms),axis=0)/(2*denominator))
        score[counts<3]=np.nan;norm[counts<3]=np.nan
        return score,counts,norm

def calibration_mask(d):
    ids=np.unique(np.r_[d.index[d.cohort=='reference'],d.Y.idxmin(),d.Y.idxmax(),d['T'].idxmin(),d['T'].idxmax(),0,50,100,200,300,400])
    keys=set(d.loc[ids,'pose'])
    return d.pose.isin(keys).to_numpy()

def main():
    full,features,array=load();cal=calibration_mask(full)
    usable=~full.suspect.to_numpy()
    folds=np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8],16)%5 for k in full.pose])
    configs=[('qr_line_bias',True,False,False),('scene_no_bias',False,False,True),('scene_bias',True,False,True),('scene_bias_gain',True,True,True)]
    records=[];summaries=[]
    for name,bias,gain,natural in configs:
        for fold in range(5):
            train=full[usable&((folds!=fold)|cal)].copy();test=full[usable&(folds==fold)&~cal].copy()
            m=SceneModel(train,features,array,bias,gain,natural).fit()
            baseline=m.score_baseline(test);external,count,norm=m.independent_feature_score(test)
            for j,(_,r) in enumerate(test.iterrows()):records.append({'config':name,'fold':fold,'path':r.path,'pose':r.pose,'session':r.session,
                'QR_component_RMS_px':baseline[j],'right_feature_component_RMS_px':external[j],'right_normalized_RMS':norm[j],'valid_right_features':int(count[j])})
            print('VALIDATION',name,fold,'QR',round(float(np.median(baseline)),4),'independent-right',round(float(np.nanmedian(external)),4),'natural A',len(m.specs)-3,flush=True)
        q=pd.DataFrame(records);q=q[q.config==name]
        summaries.append({'config':name,'n_test':len(q),'right_evaluated':int(q.right_feature_component_RMS_px.notna().sum()),
            'QR_RMSE_px':np.sqrt(np.mean(q.QR_component_RMS_px**2)),'QR_median_px':q.QR_component_RMS_px.median(),
            'right_RMSE_px':np.sqrt(np.nanmean(q.right_feature_component_RMS_px**2)),'right_median_px':q.right_feature_component_RMS_px.median(),
            'right_normalized_median':q.right_normalized_RMS.median()})
        pd.DataFrame(records).to_csv(P/'strict_cv_predictions.csv',index=False);pd.DataFrame(summaries).to_csv(P/'strict_cv_summary.csv',index=False)
    # Harder held-out contiguous pose blocks, not just scattered intermediate poses.
    blocked=np.zeros(len(full),int)
    for _,q in full.groupby('group'):
        axis='T' if q['T'].nunique()>1 else 'Y'
        order=q[[axis,'pose']].drop_duplicates().sort_values(axis).pose.tolist()
        mapping={key:min(4,int(i*5/len(order))) for i,key in enumerate(order)}
        blocked[q.index]=q.pose.map(mapping)
    blockrecords=[]
    for name,bias,gain,natural in [configs[0],configs[2],configs[3]]:
        for fold in range(5):
            train=full[usable&((blocked!=fold)|cal)].copy();test=full[usable&(blocked==fold)&~cal].copy()
            k=SceneModel(train,features,array,bias,gain,natural).fit()
            qr=k.score_baseline(test);right,count,norm=k.independent_feature_score(test)
            for j,(_,r) in enumerate(test.iterrows()):blockrecords.append({'config':name,'fold':fold,'path':r.path,'pose':r.pose,
                'QR_component_RMS_px':qr[j],'right_feature_component_RMS_px':right[j],'right_normalized_RMS':norm[j]})
            print('CONTIGUOUS',name,fold,'right',round(float(np.nanmedian(right)),4),flush=True)
    pd.DataFrame(blockrecords).to_csv(P/'contiguous_cv_predictions.csv',index=False)
    # Fit a single provisional standard with constant bias; gain alternative is a hypothesis test.
    d=full[usable].copy();m=SceneModel(d,features,array,True,False,True).fit(maxiter=600)
    probe=np.random.default_rng(507).normal(0,.1,m.npar);rob=np.ones_like(m.wq)
    analytic=m.objective_scene(probe,rob)[1];numeric=approx_derivative(lambda x:m.objective_scene(x,rob)[0],probe,method='3-point').ravel()
    graderr=np.linalg.norm(analytic-numeric)/max(1,np.linalg.norm(numeric));assert graderr<1e-5
    origins=[]
    for i,s in enumerate(m.specs):
        dim=m.dims[i];value=m.Cblocks[i,0,:dim]@np.linalg.inv(s['L'])
        origins.append({'block':s['name'],'feature_id':s['fid'],'origin_values':value.tolist(),'dimension':dim,'L':s['L'].tolist(),'importance':s['weight']})
    qr=origins[0]['origin_values']
    save('standard_origin_v2.json',{'schema':'quality_controlled_scene_origin.v2','status':'provisional_statistical_origin',
        'gauge':'Nominal-zero reference latent-coordinate mean fixed to zero; day image registration is a convention based on QR zero anchors.',
        'origin_QR_features_6':qr,'origin_blocks':origins,'scan_bias_YT':m.B,'image_day_shift_px':m.D,
        'origin_feature_partition':'A_left only; B_right reserved for validation',
        'coefficients_whitened_padded':m.Cblocks,'powers':m.powers,'quality_weights':m.importance,'gradient_relative_error':graderr,
        'optimization':m.optimization,'raw_bias_bound_hits':int(np.sum(np.abs(m.u)>2.999)),
        'selection':'Natural candidates selected on fixed calibration-only pilot; rank/quality selection and covariance recomputed within each training fold.'})
    # Actual QR point-cloud construction under the same row weights and gauge.
    import cv2
    pointrows=[]
    for r in d.itertuples():
        fs=cv2.FileStorage(str(OLD/'remeasured/cache'/f'frame{r.index}.yml.gz'),cv2.FILE_STORAGE_READ)
        pts=fs.getNode('points').mat();fs.release()
        pointrows.append(pts[:,:2]-(m.D if r.date=='2026-10-15' else 0))
    target=np.asarray(pointrows).reshape(len(d),-1);w=m.weights[0]
    H=m.A.T@(w[:,None]*m.A)+np.diag(m.reg);rhs=m.A.T@(w[:,None]*target)
    rhs[0]+=m.reg[0]*(target[m.refs]*m.wq[0,m.refs,None]).sum(0)/m.wq[0,m.refs].sum()
    pc=np.linalg.solve(H,rhs)[0].reshape(630,2)
    pd.DataFrame({'point_id':np.arange(630),'qr_id':pts[:,2].astype(int),'x_standard_px':pc[:,0],'y_standard_px':pc[:,1]}).to_csv(P/'standard_origin_v2_QR630.csv',index=False)
    naturalorigin=[]
    for s in origins:
        if s['feature_id'] is not None:
            f=features.iloc[s['feature_id']]
            naturalorigin.append({'feature_id':int(s['feature_id']),'family':f.family,'x_standard_px':f.reference_x+s['origin_values'][0],
                'y_standard_px':f.reference_y+s['origin_values'][1],'note':'candidate origin-construction feature; not an independent verification point'})
    pd.DataFrame(naturalorigin).to_csv(P/'standard_origin_v2_natural.csv',index=False)
    out=full[['path','Y','T','session','group','suspect']].copy();ids=out.group.to_numpy(int)-1
    out['Y_bias_mm']=m.B[ids,0];out['T_bias_deg']=m.B[ids,1];out['Y_standard']=out.Y+out.Y_bias_mm;out['T_standard']=out['T']+out.T_bias_deg
    out.to_csv(P/'coordinates_standard_v2.csv',index=False)
    # Post-construction reference checks: only Oct-15 zero images can be held out while
    # retaining the Oct-4 image frame definition; no fitted offset for an unseen group.
    post=[]
    for ix in d.index[(d.cohort=='reference')&(d.date=='2026-10-15')]:
        train=d.drop(index=ix);test=d.loc[[ix]]
        k=SceneModel(train,features,array,True,False,True).fit(maxiter=500)
        error=k.score_baseline(test)[0];right,count,norm=k.independent_feature_score(test)
        post.append({'path':test.iloc[0].path,'QR_component_RMS_px':error,'right_component_RMS_px':right[0],'right_normalized_RMS':norm[0],'right_features':int(count[0]),
                     'note':'held-out zero image; training anchors and gauge rebuilt without that image; Oct4 anchor not held out because it defines image frame'})
    pd.DataFrame(post).to_csv(P/'heldout_reference_checks.csv',index=False)
    print('FINAL',qr,'GRADIENT',graderr,flush=True)
if __name__=='__main__':main()
