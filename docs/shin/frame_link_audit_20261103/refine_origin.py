from forensics import *
from scipy.optimize._numdiff import approx_derivative

def audited_data(d,separate=True):
    q=d.copy();q['T_original']=q['T']
    typo=q.path.str.contains('Y+0.05 T+0.32/Y+0.05 T+0.33',regex=False)
    q.loc[typo,'T']=.32
    q['pose']=q.Y.round(6).astype(str)+':'+q['T'].round(6).astype(str)
    q['suspect']=False
    q['correction_reason']=np.where(typo,'folder + local trajectory consensus: file .33 is .32','unchanged')
    if separate:
        q.loc[q.path=='center.bmp','group']=16
        q.loc[q.path.str.endswith('center3.bmp'),'group']=17
    return q

def training_repeat_weights(train,f,a):
    r=repeat_vectors(train,f,a,'A_left')
    weights=pd.Series(1.,index=train.row_id.to_numpy())
    if len(r):
        selected=r[r.large_repeat_deviation]
        for row in selected.itertuples():weights.loc[row.row_id]=max(.2,min(1,(row.threshold_px/max(row.common_norm,1e-9))**2))
    return weights.to_dict(),r

class RefinedModel(SceneModel):
    def __init__(self,train,f,a,robust_rows=False,core=False,y5=False):
        true=train.reset_index(drop=True).copy();proxy=true.copy();proxy['group']=np.minimum(proxy.group,15)
        super().__init__(proxy,f,a,True,False,True,'A_left')
        self.train=true;self.sid=true.group.to_numpy(int)-1;self.nsession=17
        self.anchor=np.bincount(self.sid[self.refs],minlength=17)/self.refs.sum()
        self.present=np.bincount(self.sid,minlength=17)>0;self.fixed=np.flatnonzero(self.anchor)[0]
        self.free=np.array([i for i in range(17) if self.present[i] and i!=self.fixed],int)
        self.npar=self.base_npar=2*len(self.free);self.u=np.zeros(self.npar)
        self.core=core;self.robust_rows=robust_rows;self.y5=y5
        if core:
            for i,s in enumerate(self.specs):
                if s['fid'] is not None and f.iloc[s['fid']].family in ['left_panel','right_panel','lower_structure']:
                    self.importance[i]=0
        if robust_rows:
            weightmap,rep=training_repeat_weights(true,f,a)
            rw=np.array([weightmap[int(i)] for i in true.row_id]);self.wq*=rw[None,:]
            self.repeat_diagnostics=rep
        if y5:self.powers.extend([(4,0),(5,0)]);self.reg=np.r_[self.reg,.02,.02]

    def unpack_all(self,u):
        B=np.zeros((17,2));B[self.free]=u.reshape(-1,2)*[self.sy,self.st]
        B-=self.anchor@B;B[~self.present]=0
        return B,self.D0,np.zeros(17)

    def objective_scene(self,u,robust,store=False):
        B,D,S=self.unpack_all(u);z=self.q+B[self.sid];A,Ay,At=self.basis(z)
        weights=self.wq*robust
        H=np.einsum('np,bn,nq->bpq',A,weights,A,optimize=True)+np.diag(self.reg)[None,:,:]
        O=np.sum(self.Xblocks[:,self.refs,:]*self.wq[:,self.refs,None],axis=1)/np.maximum(self.wq[:,self.refs].sum(1)[:,None],1e-9)
        rhs=np.einsum('np,bn,bnd->bpd',A,weights,self.Xblocks,optimize=True);rhs[:,0,:]+=self.reg[0]*O
        C=np.linalg.solve(H,rhs);R=np.einsum('np,bpd->bnd',A,C,optimize=True)-self.Xblocks
        wr=weights[:,:,None]*R*self.importance[:,None,None]
        penalty=np.sum(self.reg[None,1:,None]*C[:,1:,:]**2,axis=(1,2))+self.reg[0]*np.sum((C[:,0,:]-O)**2,axis=1)
        value=np.sum(wr*R)+np.sum(penalty*self.importance)+np.sum((B[self.present]/[self.sy,self.st])**2)
        predy=np.einsum('np,bpd->bnd',Ay,C,optimize=True);predt=np.einsum('np,bpd->bnd',At,C,optimize=True)
        gz=2*np.c_[np.sum(wr*predy,axis=(0,2)),np.sum(wr*predt,axis=(0,2))]
        gb=np.zeros((17,2));np.add.at(gb,self.sid,gz);gb+=2*B/np.array([self.sy**2,self.st**2]);gb-=self.anchor[:,None]*gb.sum(0)
        grad=(gb[self.free]*[self.sy,self.st]).ravel()
        if store:self.Cblocks=C;self.B=B;self.D=D;self.S=S;self.A=A;self.weights=weights;self.residual_blocks=R
        return float(value),grad

def validation(model,test,f,a,bgY5=False,detail=False):
    specs,_=blocks(model.train,f,a,'B_right',True);specs=[s for s in specs if s['fid'] is not None]
    z=model.latent(model.train);zt=model.latent(test);A=model.basis(z)[0];At=model.basis(zt)[0]
    errors=[];norms=[];valids=[];weights=[];records=[]
    for s in specs:
        family=f.iloc[s['fid']].family;X=A;Xt=At;reg=model.reg.copy()
        if bgY5 and not model.y5 and family in ['left_panel','right_panel','lower_structure']:
            X=np.c_[A,(z[:,0]/2)**4,(z[:,0]/2)**5];Xt=np.c_[At,(zt[:,0]/2)**4,(zt[:,0]/2)**5];reg=np.r_[reg,.02,.02]
        Z,valid,_=block_data(s,model.train,a,f);w=valid*s['quality_weight']
        H=X.T@(w[:,None]*X)+np.diag(reg);rhs=X.T@(w[:,None]*Z);rhs[0]+=reg[0]*Z[model.refs&valid].mean(0)
        C=np.linalg.solve(H,rhs);target,vm,_=block_data(s,test,a,f);R=Xt@C-target;raw=R@np.linalg.inv(s['L'])
        errors.append(np.sum(raw**2,axis=1));norms.append(np.sum(R**2,axis=1));valids.append(vm);weights.append(s['weight'])
        if detail:
            for ii,(_,row) in enumerate(test.iterrows()):
                if vm[ii]:records.append({'path':row.path,'feature_id':s['fid'],'family':family,'dx':raw[ii,0],'dy':raw[ii,1],'component_RMS':np.sqrt(np.mean(raw[ii]**2))})
    valids=np.array(valids);W=valids*np.array(weights)[:,None];den=np.maximum(W.sum(0),1e-12)
    score=np.sqrt(np.sum(W*np.array(errors),axis=0)/(2*den));norm=np.sqrt(np.sum(W*np.array(norms),axis=0)/(2*den));count=valids.sum(0)
    score[count<3]=np.nan;norm[count<3]=np.nan
    return score,count,norm,records

CONFIGS=[
 ('audited_time',False,False,False,False,False),
 ('zero_split',True,False,False,False,False),
 ('zero_split_robust',True,True,False,False,False),
 ('core_robust',True,True,True,False,False),
 ('core_robust_bgY5',True,True,True,False,True),
 ('all_robust_Y5',True,True,False,True,False)]

def main():
    original,f,a=load();full=audited_data(original,True)
    cal=calibration_mask(full)
    folds=np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8],16)%5 for k in full.pose])
    blocked=np.zeros(len(full),int)
    for _,q in full.groupby('group'):
        axis='T' if q['T'].nunique()>1 else 'Y';order=q[[axis,'pose']].drop_duplicates().sort_values(axis).pose.tolist()
        mapping={key:min(4,int(i*5/len(order))) for i,key in enumerate(order)};blocked[q.index]=q.pose.map(mapping)
    records=[];posts=[];detail=[];done=set()
    if '--fresh' not in sys.argv and (P/'refinement_CV.csv').exists() and (P/'refinement_zero_holdouts.csv').exists():
        previous=pd.read_csv(P/'refinement_CV.csv');prior_posts=pd.read_csv(P/'refinement_zero_holdouts.csv')
        for name in previous.config.unique():
            q=previous[previous.config==name]
            if q.scheme.nunique()==2 and len(prior_posts[prior_posts.config==name])==3:
                done.add(name);records.extend(q.to_dict('records'));posts.extend(prior_posts[prior_posts.config==name].to_dict('records'))
    for name,separate,rr,core,y5,bg in CONFIGS:
        if name in done:
            print('RESUMED',name,flush=True)
            continue
        data=audited_data(original,separate)
        for scheme,assigned in [('pose',folds),('contiguous',blocked)]:
            for fold in range(5):
                train=data[(assigned!=fold)|cal].copy();test=data[(assigned==fold)&~cal].copy()
                m=RefinedModel(train,f,a,rr,core,y5).fit(maxiter=300)
                qr=m.score_baseline(test);right,count,norm,det=validation(m,test,f,a,bg,True)
                for ii,(_,r) in enumerate(test.iterrows()):records.append({'config':name,'scheme':scheme,'fold':fold,'path':r.path,'pose':r.pose,
                    'session':r.session,'QR_RMS_px':qr[ii],'right_RMS_px':right[ii],'right_features':int(count[ii]),'right_normalized':norm[ii]})
                for r in det:
                    r.update({'config':name,'scheme':scheme,'fold':fold})
                detail.extend(det)
                print('CV',name,scheme,fold,'QR',round(float(np.median(qr)),4),'right',round(float(np.nanmedian(right)),4),flush=True)
        for idx in data.index[(data.cohort=='reference')&(data.date=='2026-10-15')]:
            train=data.drop(index=idx);test=data.loc[[idx]]
            m=RefinedModel(train,f,a,rr,core,y5).fit(maxiter=450)
            qr=m.score_baseline(test)[0];right,count,norm,_=validation(m,test,f,a,bg)
            posts.append({'config':name,'path':test.iloc[0].path,'QR_RMS_px':qr,'right_RMS_px':right[0],'right_normalized':norm[0]})
        pd.DataFrame(records).to_csv(P/'refinement_CV.csv',index=False);pd.DataFrame(posts).to_csv(P/'refinement_zero_holdouts.csv',index=False)
        pd.DataFrame(detail).to_csv(P/'refinement_feature_CV.csv',index=False)
    scores=pd.DataFrame(records).groupby(['config','scheme']).agg(n=('path','size'),QR_RMSE=('QR_RMS_px',lambda x:np.sqrt(np.mean(x*x))),right_RMSE=('right_RMS_px',lambda x:np.sqrt(np.nanmean(x*x))),QR_median=('QR_RMS_px','median'),right_median=('right_RMS_px','median'))
    scores.to_csv(P/'refinement_summary.csv')
    print(scores.to_string());print(pd.DataFrame(posts).to_string(index=False));print('COMPLETE')
if __name__=='__main__':main()
