"""Independent train-only model comparison. Existing displacement caches are inputs.

No corrected pseudo ground truth is used. Outer test sets never select a model.
All arrays use image rows; whitening therefore multiplies eigenvector columns.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def load_data(hp=8):
    z = np.load(ROOT / 'analysis_v6' / f'tracks_qr_hp{hp}.npz')
    F = pd.read_csv(ROOT / 'analysis_v6' / f'features_qr_hp{hp}.csv')
    names = z['names'].astype(str)
    isc = names == 'center.bmp'
    axes = np.array([n[0] if n != 'center.bmp' else 'C' for n in names])
    values = np.array([float(n[1:-4]) if n != 'center.bmp' else 0 for n in names])
    sessions = np.array([('T' if a == 'T' else ('Y2' if abs(round(v/.05)) % 2 else 'Y1')) + ('+' if v > 0 else '-') for a,v in zip(axes,values)])
    sessions[isc] = 'C'
    D = z['D_gn'] - z['D_gn'][isc]
    return dict(D=D[~isc],quality=z['score'][~isc],F=F,names=names[~isc],axes=axes[~isc],values=values[~isc],sessions=sessions[~isc])

def design(v, degree, anchor):
    return np.column_stack([v**k for k in range(1 if anchor else 0, degree+1)])

def spatial_basis(F, degree):
    x=(F.x.to_numpy()-1900)/400
    y=(F.y.to_numpy()-1070)/120
    return np.column_stack([x**i*y**(k-i) for k in range(degree+1) for i in range(k,-1,-1)])

def fit_maps(X, axes, values, degree, anchor):
    co={}; residual=[]
    for a in ['Y','T']:
        m=axes==a; A=design(values[m],degree,anchor)
        b=np.linalg.lstsq(A,X[m],rcond=None)[0]
        if anchor: b=np.vstack([np.zeros((1,X.shape[1])),b])
        co[a]=b
        residual.append(X[m]-design(values[m],degree,False)@b)
    return co,np.vstack(residual)

class MotionModel:
    def __init__(self, config):
        self.config=config

    def fit(self,D,F,quality,axes,values):
        c=self.config
        subset=c.get('subset','all')
        if subset=='screws': keep=F.region.str.startswith('QR').to_numpy() | (F.region.to_numpy()=='판나사')
        else:
            regions={'all':['QR좌','QR중','QR우'],'L':['QR좌'],'C':['QR중'],'R':['QR우'],'LR':['QR좌','QR우']}
            keep=F.region.isin(regions[subset]).to_numpy()
        self.indices=np.flatnonzero(keep)
        k=c.get('topk',0)
        if k and k<len(self.indices):
            if c.get('selection')=='strength':
                info=F.iloc[self.indices]['strength'].to_numpy()
            else:
                Z=D[:,self.indices].reshape(len(D),-1)
                co,R=fit_maps(Z,axes,values,2,False)
                ratio=np.median(co['Y'][1,::2])/np.median(co['T'][1,::2])
                contrast=co['Y'][1]-ratio*co['T'][1]
                info=(contrast**2/np.maximum(R.var(0),.003**2)).reshape(-1,2).sum(1)
            self.indices=self.indices[np.argsort(info)[-k:]]
        self.F=F.iloc[self.indices].reset_index(drop=True)
        q=np.median(quality[:,self.indices],axis=0)
        self.point_weight=np.maximum(q, .01)**20 if c.get('quality') else np.ones(len(self.F))
        xy=(self.F[['x','y']].to_numpy()-[1900,1070])/400
        nuisance=[]
        if c.get('nuisance','none') in ('dy','both'):
            nuisance.append(np.tile([0.,1.],len(self.F)))
        if c.get('nuisance','none') in ('rot','both'):
            nuisance.append(np.column_stack([-xy[:,1],xy[:,0]]).ravel())
        self.nuisance=np.column_stack(nuisance) if nuisance else None
        if self.nuisance is not None:
            self.nuisance_q=np.linalg.qr(self.nuisance)[0]
        self.kind=c.get('kind','p12')
        if self.kind.startswith('p'):
            degree={'p6':1,'p12':2,'p20':3}[self.kind]
            self.B=spatial_basis(self.F,degree)
            self.Binv=np.linalg.pinv(self.B*self.point_weight[:,None]**.5)*self.point_weight[None,:]**.5
        elif self.kind=='h8':
            self.xy=(self.F[['x','y']].to_numpy()-[1900,1070])/400
        elif self.kind=='qr18':
            self.region_indices=[np.flatnonzero(self.F.region.to_numpy()==r) for r in ['QR좌','QR중','QR우'] if (self.F.region==r).any()]
            self.region_B=[spatial_basis(self.F.iloc[i],1) for i in self.region_indices]
        X=self.transform(D)
        self.co,R=fit_maps(X,axes,values,c.get('degree',2),c.get('anchor',False))
        sd=np.maximum(R.std(0),1e-4)
        self.sd=sd
        shrink=c.get('shrink',1.)
        if shrink<1 and X.shape[1]<100:
            cov=np.cov((R/sd).T)
            cov=(1-shrink)*cov+shrink*np.eye(len(sd))
            eig,vec=np.linalg.eigh(cov)
            self.W=(vec/np.sqrt(np.maximum(eig,1e-5))[None,:])/sd[:,None]
        else: self.W=np.diag(1/sd)
        self.cow={a:b@self.W for a,b in self.co.items()}
        self.classifier=None
        if c.get('classifier'):
            scale=np.std(X,axis=0)+1e-8
            self.class_scale=scale
            self.classifier=SVC(C=c.get('C',1),kernel=c['classifier'],class_weight='balanced')
            self.classifier.fit(X/scale,axes)
        return self

    def transform(self,D):
        D=D[:,self.indices].copy(); n=len(D)
        if self.nuisance is not None:
            Z=D.reshape(n,-1); Z=Z-(Z@self.nuisance_q)@self.nuisance_q.T; D=Z.reshape(D.shape)
        if self.kind=='raw': return D.reshape(n,-1)*np.repeat(self.point_weight**.5,2)
        if self.kind.startswith('p'):
            co=np.einsum('kp,npd->nkd',self.Binv,D)
            if self.config.get('robust'):
                for j in range(n):
                    b=co[j]
                    for _ in range(4):
                        r=np.linalg.norm(D[j]-self.B@b,axis=1)
                        scale=max(np.median(r)*1.4826,.005)
                        w=np.minimum(1.,1.5*scale/np.maximum(r,1e-9))*self.point_weight
                        b=np.linalg.lstsq(self.B*np.sqrt(w[:,None]),D[j]*np.sqrt(w[:,None]),rcond=None)[0]
                    co[j]=b
            return co.reshape(n,-1)
        if self.kind=='qr18':
            return np.concatenate([np.einsum('kp,npd->nkd',np.linalg.pinv(b),D[:,i]).reshape(n,-1) for i,b in zip(self.region_indices,self.region_B)],axis=1)
        if self.kind=='h8':
            out=[]; x,y=self.xy.T; z=np.zeros(len(x)); one=np.ones(len(x))
            for d in D:
                u,v=(self.xy+d/400).T
                A=np.vstack([np.column_stack([x,y,one,z,z,z,-u*x,-u*y]),np.column_stack([z,z,z,x,y,one,-v*x,-v*y])])
                h=np.linalg.lstsq(A,np.r_[u,v],rcond=None)[0]-[1,0,0,0,1,0,0,0]
                out.append(h)
            return np.asarray(out)
        raise ValueError(self.kind)

    def predict(self,D):
        X=self.transform(D); Xw=X@self.W; results={}
        for a in ['Y','T']:
            b=self.cow[a]; z=Xw-b[0]; t=z@b[1]/np.dot(b[1],b[1])
            if len(b)>2:
                for _ in range(12):
                    pred=b[0]+t[:,None]*b[1]+t[:,None]**2*b[2]
                    der=b[1]+2*t[:,None]*b[2]
                    delta=np.sum((Xw-pred)*der,axis=1)/np.maximum(np.sum(der**2,axis=1),1e-12)
                    t=np.clip(t+delta,-3 if a=='Y' else -1.5,3 if a=='Y' else 1.5)
            pred=design(t,len(b)-1,False)@b
            cost=np.sum((Xw-pred)**2,axis=1)
            if self.config.get('grid'):
                step=.05 if a=='Y' else .01
                ts=step*(np.rint(t[:,None]/step)+np.array([-1,0,1]))
                ps=np.stack([design(ts[:,j],len(b)-1,False)@b for j in range(3)],axis=1)
                costs=np.sum((Xw[:,None,:]-ps)**2,axis=2)
                which=costs.argmin(1); t=ts[np.arange(len(ts)),which]; cost=costs[np.arange(len(ts)),which]
            results[a]=(t,cost)
        axis=np.where(results['Y'][1]<results['T'][1],'Y','T')
        if self.classifier is not None: axis=self.classifier.predict(X/self.class_scale)
        value=np.where(axis=='Y',results['Y'][0],results['T'][0])
        return dict(pred_axis=axis,pred_value=value,cost_Y=results['Y'][1],cost_T=results['T'][1],hyp_Y=results['Y'][0],hyp_T=results['T'][0])

def records(data, test, prediction):
    frame=pd.DataFrame({k:data[k][test] for k in ['names','axes','values','sessions']})
    for k,v in prediction.items(): frame[k]=v
    frame['axis_correct']=frame['axes']==frame.pred_axis
    frame['sign_correct']=np.sign(frame['values'])==np.sign(frame.pred_value)
    frame['joint_correct']=frame.axis_correct & frame.sign_correct
    frame['true_Y']=np.where(frame['axes']=='Y',frame['values'],0.)
    frame['true_T']=np.where(frame['axes']=='T',frame['values'],0.)
    frame['pred_Y']=np.where(frame.pred_axis=='Y',frame.pred_value,0.)
    frame['pred_T']=np.where(frame.pred_axis=='T',frame.pred_value,0.)
    frame['abs_error_Y']=abs(frame.pred_Y-frame.true_Y)
    frame['abs_error_T']=abs(frame.pred_T-frame.true_T)
    frame['selected_abs_error']=abs(frame.pred_value-frame['values'])
    frame['strict_pass']=frame.joint_correct & (frame.abs_error_Y<=.09) & (frame.abs_error_T<=.09)
    return frame

def metrics(frame):
    out=dict(n=len(frame),axis_accuracy=frame.axis_correct.mean(),joint_accuracy=frame.joint_correct.mean(),strict_pass_rate=frame.strict_pass.mean(),axis_errors=int((~frame.axis_correct).sum()),joint_errors=int((~frame.joint_correct).sum()),max_vector_error_Y=frame.abs_error_Y.max(),max_vector_error_T=frame.abs_error_T.max(),strict_all_pass=bool(frame.strict_pass.all()))
    for a in ['Y','T']:
        f=frame[frame['axes']==a]; g=f[f.axis_correct]
        out.update({f'{a}_n':len(f),f'{a}_axis_accuracy':f.axis_correct.mean(),f'{a}_joint_accuracy':f.joint_correct.mean(),f'{a}_strict_pass_rate':f.strict_pass.mean(),f'{a}_matched_max_error':g.selected_abs_error.max(),f'{a}_matched_mae':g.selected_abs_error.mean(),f'{a}_matched_rmse':np.sqrt((g.selected_abs_error**2).mean())})
    return out

def objective(frame):
    # Predeclared lexicographic selection: axis+sign errors, tolerance failures,
    # then worst normalized vector error and total normalized squared error.
    return (int((~frame.joint_correct).sum()),int((~frame.strict_pass).sum()),float(np.maximum(frame.abs_error_Y,frame.abs_error_T).max()),float((frame.abs_error_Y**2+frame.abs_error_T**2).mean()))

def split_folds(data,indices,n=3,seed=20261014):
    sk=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
    return [(indices[tr],indices[te]) for tr,te in sk.split(indices,data['sessions'][indices])]

def evaluate_config(data,c,folds):
    rec=[]
    for fold,(tr,te) in enumerate(folds):
        m=MotionModel(c).fit(data['D'][tr],data['F'],data['quality'][tr],data['axes'][tr],data['values'][tr])
        f=records(data,te,m.predict(data['D'][te])); f['fold']=fold; rec.append(f)
    return pd.concat(rec,ignore_index=True)

def configs():
    out=[]
    def add(**c):
        base=dict(kind='p12',degree=2,shrink=.1,anchor=False,subset='all',nuisance='none')
        base.update(c); base['id']='c%03d'%len(out); out.append(base)
    for kind in ['raw','p6','p12','p20','h8','qr18']:
        for degree in [1,2]:
            for shrink in ([1.] if kind=='raw' else [1.,.1,.001]): add(kind=kind,degree=degree,shrink=shrink)
    for subset in ['L','C','R','LR']:
        for degree in [1,2]: add(subset=subset,degree=degree)
    for nuisance in ['dy','rot','both']:
        for degree in [1,2]: add(nuisance=nuisance,degree=degree)
    for k in [10,25,50,100]:
        for kind in ['raw','p12']: add(kind=kind,topk=k,shrink=1. if kind=='raw' else .1)
    for kind in ['raw','p6','p12','p20']:
        add(kind=kind,quality=True,shrink=1. if kind=='raw' else .1)
    for kind in ['p6','p12','p20']: add(kind=kind,robust=True)
    for degree in [1,2]:
        for shrink in [.1,1.]: add(degree=degree,shrink=shrink,anchor=True)
    for clf in ['linear','rbf']:
        for C in [.1,1.,10.]: add(classifier=clf,C=C)
    for kind in ['raw','p12','p20']:
        for degree in [1,2]: add(kind=kind,degree=degree,subset='screws',shrink=1. if kind=='raw' else .1)
    return out

def main():
    data=load_data(); cs=configs(); HERE.mkdir(exist_ok=True)
    (HERE/'candidate_configs.json').write_text(json.dumps(cs,indent=2),encoding='utf-8')
    indices=np.arange(len(data['D'])); broad=[]; predf=[]
    folds=split_folds(data,indices)
    start=time.time()
    for c in cs:
        f=evaluate_config(data,c,folds); f['config']=c['id']; predf.append(f)
        row=dict(config=c['id'],**{k:v for k,v in c.items() if k!='id'},**metrics(f)); broad.append(row)
        print('broad',c['id'],c['kind'],c.get('subset'),row['axis_errors'],row['Y_matched_max_error'],row['T_matched_max_error'],'sec',round(time.time()-start),flush=True)
    pd.DataFrame(broad).to_csv(HERE/'broad_metrics.csv',index=False,encoding='utf-8-sig')
    pd.concat(predf,ignore_index=True).to_csv(HERE/'broad_predictions.csv',index=False,encoding='utf-8-sig')
    # A broad search is descriptive only. Each outer fold independently searches
    # exactly the fixed full candidate list on inner folds.
    for protocol in ['nested_stratified','nested_leave_session_out']:
        if protocol=='nested_stratified': outer=folds
        else: outer=[(indices[data['sessions']!=s],indices[data['sessions']==s]) for s in np.unique(data['sessions'])]
        outer_records=[]; selections=[]; inner_rows=[]
        for ofold,(tr,te) in enumerate(outer):
            inner=split_folds(data,tr,n=3,seed=20261015+ofold)
            best=None
            for c in cs:
                fi=evaluate_config(data,c,inner); obj=objective(fi)
                inner_rows.append(dict(outer_fold=ofold,config=c['id'],**metrics(fi)))
                if best is None or obj<best[0]: best=(obj,c)
            c=best[1]
            m=MotionModel(c).fit(data['D'][tr],data['F'],data['quality'][tr],data['axes'][tr],data['values'][tr])
            fo=records(data,te,m.predict(data['D'][te])); fo['outer_fold']=ofold; fo['config']=c['id']; outer_records.append(fo)
            selections.append(dict(outer_fold=ofold,test_sessions=','.join(np.unique(data['sessions'][te])),config=c['id'],inner_objective=list(best[0]),**metrics(fo)))
            pd.concat(outer_records,ignore_index=True).to_csv(HERE/f'{protocol}_predictions.csv',index=False,encoding='utf-8-sig')
            print(protocol,ofold,c['id'],'test',len(te),'errors',int((~fo.axis_correct).sum()),'sec',round(time.time()-start),flush=True)
        pd.DataFrame(selections).to_csv(HERE/f'{protocol}_selections.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(inner_rows).to_csv(HERE/f'{protocol}_inner_metrics.csv',index=False,encoding='utf-8-sig')
        (HERE/f'{protocol}_metrics.json').write_text(json.dumps(metrics(pd.concat(outer_records)),indent=2),encoding='utf-8')
    (HERE/'metadata.json').write_text(json.dumps(dict(seconds=time.time()-start,inputs=['analysis_v6/features_qr_hp8.csv','analysis_v6/tracks_qr_hp8.npz'],n=len(indices),points_by_region=data['F'].region.value_counts().to_dict(),label_basis='raw filename command labels',selection_objective='axis+sign errors then tolerance failures then worst vector error then mean vector squared error',seed=20261014,warning='Session identifiers reconstructed from filename sign/spacing; six sessions are not independent repeat sessions for each axis/sign.'),indent=2),encoding='utf-8')

if __name__=='__main__': main()

