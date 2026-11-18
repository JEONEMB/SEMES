"""Independent zero-offset audit. No prior executable modules are imported.

Raw filename commands are the only available ground truth. Image-derived
center-relative labels are sensitivity targets, never measured actual position.
All nuisance/offset parameters are refitted using each outer training split.
"""
from pathlib import Path
import json, os, hashlib
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
OUT = HERE / 'zero'
ROOT = Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')

def manifest(names):
    rows=[]
    for n in names:
        a='C' if n=='center.bmp' else n[0]
        v=0. if a=='C' else float(n[1:-4])
        s='C' if a=='C' else ('T' if a=='T' else ('Y2' if abs(round(v/.05))%2 else 'Y1'))+('+' if v>0 else '-')
        p=ROOT / ('center.bmp' if a=='C' else ('Theta/' if a=='T' else 'Y/')+n)
        rows.append(dict(file=n,axis=a,value=v,session=s,mtime=p.stat().st_mtime,size=p.stat().st_size))
    f=pd.DataFrame(rows)
    f['minutes']=(f.mtime-f.mtime.min())/60
    return f

def load(measurement=None):
    direct=bool(measurement and measurement.startswith('direct'))
    if direct:
        z=np.load(HERE/'direct_alignment'/f'{measurement}.npz')
        f=np.load(HERE/'measurement'/'features.npz'); pts=f['points'][f['regions']<3].astype(float)
        x=(pts[:,0]-1906)/486; y=(pts[:,1]-1070)/122
        B=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
        d=np.einsum('pk,nkd->npd',B,z['X'].reshape(-1,6,2))
    elif measurement:
        z=np.load(HERE/'measurement'/f'{measurement}.npz')
        qr=z['regions']<3; pts=z['points'][qr].astype(float)
        d=z['D'][:,qr].astype(float)
        assert z['valid'][:,qr].all(), 'Use a per-image masked fit for invalid tracks.'
    else:
        z=np.load(PROJ/'analysis_v6/tracks_qr_hp8.npz')
        f=pd.read_csv(PROJ/'analysis_v6/features_qr_hp8.csv')
        qr=f.region.str.startswith('QR').to_numpy()
        pts=f.loc[qr,['x','y']].to_numpy(float)
        d=z['D_gn'][:,qr].astype(float)
    names=z['names'].astype(str)
    d-=d[names=='center.bmp']
    x=(pts[:,0]-1906)/486; y=(pts[:,1]-1070)/122
    B=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
    X=np.linalg.lstsq(B,d.transpose(1,0,2).reshape(len(pts),-1),rcond=None)[0].reshape(6,len(d),2).transpose(1,0,2).reshape(len(d),12)
    N=np.stack([np.c_[x*0,x*0+1], np.c_[-(pts[:,1]-1070)/100,(pts[:,0]-1906)/100]])
    U=np.linalg.lstsq(B,N.transpose(1,0,2).reshape(len(pts),-1),rcond=None)[0].reshape(6,2,2).transpose(1,0,2).reshape(2,12).T
    return manifest(names),X,d.mean(1)[:,0],U

def splits(m,kind):
    ids=np.flatnonzero(m.axis!='C')
    if kind=='interleaved':
        for k in range(2):
            te=[]
            for a in ['Y','T']:
                ix=m.index[m.axis==a].to_numpy()
                ix=ix[np.argsort(m.mtime.to_numpy()[ix])]
                te.extend(ix[k::2])
            te=np.array(sorted(te)); yield f'f{k}',np.setdiff1d(ids,te),te
    elif kind=='session_out':
        for s in sorted(m.session[m.axis!='C'].unique()):
            te=np.flatnonzero(m.session==s); yield s,np.setdiff1d(ids,te),te
    else:
        for k,(itr,ite) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261014).split(ids,m.session.iloc[ids])):
            yield f'f{k}',ids[itr],ids[ite]

def score(df):
    ok=df.axis.to_numpy()==df.pred_axis.to_numpy()
    sign=np.sign(df.value.to_numpy())==np.sign(df.pred.to_numpy())
    err=np.abs(df.value.to_numpy()-df.pred.to_numpy())
    true=np.c_[np.where(df.axis=='Y',df.value,0),np.where(df.axis=='T',df.value,0)]
    pred=np.c_[np.where(df.pred_axis=='Y',df.pred,0),np.where(df.pred_axis=='T',df.pred,0)]
    ve=np.abs(true-pred)
    r=dict(n=len(df),axis_accuracy=ok.mean(),axis_sign_accuracy=(ok&sign).mean(),strict_pass09=(ok&sign&(err<=.09)).mean(),both_vector_pass09=(ve<=.09).all(1).mean(),vector_Y_max=ve[:,0].max(),vector_T_max=ve[:,1].max())
    for a in ['Y','T']:
        t=(df.axis==a).to_numpy(); good=t&ok
        r.update({a+'_n':int(t.sum()),a+'_axis_accuracy':ok[t].mean() if t.any() else np.nan,a+'_axis_sign_accuracy':(ok&sign)[t].mean() if t.any() else np.nan,a+'_matched_MAE':err[good].mean() if good.any() else np.nan,a+'_matched_max':err[good].max() if good.any() else np.nan,a+'_wrong_axis':int((t&~ok).sum()),a+'_wrong_sign_when_axis_right':int((good&~sign).sum())})
    return r

def correct_labels(m,x,tr,mode):
    """Training-only offsets. Test labels remain raw; no test session is fitted."""
    val=m.value.to_numpy().copy(); detail=[]
    for a in ['Y','T']:
        ii=tr[m.axis.to_numpy()[tr]==a]
        v=val[ii]; ss=m.session.to_numpy()[ii]
        sessions=np.unique(ss)
        if mode=='session':
            S=(ss[:,None]==sessions).astype(float)
            A=np.c_[v,S]
            co=np.linalg.lstsq(A,x[ii],rcond=None)[0]
            off=S@co[1:]
        elif mode=='drift':
            tm=m.minutes.to_numpy()[ii]
            dirs=np.array([1 if s.endswith('+') or s=='Y2-' else -1 for s in ss])
            A=np.c_[v,np.ones(len(v)),dirs,tm,1-np.exp(-tm/3)]
            co=np.linalg.lstsq(A,x[ii],rcond=None)[0]; off=A[:,1:]@co[1:]
        else:
            A=np.c_[v,np.ones(len(v))]; co=np.linalg.lstsq(A,x[ii],rcond=None)[0]; off=np.repeat(co[1],len(ii))
        val[ii]=v+off/co[0]
        detail.append(dict(axis=a,mode=mode,n=len(ii),rank=int(np.linalg.matrix_rank(A)),columns=A.shape[1],condition=float(np.linalg.cond(A)),slope=float(co[0]),offset_min=float(np.min(off/co[0])),offset_max=float(np.max(off/co[0]))))
    return val,detail

def fit(m,X,tr,lab,mode,quad,U=None):
    branches={}; residual=[]; parameter_rank=0
    use_sessions=mode in ['oracle_zero','inferred_session']
    for a in ['Y','T']:
        ii=tr[m.axis.to_numpy()[tr]==a]; v=lab[ii]; Z=X[ii]
        ss=m.session.to_numpy()[ii]; sessions=np.unique(ss)
        basis=np.c_[v,v*v] if quad else v[:,None]
        if use_sessions:
            A=np.c_[basis,(ss[:,None]==sessions).astype(float)]
        elif mode=='free': A=np.c_[basis,np.ones(len(v))]
        else: A=basis
        co=np.linalg.lstsq(A,Z,rcond=None)[0]
        parameter_rank+=np.linalg.matrix_rank(A)
        residual.append(Z-A@co)
        s=co[0]; c=co[1] if quad else np.zeros(X.shape[1]); nbase=2 if quad else 1
        if use_sessions:
            for j,session in enumerate(sessions): branches[session]=(a,co[nbase+j],s,c)
        else: branches[a]=(a,co[-1] if mode=='free' else np.zeros(X.shape[1]),s,c)
    R=np.vstack(residual)
    S=R.T@R/max(len(R)-parameter_rank,1)
    std=np.sqrt(np.maximum(np.diag(S),1e-14)); C=S/std[:,None]/std[None,:]
    C=.999*C+.001*np.eye(len(C))
    W=np.linalg.cholesky(np.linalg.inv(C)).T/std[None,:]
    if U is not None: W=null_space((W@U).T).T@W
    return branches,W

def predict(m,X,te,model,mode):
    branches,W=model
    allp=[]
    if mode=='oracle_zero':
        known_offsets=np.stack([branches[s][1] for s in m.session.to_numpy()[te]])
        branches={a:(a,known_offsets,next(p[2] for p in branches.values() if p[0]==a),next(p[3] for p in branches.values() if p[0]==a)) for a in ['Y','T']}
    for b,(a,base,s,c) in branches.items():
        Z=(X[te]-base)@W.T; sw=s@W.T; cw=c@W.T
        v=Z@sw/(sw@sw)
        for _ in range(8):
            J=sw+2*v[:,None]*cw
            r=Z-v[:,None]*sw-(v*v)[:,None]*cw
            v+=np.sum(J*r,axis=1)/np.maximum(np.sum(J*J,axis=1),1e-20)
        r=Z-v[:,None]*sw-(v*v)[:,None]*cw
        cost=np.sum(r*r,axis=1)
        # Branch sign constraints are learned session domain, not test labels.
        if mode=='inferred_session': cost=np.where(v*(1 if b.endswith('+') else -1)>=0,cost,np.inf)
        allp.append((b,a,v,cost))
    cost=np.stack([p[3] for p in allp],axis=1); j=np.argmin(cost,axis=1)
    rows=[]
    for k,i in enumerate(te):
        b,a,v,_=allp[j[k]]
        rows.append(dict(file=m.file.iloc[i],axis=m.axis.iloc[i],value=m.value.iloc[i],session=m.session.iloc[i],pred_axis=a,pred=float(v[k]),branch=b,cost=float(cost[k,j[k]])))
    return pd.DataFrame(rows)

def main():
    import argparse
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--measurement');args=parser.parse_args()
    if args.measurement: OUT=HERE/('zero_'+args.measurement)
    OUT.mkdir(exist_ok=True)
    m,X,x,U=load(args.measurement)
    if not args.measurement: m.to_csv(HERE/'manifest.csv',index=False,encoding='utf-8-sig')
    zero=[]
    for s,g in m[m.axis!='C'].groupby('session'):
        ii=g.index.to_numpy(); v=g.value.to_numpy(); A=np.c_[np.ones(len(g)),v]
        p=np.linalg.lstsq(A,x[ii],rcond=None)[0]; residual=x[ii]-A@p
        cov=(residual@residual)/(len(g)-2)*np.linalg.inv(A.T@A)
        zero.append(dict(session=s,n=len(g),slope_px_per_unit=p[1],intercept_px=p[0],center_relative_offset=p[0]/p[1],intercept_se_px=np.sqrt(cov[0,0]),residual_rms_px=np.sqrt(np.mean(residual**2)),value_time_correlation=np.corrcoef(v,g.minutes)[0,1]))
    pd.DataFrame(zero).to_csv(OUT/'session_zero_estimates.csv',index=False)
    rows=[]; preds=[]; details=[]
    configs=[('raw_center','center',None),('raw_free','free',None),('center_relative_session','center','session'),('center_relative_drift','center','drift'),('center_relative_global','center','global'),('inferred_session','inferred_session',None),('oracle_zero','oracle_zero',None)]
    for cv in ['interleaved','stratified5','session_out']:
        for name,mode,cor in configs:
            if cv=='session_out' and mode=='oracle_zero': continue
            for nuisance in [False,True]:
                for quad in [False,True]:
                    records=[]
                    for fold,tr,te in splits(m,cv):
                        lab=m.value.to_numpy().copy()
                        if cor:
                            lab,inf=correct_labels(m,x,tr,cor)
                            details.extend([dict(cv=cv,fold=fold,config=name,**dd) for dd in inf])
                        model=fit(m,X,tr,lab,mode,quad,U if nuisance else None)
                        df=predict(m,X,te,model,mode)
                        df['fold']=fold
                        records.append(df)
                    df=pd.concat(records,ignore_index=True)
                    meta=dict(cv=cv,config=name,nuisance=nuisance,quadratic=quad)
                    rows.append(dict(**meta,**score(df)))
                    preds.append(df.assign(**meta))
        print('Completed zero experiment',cv,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'zero_ablation.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(OUT/'zero_predictions.csv',index=False)
    pd.DataFrame(details).drop_duplicates().to_csv(OUT/'train_only_correction_diagnostics.csv',index=False)
    # Perturb held-out image features and labels and assert fit parameters unchanged.
    _,tr,te=next(splits(m,'stratified5'))
    lab,_=correct_labels(m,x,tr,'session'); one=fit(m,X,tr,lab,'center',False,U)
    m2=m.copy(); m2.loc[te,'value']+=123.; x2=x.copy(); x2[te]+=1000.; X2=X.copy(); X2[te]+=10000.
    lab2,_=correct_labels(m2,x2,tr,'session'); two=fit(m2,X2,tr,lab2,'center',False,U)
    ok=np.allclose(one[1],two[1]) and all(np.allclose(np.asarray(one[0][k][1:]),np.asarray(two[0][k][1:])) for k in one[0])
    assert ok
    (OUT/'checks.json').write_text(json.dumps(dict(heldout_perturbation_does_not_change_fit=ok,ground_truth='raw filename command; physical pose not observed',center_count=int((m.axis=='C').sum()),evaluated_count=int((m.axis!='C').sum())),indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).sort_values(['cv','axis_sign_accuracy','strict_pass09'],ascending=[True,False,False]).groupby('cv').head(5).to_string(index=False),flush=True)

if __name__=='__main__': main()
