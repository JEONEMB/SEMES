"""Spatial QR calibration using the fixed user-requested train/test split.

Retains affine deformation information discarded by QR mean translations.
All model selection uses training-only internal grouped folds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from refit_axis_split import metrics, TICKS


def extract(source):
    features={}
    qr=pd.read_csv(source/'qr_displacements.csv')
    means=qr.pivot(index='path',columns='qr',values=['dx','dy'])
    cols=[(d,s) for s in ['L','C','R'] for d in ['dx','dy']]
    features['qr_means']=means[cols].set_axis([f'{s}_{d}' for d,s in cols],axis=1)
    ecc=pd.read_csv(source/'ecc.csv')
    for name,dims in [('ecc_translation',['dx','dy']),
                      ('ecc_affine',['dx','dy','a00','a01','a10','a11'])]:
        w=ecc.pivot(index='path',columns='qr',values=dims)
        cols=[(d,s) for s in ['L','C','R'] for d in dims]
        features[name]=w[cols].set_axis([f'ecc_{s}_{d}' for d,s in cols],axis=1)
    pts=pd.read_csv(source/'displacements.csv.gz')
    pts=pts[~pts.diagonal]
    rows=[]
    for path,g in pts.groupby('path',sort=True):
        row={'path':path}
        for slot,h in g.groupby('qr'):
            # Reference geometry only; never use labels to extract image features.
            rx=(h.old_reference_x.to_numpy()-1900)/300
            ry=(h.old_reference_y.to_numpy()-1070)/100
            a=np.column_stack([np.ones(len(h)),rx,ry])
            coeff=np.linalg.lstsq(a,h[['dx','dy']].to_numpy(),rcond=None)[0]
            for i,term in enumerate(['offset','xgradient','ygradient']):
                for j,dim in enumerate(['dx','dy']):
                    row[f'point_{slot}_{dim}_{term}']=coeff[i,j]
        rows.append(row)
    features['point_affine']=pd.DataFrame(rows).set_index('path')
    features['combined']=features['qr_means'].join(features['ecc_affine'])
    return features


def fit(x,y,s,alpha,degree,separate):
    xc=x.mean(0);xs=x.std(0);xs=np.where(xs>1e-12,xs,1.)
    z=(x-xc)/xs
    if degree==2:
        z=np.column_stack([z]+[z[:,i]*z[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])])
    centers={};a=z.copy();b=y.copy()
    for session in np.unique(s):
        mask=s==session
        zm,ym=z[mask].mean(0),y[mask].mean(0)
        centers[str(session)]={'feature':zm.tolist(),'target':ym.tolist()}
        a[mask]-=zm;b[mask]-=ym
    fs=a.std(0);fs=np.where(fs>1e-12,fs,1.)
    a/=fs
    models={}
    for name in (np.unique(s) if separate else ['shared']):
        mask=s==name if separate else np.ones(len(s),dtype=bool)
        u,v,vt=np.linalg.svd(a[mask],full_matrices=False)
        weights=v/(v*v+alpha) if alpha else np.divide(1.,v,out=np.zeros_like(v),where=v>v.max()*1e-10)
        models[str(name)]=((vt.T*weights)@(u.T@(b[mask]/TICKS))).tolist()
    return dict(x_center=xc.tolist(),x_scale=xs.tolist(),feature_scale=fs.tolist(),
                centers=centers,coefficients=models,alpha=alpha,degree=degree,separate=separate)


def predict(m,x,s):
    if 'kernel_models' in m:
        out=predict(m['baseline'],x,s)
        for session in np.unique(s):
            mask=s==session;k=m['kernel_models'][str(session)]
            z=((x[mask]-k['center'])/k['scale'])@np.array(k['projection'])
            dist=np.sum((z[:,None,:]-np.array(k['training_z'])[None,:,:])**2,axis=2)/z.shape[1]
            out[mask]+=np.exp(-dist/(2*m['length']**2))@np.array(k['coefficients'])*TICKS
        return out
    z=(x-m['x_center'])/m['x_scale']
    if m['degree']==2:
        z=np.column_stack([z]+[z[:,i]*z[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])])
    out=np.zeros((len(x),2))
    for session in np.unique(s):
        mask=s==session;c=m['centers'][str(session)]
        coef=m['coefficients'][str(session) if m['separate'] else 'shared']
        out[mask]=((z[mask]-c['feature'])/m['feature_scale'])@np.array(coef)*TICKS+c['target']
    return out


def fit_kernel(x,y,s,alpha,length,rank):
    baseline=fit(x,y,s,.0001,1,False)
    residual=(y-predict(baseline,x,s))/TICKS
    km={}
    for session in np.unique(s):
        mask=s==session;xx=x[mask];center=xx.mean(0);scale=xx.std(0)
        scale=np.where(scale>1e-12,scale,1.);z=(xx-center)/scale
        _,sv,vt=np.linalg.svd(z,full_matrices=False)
        count=min(rank,len(sv));proj=vt[:count].T/(np.maximum(sv[:count]/np.sqrt(len(z)),.05))
        zz=z@proj
        dist=np.sum((zz[:,None,:]-zz[None,:,:])**2,axis=2)/count
        kernel=np.exp(-dist/(2*length*length))
        coef=np.linalg.solve(kernel+alpha*np.eye(len(xx)),residual[mask])
        km[str(session)]=dict(center=center.tolist(),scale=scale.tolist(),projection=proj.tolist(),
                              training_z=zz.tolist(),coefficients=coef.tolist())
    return dict(baseline=baseline,kernel_models=km,alpha=alpha,length=length,rank=rank)


def main(source,split,output):
    output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(split,index_col='path')
    fs=extract(source)
    train=df.split.eq('train').to_numpy(); y=df[['y_mm','theta_deg']].to_numpy();s=df.session.to_numpy()
    fold=df.inner_fold.to_numpy().copy()
    # Keep all repeats of each mixed coordinate in the same internal fold.
    mixed=df[train&df.category.eq('mixed').to_numpy()]
    for i,((yy,tt),g) in enumerate(mixed.groupby(['y_mm','theta_deg'],sort=True)):
        fold[df.index.isin(g.index)]=i%3
    configs=[]
    for name,f in fs.items():
        x=f.loc[df.index].to_numpy()
        for degree in ([1,2] if name in ['qr_means','ecc_translation'] else [1]):
            for separate in [False,True]:
                for alpha in [0.,1e-6,1e-4,.01,1.,100.]:
                    # Avoid unconstrained high-dimensional fits.
                    if degree==2 and alpha==0:continue
                    pred=np.full_like(y,np.nan)
                    for k in range(3):
                        val=train&(fold==k);learn=train&~val
                        m=fit(x[learn],y[learn],s[learn],alpha,degree,separate)
                        pred[val]=predict(m,x[val],s[val])
                    error=((pred-y)/TICKS)**2
                    axis=train&(s=='old');mix=train&(s=='new')
                    # Equal influence for axis and mixed domains, despite counts.
                    score=float((np.mean(error[axis])+np.mean(error[mix]))/2)
                    configs.append(dict(family='polynomial',features=name,degree=degree,separate=separate,alpha=alpha,
                                        internal_score=score,axis_mse_ticks=float(np.mean(error[axis])),
                                        mixed_mse_ticks=float(np.mean(error[mix]))))
        # Nonlinear correction about a linear baseline, independently per session.
        for rank in [2,6]:
            for length in [.3,1.,3.]:
                for alpha in [.0001,.01,1.]:
                    pred=np.full_like(y,np.nan)
                    for k in range(3):
                        val=train&(fold==k);learn=train&~val
                        m=fit_kernel(x[learn],y[learn],s[learn],alpha,length,rank)
                        pred[val]=predict(m,x[val],s[val])
                    error=((pred-y)/TICKS)**2
                    axis=train&(s=='old');mix=train&(s=='new')
                    score=float((np.mean(error[axis])+np.mean(error[mix]))/2)
                    configs.append(dict(family='kernel_residual',features=name,rank=rank,length=length,alpha=alpha,
                                        internal_score=score,axis_mse_ticks=float(np.mean(error[axis])),
                                        mixed_mse_ticks=float(np.mean(error[mix]))))
    table=pd.DataFrame(configs).sort_values('internal_score')
    table.to_csv(output/'training_selection.csv',index=False)
    best=table.iloc[0].to_dict();name=best['features'];x=fs[name].loc[df.index].to_numpy()
    if best['family']=='kernel_residual':
        m=fit_kernel(x[train],y[train],s[train],float(best['alpha']),float(best['length']),int(best['rank']))
    else:
        m=fit(x[train],y[train],s[train],float(best['alpha']),int(best['degree']),bool(best['separate']))
    pred=predict(m,x,s)
    result=df.copy()
    result[['predicted_y_mm','predicted_theta_deg']]=pred
    result[['error_y_mm','error_theta_deg']]=pred-y
    result['selection_fold']=fold
    result.to_csv(output/'predictions_and_split.csv')
    scores=[metrics(n,g) for n,g in result.groupby('split')]
    pd.DataFrame(scores).to_csv(output/'summary.csv',index=False)
    fs[name].loc[df.index].to_csv(output/'selected_features.csv')
    bundle=dict(model=m,selected=best,input_columns=fs[name].columns.tolist(),summary=scores,
                source_sha256={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in
                               ['frames.csv','qr_displacements.csv','ecc.csv','displacements.csv.gz']},
                split_sha256=hashlib.sha256(split.read_bytes()).hexdigest())
    bundle['selected']={k:v for k,v in best.items() if not (isinstance(v,float) and np.isnan(v))}
    (output/'model.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    assert np.isfinite(pred).all()
    print(table.head(8).to_string(index=False));print(pd.DataFrame(scores).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--split',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.split,a.output)
