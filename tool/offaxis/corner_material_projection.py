"""Held-point planar projection with train-only material boundary retreat.

Effective printed-boundary offsets are descriptive calibration parameters, not
independent ground truth or measured physical ink/coating thickness.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.analysis_packages'))
import json
import time
import hashlib
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from corner_benchmark import geometry,project,OUT,COLORS


def signed_retreat(quads):
    result=[]
    for b in quads:
        minority=b if b.sum()==1 else 1-b
        r,c=np.unravel_index(np.argmax(minority),minority.shape)
        result.append(np.array([1 if c else -1,1 if r else -1])*(1 if b.sum()==1 else -1))
    return np.array(result,float)


def fit_projector(grid,sign,observed):
    center=observed.mean(0);scale=80.
    target=(observed-center)/scale
    H,_=cv2.findHomography(grid,target,0)
    if H is None:raise ValueError('degenerate training geometry')
    initial=np.r_[H.ravel()[:8]/H[2,2],0.,0.]
    def evaluate(p,jac=False):
        uv=grid+sign*p[8:]
        x,y=uv.T
        d=p[6]*x+p[7]*y+1
        a=p[0]*x+p[1]*y+p[2];b=p[3]*x+p[4]*y+p[5]
        u,v=a/d,b/d
        if not jac:return (np.column_stack([u,v])-target).ravel()
        J=np.zeros((len(grid),2,10))
        J[:,0,0]=x/d;J[:,0,1]=y/d;J[:,0,2]=1/d
        J[:,1,3]=x/d;J[:,1,4]=y/d;J[:,1,5]=1/d
        J[:,0,6]=-u*x/d;J[:,0,7]=-u*y/d
        J[:,1,6]=-v*x/d;J[:,1,7]=-v*y/d
        J[:,0,8]=(p[0]-u*p[6])/d*sign[:,0]
        J[:,0,9]=(p[1]-u*p[7])/d*sign[:,1]
        J[:,1,8]=(p[3]-v*p[6])/d*sign[:,0]
        J[:,1,9]=(p[4]-v*p[7])/d*sign[:,1]
        return J.reshape(-1,10)
    opt=least_squares(evaluate,initial,jac=lambda p:evaluate(p,True),
        bounds=(np.r_[np.full(8,-np.inf),-.25,-.25],np.r_[np.full(8,np.inf),.25,.25]),
        loss='soft_l1',f_scale=.25/scale,max_nfev=35,xtol=1e-8,ftol=1e-8,gtol=1e-8)
    return dict(parameters=opt.x,center=center,scale=scale,success=opt.success,evaluate=evaluate)


def predict(model,grid,sign):
    p=model['parameters'];H=np.r_[p[:8],1.].reshape(3,3)
    return project(H,grid+sign*p[8:])*model['scale']+model['center']


def check():
    rng=np.random.default_rng(41)
    grid=rng.uniform(0,21,(80,2));sign=rng.choice([-1.,1.],(80,2))
    H=np.array([[8.,.1,1800],[-.2,8.1,1000],[.0001,-.0002,1]])
    truth=project(H,grid+sign*[.08,.11])
    model=fit_projector(grid,sign,truth)
    err=float(abs(predict(model,grid,sign)-truth).max())
    assert err<1e-5,err
    p=model['parameters'];analytic=model['evaluate'](p,True)
    eps=1e-6
    numeric=np.column_stack([(model['evaluate'](p+np.eye(10)[i]*eps)-model['evaluate'](p-np.eye(10)[i]*eps))/(2*eps) for i in range(10)])
    relative=float(np.linalg.norm(analytic-numeric)/np.linalg.norm(numeric))
    assert relative<1e-6,relative
    return dict(noiseless_max_error_px=err,jacobian_relative_error=relative)


def main():
    cv2.setNumThreads(1);checks=check();start=time.time()
    real=pd.read_csv(OUT/'real_points.csv.gz')
    models={m['slot']:m for m in geometry()}
    rows=[];params=[]
    l=real[real.kind.eq('L')]
    for n,((path,qr,method),g) in enumerate(l.groupby(['path','qr','method'],sort=False)):
        m=models[qr];features={f['id']:i for i,f in enumerate(m['features'])}
        ix=[features[i] for i in g.id]
        grid=m['grid'][ix];sign=signed_retreat(m['quad'][ix]);obs=g[['x','y']].to_numpy()
        valid=g.valid.to_numpy()&np.isfinite(obs).all(1)
        fold=(grid[:,0].astype(int)//2+2*(grid[:,1].astype(int)//2))%5
        predicted=np.full_like(obs,np.nan)
        for f in range(5):
            train=valid&(fold!=f);test=valid&(fold==f)
            if train.sum()<20 or not test.any():continue
            model=fit_projector(grid[train],sign[train],obs[train])
            predicted[test]=predict(model,grid[test],sign[test])
            params.append(dict(path=path,qr=qr,method=method,fold=f,
                retreat_x_module=model['parameters'][8],retreat_y_module=model['parameters'][9],success=bool(model['success'])))
        for i,r in enumerate(g.itertuples()):
            rr=obs[i]-predicted[i]
            rows.append(dict(path=path,qr=qr,method=method,id=int(r.id),valid=bool(valid[i]),
                predicted_x=predicted[i,0],predicted_y=predicted[i,1],
                corrected_dx=rr[0],corrected_dy=rr[1],corrected_error_px=np.linalg.norm(rr)))
        if (n+1)%300==0:
            (OUT/'material_progress.txt').write_text(f'{n+1}/3069 groups; {time.time()-start:.1f}s',encoding='utf-8')
    result=pd.DataFrame(rows);result.to_csv(OUT/'material_projection.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(params).to_csv(OUT/'material_projection_parameters.csv',index=False)
    common=result.pivot(index=['path','qr','id'],columns='method',values='corrected_error_px').dropna()
    summary=[]
    for method in common:
        e=common[method]
        summary.append(dict(method=method,scored=len(e),median_px=float(e.median()),rms_px=float(np.sqrt(np.mean(e**2))),p95_px=float(e.quantile(.95))))
    pd.DataFrame(summary).to_csv(OUT/'material_projection_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for method in common:
        e=common[method].sort_values();axes[0].plot(e,np.arange(1,len(e)+1)/len(e),c=COLORS[method],label=method)
    axes[0].set_xlim(0,1);axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
    axes[0].set_title('Held L points: H + train-only boundary retreat');axes[0].set_xlabel('Residual (px); consistency, not absolute truth')
    for method,g in pd.DataFrame(params).groupby('method'):
        axes[1].scatter(g.retreat_x_module,g.retreat_y_module,s=2,alpha=.08,c=COLORS[method],label=method)
    axes[1].set_xlabel('Effective X retreat (module)');axes[1].set_ylabel('Effective Y retreat (module)');axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    axes[1].set_title('Apparent offset includes estimator bias + printing')
    fig.savefig(OUT/'08_material_projection.png',dpi=170);plt.close(fig)
    (OUT/'material_manifest.json').write_text(json.dumps(dict(checks=checks,seconds=time.time()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        real_source_sha256=hashlib.sha256((OUT/'real_points.csv.gz').read_bytes()).hexdigest(),
        note='No held point enters projector calibration. Effective retreat is not independent ground truth.'),indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(pd.DataFrame(params).groupby('method')[['retreat_x_module','retreat_y_module']].median().to_string(),flush=True)


if __name__=='__main__':main()
