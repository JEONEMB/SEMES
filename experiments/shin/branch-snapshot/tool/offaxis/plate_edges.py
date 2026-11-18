"""Raw-intensity subpixel outer-edge measurement and nested pose holdout.

No Canny and no new images. A local PSF step / ridge plus illumination slope
is fit to raw pixels, followed by robust long-edge regression. QR is only a
coarse ROI initializer; no command coordinate is used for extraction.
"""
import sys
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.analysis_packages'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.analysis_packages/mplconfig'))
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=ROOT/'docs/offaxis-eda'
OUT=BASE/'plate-edges'
DATA=ROOT/'data/SSAFY Test Image Set'
TICKS=np.array([.05,.01])
ALPHAS=[0.,.001,.01,.1,1.,10.,100.]
EDGES={
 'top':dict(vertical=False,normal=398.,anchor=1900.,scale=450.,kind='ridge',polarity=-1,
            tangent=np.r_[np.linspace(1500,1770,16),np.linspace(2050,2310,16)]),
 'bottom':dict(vertical=False,normal=1735.,anchor=1900.,scale=450.,kind='step',polarity=1,
            tangent=np.r_[np.linspace(1510,1800,16),np.linspace(2000,2290,16)]),
 'left':dict(vertical=True,normal=1407.,anchor=1150.,scale=250.,kind='step',polarity=-1,
            tangent=np.linspace(930,1370,36)),
 'right':dict(vertical=True,normal=2401.,anchor=1150.,scale=250.,kind='step',polarity=1,
            tangent=np.linspace(930,1370,36))}


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def model_value(p,n,t,kind,jac=False):
    u,sigma,amp,offset,bg,bt=p
    z=(n-u)/sigma
    density=np.exp(-.5*z*z)
    if kind=='step':
        F=.5*(1+erf(z/np.sqrt(2)))
        du=-density/(sigma*np.sqrt(2*np.pi))
        ds=-z*density/(sigma*np.sqrt(2*np.pi))
    else:
        F=density;du=z*density/sigma;ds=z*z*density/sigma
    if jac:return np.column_stack([amp*du,amp*ds,F,np.ones(len(n)),n,t])
    return offset+bg*n+bt*t+amp*F


def fit_profile(image,normal,tangent,definition):
    n0=int(round(normal));t0=int(round(tangent))
    scan=np.arange(n0-13,n0+14)
    ts=np.arange(t0-3,t0+4)
    if definition['vertical']:stripe=image[np.ix_(ts,scan)].T.astype(float)
    else:stripe=image[np.ix_(scan,ts)].astype(float)
    average=stripe.mean(1)
    if definition['kind']=='ridge':
        # Known dark outline: a virtual center of the narrow ridge, not a claim
        # to the physical outside boundary of the acrylic material.
        index=np.argmin(average[3:-3])+3
        coarse=float(scan[index])
    else:
        gradient=definition['polarity']*np.diff(average)
        mids=(scan[:-1]+scan[1:])/2
        score=gradient-.12*(mids-normal)**2
        index=int(np.argmax(score));coarse=float(mids[index])
    center=int(round(coarse));nn=np.arange(center-7,center+8)
    if definition['vertical']:patch=image[np.ix_(ts,nn)].T.astype(float)
    else:patch=image[np.ix_(nn,ts)].astype(float)
    n=np.repeat(nn-coarse,len(ts)).astype(float);t=np.tile(ts-t0,len(nn)).astype(float)
    target=patch.ravel()
    if definition['kind']=='step':
        amp=float(patch[-3:].mean()-patch[:3].mean());offset=float(patch[:3].mean())
    else:
        offset=float(np.r_[patch[:3].ravel(),patch[-3:].ravel()].mean());amp=float(patch.mean(1).min()-offset)
    initial=[0.,.8,amp,offset,0.,0.]
    def residual(p):return model_value(p,n,t,definition['kind'])-target
    opt=least_squares(residual,initial,jac=lambda p:model_value(p,n,t,definition['kind'],True),
        bounds=([-3.5,.15,-300,-50,-12,-12],[3.5,4.5,300,310,12,12]),
        loss='soft_l1',f_scale=4.,max_nfev=45,ftol=1e-7,xtol=1e-7,gtol=1e-6)
    p=opt.x;rmse=float(np.sqrt(np.mean(opt.fun**2)))
    valid=bool(opt.success and abs(p[0])<3.4 and p[1]<4.4 and abs(p[2])>8 and rmse<20)
    return dict(normal=coarse+p[0],tangent=float(t0),coarse=coarse,sigma=p[1],contrast=p[2],rmse=rmse,
        valid=valid,success=bool(opt.success),parameters=p.tolist(),n=n,t=t,observed=target,fitted=model_value(p,n,t,definition['kind']))


def line_fit(samples,definition):
    valid=np.array([s['valid'] for s in samples]);t=np.array([s['tangent'] for s in samples])
    n=np.array([s['normal'] for s in samples]);rmse=np.array([s['rmse'] for s in samples])
    z=(t-definition['anchor'])/definition['scale']
    if valid.sum()<12:return [np.nan,np.nan],np.full(len(samples),np.nan),valid
    X=np.column_stack([np.ones(len(z)),z]);w=valid.astype(float)/np.maximum(rmse,2.)**2
    for _ in range(8):
        coef=np.linalg.lstsq(X*np.sqrt(w[:,None]),n*np.sqrt(w),rcond=None)[0]
        r=n-X@coef
        w=valid.astype(float)/np.maximum(rmse,2.)**2*np.minimum(1,.35/np.maximum(abs(r),1e-12))
    inlier=valid&(abs(r)<1.5)
    if inlier.sum()<12:return [np.nan,np.nan],r,inlier
    # Keep the robust fit, do not delete its largest residual to improve scores.
    return coef.tolist(),r,inlier


def extract(second_only=False):
    meta=pd.read_csv(BASE/'ppt-method-v2/frames.csv').set_index('path')
    if second_only:meta=meta[meta.category.isin(['mixed','new-reference'])]
    ec=pd.read_csv(BASE/'raw/ecc.csv').groupby('path')[['dx','dy']].mean()
    features=[];samples=[];start=time.time()
    for k,(path,row) in enumerate(meta.iterrows()):
        raw=(DATA/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==row.sha256,path
        image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        shift=ec.loc[path].to_numpy()
        feat=dict(path=path)
        profile_examples={}
        for name,definition in EDGES.items():
            normal_shift=shift[0 if definition['vertical'] else 1]
            tangent_shift=shift[1 if definition['vertical'] else 0]
            measured=[fit_profile(image,definition['normal']+normal_shift,t+tangent_shift,definition) for t in definition['tangent']]
            coefs,res,inlier=line_fit(measured,definition)
            feat[name+'_position']=coefs[0];feat[name+'_slope']=coefs[1]
            feat[name+'_valid_profiles']=int(inlier.sum());feat[name+'_total_profiles']=len(inlier)
            feat[name+'_line_rms']=float(np.sqrt(np.mean(res[inlier]**2))) if inlier.any() else np.nan
            profile_examples[name]=measured[len(measured)//3]
            for i,r in enumerate(measured):
                samples.append(dict(path=path,edge=name,sample=i,tangent=r['tangent'],normal=r['normal'],
                    sigma=r['sigma'],contrast=r['contrast'],pixel_rmse=r['rmse'],valid_profile=r['valid'],
                    line_inlier=bool(inlier[i]),line_residual=res[i]))
        features.append(feat)
        if path in ['center.bmp','기준 위치/center1.bmp']:
            illustrate(image,feat,measured=None,profile_examples=profile_examples,name='old' if path=='center.bmp' else 'new')
        if (k+1)%20==0:print(f'Edges {k+1}/{len(meta)}; {time.time()-start:.1f}s',flush=True)
    pd.DataFrame(features).to_csv(OUT/'edge_features.csv',index=False)
    pd.DataFrame(samples).to_csv(OUT/'edge_profiles.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    print('Extraction seconds',time.time()-start,flush=True)


def illustrate(image,features,measured,profile_examples,name):
    fig,ax=plt.subplots(figsize=(9,10),layout='constrained');ax.imshow(image,cmap='gray',vmin=0,vmax=255)
    colors={'top':'#de632f','bottom':'#b24bca','left':'#1ca578','right':'#159ec2'}
    for edge,d in EDGES.items():
        t=d['tangent'];n=features[edge+'_position']+features[edge+'_slope']*(t-d['anchor'])/d['scale']
        if d['vertical']:ax.plot(n,t,'.',ms=4,c=colors[edge],label=edge)
        else:ax.plot(t,n,'.',ms=4,c=colors[edge],label=edge)
    ax.set_xlim(1300,2550);ax.set_ylim(1820,290);ax.legend();ax.set_title('Numerically fitted subpixel edge lines; raw image')
    fig.savefig(OUT/f'01_edges_{name}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,(edge,p) in zip(axes.ravel(),profile_examples.items()):
        ax.scatter(p['n'],p['observed'],s=8,c='#939da5',alpha=.5,label='raw pixels (7-wide strip)')
        nn=np.linspace(p['n'].min(),p['n'].max(),400)
        ax.plot(nn,model_value(p['parameters'],nn,np.zeros(len(nn)),EDGES[edge]['kind']),
                c=colors[edge],lw=2,label=EDGES[edge]['kind']+' continuous fit')
        ax.axvline(p['normal']-p['coarse'],c='#344454',ls='--',label='subpixel location')
        ax.set_title(f'{edge}: position={p["normal"]:.4f}px; sigma={p["sigma"]:.3f}px')
        ax.set_xlabel('Normal coordinate relative to coarse seed (px)');ax.set_ylabel('Raw intensity');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/f'02_profiles_{name}.png',dpi=170);plt.close(fig)


def fit_ridge(x,y,alpha):
    med=np.nanmedian(x,axis=0);x=np.where(np.isfinite(x),x,med)
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-8);z=(x-mean)/scale
    ym=y.mean(0);u,s,vt=np.linalg.svd(z,full_matrices=False)
    factors=s/(s*s+alpha) if alpha else np.divide(1.,s,out=np.zeros_like(s),where=s>s.max()*1e-10)
    coef=(vt.T*factors)@(u.T@(y-ym))
    return dict(median=med,mean=mean,scale=scale,ymean=ym,coef=coef,alpha=alpha)


def predict(m,x):
    x=np.where(np.isfinite(x),x,m['median'])
    return ((x-m['mean'])/m['scale'])@m['coef']+m['ymean']


def tune(x,y,groups):
    scores=[]
    for alpha in ALPHAS:
        errors=[]
        for g in np.unique(groups):
            test=groups==g
            m=fit_ridge(x[~test],y[~test],alpha)
            errors.extend((predict(m,x[test])-y[test]).tolist())
        scores.append(dict(alpha=alpha,mse_ticks=float(np.mean(np.square(errors)))))
    return min(scores,key=lambda r:r['mse_ticks'])['alpha'],scores


def evaluate():
    meta=pd.read_csv(BASE/'ppt-method-v2/frames.csv').set_index('path')
    mixed=meta[meta.category.eq('mixed')].sort_index()
    refs=meta[meta.category.eq('new-reference')]
    actual_reference=refs[['y_actual','theta_actual']].mean().to_numpy()
    actual=mixed[['y_actual','theta_actual']].to_numpy()-actual_reference
    nominal=mixed[['y_mm','theta_deg']].to_numpy()
    assert np.isfinite(actual).all() and np.max(abs(actual-nominal))<1e-5
    y=actual/TICKS
    coordinates=np.unique(nominal,axis=0)
    group=np.array([np.where(np.all(coordinates==q,axis=1))[0][0] for q in nominal])
    qr=pd.read_csv(BASE/'ppt-method-v2/qr_displacements.csv').pivot(index='path',columns='qr',values=['dx','dy'])
    qrcols=[(d,s) for s in 'LCR' for d in ['dx','dy']]
    qx=qr[qrcols].loc[mixed.index].to_numpy()
    ef=pd.read_csv(OUT/'edge_features.csv').set_index('path')
    edgecols=[e+'_'+v for e in EDGES for v in ['position','slope']]
    ex=ef.loc[mixed.index,edgecols].to_numpy()
    # A reference change is a fixed feature offset and cannot affect an affine
    # model with a free intercept. Use center1 solely for readable feature units.
    ex-=ef.loc['기준 위치/center1.bmp',edgecols].to_numpy(dtype=float)
    sets={'QR6_OLS':qx,'QR6_nested':qx,'edges8_nested':ex,'QR6_edges8_nested':np.column_stack([qx,ex]),
          'QR6_vertical4_nested':np.column_stack([qx,ex[:,4:]]),
          'QR6_horizontal4_nested':np.column_stack([qx,ex[:,:4]])}
    rows=[];scores=[];selections=[];fitted={};all_tuning=[]
    for name,x in sets.items():
        held=np.zeros_like(y)
        for g in range(19):
            test=group==g;train=~test
            alpha=0. if name.endswith('OLS') else tune(x[train],y[train],group[train])[0]
            m=fit_ridge(x[train],y[train],alpha);held[test]=predict(m,x[test])
            selections.append(dict(model=name,held_coordinate=g,held_y_mm=coordinates[g,0],held_yaw_deg=coordinates[g,1],alpha=alpha))
        err=held-y
        scores.append(dict(model=name,images=57,coordinates=19,Y_rmse_ticks=np.sqrt(np.mean(err[:,0]**2)),
            yaw_rmse_ticks=np.sqrt(np.mean(err[:,1]**2)),Y_rmse_mm=np.sqrt(np.mean(err[:,0]**2))*.05,
            yaw_rmse_deg=np.sqrt(np.mean(err[:,1]**2))*.01,both_within_tick=float(np.mean(np.max(abs(err),axis=1)<=1)),
            Y_max_abs_ticks=np.max(abs(err[:,0])),yaw_max_abs_ticks=np.max(abs(err[:,1]))))
        for i,path in enumerate(mixed.index):
            rows.append(dict(model=name,path=path,coordinate_group=group[i],y_mm=actual[i,0],yaw_deg=actual[i,1],
                predicted_y_mm=held[i,0]*.05,predicted_yaw_deg=held[i,1]*.01,Y_error_ticks=err[i,0],yaw_error_ticks=err[i,1]))
        alpha,tuning=tune(x,y,group) if not name.endswith('OLS') else (0.,[])
        fitted[name]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit_ridge(x,y,alpha).items()}
        all_tuning.extend(dict(model=name,**r) for r in tuning)
        print(name,scores[-1],flush=True)
    pd.DataFrame(scores).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'coordinate_holdout.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'outer_alpha_selection.csv',index=False)
    pd.DataFrame(all_tuning).to_csv(OUT/'full_data_tuning.csv',index=False)
    (OUT/'models.json').write_text(json.dumps(fitted,indent=2),encoding='utf-8')
    # Edge repeatability includes stage/image/lighting changes, not only sensor noise.
    eframe=ef.loc[mixed.index,edgecols].copy();eframe['group']=group
    rep=[]
    for col in edgecols:
        means=eframe.groupby('group')[col].transform('mean');r=eframe[col]-means
        rep.append(dict(feature=col,valid_images=int(eframe[col].notna().sum()),repeat_sd_px=float(np.sqrt(np.nansum(r*r)/38))))
    pd.DataFrame(rep).to_csv(OUT/'edge_repeatability.csv',index=False)
    visual_results(pd.DataFrame(scores),pd.DataFrame(rows))
    labelcheck=mixed[['y_actual','theta_actual','y_mm','theta_deg']].copy()
    labelcheck['relative_y_actual']=actual[:,0];labelcheck['relative_yaw_actual']=actual[:,1]
    labelcheck.to_csv(OUT/'actuator_labels.csv')
    manifest=dict(date='2026-10-27',labels='Second-data actuator reported position relative to valid new-reference mean',
        actual_reference=actual_reference.tolist(),max_difference_from_nominal=float(np.max(abs(actual-nominal))),
        coordinate_holdout='19 outer folds; all 3 repeated images held together',
        regularization='18-coordinate inner holdout; scaler and imputation learned on training only',
        edge_geometry='Independent long edges, no single-plane homography enforced',Canny_used=False,
        original_images_modified=False,qr_seed_uses_labels=False,
        edge_definitions={e:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in d.items()} for e,d in EDGES.items()},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),BASE/'ppt-method-v2/frames.csv',BASE/'ppt-method-v2/qr_displacements.csv',OUT/'edge_features.csv']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def visual_results(score,pred):
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,metric,title in [(axes[0],'Y_rmse_ticks','Y'),(axes[1],'yaw_rmse_ticks','yaw')]:
        ax.bar(np.arange(len(score)),score[metric],color=['#6c8497','#6c8497','#dd9340','#2d9a6c','#2c7eb3','#b16a9a'])
        ax.set_xticks(np.arange(len(score)),score.model,rotation=30,ha='right',fontsize=8);ax.set_ylabel('Outer coordinate holdout RMSE (ticks)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same 57 images / 19 poses; all repeats held out together; nested regularization')
    fig.savefig(OUT/'03_prediction_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,col,title in [(axes[0],'Y_error_ticks','Y'),(axes[1],'yaw_error_ticks','yaw')]:
        for name,color in [('QR6_nested','#6c8497'),('QR6_edges8_nested','#2d9a6c')]:
            g=pred[pred.model.eq(name)].groupby('coordinate_group')[col].mean()
            ax.plot(g.index,g,'o-',ms=4,c=color,label=name)
        ax.axhspan(-1,1,color='#a0d8b0',alpha=.2);ax.set_xlabel('Held coordinate ID');ax.set_ylabel('Mean error (ticks)');ax.set_title(title);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'04_coordinate_errors.png',dpi=170);plt.close(fig)


def numerical_checks():
    n,t=np.meshgrid(np.arange(-7,8,dtype=float),np.arange(-3,4,dtype=float),indexing='ij');n=n.ravel();t=t.ravel()
    errors={}
    for kind in ['step','ridge']:
        p=np.array([.37,1.1,80.,120.,.2,-.1]);A=model_value(p,n,t,kind,True);eps=1e-6
        N=np.column_stack([(model_value(p+np.eye(6)[j]*eps,n,t,kind)-model_value(p-np.eye(6)[j]*eps,n,t,kind))/(2*eps) for j in range(6)])
        error=float(np.linalg.norm(A-N)/np.linalg.norm(N));assert error<1e-7,error
        truth=model_value(p,n,t,kind)
        recovered=least_squares(lambda q:model_value(q,n,t,kind)-truth,p+[.2,.1,1,1,.1,.1],jac=lambda q:model_value(q,n,t,kind,True)).x
        assert np.max(abs(recovered-p))<1e-6
        errors[kind+'_jacobian_relative_error']=error
    return errors


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reuse',action='store_true');p.add_argument('--all-images',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    checks=numerical_checks();(OUT/'numerical_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    if not a.reuse:extract(second_only=not a.all_images)
    evaluate()
