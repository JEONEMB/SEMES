"""QR-relative background horizontal boundaries; raw model search, no Canny.

First find a long consistent step in a broad QR-relative band using direct
least-squares step models. Refine raw pixel strips with continuous parameters.
Reference line identities are then tracked with an image-derived soft prior.
No equipment-specific plate boundary or absolute row is supplied.
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
from plate_edges import model_value,fit_ridge,predict,tune,TICKS,numerical_checks

BASE=ROOT/'docs/offaxis-eda'
OUT=BASE/'common-background-edges'
DATA=ROOT/'data/SSAFY Test Image Set'
REGIONS=['upper_left','upper_right','lower_left','lower_right']


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def qr_geometry(points,width,height):
    bounds=points.groupby('qr').agg(x0=('x','min'),x1=('x','max'),y0=('y','min'),y1=('y','max'))
    qw=float(np.median(bounds.x1-bounds.x0));qh=float(np.median(bounds.y1-bounds.y0))
    left_end=float(bounds.x0.min()-1.7*qw);right_start=float(bounds.x1.max()+1.7*qw)
    left=np.linspace(.065*width,min(left_end,.45*width),30)
    right=np.linspace(max(right_start,.55*width),.935*width,30)
    top=float(bounds.y0.min());bottom=float(bounds.y1.max())
    bands={'upper':(max(20,top-2.7*qh),top-.35*qh),
           'lower':(bottom+.05*qh,min(height-20,bottom+1.25*qh))}
    return dict(left=left,right=right,bands=bands,qr_width=qw,qr_height=qh,
        qr_center=np.array([(bounds.x0.min()+bounds.x1.max())/2,(top+bottom)/2]))


def candidates(image,x,band,polarity,prior=None):
    """Fit background+step bases at candidate positions; never image gradients."""
    lo,hi=map(int,band);lo=max(10,lo);hi=min(image.shape[0]-10,hi)
    x=int(round(x));stripe=image[lo-7:hi+8,x-3:x+4].astype(float)
    profile=stripe.mean(1)
    centers=np.arange(lo,hi+1,2)
    n=np.arange(-7,8,dtype=float)
    windows=profile[(centers[:,None]-lo+7)+n.astype(int)[None,:]]
    null=np.column_stack([np.ones(len(n)),n]);nullp=np.linalg.pinv(null)
    nullfit=(windows@nullp.T)@null.T
    null_sse=np.mean((windows-nullfit)**2,axis=1)
    scores=np.full(len(centers),-np.inf);best_sigma=np.zeros(len(centers));amps=np.zeros(len(centers))
    for sigma in [.65,1.2,2.2,3.5]:
        F=.5*(1+erf(n/(sigma*np.sqrt(2))))
        X=np.column_stack([np.ones(len(n)),n,F]);coef=windows@np.linalg.pinv(X).T
        sse=np.mean((windows-coef@X.T)**2,axis=1)
        gain=np.maximum(null_sse-sse,0)
        score=np.log1p(gain/(sse+2.25))*np.sqrt(np.maximum(abs(coef[:,2]),0))
        score[polarity*coef[:,2]<5]=-np.inf
        if prior is not None:
            # Soft identity prior around a line detected from the reference,
            # shifted using current QR geometry. Not a fixed pixel row.
            score-=.5*((centers-prior)/max(12.,.10*(hi-lo)))**2
        update=score>scores;scores[update]=score[update];best_sigma[update]=sigma;amps[update]=coef[update,2]
    found=[]
    for idx in np.argsort(scores)[::-1]:
        if not np.isfinite(scores[idx]):continue
        y=float(centers[idx])
        if any(abs(y-c['y'])<10 for c in found):continue
        found.append(dict(x=float(x),y=y,score=float(scores[idx]),sigma=float(best_sigma[idx]),amp=float(amps[idx])))
        if len(found)==4:break
    return found


def consistent_line(groups,width):
    """Numerical candidate consensus across both exposed sides, initialization only."""
    rng=np.random.default_rng(271027)
    xs=np.array([g[0]['x'] for g in groups]);N=len(groups)
    ys=np.full((N,4),np.nan);scores=np.zeros((N,4))
    for i,g in enumerate(groups):
        for j,c in enumerate(g):ys[i,j]=c['y'];scores[i,j]=max(c['score'],.01)
    left=np.flatnonzero(xs<width/2);right=np.flatnonzero(xs>width/2)
    best_score=-np.inf;best=None
    for _ in range(1400):
        i=int(rng.choice(left));j=int(rng.choice(right))
        a=int(rng.integers(len(groups[i])));b=int(rng.integers(len(groups[j])))
        slope=(ys[j,b]-ys[i,a])/(xs[j]-xs[i])
        if abs(slope)>.06:continue
        intercept=ys[i,a]-slope*(xs[i]-width/2)
        dist=abs(ys-(intercept+slope*(xs[:,None]-width/2)))
        agreement=scores*np.exp(-.5*(dist/2.)**2)
        agreement[~np.isfinite(dist)]=0
        value=float(agreement.max(1).sum())
        if value>best_score:best_score=value;best=(intercept,slope)
    if best is None:raise ValueError('No horizontal line consensus')
    return best


def refine(image,x,y,polarity):
    x=int(round(x));cy=int(round(y))
    nn=np.arange(cy-7,cy+8);tt=np.arange(x-3,x+4)
    patch=image[np.ix_(nn,tt)].astype(float)
    n=np.repeat(nn-cy,len(tt)).astype(float);t=np.tile(tt-x,len(nn)).astype(float)
    amp=polarity*max(10.,polarity*float(patch[-3:].mean()-patch[:3].mean()))
    init=[float(y-cy),1.,amp,float(patch[:3].mean()),0.,0.]
    lower=[-4.,.15,1. if polarity>0 else -300.,-50.,-12.,-12.]
    upper=[4.,5.,300. if polarity>0 else -1.,310.,12.,12.]
    opt=least_squares(lambda p:model_value(p,n,t,'step')-patch.ravel(),init,
        jac=lambda p:model_value(p,n,t,'step',True),bounds=(lower,upper),loss='soft_l1',f_scale=3.,
        max_nfev=55,ftol=1e-7,xtol=1e-7,gtol=1e-6)
    p=opt.x;rmse=float(np.sqrt(np.mean(opt.fun**2)))
    valid=bool(opt.success and abs(p[0])<3.9 and p[1]<4.9 and abs(p[2])>5 and rmse<15)
    return dict(x=float(x),y=float(cy+p[0]),valid=valid,sigma=float(p[1]),contrast=float(p[2]),rmse=rmse,
        n=n,t=t,raw=patch.ravel(),parameters=p.tolist(),base_y=cy)


def polynomial(samples,anchor,scale,degree):
    x=np.array([p['x'] for p in samples]);y=np.array([p['y'] for p in samples])
    valid=np.array([p['valid'] for p in samples]);quality=np.array([p['rmse'] for p in samples])
    if valid.sum()<12:return np.full(degree+1,np.nan),np.full(len(x),np.nan)
    X=np.column_stack([((x-anchor)/scale)**k for k in range(degree+1)])
    w=valid.astype(float)/np.maximum(quality,1.5)**2
    for _ in range(10):
        coef=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        residual=y-X@coef
        w=valid.astype(float)/np.maximum(quality,1.5)**2*np.minimum(1,.5/np.maximum(abs(residual),1e-12))
    return coef,residual


def extract():
    meta=pd.read_csv(BASE/'ppt-method-v2/frames.csv').set_index('path')
    subset=meta[meta.category.isin(['mixed','new-reference'])]
    pts=pd.read_csv(BASE/'ppt-method-v2/points.csv.gz',usecols=['path','qr','diagonal','x','y'])
    pts=pts[~pts.diagonal];pointmap={p:g for p,g in pts.groupby('path')}
    paths=['기준 위치/center1.bmp']+[p for p in subset.index if p!='기준 위치/center1.bmp']
    ref=None;features=[];all_samples=[];roirows=[];start=time.time();hashes={}
    for frame,path in enumerate(paths):
        file=DATA/path;digest=sha(file);assert digest==meta.loc[path,'sha256'];hashes[path]=digest
        image=cv2.imdecode(np.fromfile(file,np.uint8),cv2.IMREAD_GRAYSCALE);h,w=image.shape
        geom=qr_geometry(pointmap[path],w,h)
        feat=dict(path=path);regions={};examples={};all_candidates={}
        for edge,polarity in [('upper',1),('lower',-1)]:
            columns=np.r_[geom['left'],geom['right']]
            groups=[]
            for x in columns:
                prior=None
                if ref is not None:
                    oldline=ref[edge];dx,dy=geom['qr_center']-ref['qr_center']
                    prior=oldline[0]+oldline[1]*(x-dx-w/2)+dy
                group=candidates(image,x,geom['bands'][edge],polarity,prior)
                if not group:raise ValueError(f'No numerical step candidate: {path} {edge} {x}')
                groups.append(group)
            initial=consistent_line(groups,w)
            samples=[]
            for x in columns:
                y=initial[0]+initial[1]*(x-w/2)
                samples.append(refine(image,x,y,polarity))
            line,residual=polynomial(samples,w/2,w/2,1)
            quad,qres=polynomial(samples,w/2,w/2,2)
            feat[edge+'_position']=line[0];feat[edge+'_slope']=line[1]
            for j,value in enumerate(quad):feat[edge+f'_quadratic_{j}']=value
            feat[edge+'_valid_samples']=sum(p['valid'] for p in samples)
            feat[edge+'_line_rms']=float(np.sqrt(np.mean(residual[np.array([p['valid'] for p in samples])&np.isfinite(residual)]**2)))
            for side,part in [('left',samples[:30]),('right',samples[30:])]:
                local,rr=polynomial(part,w*(.25 if side=='left' else .75),w/4,1)
                feat[edge+'_'+side+'_position']=local[0];feat[edge+'_'+side+'_slope']=local[1]
                regions[edge+'_'+side]=part
            examples[edge]=samples[10]
            all_candidates[edge]=groups
            for i,(s,r) in enumerate(zip(samples,residual)):
                all_samples.append(dict(path=path,edge=edge,side='left' if i<30 else 'right',sample=i,
                    x=s['x'],y=s['y'],valid=s['valid'],sigma=s['sigma'],contrast=s['contrast'],pixel_rmse=s['rmse'],line_residual=r))
            if ref is None:
                if 'newref' not in locals():newref={}
                newref[edge]=(line[0],line[1]/(w/2))
        if ref is None:
            ref=dict(**newref,qr_center=geom['qr_center'])
            illustrate(image,feat,regions,examples,geom,all_candidates)
            candidate_rows=[]
            for edge,groups in all_candidates.items():
                for i,g in enumerate(groups):
                    candidate_rows.extend(dict(edge=edge,column=i,**c) for c in g)
            pd.DataFrame(candidate_rows).to_csv(OUT/'reference_candidates.csv',index=False)
        roirows.append(dict(path=path,left_start=geom['left'][0],left_end=geom['left'][-1],right_start=geom['right'][0],right_end=geom['right'][-1],
            upper_y_min=geom['bands']['upper'][0],upper_y_max=geom['bands']['upper'][1],
            lower_y_min=geom['bands']['lower'][0],lower_y_max=geom['bands']['lower'][1]))
        features.append(feat)
        if (frame+1)%10==0:print(f'Common edges {frame+1}/60; {time.time()-start:.1f}s',flush=True)
    pd.DataFrame(features).to_csv(OUT/'edge_features.csv',index=False)
    pd.DataFrame(all_samples).to_csv(OUT/'edge_samples.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(roirows).to_csv(OUT/'dynamic_search_regions.csv',index=False)
    (OUT/'extraction_manifest.json').write_text(json.dumps(dict(original_image_sha256=hashes,seconds=time.time()-start,
        Canny=False,gradient_detector=False,manual_absolute_y=False,qr_relative_bands={'upper':['QR_top-2.7*QR_height','QR_top-.35*QR_height'],
        'lower':['QR_bottom+.05*QR_height','QR_bottom+1.25*QR_height']},
        description='Direct least-squares step-model candidate search; line consensus; continuous raw-pixel robust step fitting; soft reference identity prior.'),indent=2),encoding='utf-8')


def illustrate(image,feat,regions,examples,geom,all_candidates):
    h,w=image.shape;fig,ax=plt.subplots(figsize=(15,8),layout='constrained')
    ax.imshow(image,cmap='gray',vmin=0,vmax=255)
    colors={'upper_left':'#dd7b23','upper_right':'#e6a72e','lower_left':'#248eb8','lower_right':'#35b29a'}
    for name,samples in regions.items():
        ax.scatter([s['x'] for s in samples],[s['y'] for s in samples],s=12,c=colors[name],label=name)
    ax.set_ylim(1500,350);ax.set_xlim(0,w);ax.legend(fontsize=9);ax.set_title('Common background boundaries: raw-pixel continuous fits; central fixture excluded')
    fig.savefig(OUT/'01_common_edges.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,(edge,p) in zip(axes,examples.items()):
        ax.scatter(p['n'],p['raw'],s=9,alpha=.5,c='#8596a3',label='Raw pixels')
        n=np.linspace(p['n'].min(),p['n'].max(),500)
        ax.plot(n,model_value(p['parameters'],n,np.zeros(len(n)),'step'),c='#de7d28',lw=2,label='Continuous numerical fit')
        ax.axvline(p['y']-p['base_y'],ls='--',c='#243f50',label='Subpixel location')
        ax.set_title(f'{edge}: y={p["y"]:.5f}px');ax.set_xlabel('Normal offset (px)');ax.set_ylabel('Intensity');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'02_numeric_profiles.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,6),layout='constrained')
    for ax,(edge,groups) in zip(axes,all_candidates.items()):
        for g in groups:ax.scatter([c['x'] for c in g],[c['y'] for c in g],s=[max(4,c['score']*2) for c in g],c='#9fabb3',alpha=.6)
        xs=np.r_[geom['left'],geom['right']]
        ax.plot(xs,feat[edge+'_position']+feat[edge+'_slope']*(xs-w/2)/(w/2),c='#d9672a',label='Selected consistent line')
        ax.set_ylim(*geom['bands'][edge][::-1]);ax.set_xlim(0,w);ax.set_title(edge+' broad QR-relative search');ax.set_xlabel('Image x (px)');ax.set_ylabel('Candidate y (px)');ax.legend()
    fig.savefig(OUT/'03_broad_model_search.png',dpi=170);plt.close(fig)


def evaluate():
    meta=pd.read_csv(BASE/'ppt-method-v2/frames.csv').set_index('path')
    mixed=meta[meta.category.eq('mixed')].sort_index();refs=meta[meta.category.eq('new-reference')]
    actual=mixed[['y_actual','theta_actual']].to_numpy()-refs[['y_actual','theta_actual']].mean().to_numpy()
    nominal=mixed[['y_mm','theta_deg']].to_numpy();assert np.max(abs(actual-nominal))<1e-5
    y=actual/TICKS;coords=np.unique(nominal,axis=0);groups=np.array([np.flatnonzero(np.all(coords==q,axis=1))[0] for q in nominal])
    qr=pd.read_csv(BASE/'ppt-method-v2/qr_displacements.csv').pivot(index='path',columns='qr',values=['dx','dy'])
    qrcols=[(d,s) for s in 'LCR' for d in ['dx','dy']];qx=qr.loc[mixed.index,qrcols].to_numpy()
    ef=pd.read_csv(OUT/'edge_features.csv').set_index('path')
    shared=[e+'_'+c for e in ['upper','lower'] for c in ['position','slope']]
    segments=[e+'_'+c for e in REGIONS for c in ['position','slope']]
    quadratic=[e+f'_quadratic_{j}' for e in ['upper','lower'] for j in range(3)]
    def values(cols):
        z=ef.loc[mixed.index,cols].to_numpy();r=ef.loc['기준 위치/center1.bmp',cols].to_numpy(dtype=float)
        return z-np.where(np.isfinite(r),r,0)
    sx=values(shared);tx=values(segments);cx=values(quadratic)
    sets={'QR6':qx,'common_shared4':sx,'QR6_common_shared4':np.c_[qx,sx],
          'QR6_common_segments8':np.c_[qx,tx],'QR6_common_quadratic6':np.c_[qx,cx]}
    scores=[];rows=[];selection=[];models={};blocks=[]
    for name,x in sets.items():
        held=np.zeros_like(y)
        for g in range(19):
            test=groups==g;train=~test;alpha=tune(x[train],y[train],groups[train])[0]
            held[test]=predict(fit_ridge(x[train],y[train],alpha),x[test])
            selection.append(dict(model=name,held_group=g,alpha=alpha))
        e=held-y
        scores.append(dict(model=name,images=57,Y_rmse_mm=np.sqrt(np.mean(e[:,0]**2))*.05,
            yaw_rmse_deg=np.sqrt(np.mean(e[:,1]**2))*.01,Y_rmse_ticks=np.sqrt(np.mean(e[:,0]**2)),
            yaw_rmse_ticks=np.sqrt(np.mean(e[:,1]**2)),both_within_tick=float(np.mean(np.max(abs(e),axis=1)<=1)),
            edge_input_missing_images=int((~np.isfinite(x).all(1)).sum())))
        for i,path in enumerate(mixed.index):rows.append(dict(model=name,path=path,group=groups[i],
            y_mm=actual[i,0],yaw_deg=actual[i,1],predicted_y_mm=held[i,0]*.05,predicted_yaw_deg=held[i,1]*.01,
            Y_error_ticks=e[i,0],yaw_error_ticks=e[i,1]))
        alpha=tune(x,y,groups)[0];m=fit_ridge(x,y,alpha)
        models[name]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()}
        print(scores[-1],flush=True)
        if name in ['QR6','QR6_common_shared4','QR6_common_segments8']:
            bh=np.zeros_like(y)
            for Y in np.unique(nominal[:,0]):
                test=nominal[:,0]==Y;train=~test;alpha=tune(x[train],y[train],groups[train])[0]
                bh[test]=predict(fit_ridge(x[train],y[train],alpha),x[test])
            be=bh-y;blocks.append(dict(model=name,Y_rmse_ticks=np.sqrt(np.mean(be[:,0]**2)),yaw_rmse_ticks=np.sqrt(np.mean(be[:,1]**2)),
                both_within_tick=float(np.mean(np.max(abs(be),axis=1)<=1))))
    pd.DataFrame(scores).to_csv(OUT/'comparison.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'coordinate_holdout.csv',index=False)
    pd.DataFrame(selection).to_csv(OUT/'outer_alpha_selection.csv',index=False);pd.DataFrame(blocks).to_csv(OUT/'Y_sweep_holdout.csv',index=False)
    (OUT/'models.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    rep=[]
    for col in shared+segments:
        z=ef.loc[mixed.index,col];r=z.to_numpy()-pd.Series(z.to_numpy()).groupby(groups).transform('mean').to_numpy()
        rep.append(dict(feature=col,valid_images=int(z.notna().sum()),repeat_sigma_px=float(np.sqrt(np.nansum(r*r)/38))))
    pd.DataFrame(rep).to_csv(OUT/'repeatability.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained');score=pd.DataFrame(scores)
    for ax,col,title in [(axes[0],'Y_rmse_ticks','Y'),(axes[1],'yaw_rmse_ticks','yaw')]:
        ax.bar(range(len(score)),score[col],color=['#8193a2','#dc9c43','#2c9d6d','#3686b6','#a966a7'])
        ax.set_xticks(range(len(score)),score.model,rotation=25,ha='right',fontsize=8);ax.set_title(title);ax.set_ylabel('Held-coordinate RMSE (ticks)');ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'04_prediction_comparison.png',dpi=170);plt.close(fig)
    (OUT/'manifest.json').write_text(json.dumps(dict(date='2026-10-27',original_images_modified=False,
        reference_identity='Detected automatically in new center1; soft image-based tracking prior in other frames',
        user_edges='Upper groove and lower panel boundary on exposed left/right background; no central equipment edges',
        primary_comparison='QR6 vs QR6_common_shared4',secondary_models=['QR6_common_segments8','QR6_common_quadratic6'],
        labels='Second-data Actual record relative to valid new reference mean',outer_folds=19,inner_folds=18,
        repeated_coordinate_leakage=False,new_images_requested=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('plate_edges.py'),
            BASE/'ppt-method-v2/frames.csv',BASE/'ppt-method-v2/points.csv.gz',BASE/'ppt-method-v2/qr_displacements.csv',OUT/'edge_features.csv']}),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reuse',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    (OUT/'numerical_checks.json').write_text(json.dumps(numerical_checks(),indent=2),encoding='utf-8')
    if not a.reuse:extract()
    evaluate()
