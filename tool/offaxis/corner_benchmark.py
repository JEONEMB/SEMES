"""Physical-rendering and held-point real-image benchmark of QR corner refiners.

Synthetic truth is known K[R|t] projection; real cross-fit errors are consistency,
not ground truth. No pose model is trained and no original data are modified.
"""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.analysis_packages'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.analysis_packages/mplconfig'))
import argparse
import hashlib
import html
import json
import time
import cv2
import numpy as np
import pandas as pd
import qrcode
from qrcode.util import QRData
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import fit_corner

BASE=ROOT/'docs/offaxis-eda'
OUT=BASE/'corner-benchmark'
DATA=ROOT/'data/SSAFY Test Image Set'
METHODS=['subpix','edge_lines','quadrant_model']
COLORS={'subpix':'#2475b0','edge_lines':'#e07924','quadrant_model':'#269568'}


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load(p):return cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_GRAYSCALE)


def project(H,points):
    a=np.column_stack([points,np.ones(len(points))])@H.T
    return a[:,:2]/a[:,2:]


def geometry():
    old=json.loads((BASE/'raw/reference.json').read_text())
    models=json.loads((BASE/'ppt-method-v2/reference.json').read_text())
    for m,qr in zip(models,old):
        info=qr['qr_model']
        code=qrcode.QRCode(version=1,error_correction=info['ec'],border=0,mask_pattern=info['mask'])
        code.add_data(QRData(qr['payload'],mode=info['encoding']),optimize=0);code.make(fit=False)
        m['bits']=np.array(code.get_matrix(),np.uint8)
        m['grid']=np.array([[f['col'],f['row']] for f in m['features']],float)
        m['seed']=np.array([f['seed'] for f in m['features']],float)
        m['basis']=np.array([f['basis'] for f in m['features']],float)
        m['quad']=np.array([f['bits'] for f in m['features']],float)
        m['is_l']=~np.array([f['diagonal'] for f in m['features']],bool)
    return models


def subpix(image,seeds):
    points=np.ascontiguousarray(seeds.reshape(-1,1,2),dtype=np.float32)
    result=cv2.cornerSubPix(image,points,(3,3),(-1,-1),
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,1e-5)).reshape(-1,2).astype(float)
    valid=np.isfinite(result).all(1)&(np.linalg.norm(result-seeds,axis=1)<3.6)
    return result,valid


def edge_lines(image,seeds,bases,quads,details=False):
    """Multiple signed edge profiles -> two fitted lines -> intersection.

    Inspired by edge/line refinement, not a reproduction of AprilTag/Devernay.
    Bilinear sampling of raw image; no smoothing/thresholded coordinate fitting.
    """
    n=len(seeds)
    flat=quads.reshape(n,4)
    minority=np.where((flat.sum(1)==1)[:,None],flat,1-flat)
    k=np.argmax(minority,axis=1)
    sx=np.where(k%2==1,1.,-1.);sy=np.where(k//2==1,1.,-1.)
    along=np.linspace(2.,4.8,9)
    normal=np.linspace(-2.5,2.5,21)
    uv=np.zeros((n,2,len(along),len(normal),2))
    uv[:,0,:,:,0]=sx[:,None,None]*along[None,:,None]
    uv[:,0,:,:,1]=normal[None,None,:]
    uv[:,1,:,:,0]=normal[None,None,:]
    uv[:,1,:,:,1]=sy[:,None,None]*along[None,:,None]
    coords=seeds[:,None,None,None,:]+np.einsum('nij,natkj->natki',bases,uv)
    sampled=map_coordinates(image.astype(float),[coords[...,1],coords[...,0]],order=1,mode='nearest')
    gradient=np.diff(sampled,axis=3)/.25
    # Orientation by known quadrant colors, independent of measured location.
    qi=quads
    signh=-np.where(sx>0,qi[:,1,1]-qi[:,0,1],qi[:,1,0]-qi[:,0,0])
    signv=-np.where(sy>0,qi[:,1,1]-qi[:,1,0],qi[:,0,1]-qi[:,0,0])
    signed=gradient*np.column_stack([signh,signv])[:,:,None,None]
    # 0.75px derivative pooling is an estimator operation, not input denoising.
    smooth=(signed[...,:-2]+2*signed[...,1:-1]+signed[...,2:])/4
    peak=np.argmax(smooth,axis=3)
    left=np.take_along_axis(smooth,np.maximum(peak-1,0)[...,None],axis=3)[...,0]
    mid=np.take_along_axis(smooth,peak[...,None],axis=3)[...,0]
    right=np.take_along_axis(smooth,np.minimum(peak+1,smooth.shape[3]-1)[...,None],axis=3)[...,0]
    delta=np.divide(.5*(left-right),left-2*mid+right,out=np.zeros_like(mid),where=abs(left-2*mid+right)>1e-9)
    delta=np.clip(delta,-.75,.75)
    # Use the integrated signed gradient rather than its strongest pixel.
    # The latter locks onto interpolation knots on nearly straight edges.
    # A low quantile estimates the weak derivative background from texture.
    background=np.quantile(signed,.10,axis=3,keepdims=True)
    weights_profile=np.maximum(signed-background,0)
    normal_mid=(normal[:-1]+normal[1:])/2
    locations=(weights_profile*normal_mid).sum(3)/np.maximum(weights_profile.sum(3),1e-12)
    ok=(peak>0)&(peak<smooth.shape[3]-1)&(mid>5)
    intercept=[];slope=[];residual=[]
    for arm,sgn in [(0,sx),(1,sy)]:
        t=sgn[:,None]*along
        weights=ok[:,arm].astype(float)*np.clip(mid[:,arm],0,100)
        values=locations[:,arm]
        # Reweighted regression rejects profile outliers, retaining the raw data.
        for _ in range(3):
            sw=np.maximum(weights.sum(1),1e-12)
            mt=(weights*t).sum(1)/sw;mv=(weights*values).sum(1)/sw
            b=(weights*(t-mt[:,None])*(values-mv[:,None])).sum(1)/np.maximum((weights*(t-mt[:,None])**2).sum(1),1e-12)
            a=mv-b*mt
            r=values-a[:,None]-b[:,None]*t
            weights*=np.minimum(1,.35/np.maximum(abs(r),1e-12))
        intercept.append(a);slope.append(b);residual.append(np.sqrt((weights*r*r).sum(1)/np.maximum(weights.sum(1),1e-12)))
    ah,av=intercept;bh,bv=slope
    determinant=1-bh*bv
    u=(av+bv*ah)/np.where(abs(determinant)>.1,determinant,np.nan)
    v=ah+bh*u
    result=seeds+np.einsum('nij,nj->ni',bases,np.column_stack([u,v]))
    valid=(ok.sum(2)>=5).all(1)&np.isfinite(result).all(1)&(np.linalg.norm(result-seeds,axis=1)<3.6)
    valid&=(np.maximum(*residual)<.5)&(abs(determinant)>.3)
    if details:return result,valid,dict(uv=uv,coords=coords,locations=locations,ok=ok,intercept=intercept,slope=slope)
    return result,valid


def match_free(image,seeds,bounds):
    x0,y0,x1,y1=bounds
    crop=np.ascontiguousarray(image[y0:y1,x0:x1])
    pts=cv2.goodFeaturesToTrack(crop,maxCorners=900,qualityLevel=.01,minDistance=3,blockSize=3)
    if pts is None:return np.empty((0,2)),np.full(len(seeds),np.nan),np.full(len(seeds),-1)
    pts=cv2.cornerSubPix(crop,pts,(3,3),(-1,-1),(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,1e-5)).reshape(-1,2)
    pts=pts+[x0,y0]
    dist=np.linalg.norm(seeds[:,None,:]-pts[None,:,:],axis=2)
    rows,cols=linear_sum_assignment(dist)
    assignment=np.full(len(seeds),-1,int);distance=np.full(len(seeds),np.nan)
    keep=dist[rows,cols]<3.
    assignment[rows[keep]]=cols[keep];distance[rows[keep]]=dist[rows[keep],cols[keep]]
    return pts,distance,assignment


def crossfit(grid,points,valid,ids):
    """Spatial 2x2 module blocks held out; no evaluated point fits its H."""
    fold=(grid[:,0].astype(int)//2+2*(grid[:,1].astype(int)//2))%5
    residual=np.full((len(grid),2),np.nan)
    for f in range(5):
        train=valid&(fold!=f);test=valid&(fold==f)
        if train.sum()<12 or test.sum()==0:continue
        H,inliers=cv2.findHomography(grid[train],points[train],cv2.RANSAC,.75,maxIters=2000,confidence=.995)
        if H is None:continue
        residual[test]=points[test]-project(H,grid[test])
    return residual


def make_real(models):
    meta=pd.read_csv(BASE/'ppt-method-v2/frames.csv').set_index('path')
    ec=pd.read_csv(BASE/'raw/ecc.csv').set_index(['path','qr'])
    cache=pd.read_csv(BASE/'ppt-method-v2/points.csv.gz')
    bypath={p:g.set_index(['qr','id']) for p,g in cache.groupby('path')}
    rows=[];free_rows=[];start=time.time()
    for frame,(path,info) in enumerate(meta.iterrows()):
        image=load(DATA/path)
        assert digest(DATA/path)==info.sha256,path
        for m in models:
            r=ec.loc[(path,m['slot'])]
            A=np.array([[r.a00,r.a01],[r.a10,r.a11]]);offset=np.array([r.a02,r.a12])
            seeds=m['seed']@A.T+offset
            bases=np.einsum('ij,njk->nik',A,m['basis'])
            bases/=np.linalg.norm(bases,axis=1)[:,None,:]
            ids=np.array([f['id'] for f in m['features']])
            cached=bypath[path].loc[[(m['slot'],i) for i in ids]]
            modelpoints=cached[['x','y']].to_numpy()
            modelvalid=np.isfinite(modelpoints).all(1)&(np.linalg.norm(modelpoints-seeds,axis=1)<3.6)
            sp,spvalid=subpix(image,seeds)
            edge=np.full_like(seeds,np.nan);edgevalid=np.zeros(len(seeds),bool)
            edge[m['is_l']],edgevalid[m['is_l']]=edge_lines(image,seeds[m['is_l']],bases[m['is_l']],m['quad'][m['is_l']])
            mn=np.floor(seeds.min(0)-20).astype(int);mx=np.ceil(seeds.max(0)+20).astype(int)
            pts,dist,assignment=match_free(image,seeds,(max(0,mn[0]),max(0,mn[1]),min(image.shape[1],mx[0]),min(image.shape[0],mx[1])))
            free_rows.append(dict(path=path,qr=m['slot'],detected_any_corner=len(pts),
                matched_L=int(((assignment>=0)&m['is_l']).sum()),candidate_L=int(m['is_l'].sum()),
                matched_X=int(((assignment>=0)&~m['is_l']).sum()),candidate_X=int((~m['is_l']).sum()),
                unmatched_detections=int(len(pts)-(assignment>=0).sum())))
            estimates={'subpix':(sp,spvalid),'edge_lines':(edge,edgevalid),'quadrant_model':(modelpoints,modelvalid)}
            for method,(p,valid) in estimates.items():
                for kind,mask in [('L',m['is_l']),('X',~m['is_l'])]:
                    if method=='edge_lines' and kind=='X':continue
                    rr=crossfit(m['grid'],p,valid&mask,ids)
                    for j in np.flatnonzero(mask):
                        rows.append(dict(path=path,category=info.category,qr=m['slot'],id=int(ids[j]),
                            col=m['grid'][j,0],row=m['grid'][j,1],method=method,kind=kind,
                            x=p[j,0],y=p[j,1],valid=bool(valid[j]),
                            cv_dx=rr[j,0],cv_dy=rr[j,1],cv_error_px=np.linalg.norm(rr[j]),
                            y_mm=info.y_mm,theta_deg=info.theta_deg))
            if path=='center.bmp' and m['slot']=='C':
                visualize_real(image,m,seeds,estimates,pts,assignment,bases)
        if (frame+1)%30==0:print(f'Real: {frame+1}/{len(meta)}; {time.time()-start:.1f}s',flush=True)
    real=pd.DataFrame(rows)
    real.to_csv(OUT/'real_points.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(free_rows).to_csv(OUT/'free_detection.csv',index=False)
    return real


def visualize_real(image,m,seeds,estimates,detected,assignment,bases):
    mn=np.floor(seeds.min(0)-18).astype(int);mx=np.ceil(seeds.max(0)+18).astype(int)
    fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for a in ax:a.imshow(image,cmap='gray',vmin=0,vmax=255);a.set_xlim(mn[0],mx[0]);a.set_ylim(mx[1],mn[1])
    ax[0].scatter(seeds[m['is_l'],0],seeds[m['is_l'],1],s=9,facecolors='none',edgecolors='#31ca72',label=f'L corners ({m["is_l"].sum()})')
    ax[0].scatter(seeds[~m['is_l'],0],seeds[~m['is_l'],1],s=16,c='#ff4260',marker='x',label='X virtual crossings')
    ax[0].legend(fontsize=8);ax[0].set_title('1. All angled QR-grid vertices')
    ax[1].scatter(detected[:,0],detected[:,1],s=9,facecolors='none',edgecolors='#10c7e2')
    ax[1].set_title(f'2. Free Shi-Tomasi detector: {len(detected)} corners')
    for method,(p,valid) in estimates.items():
        v=valid&m['is_l'];ax[2].scatter(p[v,0],p[v,1],s=7,c=COLORS[method],alpha=.65,label=method)
    ax[2].legend(fontsize=8);ax[2].set_title('3. L-corner refiners on raw pixels')
    fig.savefig(OUT/'01_real_candidates.png',dpi=170);plt.close(fig)
    # Fixed representative examples, selected by geometry not measured error.
    chosen=[np.flatnonzero(~m['is_l'])[4],np.flatnonzero(m['is_l'])[35],np.flatnonzero(m['is_l'])[80]]
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for a,j in zip(axes,chosen):
        x,y=seeds[j];a.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        a.set_xlim(x-6,x+6);a.set_ylim(y+6,y-6)
        a.scatter([x],[y],c='yellow',s=60,marker='+',label='grid seed (not truth)')
        for method,(p,valid) in estimates.items():
            if np.isfinite(p[j]).all():a.scatter(*p[j],s=30,facecolors='none',edgecolors=COLORS[method],label=method)
        a.set_title(f'{"L" if m["is_l"][j] else "X"} | row {m["grid"][j,1]:g}, col {m["grid"][j,0]:g}')
    axes[1].legend(fontsize=7,loc='upper right')
    fig.suptitle('Original pixels: gaps, coating texture, and competing point definitions')
    fig.savefig(OUT/'02_raw_corner_closeups.png',dpi=190);plt.close(fig)
    j=chosen[1]
    _,_,details=edge_lines(image,seeds[j:j+1],bases[j:j+1],m['quad'][j:j+1],True)
    fig,a=plt.subplots(figsize=(7,6),layout='constrained');a.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    x,y=seeds[j];a.set_xlim(x-7,x+7);a.set_ylim(y+7,y-7)
    for arm,color in [(0,'#ff8c23'),(1,'#29c9e2')]:
        uv=details['uv'][0,arm,:,0].copy()
        if arm==0:uv[:,1]=details['locations'][0,arm]
        else:uv[:,0]=details['locations'][0,arm]
        samples=seeds[j]+uv@bases[j].T
        a.scatter(samples[:,0],samples[:,1],s=35,c=color,label=f'edge {arm+1}: subpixel profiles')
        t=np.linspace(-1,6,50)
        aa=details['intercept'][arm][0];bb=details['slope'][arm][0]
        # Draw lines across the local patch irrespective of arm sign.
        t=np.linspace(-6,6,50)
        line=np.column_stack([t,aa+bb*t]) if arm==0 else np.column_stack([aa+bb*t,t])
        pp=seeds[j]+line@bases[j].T;a.plot(pp[:,0],pp[:,1],c=color,lw=1)
    point=estimates['edge_lines'][0][j];a.scatter(*point,marker='+',s=180,c='#ff3a5d',label='line intersection')
    a.legend(fontsize=9);a.set_title('Edge profiles -> two robust line fits -> corner')
    fig.savefig(OUT/'03_edge_intersection.png',dpi=170);plt.close(fig)


def physical_scene(bits,view,regime):
    rng=np.random.default_rng(7300+view)
    rotvec=np.array([rng.uniform(-.12,.12),rng.uniform(-.12,.12),rng.uniform(-.08,.08)])
    R=cv2.Rodrigues(rotvec)[0]
    K=np.array([[450.,0,128],[0,450.,128],[0,0,1.]])
    t=np.array([rng.uniform(-.12,.12),rng.uniform(-.12,.12),rng.uniform(48,54)])
    H=K@np.column_stack([R[:,:2],t-R[:,:2]@np.array([10.5,10.5])])
    # Known material pattern: rasterized at 64 samples per module, optionally
    # eroded by a 5x5 box (2/64 module retreat per exposed black edge).
    unit=64;border=3
    texture=np.pad(np.kron(bits,np.ones((unit,unit),np.uint8)),border*unit)
    retreat=0 if regime=='clean' else 2/unit
    if retreat:texture=cv2.erode(texture,np.ones((5,5),np.uint8))
    S=6;size=256
    yy,xx=np.mgrid[:size*S,:size*S]
    # Sensor integer coordinate is pixel center; integrate six subpixel samples.
    xy=np.column_stack([(xx.ravel()+.5)/S-.5,(yy.ravel()+.5)/S-.5])
    uv=project(np.linalg.inv(H),xy)
    sx=((uv[:,0]+border)*unit-.5).reshape(size*S,size*S).astype(np.float32)
    sy=((uv[:,1]+border)*unit-.5).reshape(size*S,size*S).astype(np.float32)
    source=220.-180.*texture.astype(np.float32)
    if regime=='gap_texture':
        # A fixed material-space reflectance texture is observed from all views.
        # It is not regenerated in camera coordinates after each movement.
        material_rng=np.random.default_rng(58217)
        field=material_rng.normal(size=source.shape).astype(np.float32)
        field=cv2.GaussianBlur(field,(0,0),6.)
        field/=field.std()
        gy,gx=np.mgrid[:source.shape[0],:source.shape[1]]
        source+=9*field+5*np.sin(.055*gx+.03*gy)
    raw=cv2.remap(source,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=220)
    # Optical blur is part of the forward simulation, not image preprocessing.
    sigmax=rng.uniform(.45,.9);sigmay=rng.uniform(.45,.9)
    raw=cv2.GaussianBlur(raw,(0,0),sigmax*S,sigmaY=sigmay*S)
    raw=raw.reshape(size,S,size,S).mean((1,3))
    if regime=='gap_texture':
        # Illustrative correlated reflectance; not a calibrated Teflon BRDF.
        gy,gx=np.mgrid[:size,:size]
        raw+=.025*(gx-128)
    raw+=rng.normal(0,1.5,raw.shape)
    image=np.clip(np.rint(raw),0,255).astype(np.uint8)
    return image,H,retreat,dict(K=K.tolist(),R=R.tolist(),t=t.tolist(),sigma_x_px=sigmax,sigma_y_px=sigmay)


def make_synthetic(models,views):
    m=models[1]
    # Fixed sampling spans QR; not selected by benchmark outcome.
    l=np.flatnonzero(m['is_l']);x=np.flatnonzero(~m['is_l'])
    chosen=np.r_[l[np.linspace(0,len(l)-1,48,dtype=int)],x[np.linspace(0,len(x)-1,24,dtype=int)]]
    grid=m['grid'][chosen];quads=m['quad'][chosen];isl=m['is_l'][chosen]
    rows=[];camera=[];start=time.time()
    for regime in ['clean','gap','gap_texture']:
        for view in range(views):
            image,H,retreat,params=physical_scene(m['bits'],view,regime)
            ideal=project(H,grid)
            physical=grid.copy()
            for j,b in enumerate(quads):
                if not isl[j]:continue
                minority=b if b.sum()==1 else 1-b
                rr,cc=np.unravel_index(np.argmax(minority),minority.shape)
                sign=np.array([1 if cc else -1,1 if rr else -1])
                physical[j]+=sign*retreat*(1 if b.sum()==1 else -1)
            truth=project(H,physical)
            near=project(H,grid+np.array([1.,0]))-ideal
            up=project(H,grid+np.array([0.,1]))-ideal
            bases=np.stack([near/np.linalg.norm(near,axis=1)[:,None],up/np.linalg.norm(up,axis=1)[:,None]],axis=2)
            rng=np.random.default_rng(9300+view)
            seeds=ideal+rng.uniform(-.45,.45,ideal.shape)
            sp,spvalid=subpix(image,seeds)
            edge=np.full_like(seeds,np.nan);ev=np.zeros(len(seeds),bool)
            edge[isl],ev[isl]=edge_lines(image,seeds[isl],bases[isl],quads[isl])
            mp=[];mv=[]
            for seed,b,basis in zip(seeds,quads,bases):
                r=fit_corner(image,seed,b,basis,4)
                mp.append([r['x'],r['y']]);mv.append(r['success'] and not r['boundary'])
            estimates={'subpix':(sp,spvalid),'edge_lines':(edge,ev),'quadrant_model':(np.array(mp),np.array(mv))}
            for method,(p,valid) in estimates.items():
                for j in range(len(grid)):
                    if method=='edge_lines' and not isl[j]:continue
                    rows.append(dict(regime=regime,view=view,id=int(m['features'][chosen[j]]['id']),
                        method=method,kind='L' if isl[j] else 'X',valid=bool(valid[j]),
                        x=p[j,0],y=p[j,1],truth_x=truth[j,0],truth_y=truth[j,1],
                        ideal_x=ideal[j,0],ideal_y=ideal[j,1],
                        error_px=float(np.linalg.norm(p[j]-truth[j])),
                        ideal_grid_error_px=float(np.linalg.norm(p[j]-ideal[j]))))
            camera.append(dict(regime=regime,view=view,retreat_module=retreat,H=H.tolist(),**params))
            if view==0:
                fig,ax=plt.subplots(figsize=(6,6),layout='constrained');ax.imshow(image,cmap='gray',vmin=0,vmax=255)
                ax.scatter(truth[isl,0],truth[isl,1],s=14,facecolors='none',edgecolors='#26e595',label='known projected L truth')
                ax.scatter(ideal[~isl,0],ideal[~isl,1],s=18,c='#ff3555',marker='x',label='virtual X grid point')
                ax.legend(fontsize=8);ax.set_title(f'Physical pinhole rendering: {regime}')
                fig.savefig(OUT/f'synthetic_{regime}.png',dpi=160);plt.close(fig)
            if (view+1)%8==0:print(f'Synthetic {regime}: {view+1}/{views}; {time.time()-start:.1f}s',flush=True)
    result=pd.DataFrame(rows);result.to_csv(OUT/'synthetic_points.csv',index=False)
    (OUT/'synthetic_cameras.json').write_text(json.dumps(camera,indent=2),encoding='utf-8')
    return result


def summarize(real,syn):
    results=[]
    for keys,g in syn.groupby(['regime','kind','method']):
        valid=g.valid&np.isfinite(g.error_px)
        e=g.loc[valid,'error_px']
        results.append(dict(regime=keys[0],kind=keys[1],method=keys[2],attempted=len(g),valid=int(valid.sum()),
            coverage=float(valid.mean()),median_px=float(e.median()),rms_px=float(np.sqrt(np.mean(e**2))),
            p95_px=float(e.quantile(.95)),within_01_all=float((valid&g.error_px.le(.1)).mean()),
            ideal_grid_rms_px=float(np.sqrt(np.mean(g.loc[valid,'ideal_grid_error_px']**2)))))
    ss=pd.DataFrame(results);ss.to_csv(OUT/'synthetic_summary.csv',index=False)
    results=[]
    # Common support on L points: valid in all three refiners and scored in all.
    l=real[real.kind.eq('L')]
    complete=l.pivot(index=['path','qr','id'],columns='method',values='cv_error_px').dropna()
    support=set(complete.index)
    real['common_L_support']=[(r.path,r.qr,r.id) in support and r.kind=='L' for r in real.itertuples()]
    for (kind,method),g in real.groupby(['kind','method']):
        for scope,h in [('all_valid',g),('common_L',g[g.common_L_support])]:
            if h.empty:continue
            e=h.loc[h.valid,'cv_error_px'].dropna()
            results.append(dict(kind=kind,method=method,scope=scope,attempted=len(h),valid=int(h.valid.sum()),
                scored=len(e),median_px=float(e.median()),rms_px=float(np.sqrt(np.mean(e**2))),p95_px=float(e.quantile(.95)),
                below_02_fraction=float((e<=.2).mean())))
    rs=pd.DataFrame(results);rs.to_csv(OUT/'real_summary.csv',index=False)
    # Frame/QR medians avoid presenting hundreds of correlated points as trials.
    real.groupby(['path','qr','kind','method'],as_index=False).agg(valid_fraction=('valid','mean'),
        median_cv_error_px=('cv_error_px','median'),p95_cv_error_px=('cv_error_px',lambda x:x.quantile(.95))).to_csv(OUT/'real_frame_summary.csv',index=False)
    # Repeatability: remove per-QR median motion inside a triplicate location.
    repeated=real[real.category.eq('mixed')&real.valid].copy()
    repeated['rep_mean_x']=repeated.groupby(['qr','id','method','y_mm','theta_deg']).x.transform('mean')
    repeated['rep_mean_y']=repeated.groupby(['qr','id','method','y_mm','theta_deg']).y.transform('mean')
    repeated['rx']=repeated.x-repeated.rep_mean_x;repeated['ry']=repeated.y-repeated.rep_mean_y
    repeated['rx']-=repeated.groupby(['path','qr','method']).rx.transform('median')
    repeated['ry']-=repeated.groupby(['path','qr','method']).ry.transform('median')
    repeated['repeat_shape_residual_px']=np.hypot(repeated.rx,repeated.ry)
    repeated.groupby(['method','kind']).agg(points=('id','count'),median_shape_residual_px=('repeat_shape_residual_px','median'),
        p95_shape_residual_px=('repeat_shape_residual_px',lambda a:a.quantile(.95))).to_csv(OUT/'repeatability.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,regime in zip(axes,['clean','gap','gap_texture']):
        for method in METHODS:
            e=syn[syn.regime.eq(regime)&syn.kind.eq('L')&syn.method.eq(method)&syn.valid].error_px.sort_values()
            ax.plot(e,np.arange(1,len(e)+1)/len(e),c=COLORS[method],label=method)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.set_xlabel('L physical-corner error (px)');ax.set_title(regime)
    axes[0].set_ylabel('Fraction of valid estimates');axes[-1].legend(fontsize=8)
    fig.suptitle('Known synthetic truth: identical views and seeds; conditional CDF (coverage reported separately)')
    fig.savefig(OUT/'05_synthetic_accuracy.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for method in METHODS:
        e=complete[method].sort_values()
        axes[0].plot(e,np.arange(1,len(e)+1)/len(e),c=COLORS[method],label=method)
    axes[0].set_xlim(0,1.5);axes[0].set_ylim(0,1);axes[0].set_xlabel('Held-point planar reprojection residual (px)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[0].set_title('Real L corners: common support; NOT absolute truth')
    for method in ['subpix','quadrant_model']:
        for kind,style in [('L','-'),('X','--')]:
            e=real[real.method.eq(method)&real.kind.eq(kind)&real.valid].cv_error_px.dropna().sort_values()
            axes[1].plot(e,np.arange(1,len(e)+1)/len(e),c=COLORS[method],ls=style,label=f'{method} {kind}')
    axes[1].set_xlim(0,1.5);axes[1].set_ylim(0,1);axes[1].grid(alpha=.2);axes[1].legend(fontsize=8);axes[1].set_xlabel('Held-point planar reprojection residual (px)');axes[1].set_title('L vs X: different locations; contextual comparison')
    fig.savefig(OUT/'06_real_consistency.png',dpi=170);plt.close(fig)
    # Point quality map from all frames, no display-based rejection.
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,method in zip(axes,METHODS):
        g=real[real.qr.eq('C')&real.kind.eq('L')&real.method.eq(method)].groupby(['col','row']).cv_error_px.median().reset_index()
        sc=ax.scatter(g.col,g.row,c=g.cv_error_px,s=22,cmap='magma',vmin=0,vmax=.75)
        ax.invert_yaxis();ax.set_aspect('equal');ax.set_title(method);ax.set_xlabel('module col')
    fig.colorbar(sc,ax=axes,label='Median held-point residual (px)');fig.suptitle('Real center QR: spatial consistency map (341 images)')
    fig.savefig(OUT/'07_spatial_quality.png',dpi=170);plt.close(fig)
    print(ss.to_string(index=False),flush=True);print(rs.to_string(index=False),flush=True)
    return ss,rs


def numerical_checks():
    grid=np.array([[0.,0.],[1,0],[0,1],[1,1]])
    H=np.array([[8,.2,30],[-.1,8.2,40],[.001,-.002,1]])
    points=project(H,grid)
    assert np.max(abs(project(np.linalg.inv(H),points)-grid))<1e-12
    # Rotation camera homography and direct 3D projection agree.
    R=cv2.Rodrigues(np.array([.1,-.05,.02]))[0];K=np.diag([450.,450.,1.]);t=np.array([0.,0.,50.])
    H=K@np.column_stack([R[:,:2],t])
    world=np.column_stack([grid,np.zeros(len(grid))]);cam=(world@R.T+t)@K.T
    assert np.max(abs(project(H,grid)-cam[:,:2]/cam[:,2:]))<1e-12
    return dict(project_inverse_max_error=1e-12,direct_3d_projection_agreement=1e-12)


def main():
    p=argparse.ArgumentParser();p.add_argument('--views',type=int,default=24);p.add_argument('--reuse-real',action='store_true');p.add_argument('--reuse-synthetic',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1);cv2.setRNGSeed(251026)
    checks=numerical_checks();models=geometry();start=time.time()
    print('Candidates:',[(m['slot'],int(m['is_l'].sum()),int((~m['is_l']).sum())) for m in models],flush=True)
    real=pd.read_csv(OUT/'real_points.csv.gz') if a.reuse_real else make_real(models)
    syn=pd.read_csv(OUT/'synthetic_points.csv') if a.reuse_synthetic else make_synthetic(models,a.views)
    summarize(real,syn)
    sources=[BASE/'raw/reference.json',BASE/'ppt-method-v2/reference.json',BASE/'raw/ecc.csv',BASE/'ppt-method-v2/points.csv.gz',BASE/'ppt-method-v2/frames.csv',Path(__file__),Path(__file__).with_name('extract.py')]
    manifest=dict(seconds=time.time()-start,checks=checks,original_frames=341,synthetic_views_per_regime=a.views,
        synthetic_seed=7300,seed_jitter_seed=9300,raw_preprocessing='none; gradients/profile sampling are estimator operations',
        real_truth_available=False,camera_calibration_available=False,teflon='User reports coating; correlated texture is illustrative, not measured BRDF',
        synthetic_pixel_supersampling=6,synthetic_pattern_samples_per_module=64,
        versions=dict(numpy=np.__version__,opencv=cv2.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
        source_sha256={str(f.relative_to(ROOT)):digest(f) for f in sources})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Complete',manifest['seconds'],flush=True)


if __name__=='__main__':main()
