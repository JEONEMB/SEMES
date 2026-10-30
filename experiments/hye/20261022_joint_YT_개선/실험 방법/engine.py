"""Fixed-ID observations -> weighted geometry -> joint regressor -> stabilization.
All train-dependent calculations live in fit_* functions; predict receives X only.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, BayesianRidge, ARDRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.covariance import LedoitWolf, OAS

H=Path(__file__).resolve().parent;B=H.parent
MEASUREMENTS=['uniform','photo','stability','contrast','huber','soft_l1','cauchy']
FEATURES=['global2','global1','global3','local1','local2','quadrants','homography']
POLICIES=['none','bootstrap_mean','bootstrap_median','subagging','noise','mixup','additive_mix','balanced']

def load_observations():
    a=np.load(B/'center_patch_v2_20261021/tracks.npz'); names=a['names'].astype(str);ids=a['ids'];points=a['points']
    c=pd.read_csv(B.parent/'analysis_grid_corners_20261019/out/corners_280.csv')
    def table(key):return c.pivot(index='file',columns='id',values=key).loc[names,ids].to_numpy()
    xy=np.stack([table('x'),table('y')],axis=-1)
    D=xy-xy[list(names).index('center.bmp')]
    q=np.stack([table('rmse'),table('window_stability_px'),table('contrast')],axis=-1)
    return dict(names=names,ids=ids,points=points,D=D,q=q,valid=table('valid').astype(bool))

def polynomial(points,degree,center=None):
    if center is None:u=(points[:,0]-1906)/486;v=(points[:,1]-1070)/122
    else:u=(points[:,0]-center)/100;v=(points[:,1]-1070)/100
    powers=[(i-j,j) for i in range(degree+1) for j in range(i+1)]
    return np.stack([u**i*v**j for i,j in powers],axis=1)

def aggregate(design,d,w,robust=None):
    ww=w.copy();coef=None
    for _ in range(8 if robust else 1):
        sw=np.sqrt(np.maximum(ww,.001));coef=np.linalg.lstsq(design*sw[:,None],d*sw[:,None],rcond=None)[0]
        if robust:
            r=np.linalg.norm(d-design@coef,axis=1)
            scale=max(float(1.4826*np.median(abs(r-np.median(r)))),.015)
            z=r/(2.5*scale)
            rw=np.minimum(1,1/np.maximum(z,1e-10)) if robust=='huber' else (1/np.sqrt(1+z*z) if robust=='soft_l1' else 1/(1+z*z))
            ww=np.maximum(w*rw,.001)
    return coef.ravel(),ww

def extract(points,d,q,measurement,feature):
    n=len(points);w=np.ones(n)
    if measurement=='photo':w=1/(1+(q[:,0]/10)**2)
    elif measurement=='stability':w=1/(1+(q[:,1]/.2)**2)
    elif measurement=='contrast':w=np.minimum(q[:,2]/100,2)**2/(1+(q[:,0]/10)**2+(q[:,1]/.2)**2)
    w=np.clip(w,.001,4);robust=measurement if measurement in ['huber','soft_l1','cauchy'] else None
    regs=np.argmin(abs(points[:,0,None]-[1644,1901,2159]),axis=1)
    if feature.startswith('global'):
        return aggregate(polynomial(points,int(feature[-1])),d,w,robust)[0]
    if feature.startswith('local'):
        return np.concatenate([aggregate(polynomial(points[regs==r],int(feature[-1]),[1644,1901,2159][r]),d[regs==r],w[regs==r],robust)[0] for r in range(3)])
    if feature=='quadrants':
        # Every point contributes with positive weight, no representative subset.
        chunks=[]
        for r in range(3):
            rr=regs==r;px=points[rr];dd=d[rr];qw=w[rr]
            _,rw=aggregate(polynomial(px,1,[1644,1901,2159][r]),dd,qw,robust)
            for lr in [False,True]:
                for tb in [False,True]:
                    sel=((px[:,0]>np.median(px[:,0]))==lr)&((px[:,1]>np.median(px[:,1]))==tb)
                    chunks.extend(np.average(dd[sel],axis=0,weights=rw[sel]))
        return np.array(chunks)
    if feature=='homography':
        xy=(points-[1906,1070])/[486,122];dest=(points+d-[1906,1070])/[486,122]
        u,v=xy.T;xx,yy=dest.T
        A=np.zeros((n,2,8));A[:,0,:3]=np.c_[u,v,np.ones(n)];A[:,1,3:6]=np.c_[u,v,np.ones(n)]
        A[:,0,6:]=-np.c_[xx*u,xx*v];A[:,1,6:]=-np.c_[yy*u,yy*v]
        ww=w.copy()
        for _ in range(8 if robust else 1):
            sw=np.repeat(np.sqrt(ww),2);co=np.linalg.lstsq(A.reshape(-1,8)*sw[:,None],dest.ravel()*sw,rcond=None)[0]
            if robust:
                err=np.linalg.norm((A@co)-dest,axis=1);scale=max(np.median(err)*1.4826,1e-5);z=err/(2.5*scale)
                rw=np.minimum(1,1/np.maximum(z,1e-10)) if robust=='huber' else (1/np.sqrt(1+z*z) if robust=='soft_l1' else 1/(1+z*z))
                ww=np.maximum(w*rw,.001)
        return co-np.array([1,0,0,0,1,0,0,0])
    raise ValueError(feature)

class Forward:
    def __init__(self,cov='shrink',quad=False,loss='linear'):self.cov=cov;self.quad=quad;self.loss=loss
    def design(self,y):return np.c_[np.ones(len(y)),y,y*y] if self.quad else np.c_[np.ones(len(y)),y]
    def fit(self,X,y):
        A=self.design(y);self.co=np.linalg.lstsq(A,X,rcond=None)[0];R=X-A@self.co
        std=np.maximum(np.sqrt(np.mean(R*R,axis=0)),1e-8);Z=R/std
        if self.cov=='ledoit':C=LedoitWolf(assume_centered=True).fit(Z).covariance_
        elif self.cov=='oas':C=OAS(assume_centered=True).fit(Z).covariance_
        elif self.cov=='diag':C=np.eye(X.shape[1])
        elif self.cov=='mad':
            std=np.maximum(np.median(abs(R-np.median(R,axis=0)),axis=0)*1.4826,1e-8);C=np.eye(X.shape[1])
        else:C=.999*(Z.T@Z/len(R))+.001*np.eye(X.shape[1])
        ee,U=np.linalg.eigh(C);self.W=((U/np.sqrt(np.maximum(ee,1e-8)))@U.T)/std[:,None]
        return self
    def predict(self,X):
        A=self.co@self.W;z=X@self.W-A[0];s=A[1:3]
        init=np.linalg.lstsq(s.T,z.T,rcond=None)[0].T
        if not self.quad and self.loss=='linear':return init
        out=[]
        for zz,start in zip(z,init):
            def fun(v):return v@s+(v*v)@A[3:5]-zz if self.quad else v@s-zz
            def jac(v):return (s+2*v[:,None]*A[3:5]).T if self.quad else s.T
            r=least_squares(fun,start,jac=jac,loss=self.loss,f_scale=1.5,max_nfev=80)
            if not r.success:raise RuntimeError('forward solver failed')
            out.append(r.x)
        return np.array(out)

def model_configs():
    c=[dict(family='forward',cov=cov,quad=False,loss='linear') for cov in ['shrink','diag','ledoit','oas','mad']]
    c += [dict(family='forward',cov='shrink',quad=True,loss=loss) for loss in ['linear','huber','soft_l1','cauchy']]
    c += [dict(family='ridge',alpha=a) for a in [.001,.1,10.]]
    c += [dict(family='bayes'),dict(family='ard')]
    c += [dict(family='pcr',components=n) for n in [2,4,8]]
    c += [dict(family='pls',components=n) for n in [2,4,8]]
    c += [dict(family='kernel',alpha=.01,gamma=g) for g in [.01,.1,1.]]
    c += [dict(family='svr',C=cc) for cc in [1.,10.,100.]]
    c += [dict(family='trees',leaf=n) for n in [1,3]]
    return c

def make_model(c,dim=1000):
    f=c['family']
    if f=='forward':return Forward(c['cov'],c['quad'],c['loss'])
    if f=='ridge':model=Ridge(alpha=c['alpha'])
    elif f=='bayes':model=MultiOutputRegressor(BayesianRidge(max_iter=400))
    elif f=='ard':model=MultiOutputRegressor(ARDRegression(max_iter=400))
    elif f=='pcr':return make_pipeline(StandardScaler(),PCA(n_components=min(c['components'],dim)),Ridge(alpha=.01))
    elif f=='pls':return make_pipeline(StandardScaler(),PLSRegression(n_components=min(c['components'],dim),max_iter=500,scale=False))
    elif f=='kernel':model=KernelRidge(alpha=c['alpha'],kernel='rbf',gamma=c['gamma'])
    elif f=='svr':model=MultiOutputRegressor(SVR(C=c['C'],epsilon=.005,kernel='rbf',gamma='scale'))
    elif f=='trees':model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=c['leaf'],max_features=1.,random_state=20261022,n_jobs=1)
    else:raise ValueError(f)
    return make_pipeline(StandardScaler(),model)

class Stabilized:
    def __init__(self,config,policy='none'):self.config=config;self.policy=policy
    def fit(self,X,y):
        rng=np.random.default_rng(20261022);self.members=[];n=len(X);pol=self.policy
        if pol in ['bootstrap_mean','bootstrap_median','subagging','balanced']:
            for _ in range(9):
                if pol=='subagging':ii=rng.choice(n,int(n*.8),replace=False)
                elif pol=='balanced':
                    # True training labels used only for resampling, never at inference.
                    yids=np.flatnonzero(abs(y[:,0])>0);tids=np.flatnonzero(abs(y[:,1])>0)
                    ii=np.r_[rng.choice(yids,n//2,replace=True),rng.choice(tids,n-n//2,replace=True)] if len(yids) and len(tids) else rng.choice(n,n)
                else:ii=rng.choice(n,n,replace=True)
                self.members.append(make_model(self.config,X.shape[1]).fit(X[ii],y[ii]))
            return self
        xx,yy=X,y
        if pol=='noise':
            A=np.c_[np.ones(n),y,y*y];r=X-A@np.linalg.lstsq(A,X,rcond=None)[0]
            scale=np.std(r,axis=0)
            xx=np.concatenate([X]+[X+rng.normal(size=X.shape)*scale*.5 for _ in range(3)]);yy=np.tile(y,(4,1))
        elif pol=='mixup':
            i=rng.integers(n,size=n*3);j=rng.integers(n,size=n*3);lam=rng.beta(.4,.4,size=(n*3,1))
            xx=np.r_[X,lam*X[i]+(1-lam)*X[j]];yy=np.r_[y,lam*y[i]+(1-lam)*y[j]]
        elif pol=='additive_mix':
            yi=np.flatnonzero(abs(y[:,0])>0);ti=np.flatnonzero(abs(y[:,1])>0)
            if len(yi) and len(ti):
                ii=rng.choice(yi,n*3);jj=rng.choice(ti,n*3)
                # Common origin estimated only from this fit's training coordinates.
                origin=np.linalg.lstsq(np.c_[np.ones(n),y,y*y],X,rcond=None)[0][0]
                xx=np.r_[X,X[ii]+X[jj]-origin];yy=np.r_[y,y[ii]+y[jj]]
        self.members=[make_model(self.config,X.shape[1]).fit(xx,yy)];return self
    def predict(self,X):
        p=np.stack([m.predict(X) for m in self.members])
        return np.median(p,axis=0) if self.policy=='bootstrap_median' else p.mean(axis=0)
