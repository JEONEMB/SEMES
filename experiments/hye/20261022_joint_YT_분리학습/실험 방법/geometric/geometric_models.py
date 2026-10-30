"""Geometric contrasts and repeat-noise-aware two-axis forward calibration."""
import numpy as np

def geometry_features(D,points,regions,kind):
    D=np.asarray(D);D=D[None] if D.ndim==2 else D
    raw=D.reshape(len(D),-1);shift=D.mean(axis=1);h=shift[:,0:1]/100
    if kind=='raw':return raw
    if kind=='raw_shift':return np.c_[raw,raw*h]
    if kind in ['shape','shape_shift']:
        shape=(D-shift[:,None]).reshape(len(D),-1)
        return shape if kind=='shape' else np.c_[shape,shape*h]
    # Translation-invariant central point moments. Every fixed point contributes.
    features=[]
    for mask in [np.ones(len(points),bool)]+[regions==k for k in range(3)]:
        p=points[mask];center=p.mean(0);base=(p-center)/100
        dest=(p[None]+D[:,mask]);dest=(dest-dest.mean(1)[:,None])/100
        if kind=='moments':
            for degree in [2,3,4]:
                for power in range(degree+1):
                    features.append(np.mean(dest[:,:,0]**power*dest[:,:,1]**(degree-power),axis=1)-np.mean(base[:,0]**power*base[:,1]**(degree-power)))
        else:
            # Symmetric local strain and antisymmetric infinitesimal rotation.
            A=np.c_[np.ones(len(p)),base];coef=np.einsum('ap,npd->nad',np.linalg.pinv(A),D[:,mask]/100)
            u,v=coef[:,1],coef[:,2]
            features.extend([u[:,0],v[:,1],(u[:,1]+v[:,0])/2,(u[:,1]-v[:,0])/2])
    shape=np.array(features).T
    return np.c_[shift/100,shape] if kind in ['moments','strain'] else shape

class RepeatGLS:
    def __init__(self,shrink=.1,repeat_mix=1.,fit_mode='all'):
        self.shrink=shrink;self.repeat_mix=repeat_mix;self.fit_mode=fit_mode
    def fit(self,X,y,groups,new):
        X=np.asarray(X);y=np.asarray(y);groups=np.asarray(groups);new=np.asarray(new)
        group_names=sorted(set(groups[new]));gx=np.array([X[(groups==g)&new].mean(0) for g in group_names]);gy=np.array([y[(groups==g)&new].mean(0) for g in group_names])
        if self.fit_mode=='new_condition_means':
            coef=np.linalg.lstsq(np.c_[np.ones(len(gy)),gy],gx,rcond=None)[0]
        elif self.fit_mode=='axis_pairs':
            slopes=[]
            for axis in range(2):
                dx=[];dy=[]
                for i in range(len(gy)):
                    for j in range(i):
                        if abs(gy[i,1-axis]-gy[j,1-axis])<1e-9 and abs(gy[i,axis]-gy[j,axis])>1e-8:
                            dx.append(gx[i]-gx[j]);dy.append(gy[i,axis]-gy[j,axis])
                dx=np.array(dx);dy=np.array(dy);assert len(dy)>1
                slopes.append((dy@dx)/(dy@dy))
            slopes=np.array(slopes);origin=np.mean(gx-gy@slopes,axis=0);coef=np.r_[origin[None],slopes]
        else:coef=np.linalg.lstsq(np.c_[np.ones(len(y)),y],X,rcond=None)[0]
        self.origin=coef[0];self.slope=coef[1:]
        model_res=X-np.c_[np.ones(len(y)),y]@coef
        repeat=[];degrees=0
        for g in group_names:
            xx=X[(groups==g)&new]
            if len(xx)>1:repeat.extend(xx-xx.mean(0));degrees+=len(xx)-1
        repeat=np.array(repeat)
        # R.T@R approximates covariance; repeat degrees count separately.
        rr=model_res/np.sqrt(len(X));rrep=repeat/np.sqrt(max(degrees,1))
        R=np.r_[rr*np.sqrt(1-self.repeat_mix),rrep*np.sqrt(self.repeat_mix)]
        self.std=np.maximum(np.sqrt(np.sum(R*R,axis=0)),1e-6)
        Z=R/self.std;V=(self.slope/self.std).T;a=self.shrink;b=1-a
        Q=(V-b*Z.T@np.linalg.solve(a*np.eye(len(Z))+b*Z@Z.T,Z@V))/a
        self.beta=Q@np.linalg.pinv((self.slope/self.std)@Q)/self.std[:,None]
        self.response=self.slope@self.beta
        self.repeat_dof=degrees
        return self
    def predict(self,X):return (np.atleast_2d(X)-self.origin)@self.beta
