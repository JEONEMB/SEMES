"""Train-only repeat-based subpixel phase nuisance model.

Experimental hypothesis, NOT proof that these errors are pixel locking.
Original D is never changed: a private model feature array preserves all630 IDs.
Image-only phase variables, no condition identity or label in predict.
"""
from pathlib import Path
import sys,numpy as np
G=Path(__file__).resolve().parent.parent/'joint_yt_disentangle_20261022'/'geometric'
sys.path.insert(0,str(G))
from geometric_models import RepeatGLS

def features(D):
    raw=D.reshape(len(D),-1);h=D[:,:,0].mean(axis=1,keepdims=True)/100
    return np.c_[raw,raw*h]

class PhaseNuisance:
    def __init__(self,config):self.config=dict(config)
    def phase(self,D):
        shift=D.mean(axis=1);p=shift[:,:1] if self.config['phase']=='x' else shift
        return np.c_[np.sin(2*np.pi*p),np.cos(2*np.pi*p)-1]
    def transform(self,D):
        d=np.asarray(D);d=d[None] if d.ndim==2 else d
        if self.config['enabled']:
            return d-(self.phase(d)@self.coef).reshape(d.shape)
        return d.copy()
    def fit(self,D,y,metadata):
        D=np.asarray(D);groups=metadata.cv_group.to_numpy();new=metadata.dataset.to_numpy()==20261020
        self.training_indices=metadata['index'].to_numpy()
        if self.config['enabled']:
            P=self.phase(D);shape=(D-D.mean(axis=1)[:,None]).reshape(len(D),-1);pp=[];ss=[]
            for group in sorted(set(groups[new])):
                ix=np.flatnonzero(new&(groups==group))
                if len(ix)>1:
                    pp.extend(P[ix]-P[ix].mean(axis=0));ss.extend(shape[ix]-shape[ix].mean(axis=0))
            pp=np.array(pp);ss=np.array(ss);n=max(len(pp),1)
            self.coef=np.linalg.solve(pp.T@pp/n+self.config['alpha']*np.eye(pp.shape[1]),pp.T@ss/n)
            # Keep mean translation unchanged; every structural point contributes.
            self.coef=self.coef.reshape(-1,630,2);self.coef-=self.coef.mean(axis=1)[:,None];self.coef=self.coef.reshape(-1,1260)
            self.nuisance_fit_rows=n
        self.model=RepeatGLS(shrink=.01,repeat_mix=0.,fit_mode='new_condition_means').fit(features(self.transform(D)),y,groups,new)
        return self
    def predict(self,D):return self.model.predict(features(self.transform(D)))

CONFIGS=[dict(enabled=False,phase='x',alpha=1.)]+[dict(enabled=True,phase=p,alpha=a) for p in ['x','xy'] for a in [.01,.1,1.,10.]]
