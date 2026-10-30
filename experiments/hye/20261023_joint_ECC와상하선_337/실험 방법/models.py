"""Train-only command calibration on QR/line observations.

Inference accepts observed features, and optionally known relative probe inputs.
It never receives the unknown initial command or the image filename.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'joint_yt_generalization_20261022'))
from forward_model import ForwardGLS,design

CONFIGS=[dict(degree=d,shrink=s,scope=scope) for d,scope in [(1,'all'),(2,'all'),(1,'mixed')] for s in [.01,.1]]

def feature_map(X):
    X=np.atleast_2d(X);shift=X[:,:54:2].mean(axis=1,keepdims=True)/100
    return np.c_[X,X*shift]

class ObservationModel:
    def __init__(self,config):self.config=dict(config)
    def fit(self,X,y,metadata):
        X=feature_map(X);y=np.asarray(y);dates=metadata.dataset.to_numpy();select=(dates==20261020) if self.config['scope']=='mixed' else np.ones(len(y),bool)
        dates=dates[select];weights=np.array([1/np.sum(dates==d) for d in dates])
        self.model=ForwardGLS(self.config['degree'],self.config['shrink']).fit(X[select],y[select],weights=weights)
        self.training_indices=metadata['index'].to_numpy()[select]
        return self
    def predict(self,X):return self.model.predict(feature_map(X))
    def predict_sequence(self,X,relative_inputs,bounded=False):
        """Joint fit to static images at known offsets; NOT a robot rollout."""
        X=np.atleast_2d(X);u=np.asarray(relative_inputs,dtype=float)
        if u.shape!=(len(X),2) or not np.allclose(u[0],0):raise ValueError('Cumulative offsets must start at zero')
        f=self.model;h=(feature_map(X)-f.origin)@f.projector;G=f.gram
        vals,V=np.linalg.eigh(G);S=(V*np.sqrt(np.maximum(vals,1e-12)))@V.T
        target=h@np.linalg.pinv(G)
        initial=f.predict(feature_map(X[:1]))[0]/f.scale;us=u/f.scale
        def residual(q):return ((design(q[None]+us,f.degree)[0]-target)@S).ravel()
        def jac(q):
            _,jy,jt=design(q[None]+us,f.degree)
            return np.stack([jy@S,jt@S],axis=-1).reshape(-1,2)
        bounds=(f.lower,f.upper) if bounded else (-np.inf,np.inf)
        if bounded:initial=np.clip(initial,f.lower+1e-10,f.upper-1e-10)
        result=least_squares(residual,initial,jac=jac,bounds=bounds,max_nfev=60,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        return result.x*f.scale,dict(success=bool(result.success),nfev=int(result.nfev),cost=float(result.cost),outside_training_margin=bool(np.any(result.x<f.lower)|np.any(result.x>f.upper)))
