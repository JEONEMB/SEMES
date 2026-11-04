"""Coordinate-only joint Y/T model. Inference never reads filename or trial labels."""
import numpy as np
from forward_model import ForwardGLS

def feature_map(D,kind):
    d=np.asarray(D);d=d[None] if d.ndim==2 else d
    raw=d.reshape(len(d),-1);h=d[:,:,0].mean(axis=1,keepdims=True)/100
    if kind=='raw1260':return raw
    if kind=='raw_shift':return np.c_[raw,raw*h]
    raise ValueError(kind)

class JointRegressor:
    def __init__(self,config):self.config=dict(config)
    def fit(self,D,y):
        X=feature_map(D,self.config['feature']);self.shifts=np.asarray(D)[:,:,0].mean(axis=1)
        if self.config['family']=='global':self.model=ForwardGLS(1,self.config['shrink']).fit(X,y)
        else:self.X=X.copy();self.y=np.array(y,copy=True)
        return self
    def predict(self,D):
        D=np.asarray(D);D=D[None] if D.ndim==2 else D;X=feature_map(D,self.config['feature'])
        if self.config['family']=='global':return self.model.predict(X)
        out=[]
        for row,h in zip(X,D[:,:,0].mean(axis=1)):
            w=.01+np.exp(-.5*((self.shifts-h)/self.config['width'])**2)
            model=ForwardGLS(1,self.config['shrink']).fit(self.X,self.y,w)
            out.append(model.predict(row)[0])
        return np.array(out)
