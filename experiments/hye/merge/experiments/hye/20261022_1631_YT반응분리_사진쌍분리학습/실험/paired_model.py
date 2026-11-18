"""Training-only matched-axis differences and repeat consistency regularization."""
import numpy as np

def graph_operator(y,dates,kind):
    n=len(y);L=np.zeros((n,n))
    if kind=='repeat':keys=[(d,*v) for d,v in zip(dates,y)]
    else:keys=[(d,v[1 if kind=='sameT' else 0]) for d,v in zip(dates,y)]
    groups={}
    for i,key in enumerate(keys):groups.setdefault(key,[]).append(i)
    count=0
    for indices in groups.values():
        if len(indices)<2:continue
        ix=np.array(indices);L[np.ix_(ix,ix)]+=np.eye(len(ix))-np.ones((len(ix),len(ix)))/len(ix);count+=len(ix)-1
    return L*(n/max(count,1))

class PairedRidge:
    def __init__(self,config):self.config=dict(config)
    def fit(self,X,y,dates):
        c=self.config;X=np.asarray(X);self.mean=X.mean(0);self.std=np.maximum(X.std(0),1e-6);Z=(X-self.mean)/self.std
        Lr=graph_operator(y,dates,'repeat');K=Z@Z.T
        self.coefs=[];self.origins=[]
        for out,kind in [(0,'sameT'),(1,'sameY')]:
            La=graph_operator(y,dates,kind);W=np.eye(len(y))+c['pair_weight']*La+c.get('repeat_weight',0)*Lr
            # Paired target differences are supervised; repeat differences target zero.
            Wy=(np.eye(len(y))+c['pair_weight']*La)@y[:,out]
            # Weighted intercept: pair Laplacians annihilate constants.
            origin=float(y[:,out].mean());target=Wy-origin
            dual=np.linalg.solve(W@K+c['alpha']*np.eye(len(y)),target)
            self.coefs.append(Z.T@dual);self.origins.append(origin)
        self.beta=np.array(self.coefs).T;self.origins=np.array(self.origins)
        return self
    def predict(self,X):return ((np.asarray(X)-self.mean)/self.std)@self.beta+self.origins
