"""Cross-fitted nuisance removal for joint image-coordinate Y/T regression."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from forward_model import ForwardGLS

class ScalarGLS:
    def __init__(self,shrink=.1):self.shrink=shrink
    def fit(self,X,y):
        A=np.c_[np.ones(len(y)),y];co=np.linalg.lstsq(A,X,rcond=1e-10)[0]
        self.origin=co[0];r=X-A@co;std=np.maximum(np.sqrt(np.mean(r*r,axis=0)),1e-6);R=r/std
        a=self.shrink;b=(1-a)/len(y);v=(co[1]/std)[:,None]
        q=(v-b*R.T@np.linalg.solve(a*np.eye(len(y))+b*R@R.T,R@v))/a
        self.beta=(q/float(v.T@q))[:,0]/std
        return self
    def predict(self,X):return (X-self.origin)@self.beta

class ConditionalDecoder:
    def __init__(self,config):self.config=config
    def fit(self,X,Xbase,y,groups,strata):
        c=self.config;pred=np.full(y.shape,np.nan)
        splits=list(StratifiedGroupKFold(3,shuffle=True,random_state=20261023).split(X,y=strata,groups=groups))
        self.crossfit_splits=[]
        for tr,te in splits:
            assert not set(groups[tr])&set(groups[te])
            mod=ForwardGLS(1,c['base_shrink']).fit(Xbase[tr],y[tr]);pred[te]=mod.predict(Xbase[te]);self.crossfit_splits.append((tr,te))
        self.base=ForwardGLS(1,c['base_shrink']).fit(Xbase,y)
        # Forward nuisance response is estimated from only training truth/observations.
        yy,tt=y.T;A=np.c_[np.ones(len(y)),yy,tt,yy*yy,yy*tt,tt*tt]
        self.co=np.linalg.lstsq(A,X,rcond=1e-10)[0]
        xx=self.transform(X,pred)
        if c['family']=='gls':self.head=ScalarGLS(c['shrink']).fit(xx,y[:,0])
        else:self.head=make_pipeline(StandardScaler(),Ridge(alpha=c['shrink'])).fit(xx,y[:,0])
        return self
    def transform(self,X,pred):
        t=pred[:,1];y=pred[:,0];c=self.config
        if c['mode']=='append_t':return np.c_[X,X*t[:,None]]
        if c['mode']=='remove_linear':return X-t[:,None]*self.co[2]
        z=X-t[:,None]*self.co[2]-t[:,None]**2*self.co[5]
        if c['mode']=='remove_quadratic':return z
        if c['mode']=='remove_cross':return z-(t*y)[:,None]*self.co[4]
        if c['mode']=='conditional_residual':return np.c_[z,z*t[:,None]]
        raise ValueError(c['mode'])
    def predict(self,X,Xbase):
        p=self.base.predict(Xbase);p[:,0]=self.head.predict(self.transform(X,p));return p

class NuisanceProjected:
    """Project out training-estimated T response; no predicted/true T input at inference."""
    def __init__(self,shrink=.1,degree=1):self.shrink=shrink;self.degree=degree
    def fit(self,X,y):
        yy,tt=y.T;A=np.c_[np.ones(len(y)),yy,tt,yy*yy,yy*tt,tt*tt]
        co=np.linalg.lstsq(A,X,rcond=1e-10)[0];res=X-A@co
        self.scale=np.maximum(np.sqrt(np.mean(res*res,axis=0)),1e-6)
        nuisance=co[[2]] if self.degree==1 else co[[2,5]]
        # Euclidean orthogonality in residual-standardized space; all coordinates retained.
        self.basis=np.linalg.qr((nuisance/self.scale).T)[0]
        self.head=ScalarGLS(self.shrink).fit(self.transform(X),yy)
        self.base=ForwardGLS(1,self.shrink).fit(X,y);return self
    def transform(self,X):
        z=X/self.scale;return z-(z@self.basis)@self.basis.T
    def predict(self,X):
        out=self.base.predict(X);out[:,0]=self.head.predict(self.transform(X));return out
