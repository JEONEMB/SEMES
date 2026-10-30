"""Train-only origin candidates with full residual covariance and image-only selection."""
import numpy as np
from sklearn.cluster import KMeans

class OriginGLS:
 def __init__(self,shrink=.1,mode='kmeans',k=3):self.shrink=shrink;self.mode=mode;self.k=k
 def fit(self,X,y,groups=None):
  n=len(X);self.scale=np.maximum(abs(y).max(0),.05);P=y/self.scale
  if self.mode=='capture':
   names=sorted(set(groups));labels=np.array([names.index(g) for g in groups]);self.origin_names=names
  else:
   co=np.linalg.lstsq(np.c_[np.ones(n),P],X,rcond=None)[0];res=X-np.c_[np.ones(n),P]@co
   scale=np.maximum(np.std(res,axis=0),1e-6)
   labels=KMeans(n_clusters=self.k,random_state=20261022,n_init=5).fit_predict(res/scale)
   self.origin_names=[f'training_residual_cluster{i}' for i in range(self.k)]
  for iteration in range(3 if self.mode=='kmeans' and self.k>1 else 1):
   G=np.eye(len(self.origin_names))[labels];design=np.c_[G,P]
   co=np.linalg.lstsq(design,X,rcond=1e-10)[0]
   if self.mode=='kmeans' and iteration<2:
    res=X-P@co[-2:];labels=KMeans(n_clusters=self.k,random_state=20261022,n_init=5).fit_predict(res/scale)
  self.centers=co[:-2];self.A=co[-2:];res=X-design@co;self.std=np.maximum(np.sqrt(np.mean(res*res,axis=0)),1e-6);R=res/self.std
  self.R=R;self.a=self.shrink;self.b=(1-self.a)/n
  M=self.a*np.eye(n)+self.b*R@R.T;self.Minv=np.linalg.inv(M)
  V=(self.A/self.std).T
  Q=(V-self.b*R.T@(self.Minv@(R@V)))/self.a
  self.projector=Q/self.std[:,None];self.gram=self.A@self.projector;self.gram=(self.gram+self.gram.T)/2;self.graminv=np.linalg.pinv(self.gram)
  self.labels=labels;return self
 def predict(self,X):
  X=np.atleast_2d(X);z=X[:,None,:]-self.centers[None];flat=z.reshape(-1,X.shape[1]);h=flat@self.projector;q=h@self.graminv
  zs=flat/self.std;zr=zs@self.R.T;cost=(np.sum(zs*zs,axis=1)-self.b*np.sum((zr@self.Minv)*zr,axis=1))/self.a-np.sum(h*q,axis=1)
  costs=cost.reshape(len(X),-1);qq=q.reshape(len(X),-1,2)*self.scale;order=np.argsort(costs,axis=1);best=order[:,0];rows=np.arange(len(X));second=order[:,1] if len(self.centers)>1 else best
  self.last=dict(chosen=best,cost=costs[rows,best],gap=costs[rows,second]-costs[rows,best],alternative=qq[rows,second],all_costs=costs)
  return qq[rows,best]
