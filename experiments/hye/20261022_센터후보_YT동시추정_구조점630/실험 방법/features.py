"""Feature maps from all fixed 630 displacements; no point selection or labels."""
import numpy as np
class FeatureMaps:
 def __init__(self,points,regions):
  self.regions=regions;u,v=(points[:,0]-1906)/486,(points[:,1]-1070)/122;self.P12=np.linalg.pinv(np.c_[np.ones(len(u)),u,v,u*u,u*v,v*v]);self.affine=[];self.quadratic=[]
  for k in range(3):
   p=points[regions==k];u=(p[:,0]-p[:,0].mean())/100;v=(p[:,1]-p[:,1].mean())/100
   self.affine.append(np.linalg.pinv(np.c_[np.ones(len(u)),u,v]));self.quadratic.append(np.linalg.pinv(np.c_[np.ones(len(u)),u,v,u*u,u*v,v*v]))
 def transform(self,D,name):
  single=np.asarray(D).ndim==2
  if single:D=np.asarray(D)[None]
  if name=='P12':X=np.einsum('ap,npd->nad',self.P12,D).reshape(len(D),-1)
  elif name=='raw1260':X=np.asarray(D).reshape(len(D),-1)
  else:
   matrices=self.affine if name=='QR18' else self.quadratic
   X=np.concatenate([np.einsum('ap,npd->nad',mat,D[:,self.regions==k]).reshape(len(D),-1) for k,mat in enumerate(matrices)],axis=1)
  return X[0] if single else X
