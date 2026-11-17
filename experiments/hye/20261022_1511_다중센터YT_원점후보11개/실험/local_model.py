"""Train-only local response surfaces for each origin, with shared unobserved directions."""
import numpy as np,time
from model import JointOrigins,phi
class LocalOrigins(JointOrigins):
 def fit(self,X,truth,groups,scopes,indices,relative):
  super().fit(X,truth,groups,scopes,indices,relative);self.As=np.tile(self.A,(len(self.centers),1,1));local={}
  for group in sorted(set(np.asarray(groups)[indices])):
   ii=indices[(np.asarray(groups)[indices]==group)&(np.asarray(scopes)[indices]!='center')]
   if len(ii)<6:continue
   q=truth[ii]/self.scale;F=phi(q,self.degree)
   if group=='date20261020':columns=list(range(F.shape[1]))
   elif group.startswith('T'):columns=[1]+([4] if self.degree==2 else [])
   elif group.startswith('Y'):columns=[0]+([2] if self.degree==2 else [])
   else:continue
   untouched=[j for j in range(F.shape[1]) if j not in columns];target=X[ii]-(F[:,untouched]@self.A[untouched] if untouched else 0)
   design=np.c_[np.ones(len(ii)),F[:,columns]];coef=np.linalg.solve(design.T@design+1e-6*np.eye(design.shape[1]),design.T@target);A=self.A.copy();A[columns]=coef[1:];local[group]=(coef[0],A)
  for k,name in enumerate(self.origin_ids):
   group=name.split(':',1)[1] if name.startswith('learned:') else 'date20261020' if name.startswith('actual:20261020') else None
   if group in local:
    b,A=local[group];self.As[k]=A
    if name.startswith('learned:'):self.centers[k]=b
  self.cw=self.centers*self.W if self.W.ndim==1 else self.centers@self.W.T
  return self
 def predict(self,X):
  start=time.perf_counter();X=np.atleast_2d(X);poses=[];costs=[]
  for k in range(len(self.centers)):
   sub=JointOrigins(self.degree,self.shrink,self.origins_mode);sub.__dict__.update(self.__dict__);sub.A=self.As[k];sub.Aw=sub.A*self.W if self.W.ndim==1 else sub.A@self.W.T;sub.centers=self.centers[k:k+1];sub.cw=self.cw[k:k+1];sub.origin_ids=self.origin_ids[k:k+1];sub.linear_pinv=np.linalg.pinv(sub.Aw[:2]);out=sub.predict(X);poses.append(out['YT']);costs.append(out['cost'])
  Q=np.stack(poses,axis=1);cost=np.stack(costs,axis=1);order=np.argsort(cost,axis=1);best=order[:,0];second=order[:,1] if len(self.centers)>1 else best;rows=np.arange(len(X));v=cost[rows,best];w=cost[rows,second]
  return dict(YT=Q[rows,best],center_index=best,center_id=[self.origin_ids[i] for i in best],cost=v,second_cost=w,margin=(w-v)/(w+v+1e-12),alternative_YT=Q[rows,second],all_costs=cost,all_YT=Q,model_seconds=time.perf_counter()-start)
 def save(self,path):
  super().save(path);a=np.load(path);data={k:a[k] for k in a.files};a.close();data['As']=self.As;np.savez_compressed(path,**data)
