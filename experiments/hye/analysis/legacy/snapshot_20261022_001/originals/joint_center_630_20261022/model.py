"""Shared forward surface + discrete origins. Inference accepts feature vectors only."""
import numpy as np,time

def phi(q,degree):
    y,t=q[:,0],q[:,1]
    return np.c_[y,t] if degree==1 else np.c_[y,t,y*y,y*t,t*t]

class JointOrigins:
    def __init__(self,degree=2,shrink=.5,origins='learned_actual'):
        self.degree=degree;self.shrink=shrink;self.origins_mode=origins
    def fit(self,X,truth,groups,scopes,indices,relative):
        start=time.perf_counter();X=np.asarray(X);truth=np.asarray(truth)
        self.scale=np.maximum(np.max(abs(truth[indices]),axis=0),.05)
        moving=indices[np.asarray(scopes)[indices]!='center'];q=truth[moving]/self.scale;P=phi(q,self.degree)
        unique=sorted(set(np.asarray(groups)[moving]));G=np.array([[float(g==k) for k in unique] for g in np.asarray(groups)[moving]])
        design=P if self.origins_mode=='fixed' else np.c_[G,P]
        regularizer=np.eye(design.shape[1])*1e-6
        coef=np.linalg.solve(design.T@design+regularizer,design.T@X[moving])
        if self.origins_mode=='fixed':self.A=coef;learned=[]
        else:
            self.A=coef[len(unique):];learned=[(f'learned:{k}',coef[i]) for i,k in enumerate(unique)]
        origins=[('reference:fixed_center',np.zeros(X.shape[1]))]
        if self.origins_mode in ('learned','learned_actual'):origins+=learned
        if self.origins_mode in ('actual','learned_actual'):
            for i in indices:
                if scopes[i]=='center' and not str(relative[i]).endswith('20261014\\center.bmp'):
                    origins.append((f'actual:{relative[i]}',X[i]))
        self.origin_ids=[row[0] for row in origins];self.centers=np.array([row[1] for row in origins])
        residual=X[moving]-design@coef
        std=np.sqrt(np.maximum(np.sum(residual*residual,axis=0)/max(len(residual)-design.shape[1],1),1e-12))
        if self.shrink==1.:
            self.W=1/std;self.Aw=self.A*self.W;self.cw=self.centers*self.W
        else:
            S=residual.T@residual/max(len(residual)-design.shape[1],1);C=S/std[:,None]/std[None,:];C=(1-self.shrink)*C+self.shrink*np.eye(len(std))
            self.W=np.linalg.cholesky(np.linalg.inv(C)).T/std[None,:];self.Aw=self.A@self.W.T;self.cw=self.centers@self.W.T
        self.linear_pinv=np.linalg.pinv(self.Aw[:2])
        self.lower=np.min(truth[indices]/self.scale,axis=0)-.25;self.upper=np.max(truth[indices]/self.scale,axis=0)+.25
        self.training_indices=np.array(indices);self.fit_seconds=time.perf_counter()-start
        return self
    def predict(self,X):
        start=time.perf_counter();X=np.atleast_2d(X);n=len(X);k=len(self.centers);z=(X*self.W if self.W.ndim==1 else X@self.W.T)[:,None,:]-self.cw[None,:,:]
        z=z.reshape(-1,X.shape[1]);q=np.clip(z@self.linear_pinv,self.lower,self.upper)
        for iteration in range(16):
            y,t=q[:,0],q[:,1]
            prediction=phi(q,self.degree)@self.Aw
            if self.degree==1:
                j0=np.broadcast_to(self.Aw[0],z.shape);j1=np.broadcast_to(self.Aw[1],z.shape)
            else:
                j0=self.Aw[0]+2*y[:,None]*self.Aw[2]+t[:,None]*self.Aw[3]
                j1=self.Aw[1]+y[:,None]*self.Aw[3]+2*t[:,None]*self.Aw[4]
            r=z-prediction;a=np.sum(j0*j0,axis=1);b=np.sum(j0*j1,axis=1);c=np.sum(j1*j1,axis=1);g0=np.sum(j0*r,axis=1);g1=np.sum(j1*r,axis=1)
            damp=1e-8*(a+c+1);a+=damp;c+=damp;det=np.maximum(a*c-b*b,1e-20)
            step=np.c_[(c*g0-b*g1)/det,(a*g1-b*g0)/det];q=np.clip(q+np.clip(step,-.2,.2),self.lower,self.upper)
        cost=np.sum((z-phi(q,self.degree)@self.Aw)**2,axis=1).reshape(n,k);Q=q.reshape(n,k,2)*self.scale;order=np.argsort(cost,axis=1);best=order[:,0];second=order[:,1] if k>1 else best;rows=np.arange(n)
        output=Q[rows,best];best_cost=cost[rows,best];second_cost=cost[rows,second];alternative=Q[rows,second];gap=(second_cost-best_cost)/(second_cost+best_cost+1e-12)
        return dict(YT=output,center_index=best,center_id=[self.origin_ids[i] for i in best],cost=best_cost,second_cost=second_cost,margin=gap,alternative_YT=alternative,all_costs=cost,all_YT=Q,model_seconds=time.perf_counter()-start)
    def save(self,path):
        np.savez_compressed(path,A=self.A,centers=self.centers,W=self.W,origin_ids=self.origin_ids,scale=self.scale,lower=self.lower,upper=self.upper,degree=self.degree,shrink=self.shrink,origins_mode=self.origins_mode,training_indices=self.training_indices,feature=getattr(self,'feature_name','P12'))
    @classmethod
    def load(cls,path):
        a=np.load(path)
        if 'As' in a:
            from local_model import LocalOrigins
            cls=LocalOrigins
        self=cls(int(a['degree']),float(a['shrink']),str(a['origins_mode']));self.A=a['A'];self.centers=a['centers'];self.W=a['W'];self.origin_ids=a['origin_ids'].tolist();self.scale=a['scale'];self.lower=a['lower'];self.upper=a['upper'];self.training_indices=a['training_indices'];self.feature_name=str(a['feature']) if 'feature' in a else 'P12';self.Aw=self.A*self.W if self.W.ndim==1 else self.A@self.W.T;self.cw=self.centers*self.W if self.W.ndim==1 else self.centers@self.W.T;self.linear_pinv=np.linalg.pinv(self.Aw[:2]);
        if 'As' in a:self.As=a['As']
        return self
