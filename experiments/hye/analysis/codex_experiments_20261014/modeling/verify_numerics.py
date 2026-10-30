"""Numerical validation independent of the real-data labels."""
import json
import numpy as np
import pandas as pd
from experiment import HERE, MotionModel

rng=np.random.default_rng(20261014)
F=pd.DataFrame(dict(x=rng.uniform(1520,2280,90),y=rng.uniform(970,1170,90),region=np.repeat(['QR좌','QR중','QR우'],30)))
x=(F.x.to_numpy()-1900)/400; y=(F.y.to_numpy()-1070)/120
by=np.column_stack([14+0.015*x*x+0.05*y,0.02+0.01*x])
bt=np.column_stack([80+0.2*x+0.07*y,0.4+0.04*x])
quad=np.column_stack([.02*x,.002*y])
ax=np.repeat(['Y','T'],120);val=np.r_[rng.uniform(-2,2,120),rng.uniform(-1,1,120)]
D=np.array([(by if a=='Y' else bt)*v+quad*v*v for a,v in zip(ax,val)])
D+=rng.normal(0,.0001,D.shape)
tr=np.r_[np.arange(90),np.arange(120,210)];te=np.setdiff1d(np.arange(240),tr)
model=MotionModel(dict(kind='p12',degree=2,shrink=.1,anchor=False,subset='all',nuisance='none')).fit(D[tr],F,np.ones((len(tr),len(F))),ax[tr],val[tr])
pred=model.predict(D[te]); correct=pred['pred_axis']==ax[te]
err=np.abs(pred['pred_value']-val[te]);assert correct.all() and err.max()<.0001
# The row-vector whitening norm must equal the covariance quadratic form.
C=rng.normal(size=(12,12));C=C.T@C+np.eye(12)*.1
eig,U=np.linalg.eigh(C);W=U/np.sqrt(eig)[None,:]
z=rng.normal(size=(20,12));a=np.sum((z@W)**2,axis=1);b=np.einsum('ni,ij,nj->n',z,np.linalg.inv(C),z)
whitening_diff=float(np.max(abs(a-b))); assert whitening_diff<1e-9
# Reordering independent query images must only reorder predictions.
p2=model.predict(D[te[::-1]])
assert np.array_equal(p2['pred_axis'],pred['pred_axis'][::-1])
assert np.allclose(p2['pred_value'],pred['pred_value'][::-1])
out=dict(synthetic_test_images=len(te),synthetic_axis_accuracy=float(correct.mean()),synthetic_max_abs_error=float(err.max()),whitening_max_absolute_difference=whitening_diff,query_permutation_invariant=True)
(HERE/'numerical_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
