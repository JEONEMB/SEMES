import numpy as np,json
from scipy.optimize._numdiff import approx_derivative
from baseline import CORNERS,field
from fast_fit import evaluator
from compiled import objective
from pathlib import Path
if __name__=='__main__':
 checks=[]
 for c in CORNERS[::70]:
  xy=np.array([[c['cx']+x,c['cy']+y] for y in range(-4,5) for x in range(-4,5)])
  p=np.array([c['cx']+.13,c['cy']-.21,1.1,.8,30.,150.]);axes=np.array(c['axes'],np.float32);patch=np.array(c['patch']);I=np.arange(81,dtype=float)
  fun,jac=evaluator(xy,I,axes,patch);r=p[4]+p[5]*(1-field(xy,p[:2],axes,p[2:4],patch))-I
  assert np.max(abs(fun(p)-r))<1e-10
  inv=np.linalg.inv(axes).astype(float);cost,cr,cJ=objective(p,xy,I,inv,patch.ravel())
  assert np.max(abs(cr-r))<1e-10 and np.max(abs(cJ-jac(p)))<1e-9
  fd=np.empty_like(cJ)
  for j in range(6):
   step=1e-5;lo=p.copy();hi=p.copy();lo[j]-=step;hi[j]+=step;fd[:,j]=(fun(hi)-fun(lo))/(2*step)
  error=float(np.max(abs(fd-cJ)));assert error<1e-5,error
  checks.append(dict(id=c['id'],max_gradient_difference=error))
 Path(__file__).with_name('results').joinpath('math_checks.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2));print('Field, residual, analytic Jacobian: passed',len(checks),'patterns')
