"""Same bounded soft_l1 objective, exact analytic Jacobian, invariant geometry cached."""
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
SQRT2=np.sqrt(2.);SQRT2PI=np.sqrt(2*np.pi)

def evaluator(xy,I,axes,patch):
    inv=np.linalg.inv(axes) # Preserve original float32 inversion, then reuse.
    transformed=xy@inv.T
    a=patch[0,0];du=patch[0,1]-a;dv=patch[1,0]-a;cross=patch[1,1]-patch[0,1]-patch[1,0]+a
    last=None;cached=None
    def compute(p):
        nonlocal last,cached
        if last is not None and np.array_equal(p,last):return cached
        # Same subtraction-before-multiply evaluation order as the original.
        loc=(xy-p[:2])@inv.T
        zu=loc[:,0]/p[2];zv=loc[:,1]/p[3]
        U=.5*(1+erf(zu/SQRT2));V=.5*(1+erf(zv/SQRT2))
        F=patch[0,0]*(1-U)*(1-V)+patch[0,1]*U*(1-V)+patch[1,0]*(1-U)*V+patch[1,1]*U*V
        res=p[4]+p[5]*(1-F)-I
        gu=(du+cross*V)*np.exp(-.5*zu*zu)/SQRT2PI
        gv=(dv+cross*U)*np.exp(-.5*zv*zv)/SQRT2PI
        J=np.empty((len(I),6));J[:,0]=p[5]*(gu/p[2]*inv[0,0]+gv/p[3]*inv[1,0]);J[:,1]=p[5]*(gu/p[2]*inv[0,1]+gv/p[3]*inv[1,1])
        J[:,2]=p[5]*gu*loc[:,0]/(p[2]*p[2]);J[:,3]=p[5]*gv*loc[:,1]/(p[3]*p[3]);J[:,4]=1;J[:,5]=1-F
        last=p.copy();cached=(res,J);return cached
    def fun(p):return compute(p)[0].copy() # SciPy robust loss rescales returned arrays in-place.
    def jac(p):return compute(p)[1].copy()
    return fun,jac

def fit(xy,I,start,axes,patch):
    low,high=np.percentile(I,[10,90]);contrast=max(high-low,20)
    p0=[*start,1.,1.,low,contrast]
    lo=[start[0]-3.8,start[1]-3.8,.35,.35,0,10];hi=[start[0]+3.8,start[1]+3.8,2.7,2.7,240,255]
    fun,jac=evaluator(xy,I,axes,patch)
    r=least_squares(fun,p0,jac=jac,bounds=(lo,hi),loss='soft_l1',f_scale=6,max_nfev=85,xtol=1e-7,ftol=1e-7,gtol=1e-7)
    return r.x,float(np.sqrt(np.mean(fun(r.x)**2))),r.success
