"""Constrained plane-normal rotation model, scaled parameters only.

Plane n.X = 1, translation direction tangent to plane. A camera rotating
about the plane normal through p moves C = a*Y + (I-R)*p. Consequently
H = K R.T (I-C n.T) inv(K). The n component of p cancels identically.
No distance or pivot depth can be recovered from this component.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def geometry(p):
    # Plane normal sign chosen toward wall; scale convention n.X=1.
    nx,ny=p[:2]
    n=np.array([nx,ny,1.]);n/=np.linalg.norm(n)
    a=np.array([1.,0.,0.]);a-=n*np.dot(a,n);a/=np.linalg.norm(a)
    b=np.cross(n,a)
    direction=np.cos(p[2])*a+np.sin(p[2])*b
    translation=p[3]*direction
    pivot=p[4]*a+p[5]*b
    return n,translation,pivot


def predict(p, ref, y, theta, f):
    n,a,pivot=geometry(p)
    K=np.array([[f,0,1920],[0,f,1080],[0,0,1.]])
    rays=np.column_stack([ref,np.ones(len(ref))])@np.linalg.inv(K).T
    R=Rotation.from_rotvec(theta[:,None]*(np.pi/180)*n).as_matrix()
    C=y[:,None]*a+np.einsum('nij,j->ni',np.eye(3)[None]-R,pivot)
    moved=np.einsum('nji,nj->ni',R,rays-C*np.einsum('ni,i->n',rays,n)[:,None])
    uv=moved@K.T
    return uv[:,:2]/uv[:,2:]-ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    qr=pd.read_csv(out/'qr_displacements.csv')
    inventory=pd.read_csv(out/'actual_coordinate_check.csv')
    # Independent measured QR translations: use ECC, actual relative stage readings.
    mixed=qr[(qr.method=='ECC')&(qr.category=='mixed')].merge(
        inventory[['path','y_actual_delta','theta_actual_delta']],on='path')
    ref=qr[(qr.method=='ECC')&(qr.category=='new-reference')].groupby('qr')[['dx','dy']].mean()
    center={'L':[1644.,1068.],'C':[1902.,1069.],'R':[2160.,1070.]}
    samples=[]
    for (slot,y,t),g in mixed.groupby(['qr','y_nominal_mm','theta_nominal_assumed_deg']):
        samples.append(dict(qr=slot,y=y,theta=t,actual_y=g.y_actual_delta.mean(),actual_theta=g.theta_actual_delta.mean(),
                            refx=center[slot][0]+ref.loc[slot,'dx'],refy=center[slot][1]+ref.loc[slot,'dy'],
                            dx=g.dx.mean()-ref.loc[slot,'dx'],dy=g.dy.mean()-ref.loc[slot,'dy']))
    df=pd.DataFrame(samples)
    coordinates=list(df[['y','theta']].drop_duplicates().itertuples(index=False,name=None))
    # Fixed coordinate holdout, no repeats from the same pose on both sides.
    hold=set(sorted(coordinates)[2::4])
    mask=np.array([(r.y,r.theta) in hold for r in df.itertuples()])
    xy=df[['refx','refy']].to_numpy();y=df.actual_y.to_numpy();t=df.actual_theta.to_numpy();z=df[['dx','dy']].to_numpy()
    scale=max(np.std(z),1.)
    rows=[];models=[];comparisons=[]
    for interaction in [False,True]:
        errors_train=[];errors_test=[]
        for slot,g in df.groupby('qr'):
            idx=g.index.to_numpy();yy=y[idx];tt=t[idx]
            X=np.column_stack([np.ones(len(g)),yy,tt,yy*yy,tt*tt]+([yy*tt] if interaction else []))
            keep=~mask[idx]
            coeff=np.linalg.lstsq(X[keep],z[idx][keep],rcond=None)[0]
            error=X@coeff-z[idx]
            errors_train.extend(error[keep].ravel());errors_test.extend(error[~keep].ravel())
        comparisons.append(dict(model='quadratic_interaction' if interaction else 'quadratic_additive',
             parameters=18 if interaction else 15,train_rmse_xy=float(np.sqrt(np.mean(np.square(errors_train)))),
             holdout_rmse_xy=float(np.sqrt(np.mean(np.square(errors_test))))))
    # Focal-length profile shows sensitivity/degeneracy, rather than pretending K is known.
    for focal in [2000.,4500.,8000.]:
        def residual(p,selected):
            return ((predict(p,xy[selected],y[selected],t[selected],focal)-z[selected])/scale).ravel()
        solutions=[]
        for tilt in [0.,.25,-.25]:
            # Positive measured +Y displacement corresponds to negative camera translation.
            initial=np.array([tilt,0,0,-.0032,0,-1.])
            fit=least_squares(lambda p:residual(p,~mask),initial,
                 bounds=([-2,-2,-np.pi,-.1,-10,-10],[2,2,np.pi,.1,10,10]),
                 max_nfev=700,x_scale='jac',ftol=1e-9,xtol=1e-9,gtol=1e-9)
            solutions.append(fit)
        fit=min(solutions,key=lambda o:np.sum(o.fun**2));pred=predict(fit.x,xy,y,t,focal)
        singular=np.linalg.svd(fit.jac,compute_uv=False)
        models.append(dict(focal_fixed=focal,parameters=fit.x.tolist(),converged=bool(fit.success),
            train_rmse_xy=float(np.sqrt(np.mean((pred[~mask]-z[~mask])**2))),
            holdout_rmse_xy=float(np.sqrt(np.mean((pred[mask]-z[mask])**2))),
            jacobian_singular_values=singular.tolist(),jacobian_condition=float(singular[0]/max(singular[-1],1e-15)),
            parameter_boundary=fit.active_mask.tolist()))
        for i,r in enumerate(df.itertuples()):
            rows.append(dict(focal_fixed=focal,qr=r.qr,y=r.y,theta=r.theta,holdout=bool(mask[i]),
                measured_dx=r.dx,measured_dy=r.dy,predicted_dx=pred[i,0],predicted_dy=pred[i,1],
                residual_dx=r.dx-pred[i,0],residual_dy=r.dy-pred[i,1]))
    best=min(models,key=lambda m:m['train_rmse_xy'])
    free_initial=np.r_[best['parameters'],np.log(best['focal_fixed'])]
    free=least_squares(lambda p:((predict(p[:6],xy[~mask],y[~mask],t[~mask],np.exp(p[6]))-z[~mask])/scale).ravel(),
        free_initial,bounds=([-2,-2,-np.pi,-.1,-10,-10,np.log(1000)],[2,2,np.pi,.1,10,10,np.log(20000)]),
        x_scale='jac',max_nfev=1000,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    focal=float(np.exp(free.x[6]));pred=predict(free.x[:6],xy,y,t,focal)
    singular=np.linalg.svd(free.jac,compute_uv=False)
    models.append(dict(focal_fitted=focal,parameters=free.x.tolist(),converged=bool(free.success),
        train_rmse_xy=float(np.sqrt(np.mean((pred[~mask]-z[~mask])**2))),
        holdout_rmse_xy=float(np.sqrt(np.mean((pred[mask]-z[mask])**2))),
        jacobian_singular_values=singular.tolist(),jacobian_condition=float(singular[0]/max(singular[-1],1e-15)),
        parameter_boundary=free.active_mask.tolist()))
    for i,r in enumerate(df.itertuples()):
        rows.append(dict(focal_fixed='free',qr=r.qr,y=r.y,theta=r.theta,holdout=bool(mask[i]),
                measured_dx=r.dx,measured_dy=r.dy,predicted_dx=pred[i,0],predicted_dy=pred[i,1],
                residual_dx=r.dx-pred[i,0],residual_dy=r.dy-pred[i,1]))
    pd.DataFrame(rows).to_csv(out/'physics_raw_residuals.csv',index=False)
    result=dict(axis_assumption='rotation axis normal to QR wall; Y tangent to wall',
      K_assumption='square pixels, fixed principal point (1920,1080), no lens distortion; focal-length profile',
      parameter_order=['normal_x_over_z','normal_y_over_z','tangent_direction_angle','translation_per_mm_over_plane_distance','pivot_tangent_a_over_distance','pivot_tangent_b_over_distance'],
      input='new-session mean ECC QR-center displacements versus sidecar coordinates minus new reference mean',
      train_coordinates=len(coordinates)-len(hold),holdout_coordinates=len(hold),models=models,
      same_split_empirical=comparisons,
      limits=['QR centers only; not a dense feature calibration','theta degree units assumed',
              'camera-behind pivot normal component unobservable: (I-R)*(p+lambda*n)=(I-R)*p',
              'absolute distance, focal length, and pivot depth are not separately identified',
              'camera translation/rotation order is assumed, not confirmed by acquisition log',
              'nonlinear fit is a conditional diagnostic; no accuracy certification'])
    (out/'physics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
