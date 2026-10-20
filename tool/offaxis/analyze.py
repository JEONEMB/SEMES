"""Preserve measured residuals; distinguish reference offsets and extrapolation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def session(path):
    stem=Path(path).stem
    if stem.startswith('Y') and '/' in path and path.split('/')[0]=='Y':
        v=float(stem[1:]); return 'Y'+('+' if v>0 else '-')+('even' if round(abs(v)*100)%10==0 else 'odd')
    if path.startswith('Theta/'):
        return 'T'+('+' if float(stem[1:])>0 else '-')
    return 'new' if path!='center.bmp' else 'old-reference'


def curve(g, value, axis, column, deoffset):
    x=g[axis].to_numpy(); z=g[column].to_numpy()
    # Low-complexity axis model; separate candidate session intercepts.
    groups=sorted(g.session.unique())
    X=np.column_stack([*[g.session.eq(s).astype(float) for s in groups],x,x*x])
    c=np.linalg.lstsq(X,z,rcond=None)[0]
    return c[-2]*value+c[-1]*value**2 if deoffset else np.interp(value,*zip(*sorted(zip(x,z))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output
    inventory=pd.read_csv(out/'sampling.csv')
    points=pd.read_csv(out/'raw/points.csv')
    ecc=pd.read_csv(out/'raw/ecc.csv')
    # Set both fitting methods' displacement exactly relative to their own center fit.
    baseline=points[points.path=='center.bmp'].set_index(['qr','id'])
    points=points.join(baseline[['x','y']].rename(columns={'x':'center_x','y':'center_y'}),on=['qr','id'])
    points['dx']=points.x-points.center_x; points['dy']=points.y-points.center_y
    points=points.merge(inventory,on='path',validate='many_to_one')
    # Only positions/quality and label provenance exported; raw extraction is also retained.
    points.to_csv(out/'point_displacements.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    means=points.groupby(['path','qr'],as_index=False).agg(dx=('dx','mean'),dy=('dy','mean'),
         points=('id','count'),accepted=('accepted','sum'),fitting_rmse=('rmse','mean'),
         window_difference=('window_difference','mean'),condition_max=('condition','max'))
    means['method']='corners630'
    em=ecc[ecc.status=='ok'][['path','qr','dx','dy','correlation','photometric_rmse']].copy(); em['method']='ECC'
    qr=pd.concat([means,em],ignore_index=True).merge(inventory,on='path',validate='many_to_one')
    qr['session']=qr.path.map(session)
    qr.to_csv(out/'qr_displacements.csv',index=False)
    frames=qr.groupby(['method','path'],as_index=False).agg(dx=('dx','mean'),dy=('dy','mean'))
    frames=frames.merge(inventory,on='path',validate='many_to_one')
    frames['session']=frames.path.map(session)
    frames.to_csv(out/'frame_displacements.csv',index=False)
    compare=frames.pivot(index='path',columns='method',values=['dx','dy'])
    comparison=pd.DataFrame({'path':compare.index,'dx_corner_minus_ecc':compare['dx']['corners630']-compare['dx']['ECC'],
                            'dy_corner_minus_ecc':compare['dy']['corners630']-compare['dy']['ECC']})
    comparison.to_csv(out/'method_comparison.csv',index=False)
    offsets=[]; axisres=[]
    for (method,slot,group),g in qr[qr.category.isin(['Y-only','theta-only'])].groupby(['method','qr','session']):
        x=g.y_nominal_mm.to_numpy() if group.startswith('Y') else g.theta_nominal_assumed_deg.to_numpy()
        for dim in ['dx','dy']:
            for degree in [1,2]:
                X=np.column_stack([x**j for j in range(degree+1)])
                c=np.linalg.lstsq(X,g[dim],rcond=None)[0]; res=g[dim].to_numpy()-X@c
                offsets.append(dict(method=method,qr=slot,session=group,dimension=dim,degree=degree,
                                    intercept=c[0],slope=c[1],quadratic=c[2] if degree==2 else 0,rmse=rmse(res),n=len(g)))
                if degree==2:
                    axisres.extend(dict(method=method,qr=slot,path=p,dimension=dim,residual=r) for p,r in zip(g.path,res))
    pd.DataFrame(offsets).to_csv(out/'axis_fits_and_offsets.csv',index=False)
    pd.DataFrame(axisres).to_csv(out/'axis_raw_model_residuals.csv',index=False)
    residuals=[]; repeats=[]; validations=[];jac=[]
    for (method,slot),g in qr.groupby(['method','qr']):
        reference=g[g.category=='new-reference']
        mixed=g[g.category=='mixed']
        for dim in ['dx','dy']:
            refmean=reference[dim].mean()
            repeats.append(dict(method=method,qr=slot,dimension=dim,y=0,theta=0,n=len(reference),
                                mean_old_reference=refmean,std=reference[dim].std(ddof=1)))
            for (y,t),r in mixed.groupby(['y_nominal_mm','theta_nominal_assumed_deg']):
                repeats.append(dict(method=method,qr=slot,dimension=dim,y=y,theta=t,n=len(r),
                                    mean_old_reference=r[dim].mean(),std=r[dim].std(ddof=1)))
            yg=g[(g.category=='Y-only')&(g.y_nominal_mm>0)]
            tg=g[(g.category=='theta-only')&(g.theta_nominal_assumed_deg>0)]
            for r in mixed.itertuples():
                y,t=r.y_nominal_mm,r.theta_nominal_assumed_deg
                normalized=curve(yg,y,'y_nominal_mm',dim,True)+curve(tg,t,'theta_nominal_assumed_deg',dim,True)
                raw=np.nan if t>1 else curve(yg,y,'y_nominal_mm',dim,False)+curve(tg,t,'theta_nominal_assumed_deg',dim,False)
                measured=getattr(r,dim)
                residuals.append(dict(path=r.path,method=method,qr=slot,dimension=dim,y=y,theta=t,
                     measured_old_reference=measured,new_reference_offset=refmean,
                     additive_raw_interpolated=raw,raw_residual=measured-raw,
                     additive_axis_fit_zeroed=normalized,reference_corrected_residual=measured-refmean-normalized,
                     theta_extrapolated=t>1))
            # Group by coordinate BEFORE holdout: all repeats of a coordinate stay together.
            pose=mixed.groupby(['y_nominal_mm','theta_nominal_assumed_deg'],as_index=False)[dim].mean()
            y=pose.y_nominal_mm.to_numpy();t=pose.theta_nominal_assumed_deg.to_numpy();z=pose[dim].to_numpy()
            for interaction in [False,True]:
                X=np.column_stack([np.ones(len(y)),y,t,y*y,t*t]+([y*t] if interaction else []))
                pred=[]
                for i in range(len(y)):
                    keep=np.arange(len(y))!=i
                    pred.append(X[i]@np.linalg.lstsq(X[keep],z[keep],rcond=None)[0])
                c=np.linalg.lstsq(X,z,rcond=None)[0]
                validations.append(dict(method=method,qr=slot,dimension=dim,interaction=interaction,
                    n_coordinates=len(y),parameters=X.shape[1],train_rmse=rmse(z-X@c),loco_rmse=rmse(z-np.array(pred)),
                    cross_coefficient=c[-1] if interaction else 0,design_condition=np.linalg.cond(X)))
            for axis,fixed in [('y_nominal_mm','theta_nominal_assumed_deg'),('theta_nominal_assumed_deg','y_nominal_mm')]:
                for constant,subset in pose.groupby(fixed):
                    subset=subset.sort_values(axis)
                    for i in range(len(subset)-1):
                        a,b=subset.iloc[i],subset.iloc[i+1]
                        jac.append(dict(method=method,qr=slot,dimension=dim,varying_axis=axis,fixed_coordinate=constant,
                                        from_value=a[axis],to_value=b[axis],sensitivity=(b[dim]-a[dim])/(b[axis]-a[axis])))
    res=pd.DataFrame(residuals);res.to_csv(out/'additive_residuals.csv',index=False)
    repeat=pd.DataFrame(repeats);repeat.to_csv(out/'repeatability.csv',index=False)
    pd.DataFrame(validations).to_csv(out/'empirical_coordinate_holdout.csv',index=False)
    pd.DataFrame(jac).to_csv(out/'local_jacobian_qr.csv',index=False)
    # Individual feature sensitivities on the actual irregular sampling, finite intervals.
    feature_jac=[]
    mix=points[points.category=='mixed']
    posepoints=mix.groupby(['qr','id','y_nominal_mm','theta_nominal_assumed_deg'],as_index=False)[['dx','dy']].mean()
    for (slot,key),g in posepoints.groupby(['qr','id']):
        for axis,fixed in [('y_nominal_mm','theta_nominal_assumed_deg'),('theta_nominal_assumed_deg','y_nominal_mm')]:
            for constant,subset in g.groupby(fixed):
                subset=subset.sort_values(axis)
                for i in range(len(subset)-1):
                    a,b=subset.iloc[i],subset.iloc[i+1];step=b[axis]-a[axis]
                    feature_jac.append(dict(qr=slot,id=key,varying_axis=axis,fixed_coordinate=constant,
                        from_value=a[axis],to_value=b[axis],dx_derivative=(b.dx-a.dx)/step,dy_derivative=(b.dy-a.dy)/step))
    pd.DataFrame(feature_jac).to_csv(out/'local_jacobian_points.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    point_residuals=[];point_axis=[]
    points['session']=points.path.map(session)
    for (slot,key),g in points.groupby(['qr','id']):
        refnew=g[g.category=='new-reference'][['dx','dy']].mean()
        axis_coeff={}
        for label,axis in [('Y-only','y_nominal_mm'),('theta-only','theta_nominal_assumed_deg')]:
            ag=g[(g.category==label)&(g[axis]>0)]
            sessions=sorted(ag.session.unique());v=ag[axis].to_numpy()
            X=np.column_stack([*[ag.session.eq(s).astype(float) for s in sessions],v,v*v])
            c=np.linalg.lstsq(X,ag[['dx','dy']],rcond=None)[0]
            axis_coeff[label]=c[-2:]
            point_axis.append(dict(qr=slot,id=key,axis=label,dx_linear=c[-2,0],dy_linear=c[-2,1],
                                   dx_quadratic=c[-1,0],dy_quadratic=c[-1,1],
                                   reference_x=g.reference_x.iloc[0],reference_y=g.reference_y.iloc[0]))
        for r in g[g.category=='mixed'].itertuples():
            yy,tt=r.y_nominal_mm,r.theta_nominal_assumed_deg
            pred=np.array([yy,yy*yy])@axis_coeff['Y-only']+np.array([tt,tt*tt])@axis_coeff['theta-only']
            point_residuals.append(dict(path=r.path,qr=slot,id=key,y=yy,theta=tt,theta_extrapolated=tt>1,
                   measured_dx=r.dx,measured_dy=r.dy,additive_dx=pred[0],additive_dy=pred[1],
                   new_reference_offset_dx=refnew.dx,new_reference_offset_dy=refnew.dy,
                   residual_dx=r.dx-refnew.dx-pred[0],residual_dy=r.dy-refnew.dy-pred[1]))
    pd.DataFrame(point_residuals).to_csv(out/'additive_point_residuals.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(point_axis).to_csv(out/'point_axis_sensitivity.csv',index=False)
    # Point repeat residuals and common frame residual: actual repeats only.
    mix=mix.copy()
    keys=['qr','id','y_nominal_mm','theta_nominal_assumed_deg']
    for dim in ['dx','dy']:
        mix[f'{dim}_repeat_residual']=mix[dim]-mix.groupby(keys)[dim].transform('mean')
        mix[f'{dim}_common']=mix.groupby(['path'])[f'{dim}_repeat_residual'].transform('mean')
        mix[f'{dim}_individual']=mix[f'{dim}_repeat_residual']-mix[f'{dim}_common']
    mix[['path','qr','id','dx_repeat_residual','dy_repeat_residual','dx_common','dy_common','dx_individual','dy_individual']].to_csv(out/'repeat_raw_point_residuals.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    actual=inventory[inventory.category.isin(['mixed','new-reference'])].copy()
    zero=actual[actual.category=='new-reference'][['y_actual','theta_actual']].mean()
    actual['y_actual_delta']=actual.y_actual-zero.y_actual
    actual['theta_actual_delta']=actual.theta_actual-zero.theta_actual
    actual['y_actual_minus_nominal']=actual.y_actual_delta-actual.y_nominal_mm
    actual['theta_actual_minus_nominal']=actual.theta_actual_delta-actual.theta_nominal_assumed_deg
    actual.to_csv(out/'actual_coordinate_check.csv',index=False)
    summary=dict(images=len(inventory),points=len(points),point_count_per_frame=points.groupby('path').size().value_counts().to_dict(),
        fitting_acceptance=float(points.accepted.mean()),ecc_failures=int((ecc.status!='ok').sum()),
        ecc_correlation_min=float(ecc.correlation.min()),ecc_correlation_median=float(ecc.correlation.median()),
        method_dx_difference_rmse=rmse(comparison.dx_corner_minus_ecc),method_dx_difference_max=float(abs(comparison.dx_corner_minus_ecc).max()),
        common_repeat_dx_rms=rmse(mix.groupby('path').dx_common.first()),individual_repeat_dx_rms=rmse(mix.dx_individual),
        actual_y_nominal_max_difference=float(abs(actual.y_actual_minus_nominal).max()),
        actual_theta_nominal_max_difference=float(abs(actual.theta_actual_minus_nominal).max()),
        new_reference_actual=zero.to_dict())
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Scatter only at measured coordinates: no rectangular surface fabrication.
    fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    for row,method in enumerate(['corners630','ECC']):
        for ax,slot in zip(axes[row],'LCR'):
            r=res[(res.method==method)&(res.qr==slot)&(res.dimension=='dx')]
            g=r.groupby(['y','theta'],as_index=False).reference_corrected_residual.mean()
            s=ax.scatter(g.y,g.theta,c=g.reference_corrected_residual,cmap='coolwarm',s=90)
            fig.colorbar(s,ax=ax,label='measured - additive (px)')
            ax.set(title=f'{method}, QR {slot}',xlabel='Y (mm)',ylabel='theta (degree assumed)')
    fig.suptitle('Reference-corrected residual at measured poses (theta > 1: extrapolation)')
    fig.tight_layout();fig.savefig(out/'coupling_sampled_residual.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in ['corners630','ECC']:
        r=repeat[(repeat.method==method)&(repeat.dimension=='dx')]
        axes[0].scatter(r.theta,r['std'],s=15,label=method)
    axes[0].set(xlabel='theta (degree assumed)',ylabel='Within-pose sample std dx (px)');axes[0].legend()
    axes[1].scatter(comparison.dx_corner_minus_ecc,comparison.dy_corner_minus_ecc,s=8)
    axes[1].set(xlabel='corner - ECC dx (px)',ylabel='corner - ECC dy (px)')
    fig.tight_layout();fig.savefig(out/'repeatability_and_methods.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
