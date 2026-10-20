"""PPT protocol statistics: fixed-center displacements, OLS, steps, and noise.

Reports sample standard deviations (ddof=1), not residual RMS, where the PPT
uses sigma. The new session uses center1 alone, retaining center2/3 as checks.
Original command groups and mixed fixed-axis analysis groups are distinguished.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def std(x):return float(np.std(x,ddof=1)) if len(x)>1 else np.nan


def old_group(path):
    name=Path(path).stem
    if path.startswith('Y/'):
        q=float(name[1:]);return 'Y'+('+' if q>0 else '-')+('0.1' if round(abs(q)*100)%10==0 else '0.05odd')
    if path.startswith('Theta/'):
        return 'T'+('+' if float(name[1:])>0 else '-')
    return None


def analyze(output,previous):
    inv=pd.read_csv(previous/'sampling.csv')
    inv['theta_deg']=inv.theta_nominal_assumed_deg
    inv['y_mm']=inv.y_nominal_mm
    inv['time_kst']=pd.to_datetime(inv.mtime_ns,unit='ns',utc=True).dt.tz_convert('Asia/Seoul').astype(str)
    inv['time_proxy_minutes']=(inv.mtime_ns-inv[inv.path=='기준 위치/center1.bmp'].mtime_ns.iloc[0])/60e9
    inv['repeat']=inv.path.str.extract(r'\((\d+)\)')[0].astype('Int64')
    inv.to_csv(output/'frames.csv',index=False)
    pts=pd.read_csv(output/'points.csv.gz')
    pts['key']=pts.qr+':'+pts.id.astype(str)
    oldref=pts[pts.path=='center.bmp'].set_index('key')[['x','y']]
    newref=pts[pts.path=='기준 위치/center1.bmp'].set_index('key')[['x','y']]
    pts=pts.join(oldref.rename(columns={'x':'old_reference_x','y':'old_reference_y'}),on='key')
    pts=pts.join(newref.rename(columns={'x':'new_reference_x','y':'new_reference_y'}),on='key')
    pts=pts.merge(inv[['path','category','y_mm','theta_deg']],on='path',validate='many_to_one')
    isnew=pts.category.isin(['new-reference','mixed'])
    for coord,dim in [('x','dx'),('y','dy')]:
        pts['old_'+dim]=pts[coord]-pts['old_reference_'+coord]
        pts[dim]=pts[coord]-np.where(isnew,pts['new_reference_'+coord],pts['old_reference_'+coord])
    pts.to_csv(output/'displacements.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    ec=pd.read_csv(previous/'raw/ecc.csv').merge(inv,on='path',validate='many_to_one')
    refec=ec[ec.path=='기준 위치/center1.bmp'].set_index('qr')[['dx','dy']]
    for dim in ['dx','dy']:
        ec['old_'+dim]=ec[dim]
        ec.loc[ec.category.isin(['new-reference','mixed']),dim]-=ec.loc[ec.category.isin(['new-reference','mixed']),'qr'].map(refec[dim])
    ec.to_csv(output/'ecc.csv',index=False)
    frames=[]
    variants={'all755':pts,'no_diagonal630':pts[~pts.diagonal], 'accepted_dynamic':pts[pts.accepted]}
    for method,g in variants.items():
        agg=g.groupby('path',as_index=False).agg(dx=('dx','mean'),dy=('dy','mean'),n_points=('id','count'))
        agg['method']=method;frames.append(agg)
    ep=ec.groupby('path',as_index=False).agg(dx=('dx','mean'),dy=('dy','mean'),n_points=('qr','count'));ep['method']='ECC';frames.append(ep)
    means=pd.concat(frames,ignore_index=True).merge(inv,on='path',validate='many_to_one')
    means.to_csv(output/'mean_displacements.csv',index=False)
    qr=pts[~pts.diagonal].groupby(['path','qr'],as_index=False)[['dx','dy']].mean()
    qr.to_csv(output/'qr_displacements.csv',index=False)
    groups=[]
    old=inv[inv.category.isin(['Y-only','theta-only'])].copy();old['group']=old.path.map(old_group)
    for name,g in old.groupby('group'):
        axis='y_mm' if name.startswith('Y') else 'theta_deg'
        groups.append(dict(group='old_'+name,scope='old',varying=axis,fixed_axis='theta_deg' if axis=='y_mm' else 'y_mm',fixed_value=0.,paths=g.path.tolist()))
    mixed=inv[inv.category=='mixed']
    for y,g in mixed.groupby('y_mm'):
        groups.append(dict(group=f'new_Y={y:g}',scope='new',varying='theta_deg',fixed_axis='y_mm',fixed_value=y,paths=g.path.tolist()))
    for t,g in mixed.groupby('theta_deg'):
        if g.y_mm.nunique()<3:continue
        groups.append(dict(group=f'new_T={t:g}',scope='new',varying='y_mm',fixed_axis='theta_deg',fixed_value=t,paths=g.path.tolist()))
    (output/'groups.json').write_text(json.dumps(groups,indent=2),encoding='utf-8')
    fits=[];frame_res=[];noise=[];point_res=[];steps=[]
    for group in groups:
        selected=means[means.path.isin(group['paths'])].copy()
        axis=group['varying'];meta={k:group[k] for k in ['group','scope','varying','fixed_axis','fixed_value']}
        for method,g in selected.groupby('method'):
            q=g[axis].to_numpy();X=np.column_stack([np.ones(len(q)),q])
            z=g[['dx','dy']].to_numpy();c=np.linalg.lstsq(X,z,rcond=None)[0];res=z-X@c
            fits.append(dict(**meta,method=method,n_images=len(g),n_coordinates=g[axis].nunique(),
                intercept_dx=c[0,0],slope_dx=c[1,0],intercept_dy=c[0,1],slope_dy=c[1,1],
                residual_sigma_dx=std(res[:,0]),residual_sigma_dy=std(res[:,1]),residual_rms_dx=float(np.sqrt(np.mean(res[:,0]**2)))))
            frame_res.extend(dict(**meta,method=method,path=path,command=value,dx=delta[0],dy=delta[1],
                slope_removed_dx=delta[0]-c[1,0]*value,residual_dx=e[0],residual_dy=e[1]) for path,value,delta,e in zip(g.path,q,z,res))
            # Adjacent commanded coordinates within same repeat ID for mixed frames.
            subsets=[(0,g)] if group['scope']=='old' else list(g.groupby('repeat'))
            for rep,h in subsets:
                h=h.assign(magnitude=abs(h[axis])).sort_values('magnitude')
                for i in range(1,len(h)):
                    a,b=h.iloc[i-1],h.iloc[i];dq=b[axis]-a[axis];sign=np.sign(dq)
                    steps.append(dict(**meta,method=method,repeat=int(rep),from_path=a.path,to_path=b.path,
                        from_command=a[axis],to_command=b[axis],command_step=round(abs(dq),8),
                        step_dx=sign*(b.dx-a.dx),step_dy=sign*(b.dy-a.dy),first_from_center=False))
                first=h.iloc[0]
                # In mixed motion the varying axis starts at a nonzero fixed axis,
                # so center-to-first is not a one-axis step and is not pooled.
                if group['scope']=='old':
                    sign=np.sign(first[axis]);steps.append(dict(**meta,method=method,repeat=0,from_path='center.bmp',to_path=first.path,
                        from_command=0,to_command=first[axis],command_step=abs(first[axis]),step_dx=sign*first.dx,
                        step_dy=sign*first.dy,first_from_center=True))
        # Point-specific straight line, followed by the 630-point common residual.
        gp=pts[pts.path.isin(group['paths'])].copy()
        gp['command']=gp[axis]
        residue=[]
        for key,g in gp.groupby('key'):
            X=np.column_stack([np.ones(len(g)),g.command]);c=np.linalg.lstsq(X,g[['dx','dy']],rcond=None)[0]
            r=g[['dx','dy']].to_numpy()-X@c
            residue.append(pd.DataFrame(dict(path=g.path.to_numpy(),key=key,qr=g.qr.to_numpy(),diagonal=g.diagonal.to_numpy(),
                residual_dx=r[:,0],residual_dy=r[:,1],x=g.old_reference_x.to_numpy(),y=g.old_reference_y.to_numpy())))
        residue=pd.concat(residue,ignore_index=True)
        common=residue[~residue.diagonal].groupby('path')[['residual_dx','residual_dy']].mean().rename(columns={'residual_dx':'common_dx','residual_dy':'common_dy'})
        residue=residue.join(common,on='path')
        residue['individual_dx']=residue.residual_dx-residue.common_dx;residue['individual_dy']=residue.residual_dy-residue.common_dy
        for key,g in residue.groupby('key'):
            noise.append(dict(**meta,key=key,qr=g.qr.iloc[0],diagonal=bool(g.diagonal.iloc[0]),
                  x=g.x.iloc[0],y=g.y.iloc[0],sigma_dx=std(g.individual_dx),sigma_dy=std(g.individual_dy),
                  common_sigma_dx=std(common.common_dx),common_sigma_dy=std(common.common_dy)))
        residue['group']=group['group'];residue['scope']=group['scope'];point_res.append(residue)
    fits=pd.DataFrame(fits);fits.to_csv(output/'linear_fits.csv',index=False)
    pd.DataFrame(frame_res).to_csv(output/'frame_linear_residuals.csv',index=False)
    noise=pd.DataFrame(noise);noise.to_csv(output/'point_noise.csv',index=False)
    pd.concat(point_res,ignore_index=True).to_csv(output/'point_linear_residuals.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    steps=pd.DataFrame(steps)
    nonfirst=steps[~steps.first_from_center]
    step_stats=nonfirst.groupby(['scope','varying','method','command_step'],as_index=False).agg(n=('step_dx','count'),mean_dx=('step_dx','mean'),sigma_dx=('step_dx','std'))
    steps=steps.merge(step_stats,on=['scope','varying','method','command_step'],how='left',validate='many_to_one')
    steps['outlier_3sigma']=~steps.first_from_center&((steps.step_dx-steps.mean_dx).abs()>3*steps.sigma_dx)
    steps.to_csv(output/'steps.csv',index=False);step_stats.to_csv(output/'step_statistics.csv',index=False)
    diagnostics=pts.groupby(['category','qr','diagonal'],as_index=False).agg(n=('id','count'),pass_rate=('accepted','mean'),mean_rmse=('rmse','mean'))
    diagnostics.to_csv(output/'fitting_summary.csv',index=False)
    diagflags=[]
    for flag in ['converged','rmse_ok','contrast_ok','window_ok','shift_ok','sigma_ok']:
        diagflags.append(dict(criterion=flag,failed=int((~pts[flag]).sum())))
    pd.DataFrame(diagflags).to_csv(output/'diagnostic_failures.csv',index=False)
    # Independent original-data numerical checks, without tuning toward the PPT.
    oldfit=fits[(fits.scope=='old')&(fits.method=='no_diagonal630')]
    baseline=pts[pts.path=='center.bmp']
    primary=noise[(noise.scope=='new')&(noise.varying=='theta_deg')]
    summary=dict(theta_unit='degree (user confirmed)',n_frames=len(inv),n_points=len(pts),
        reference_new='기준 위치/center1.bmp',reference_old='center.bmp',
        center_pass=int(baseline.accepted.sum()),center_nondiagonal_pass=int(baseline[~baseline.diagonal].accepted.sum()),
        center_diagonal_pass=int(baseline[baseline.diagonal].accepted.sum()),reference_ppt_center_pass=597,
        old_y_slopes=oldfit[oldfit.varying=='y_mm'].slope_dx.tolist(),old_theta_slopes=oldfit[oldfit.varying=='theta_deg'].slope_dx.tolist(),
        old_common_sigma_range=[float(oldfit.residual_sigma_dx.min()),float(oldfit.residual_sigma_dx.max())],
        new_common_sigma_range=[float(primary.common_sigma_dx.min()),float(primary.common_sigma_dx.max())],
        new_point_sigma_median=float(primary[~primary.diagonal].sigma_dx.median()),
        new_diagonal_sigma_median=float(primary[primary.diagonal].sigma_dx.median()),
        raw_residual_smoothing=False,
        original_fitting_code_available=False,
        known_reconstruction_limits=['PPT specifies thresholds and model family but not every optimizer setting',
          'PPT ECC Gaussian kernel unspecified; requested no-smoothing retained (kernel=1)',
          'Mixed sampling has unequal increments; step statistics are pooled only at identical actual command increments',
          'Original P12 and covariance/classification source are unavailable; no fabricated classification accuracy'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();analyze(a.output,a.previous)
