"""Diagnose frozen saved predictors; no estimator training or trajectory simulation.

All one-command results assume exact independent additive actuator motion.
An error surface regression below is descriptive only, never a new predictor.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/offaxis-eda'
OUT = BASE / 'axis-approach'
TICKS = np.array([.05, .01])
COORD = ['y_mm', 'theta_deg']
PRED = ['predicted_y_mm', 'predicted_theta_deg']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def affine(m, x):
    z = (x - m['mean']) / np.array(m['scale'])
    return np.column_stack([np.ones(len(z)), z]) @ np.array(m['coefficients'])


def plot_errors(data):
    """Static export of measured errors; lines join only same-fixed-axis samples."""
    im=Image.new('RGB',(1500,980),'white'); draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    title=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
    colors=['#1764ac','#d45b19','#24965c','#a1399b','#8b7622','#3d8790','#333333']
    draw.text((30,15),'Frozen axis+mixed model: cross-axis errors',font=title,fill='black')
    draw.text((30,52),'57 images / 19 coordinates. Solid dots: coordinate means; faint dots: individual images.',font=font,fill='black')
    panels=[('theta_deg','y_mm','error_y_mm',.05,True,'Y error vs theta (log scale); colors = fixed Y'),
            ('y_mm','theta_deg','error_theta_deg',.01,False,'Theta error vs Y; colors = fixed theta'),
            ('theta_deg','y_mm','error_theta_deg',.01,True,'Theta error vs theta; colors = fixed Y'),
            ('y_mm','theta_deg','error_y_mm',.05,False,'Y error vs Y; colors = fixed theta')]
    for k,(xc,group,ec,tick,log,titletext) in enumerate(panels):
        ox=40+(k%2)*745; oy=110+(k//2)*420
        left,top,right,bottom=ox+65,oy+45,ox+675,oy+335
        draw.text((ox,oy),titletext,font=font,fill='black')
        xmin,xmax=(np.log10(.01),np.log10(2)) if log else (0,1)
        limit=max(4,float(np.ceil(abs(data[ec]/tick).max())))
        def px(x):return left+((np.log10(x) if log else x)-xmin)/(xmax-xmin)*(right-left)
        def py(y):return bottom-(y+limit)/(2*limit)*(bottom-top)
        for y in [-limit,-1,0,1,limit]:
            draw.line((left,py(y),right,py(y)),fill='#de9999' if abs(y)==1 else '#dddddd')
            draw.text((left-48,py(y)-10),f'{y:g}',font=font,fill='black')
        for x in ([.01,.03,.1,.3,1,2] if log else [0,.1,.5,1]):
            xx=px(x);draw.line((xx,top,xx,bottom),fill='#eeeeee')
            draw.text((xx-15,bottom+10),f'{x:g}',font=font,fill='black')
        draw.rectangle((left,top,right,bottom),outline='#444444')
        draw.text((left,top-25),'error (ticks); red lines = +/-1 tick',font=font,fill='#555555')
        for j,(value,g) in enumerate(data.groupby(group)):
            color=colors[j%len(colors)]
            for _,r in g.iterrows():
                xx,yy=px(r[xc]),py(r[ec]/tick)
                draw.ellipse((xx-2,yy-2,xx+2,yy+2),fill='#bfc5cb')
            means=g.groupby(xc)[ec].mean().sort_index()
            points=[(px(x),py(y/tick)) for x,y in means.items()]
            if len(points)>1:draw.line(points,fill=color,width=2)
            for xx,yy in points:draw.ellipse((xx-5,yy-5,xx+5,yy+5),fill=color)
            draw.text((left+j*95,bottom+40),f'{value:g}',font=font,fill=color)
    draw.text((35,945),'No monotone trend is assumed. Training samples are included; this is a diagnostic, not a trajectory test.',font=font,fill='black')
    im.save(OUT/'cross_axis_errors.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []
    def read(rel):
        p = BASE / rel
        sources.append(p)
        return pd.read_csv(p)
    def model(rel):
        p = BASE / rel
        sources.append(p)
        return json.loads(p.read_text(encoding='utf-8'))

    frames = read('ppt-method-v2/frames.csv').set_index('path')
    mixed = frames[frames.category.eq('mixed')].sort_index()
    assert len(mixed) == 57 and len(mixed[COORD].drop_duplicates()) == 19
    # Verify the original images against the source inventory before reusing measurements.
    verified = []
    for path, row in frames.iterrows():
        p = ROOT / 'data/SSAFY Test Image Set' / path
        digest = sha(p)
        assert digest == row.sha256, path
        verified.append(dict(path=path, sha256=digest))
    pd.DataFrame(verified).to_csv(OUT/'verified_images.csv', index=False)

    qr = read('ppt-method-v2/qr_displacements.csv')
    cols = [(d,s) for s in 'LCR' for d in ['dx','dy']]
    wide = qr.pivot(index='path', columns='qr', values=['dx','dy'])[cols]
    x = wide.loc[mixed.index].to_numpy()
    tables = []
    def add(name, data, evaluation):
        d = data.copy()
        d['model'] = name
        d['evaluation'] = evaluation
        tables.append(d)
    # The mixed+axis model is frozen; only two mixed coordinates were held out.
    bundle = model('refit-axis-split/model.json')
    for f,digest in bundle['source_sha256'].items():
        assert sha(BASE/'ppt-method-v2'/f) == digest
    m = bundle['model']; c = m['centers']['new']
    pred = ((x-c['x'])/m['scale']) @ np.array(m['coefficients'])*TICKS+c['y']
    stored = read('refit-axis-split/predictions_and_split.csv').set_index('path').loc[mixed.index]
    assert np.max(abs(pred-stored[PRED].to_numpy())) < 1e-9
    d = mixed[COORD].copy(); d[PRED] = pred
    d['split'] = stored['split']
    add('axis_split_frozen', d, '51 training images + 6 held images; frozen single model')

    bm = model('refit-model/model.json')
    d = mixed[COORD].copy(); d[PRED] = affine(bm['models']['three_qr_affine'], x)
    add('corners_mixed_frozen', d, 'all 57 training images; descriptive only')
    gn = read('ecc-gauss-newton/measurements.csv')
    gw = gn.pivot(index='path', columns='qr', values=['dx','dy'])[cols]
    gx = (gw-gw.loc['기준 위치/center1.bmp']).loc[mixed.index].to_numpy()
    gm = model('ecc-gauss-newton/models.json')['GN_three_qr_affine']
    d = mixed[COORD].copy(); d[PRED] = affine(gm,gx)
    add('gn_mixed_frozen', d, 'all 57 training images; descriptive only')
    held = read('ecc-gauss-newton/coordinate_holdout.csv')
    for method,name in [('GN','gn_coordinate_holdout'),('Corners630','corners_coordinate_holdout')]:
        d = held[held.method.eq(method)&held.model.eq('three_qr_affine')].set_index('path')
        add(name,d[COORD+PRED], '19 coordinate holdouts; different predictor each coordinate, not a controller')
    all_data = pd.concat(tables).reset_index()
    all_data['error_y_mm'] = all_data.predicted_y_mm-all_data.y_mm
    all_data['error_theta_deg'] = all_data.predicted_theta_deg-all_data.theta_deg
    all_data['theta_axis_extrapolation'] = all_data.theta_deg > 1
    all_data.to_csv(OUT/'predictions.csv',index=False)
    plot_errors(all_data[all_data.model.eq('axis_split_frozen')])
    summaries=[]; groups=[]; secants=[]; diagnostic=[]; commands=[]
    for name,g in all_data.groupby('model',sort=False):
        q = g[COORD].to_numpy()/TICKS
        p = g[PRED].to_numpy()/TICKS
        e = p-q
        for subset, mask in [('all',np.ones(len(g),bool)),('theta_le_1',g.theta_deg.to_numpy()<=1),
                             ('held_mixed',g.get('split',pd.Series('',index=g.index)).eq('test_mixed').to_numpy())]:
            if not mask.any(): continue
            summaries.append(dict(model=name,subset=subset,images=int(mask.sum()),
                y_rmse_ticks=float(np.sqrt(np.mean(e[mask,0]**2))),
                theta_rmse_ticks=float(np.sqrt(np.mean(e[mask,1]**2))),
                y_within_tick=float(np.mean(abs(e[mask,0])<=1)),
                theta_within_tick=float(np.mean(abs(e[mask,1])<=1)),
                either_error_within_tick=float(np.mean(np.min(abs(e[mask]),axis=1)<=1)),
                both_within_tick=float(np.mean(np.max(abs(e[mask]),axis=1)<=1)),
                error_correlation=float(np.corrcoef(e[mask].T)[0,1])))
        # Commands choose only predicted values, never true coordinates.
        for policy in ['Y_only','theta_only','predicted_nearest','both']:
            u=np.zeros_like(p)
            if policy=='Y_only':u[:,0]=p[:,0]
            elif policy=='theta_only':u[:,1]=p[:,1]
            elif policy=='both':u=p.copy()
            else:
                chosen=np.argmin(abs(p),axis=1)
                u[np.arange(len(p)),chosen]=p[np.arange(len(p)),chosen]
            after=q-u
            before_distance=np.min(abs(q),axis=1)
            after_distance=np.min(abs(after),axis=1)
            eligible=before_distance>1
            for i,(_,row) in enumerate(g.iterrows()):
                commands.append(dict(model=name,policy=policy,path=row.path,
                    before_distance_ticks=before_distance[i],after_distance_ticks=after_distance[i],
                    before_already_in_axis_band=bool(before_distance[i]<=1),
                    delta_y_mm=-u[i,0]*TICKS[0],delta_theta_deg=-u[i,1]*TICKS[1],
                    next_y_mm=after[i,0]*TICKS[0],next_theta_deg=after[i,1]*TICKS[1],
                    reaches_axis_band=bool(after_distance[i]<=1),
                    closer_to_axis=bool(after_distance[i]<before_distance[i])))
        # Coordinate means retain equal weight per physical command location.
        coord=g.groupby(COORD,as_index=False)[PRED+['error_y_mm','error_theta_deg']].mean()
        repeat=g.groupby(COORD)[['error_y_mm','error_theta_deg']].std()
        coord=coord.merge(repeat.add_suffix('_repeat_sd'),on=COORD)
        coord['model']=name
        groups.append(coord)
        for varying,fixed in [('theta_deg','y_mm'),('y_mm','theta_deg')]:
            for val,h in coord.groupby(fixed):
                h=h.sort_values(varying)
                for i in range(1,len(h)):
                    a,b=h.iloc[i-1],h.iloc[i]; step=b[varying]-a[varying]
                    secants.append(dict(model=name,varying=varying,fixed=fixed,fixed_value=val,
                        start=a[varying],end=b[varying],step=step,
                        error_y_change_per_unit=(b.error_y_mm-a.error_y_mm)/step,
                        error_theta_change_per_unit=(b.error_theta_deg-a.error_theta_deg)/step))
        # Descriptive affine error surface in tick coordinates. Cross terms alone
        # do not establish contraction: include own-axis gain and intercept.
        qt=coord[COORD].to_numpy()/TICKS
        et=coord[['error_y_mm','error_theta_deg']].to_numpy()/TICKS
        design=np.column_stack([np.ones(len(qt)),qt])
        beta=np.linalg.lstsq(design,et,rcond=None)[0]
        B=beta[1:].T; J=np.eye(2)+B
        ry=np.eye(2);ry[0]-=J[0]
        rt=np.eye(2);rt[1]-=J[1]
        cv=np.zeros_like(et)
        for i in range(len(qt)):
            keep=np.arange(len(qt))!=i
            cv[i]=design[i]@np.linalg.lstsq(design[keep],et[keep],rcond=None)[0]
        diagnostic.append(dict(model=name,intercept_ticks=beta[0].tolist(),
            error_derivative_per_tick=B.tolist(),prediction_derivative_per_tick=J.tolist(),
            descriptive_error_surface_loco_rmse_ticks=np.sqrt(np.mean((cv-et)**2,axis=0)).tolist(),
            simultaneous_linearization_spectral_radius=float(max(abs(np.linalg.eigvals(-B)))),
            alternating_linearization_spectral_radius=float(max(abs(np.linalg.eigvals(rt@ry)))),
            cross_only_product=float(B[0,1]*B[1,0]),
            warning='Global descriptive fit across positive samples, not near-origin Jacobian or validated convergence'))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.concat(groups).to_csv(OUT/'coordinate_errors.csv',index=False)
    pd.DataFrame(secants).to_csv(OUT/'cross_axis_secants.csv',index=False)
    cmd=pd.DataFrame(commands);cmd.to_csv(OUT/'ideal_one_command.csv',index=False)
    rates=[]
    for (name,policy),g in cmd.groupby(['model','policy']):
        eligible=~g.before_already_in_axis_band
        rates.append(dict(model=name,policy=policy,images=len(g),
            axis_band_success=int(g.reaches_axis_band.sum()),closer=int(g.closer_to_axis.sum()),
            initially_outside_band=int(eligible.sum()),
            outside_to_band=int((eligible&g.reaches_axis_band).sum()),
            outside_closer=int((eligible&g.closer_to_axis).sum()),
            median_axis_distance_after_ticks=float(g.after_distance_ticks.median())))
    pd.DataFrame(rates).to_csv(OUT/'ideal_one_command_summary.csv',index=False)
    (OUT/'error_surface_diagnostics.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    manifest=dict(original_images_verified=len(verified),tick_sizes=TICKS.tolist(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        analysis_source_sha256=sha(Path(__file__)),new_estimator_training=False,
        actual_trajectory_data=False,actuator_assumption='q_next=q-command; independent exact axes',
        missing_requested_model='local origin/chore/hw-merge ef01a82 alignment() is a zero-valued placeholder',
        remote_check='ls-remote failed: could not resolve lab.ssafy.com',
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(rates).to_string(index=False))
    print(json.dumps(diagnostic,indent=2))


if __name__=='__main__':main()
