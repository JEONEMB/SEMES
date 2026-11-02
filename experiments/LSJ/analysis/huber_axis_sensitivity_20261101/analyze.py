"""Per-condition descriptive Pearson correlation, OLS, and sensitivity. No p-value gates."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
def read(name): return json.loads((HERE/name).read_text(encoding='utf-8'))
def dump(name,data): (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def linear(x,z):
    x=np.asarray(x,float);z=np.asarray(z,float)
    xx=x-x.mean();zz=z-z.mean()
    sx=float(xx@xx);sz=float(zz@zz)
    a=float((xx@zz)/sx);b=float(z.mean()-a*x.mean())
    pred=a*x+b;res=z-pred
    r=float(np.clip((xx@zz)/np.sqrt(sx*sz),-1,1)) if sz>0 else None
    return {'pearson_r':r,'slope':a,'intercept':b,
            'r_squared':r*r if r is not None else None,
            'rmse':float(np.sqrt(np.mean(res**2))),
            'max_abs_residual':float(np.max(abs(res))),
            'observed_min':float(z.min()),'observed_max':float(z.max()),
            'observed_range':float(np.ptp(z)),
            'residuals':res.tolist(),'fitted':pred.tolist()}

def main():
    images=read('manifest.json');measurements=read('measurements.json')
    index={(r['id'],r['target']):r for r in measurements}
    groups=[]
    def add(gid,label,dataset,axis,entries,fixed_axis,fixed_value,role='primary'):
        groups.append({'id':gid,'label':label,'dataset':dataset,'axis':axis,
                       'fixed_axis':fixed_axis,'fixed_value':fixed_value,'role':role,
                       'image_ids':[r['id'] for r in entries]})
    first=[r for r in images if r['dataset']==1 and not r['is_reference']]
    for cap in sorted({r['capture_group'] for r in first}):
        subset=[r for r in first if r['capture_group']==cap];axis=subset[0]['axis']
        add(f'D1_G{cap}',f'1차 {cap}번 촬영구간',1,axis,subset,'theta' if axis=='Y' else 'Y',None)
    second=[r for r in images if r['dataset']==2]
    for axis,fixed,fkey in [('Y','theta','theta_deg'),('theta','Y','y_mm')]:
        for value in sorted({r[fkey] for r in second}):
            subset=[r for r in second if r[fkey]==value]
            add(f'D2_{axis}_fixed_{value:g}',f'2차 {fixed}={value:g} 고정',2,axis,subset,fixed,value)
    third=[r for r in images if r['dataset']==3]
    for value in sorted({r['y_mm'] for r in third}):
        add(f'D3_theta_fixed_{value:g}',f'3차 Y={value:g}mm 고정',3,'theta',
            [r for r in third if r['y_mm']==value],'Y',value)
    movable=[r for r in second if not r['is_reference']]
    ylevels=sorted({r['y_mm'] for r in movable})
    common=set.intersection(*[{r['theta_deg'] for r in movable if r['y_mm']==v} for v in ylevels])
    for v in ylevels:
        add(f'D2_common_theta_Y_{v:g}',f'2차 공통 θ 조건 / Y={v:g}mm',2,'theta',
            [r for r in movable if r['y_mm']==v and r['theta_deg'] in common],
            'Y',v,'common_theta_levels_comparison')
    image_map={r['id']:r for r in images};metrics=[]
    for g in groups:
        field='y_mm' if g['axis']=='Y' else 'theta_deg'
        for target in ['upper_structure','lower_structure']:
            members=[];unavailable=[]
            for iid in g['image_ids']:
                row=index[iid,target];im=image_map[iid]
                if row.get('image_angle_deg') is None or not row['finite']:
                    unavailable.append({'id':iid,'reason':row['stop_reason']});continue
                members.append({'id':iid,'file':im['file'],'x':im[field],
                                'repeat':im['repeat'],'time':im['time'],
                                'image_angle_deg':row['image_angle_deg'],'image_slope_m':row['slope'],
                                'stop_reason':row['stop_reason'],
                                'band_endpoint_points':row['band_endpoint_points']})
            m={**g,'target':target,'attempted_images':len(g['image_ids']),
               'n':len(members),'members':members,'unavailable':unavailable,
               'nonconverged_included':sum(r['stop_reason']!='tolerance' for r in members)}
            levels=sorted({r['x'] for r in members});m['levels']=levels;m['distinct_levels']=len(levels)
            if len(levels)<2:
                m['status']='insufficient_axis_variation'
            else:
                x=[r['x'] for r in members]
                m['angle_regression']=linear(x,[r['image_angle_deg'] for r in members])
                m['slope_m_regression']=linear(x,[r['image_slope_m'] for r in members])
                m['input_step']=.05 if g['axis']=='Y' else .01
                m['predicted_angle_change_per_step_deg']=m['angle_regression']['slope']*m['input_step']
                m['status']='descriptive_fit'
            metrics.append(m)
    dump('groups.json',groups);dump('correlations.json',metrics)
    primary=[r for r in metrics if r['role']=='primary' and r['status']=='descriptive_fit']
    interactions={}
    for target in ['upper_structure','lower_structure']:
        rr=[r for r in metrics if r['role']=='common_theta_levels_comparison' and r['target']==target]
        interactions[target]={'common_theta_levels_deg':sorted(common),
            'sensitivities_by_y':[{ 'y_mm':r['fixed_value'], 'n':r['n'],
                                  'a_theta':r['angle_regression']['slope'],
                                  'pearson_r':r['angle_regression']['pearson_r'],
                                  'rmse_deg':r['angle_regression']['rmse'],
                                  'image_angle_change_per_tick_deg':r['predicted_angle_change_per_step_deg']} for r in rr],
            'a_theta_max_minus_min':float(np.ptp([r['angle_regression']['slope'] for r in rr])),
            'interpretation':'Observed differences over matched theta levels; no significance or causal claim.'}
    summary={'images_measured':len(images),'lines_measured':len(measurements),
             'primary_group_count':len([g for g in groups if g['role']=='primary']),
             'primary_computable_group_count':len(primary)//2,
             'primary_computable_line_groups':len(primary),
             'common_condition_line_groups':8,
             'not_computable':[{k:r[k] for k in ['id','label','axis','target','n','levels','status']}
                               for r in metrics if r['status']!='descriptive_fit'],
             'fit_stops':dict(Counter(r['stop_reason'] for r in measurements)),
             'extraction_endpoint_counts':{'total':sum(r['band_endpoint_points'] for r in measurements),
                  'lines_with_endpoint_points':sum(r['band_endpoint_points']>0 for r in measurements)},
             'interactions':interactions,
             'policy':'No image/point rejection, no condition averages, no data pooling across datasets/capture segments, no physical theta error conversion, no arbitrary r acceptance threshold or hypothesis-test p-value.'}
    dump('summary.json',summary)
    for r in primary:
        a=r['angle_regression']
        print(f"{r['id']:24} {r['target']:16} n={r['n']:3} r={a['pearson_r']:+.6f} a={a['slope']:+.8f} RMS={a['rmse']:.6f} step={r['predicted_angle_change_per_step_deg']:+.8f}")
    print(json.dumps({'fit_stops':summary['fit_stops'],'endpoint_counts':summary['extraction_endpoint_counts'],
                      'interactions':interactions},ensure_ascii=True,indent=2))

if __name__=='__main__': main()
