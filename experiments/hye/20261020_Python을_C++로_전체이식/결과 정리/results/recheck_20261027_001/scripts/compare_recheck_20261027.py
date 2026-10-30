"""Compare newly executed C++ results with retained C++ results, not accuracy claims."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/recheck_20261027_001'
BASELINES = {
 'full280':'full280_001', 'oof4':'cpp_oof4_001',
 'agent':'historical_agent_eval_003', 'theta':'historical_theta_only_001',
 'affine':'historical_agent_measure_001',
 'v2_estimate':'historical_v2_estimate_002', 'homography':'historical_v2_homography_001',
 'v2_judge':'historical_v2_judge_001', 'v2_outputs':'historical_v2_outputs_001',
 'v2_workbook':'historical_v2_workbook_parity_001',
 'v1_grid':'historical_v1_grid_002','v1_track':'historical_v1_track_frozen_001',
 'v1_combos':'historical_v1_combos_001','v1_stable':'historical_v1_stable_001',
 'v3_evaluate':'historical_v3_evaluate_001','v3_outputs':'historical_v3_outputs_20261023_002',
 'twin_lines':'historical_twin_lines_20261023_001','twin_photos':'historical_twin_photos_20261023_001',
 'slide_data':'historical_slide_data_20261023_001','slide_panels':'historical_slide_panels_20261023_001',
 'zero_offset':'historical_zero_offset_20261023_003','qr_only':'historical_qr_only_20261023_001',
 'ecc_tune':'historical_ecc_tune_20261023_002','spline':'historical_spline_20261023_001',
 'ecc_tune_parallel':'historical_ecc_tune_20261023_002',
 'pixel_fast':'historical_pixel_fast_20261023_001','pixel_slow':'historical_pixel_slow_20261023_001',
 'g1':'historical_run_g1_20261023_001','g1_parity':'historical_g1_parity_20261023_001',
 'robot':'historical_robot_motion_002','video':'historical_video_preview_001',
 'joint_fresh341':'joint_fresh341_001','joint_replay341':'joint_replay341_002',
 'aggregation':'aggregation_parity_001','parity280':'parity280_001',
}

def equal(a,b):
    if a == b:
        return True, 0.0
    try:
        x,y=float(a),float(b)
        if math.isnan(x) and math.isnan(y):return True,0.0
        if not math.isfinite(x) or not math.isfinite(y):return x==y,0.0
        delta=abs(x-y)
        return delta <= 1e-9,delta
    except (ValueError,TypeError):
        return False,0.0

def compare_csv(old,new):
    def read(p):
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.reader(f))
    a,b=read(old),read(new)
    if not a or not b:return {'passed':a==b,'rows':len(b)}
    if a[0] != b[0] or len(a)!=len(b):
        return {'passed':False,'reason':'header or row count mismatch','old_rows':len(a),'new_rows':len(b)}
    skip={i for i,k in enumerate(a[0]) if k in ('seconds','created_kst') or 'sha256' in k}
    mismatch=[]; n=0; maximum=0.0
    for row,(x,y) in enumerate(zip(a[1:],b[1:]),2):
        if len(x)!=len(y):mismatch.append({'row':row,'reason':'column count'});continue
        for col,(u,v) in enumerate(zip(x,y)):
            if col in skip:continue
            ok,delta=equal(u,v);n+=1;maximum=max(maximum,delta)
            if not ok and len(mismatch)<10:mismatch.append({'row':row,'column':a[0][col],'old':u,'new':v})
    return {'passed':not mismatch,'rows':len(b)-1,'cells':n,'max_abs_numeric_difference':maximum,'mismatch_examples':mismatch,'excluded_columns':[a[0][i] for i in skip]}

def compare_json(old,new):
    a=json.loads(old.read_text(encoding='utf-8-sig'));b=json.loads(new.read_text(encoding='utf-8-sig'))
    excluded=[];diff=[];count=0;maximum=0
    def walk(x,y,path=''):
        nonlocal count,maximum
        if isinstance(x,dict) and isinstance(y,dict):
            for k in sorted(set(x)|set(y)):
                p=path+'/'+k
                if k in ('created_kst','created_utc','seconds','seconds_mean','seconds_median','seconds_max','feature_source','output','output_directory','input_path') or k.endswith('_seconds') or 'sha256' in k:
                    excluded.append(p);continue
                if k not in x or k not in y:diff.append({'path':p,'reason':'key mismatch'});continue
                walk(x[k],y[k],p)
        elif isinstance(x,list) and isinstance(y,list):
            if len(x)!=len(y):diff.append({'path':path,'reason':'length mismatch'});return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'/'+str(i))
        else:
            count+=1;ok,delta=equal(x,y);maximum=max(maximum,delta)
            if not ok and len(diff)<10:diff.append({'path':path,'old':x,'new':y})
    walk(a,b)
    return {'passed':not diff,'values':count,'max_abs_numeric_difference':maximum,'mismatch_examples':diff,'excluded_fields':excluded}

def main():
    results=[]
    commands=[]
    for log in ('commands.json','supplement_commands.json','ecc_parallel_commands.json'):
        if (OUT/log).is_file():
            commands.extend(json.loads((OUT/log).read_text(encoding='utf-8')))
    for name,baseline in BASELINES.items():
        fresh=OUT/name;old=ROOT/'results'/baseline
        if not fresh.is_dir() or not any(c['exit_code']==0 and str(fresh) in c['command'] for c in commands):continue
        item={'run':name,'baseline':baseline,'files':[]}
        for p in sorted(fresh.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(fresh);q=old/rel
            if not q.is_file():continue
            if p.suffix=='.csv':result=compare_csv(q,p)
            elif p.suffix=='.json':result=compare_json(q,p)
            elif p.suffix=='.png':
                from PIL import Image
                with Image.open(q) as a,Image.open(p) as b:
                    result={'passed':a.size==b.size and a.mode==b.mode and a.tobytes()==b.tobytes(),'comparison':'decoded pixels','size':b.size}
            else:continue
            item['files'].append({'file':rel.as_posix(),**result})
        item['passed']=bool(item['files']) and all(f['passed'] for f in item['files'])
        results.append(item)
    (OUT/'reproduction_comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in results:
        bad=[f['file'] for f in r['files'] if not f['passed']]
        print(r['run'],r['passed'],len(r['files']),'files',bad[:8])

if __name__=='__main__':main()
