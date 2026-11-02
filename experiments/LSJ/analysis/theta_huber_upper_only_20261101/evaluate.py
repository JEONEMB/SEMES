"""Run upper-only native model on exactly the prior 254 images; paired comparison."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
import runpy
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MODEL=ROOT/'algorithm/theta_huber_upper_only_20261101'
BASE=ROOT/'algorithm/theta_huber_20261101'
OLD=ROOT/'analysis/theta_huber_holdout_20261101'
EXE=ROOT/'.tmp/theta_huber_upper_only_20261101_build/Release/theta_upper_alignment.exe'
stats=runpy.run_path(str(OLD/'evaluate.py'))['stats']


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data): (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def stamp(): return datetime.now(timezone(timedelta(hours=9))).isoformat()


def paired(rows,old):
    both=[r for r in rows if r['execution_success'] and old[r['id']]['execution_success']]
    return {'comparable_images':len(both),
        'absolute_error_decreased':sum(r['absolute_theta_error_deg']<old[r['id']]['absolute_theta_error_deg'] for r in both),
        'absolute_error_increased':sum(r['absolute_theta_error_deg']>old[r['id']]['absolute_theta_error_deg'] for r in both),
        'absolute_error_equal':sum(r['absolute_theta_error_deg']==old[r['id']]['absolute_theta_error_deg'] for r in both),
        'both_pass':sum(r['within_tolerance'] and old[r['id']]['within_tolerance'] for r in both),
        'old_fail_new_pass':sum(r['within_tolerance'] and not old[r['id']]['within_tolerance'] for r in both),
        'old_pass_new_fail':sum(not r['within_tolerance'] and old[r['id']]['within_tolerance'] for r in both),
        'both_fail':sum(not r['within_tolerance'] and not old[r['id']]['within_tolerance'] for r in both)}


def main():
    if (HERE/'results.json').exists(): raise RuntimeError('Preserve completed evaluation; do not overwrite.')
    split=read(MODEL/'calibration/data_split.json')
    assert sha(MODEL/'calibration/data_split.json')==sha(BASE/'calibration/data_split.json')
    frozen_base=read(BASE/'validation/source_provenance.json')
    for p,h in frozen_base['implemented_sources'].items(): assert sha(BASE/p)==h,p
    oldrows=read(OLD/'results.json'); old={r['id']:r for r in oldrows}
    holdout=split['held_out']; train_ids={r['id'] for r in split['calibration']}
    assert len(holdout)==254 and [r['id'] for r in holdout]==[r['id'] for r in oldrows]
    assert train_ids.isdisjoint(old)
    assert {r['expected_sha256'] for r in holdout}.isdisjoint(r['expected_sha256'] for r in split['calibration'])
    config=HERE/'frozen_calibration.json'
    config.write_bytes((MODEL/'config/calibration.json').read_bytes())
    cfg=read(config); oldcfg=read(BASE/'config/calibration.json')
    for key in ('image','upper_region','huber'): assert cfg[key]==oldcfg[key],key
    assert sha(config)==read(MODEL/'calibration/provenance.json')['calibration_config_sha256']
    target=cfg['model']['target_theta_deg']
    tolerance=read(OLD/'protocol.json')['criteria']['absolute_theta_error_deg_lte']
    protected={str(p.relative_to(ROOT)):sha(p) for p in [EXE,MODEL/'config/calibration.json',config,
        MODEL/'calibration/data_split.json',OLD/'results.json',OLD/'summary.json',
        *MODEL.glob('src/*.cpp'),*MODEL.glob('include/*.hpp'),MODEL/'CMakeLists.txt',
        BASE/'src/edge_points.cpp',BASE/'src/huber_line.cpp',BASE/'src/utf8_io.cpp']}
    protocol={'started_kst':stamp(),'train_images':40,'test_images':254,
        'test_second':20,'test_third':234,'tolerance_deg':tolerance,
        'comparison':'Existing two-angle inverse vs upper-only forward OLS inverse.',
        'same_components':['training image IDs','test image IDs and labels','upper ROI and mask','Sobel settings','Huber source and settings','target theta','scoring function and tolerance'],
        'changed_components':['no lower point extraction or line fit','no two-equation Y/theta inverse','refit b and a on the same 40 training upper angles with phi=b+a*theta'],
        'Y_effect':'Not corrected. Runtime takes no Y value and estimates no Y.',
        'input':'Only config path and image path; no test labels given to native executable.',
        'no_other_changes':'No Gaussian, new thresholds, clipping, tick rounding, point/image removal or post-test coefficient changes.',
        'interpretation_limit':'Paired feature/model ablation, not proof that lower-edge extraction itself is wrong. Same test set reused after observing baseline failures; exploratory comparison.',
        'physical_robot_executed':False,'original_model_files_unchanged':True,
        'protected_hashes_before':protected,'script_sha256':sha(Path(__file__)),
        'scoring_source_sha256':sha(OLD/'evaluate.py')}
    write('protocol.json',protocol); write('holdout_manifest.json',holdout)
    rows=[]; start_all=time.perf_counter()
    for i,item in enumerate(holdout,1):
        assert sha(Path(item['path']))==item['expected_sha256'],item['id']
        assert item['theta_deg']==old[item['id']]['theta_deg'] and item['y_mm']==old[item['id']]['y_mm']
        r={k:item[k] for k in ['id','dataset','file','path','expected_sha256','y_mm','theta_deg','repeat','is_reference']}
        r['label_correction']=item.get('label_correction')
        t=time.perf_counter()
        proc=subprocess.run([str(EXE),str(config),item['path']],capture_output=True,text=True,encoding='utf-8',errors='replace')
        r.update({'cli_elapsed_ms':(time.perf_counter()-t)*1000,'return_code':proc.returncode,
            'stderr':proc.stderr,'execution_success':False,'within_tolerance':False})
        if proc.returncode==0:
            try:
                out=json.loads(proc.stdout)
                if not all(math.isfinite(out[k]) for k in ('theta_estimate_deg','theta_correction_deg')): raise ValueError('nonfinite output')
                if out.get('lower_used') is not False or 'lower' in out: raise ValueError('lower feature unexpectedly used')
                if out.get('y_effect_corrected') is not False or out.get('y_command_available') is not False or out['y_correction_mm']!=0: raise ValueError('unexpected Y behavior')
                if out['theta_correction_deg']!=target-out['theta_estimate_deg']: raise ValueError('command sign mismatch')
                if out['upper']!=old[item['id']]['output']['upper']: raise ValueError('upper extraction/Huber differs from baseline')
                e=out['theta_estimate_deg']-item['theta_deg']
                r.update({'output':out,'theta_error_deg':e,'absolute_theta_error_deg':abs(e),
                    'expected_theta_correction_deg':target-item['theta_deg'],
                    'command_error_deg':out['theta_correction_deg']-(target-item['theta_deg']),
                    'implied_residual_deg':item['theta_deg']+out['theta_correction_deg']-target,
                    'execution_success':True,'within_tolerance':abs(e)<=tolerance,
                    'upper_measurement_exactly_matches_baseline':True,
                    'baseline_theta_error_deg':old[item['id']]['theta_error_deg'],
                    'baseline_within_tolerance':old[item['id']]['within_tolerance']})
            except (ValueError,TypeError,KeyError) as e:
                r['failure']='invalid output/comparison: '+str(e);r['stdout']=proc.stdout
        else: r['failure']='native execution failed';r['stdout']=proc.stdout
        rows.append(r)
        if i%25==0 or i==len(holdout):
            write('progress.json',{'completed':i,'total':254,'execution_failures':sum(not r['execution_success'] for r in rows)})
            print(f'{i}/254 complete; failures={sum(not x["execution_success"] for x in rows)}',flush=True)
    for p,h in protected.items(): assert sha(ROOT/p)==h,p
    for p,h in frozen_base['implemented_sources'].items(): assert sha(BASE/p)==h,p
    write('results.json',rows)
    summary={'completed_kst':stamp(),'elapsed_seconds':time.perf_counter()-start_all,'tolerance_deg':tolerance,
        'overall':stats(rows),'baseline_overall':stats(oldrows),
        'by_dataset':{str(d):stats([r for r in rows if r['dataset']==d]) for d in (2,3)},
        'baseline_by_dataset':{str(d):stats([r for r in oldrows if r['dataset']==d]) for d in (2,3)},
        'paired':paired(rows,old),'paired_by_dataset':{str(d):paired([r for r in rows if r['dataset']==d],old) for d in (2,3)},
        'by_dataset_y':[{'dataset':d,'y_mm':y,**stats([r for r in rows if r['dataset']==d and r['y_mm']==y])}
            for d in (2,3) for y in sorted({r['y_mm'] for r in rows if r['dataset']==d})],
        'worst_images':[{'id':r['id'],'dataset':r['dataset'],'y_mm':r['y_mm'],'theta_deg':r['theta_deg'],
            'estimate_deg':r['output']['theta_estimate_deg'],'error_deg':r['theta_error_deg'],'baseline_error_deg':old[r['id']]['theta_error_deg']}
            for r in sorted([r for r in rows if r['execution_success']],key=lambda r:r['absolute_theta_error_deg'],reverse=True)[:10]],
        'integrity':{'baseline_sources_unchanged':True,'upper_results_identical':sum(r.get('upper_measurement_exactly_matches_baseline',False) for r in rows),
                     'protected_files_unchanged':True,'same_test_ids_and_labels':True,'all_image_hashes_matched':True},
        'physical_robot_executed':False}
    write('summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('overall','by_dataset','paired','worst_images')},indent=2),flush=True)


if __name__=='__main__':main()
