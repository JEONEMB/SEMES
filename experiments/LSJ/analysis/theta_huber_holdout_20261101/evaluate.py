"""Frozen-model evaluation on the user-selected 254 held-out images.
No fitting, filtering of observations, threshold tuning, or model changes.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import subprocess
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / 'algorithm/theta_huber_20261101'
EXE = ROOT / '.tmp/theta_huber_20261101_build/Release/theta_alignment.exe'
TOLERANCE = 0.005


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def stats(rows):
    ok = [r for r in rows if r['execution_success']]
    errors = np.array([r['theta_error_deg'] for r in ok])
    passes = sum(r['within_tolerance'] for r in rows)
    s = {'images': len(rows), 'execution_success': len(ok),
         'execution_failures': len(rows)-len(ok), 'within_tolerance': passes,
         'outside_tolerance': len(ok)-passes,
         'pass_rate_percent_all_images': passes/len(rows)*100 if rows else None}
    if not ok:
        return s
    s.update({'mean_error_deg': float(errors.mean()), 'mae_deg': float(np.abs(errors).mean()),
              'rmse_deg': float(np.sqrt(np.mean(errors**2))),
              'max_absolute_error_deg': float(np.abs(errors).max()),
              'p95_absolute_error_deg': float(np.percentile(np.abs(errors),95)),
              'min_error_deg': float(errors.min()), 'max_error_deg': float(errors.max()),
              'cli_elapsed_ms_median': float(np.median([r['cli_elapsed_ms'] for r in ok])),
              'cli_elapsed_ms_p95': float(np.percentile([r['cli_elapsed_ms'] for r in ok],95))})
    return s


def main():
    if (HERE/'results.json').exists():
        raise RuntimeError('Completed run exists; preserve it instead of overwriting.')
    split = read(MODEL/'calibration/data_split.json')
    frozen = read(MODEL/'validation/source_provenance.json')
    holdout, train = split['held_out'], split['calibration']
    assert Counter(r['dataset'] for r in holdout)=={2:20,3:234}
    assert len(train)==40
    assert {r['id'] for r in holdout}.isdisjoint(r['id'] for r in train)
    assert {r['expected_sha256'] for r in holdout}.isdisjoint(r['expected_sha256'] for r in train)
    assert sha(EXE)==frozen['binaries']['theta_alignment.exe']
    for p,h in frozen['implemented_sources'].items():
        assert sha(MODEL/p)==h,p
    original_config = MODEL/'config/calibration.json'
    config = HERE/'frozen_calibration.json'
    config.write_bytes(original_config.read_bytes())
    target = read(config)['model']['target_theta_deg']
    protected = {'executable': EXE, 'original_config': original_config, 'frozen_config': config,
                 'split': MODEL/'calibration/data_split.json'}
    before = {k:sha(p) for k,p in protected.items()}
    protocol = {
        'started_kst':timestamp(), 'calibration_images':40,
        'test_images':254, 'test_second':20, 'test_third':234,
        'selection':'All second-data repeat 3 including center3; all third data.',
        'criteria':{'absolute_theta_error_deg_lte':TOLERANCE,'target_theta_deg':target},
        'theta_error_definition':'theta_estimate - corrected nominal theta label',
        'command_error_definition':'returned correction - (target - theta label) = -theta_error',
        'implied_residual_definition':'theta label + returned correction - target; assumes command executes exactly, not a measured post-move result.',
        'invalid_output_policy':'Retain failed image in the overall denominator; no imputed correction.',
        'precision':'Unrounded double values for all pass decisions; no additional epsilon.',
        'source_labels':'Existing user-confirmed label corrections retained; no edits to original images/TXTs.',
        'runtime_inputs':'Executable receives config path and image path only; labels used by scorer after inference.',
        'changes':'No fitting, retraining, image omission, clipping, tick rounding, Gaussian, or new point rejection.',
        'timing':'Per image CLI wall time includes process startup, config/image read, extraction, fitting, decoding, output; excludes SHA256 precheck.',
        'prior_exposure':'Internal fixed holdout: these datasets were explored previously, but held-out angles/labels were not used for this model coefficient fit.',
        'physical_robot_executed':False,
        'hashes_before':before,
        'script_sha256':sha(Path(__file__)),
    }
    write('protocol.json',protocol)
    write('holdout_manifest.json',holdout)
    rows=[]
    t_all=time.perf_counter()
    for i,item in enumerate(holdout,1):
        path=Path(item['path'])
        if sha(path)!=item['expected_sha256']:
            raise RuntimeError('Input image changed: '+item['id'])
        row={k:item[k] for k in ['id','dataset','file','path','expected_sha256','y_mm','theta_deg','repeat','is_reference']}
        row['label_correction']=item.get('label_correction')
        start=time.perf_counter()
        process=subprocess.run([str(EXE),str(config),str(path)],capture_output=True,
                               text=True,encoding='utf-8',errors='replace')
        row['cli_elapsed_ms']=(time.perf_counter()-start)*1000
        row.update({'return_code':process.returncode,'stderr':process.stderr,
                    'execution_success':False,'within_tolerance':False})
        if process.returncode==0:
            try:
                result=json.loads(process.stdout)
                for key in ['theta_estimate_deg','theta_correction_deg','nuisance_y_estimate_mm']:
                    if not math.isfinite(result[key]):
                        raise ValueError('non-finite '+key)
                if result['y_command_available'] is not False or result['y_correction_mm']!=0:
                    raise ValueError('unexpected Y command')
                if result['theta_correction_deg']!=target-result['theta_estimate_deg']:
                    raise ValueError('correction/estimate mismatch')
                row['output']=result
                row['expected_theta_correction_deg']=target-item['theta_deg']
                row['theta_error_deg']=result['theta_estimate_deg']-item['theta_deg']
                row['absolute_theta_error_deg']=abs(row['theta_error_deg'])
                row['command_error_deg']=result['theta_correction_deg']-row['expected_theta_correction_deg']
                row['implied_residual_deg']=item['theta_deg']+result['theta_correction_deg']-target
                row['execution_success']=True
                row['within_tolerance']=row['absolute_theta_error_deg']<=TOLERANCE
            except (ValueError,KeyError,TypeError) as e:
                row['failure']='invalid output: '+str(e)
                row['stdout']=process.stdout
        else:
            row['failure']='native executable failed'
            row['stdout']=process.stdout
        rows.append(row)
        if i%25==0 or i==len(holdout):
            write('progress.json',{'completed':i,'total':len(holdout),'execution_failures':sum(not r['execution_success'] for r in rows)})
            print(f'{i}/{len(holdout)} complete; execution failures={sum(not r["execution_success"] for r in rows)}',flush=True)
    after={k:sha(p) for k,p in protected.items()}
    assert before==after,'Model/split changed during evaluation'
    for p,h in frozen['implemented_sources'].items():
        assert sha(MODEL/p)==h,p
    write('results.json',rows)
    summary={'completed_kst':timestamp(),'total_wall_seconds':time.perf_counter()-t_all,
        'tolerance_deg':TOLERANCE,'overall':stats(rows),
        'by_dataset':{str(d):stats([r for r in rows if r['dataset']==d]) for d in (2,3)},
        'by_dataset_y':[{'dataset':d,'y_mm':y,**stats([r for r in rows if r['dataset']==d and r['y_mm']==y])}
            for d in (2,3) for y in sorted({r['y_mm'] for r in rows if r['dataset']==d})],
        'worst_images':[{'id':r['id'],'dataset':r['dataset'],'y_mm':r['y_mm'],'theta_deg':r['theta_deg'],
                         'estimate_deg':r['output']['theta_estimate_deg'],'error_deg':r['theta_error_deg']}
                        for r in sorted([r for r in rows if r['execution_success']],key=lambda r:r['absolute_theta_error_deg'],reverse=True)[:10]],
        'integrity':{'model_and_split_unchanged':before==after,'all_image_hashes_matched':True,
                     'no_training_test_hash_overlap':True,'hashes_after':after},
        'physical_robot_executed':False}
    write('summary.json',summary)
    write('condition_summary.json',[{'dataset':d,'y_mm':y,'theta_deg':t,
        **stats([r for r in rows if r['dataset']==d and r['y_mm']==y and r['theta_deg']==t])}
        for d,y,t in sorted({(r['dataset'],r['y_mm'],r['theta_deg']) for r in rows})])
    print(json.dumps({'overall':summary['overall'],'by_dataset':summary['by_dataset'],'worst_images':summary['worst_images'][:3]},indent=2),flush=True)


if __name__=='__main__':
    main()
