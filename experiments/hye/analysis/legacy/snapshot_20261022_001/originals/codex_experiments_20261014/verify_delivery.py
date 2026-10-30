"""Meaningful pipeline checks, not a second training-set accuracy claim."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import zero_experiment as Z
from infer import run,solve_calibrated,clean

def main():
    out=Z.HERE/'verification';out.mkdir(exist_ok=True)
    checks={}
    manifest=pd.read_csv(Z.HERE/'manifest.csv')
    checks['source_files_unchanged_size_and_mtime']=all((Z.ROOT/('center.bmp' if r.axis=='C' else ('Theta/' if r.axis=='T' else 'Y/')+r.file)).stat().st_size==r.size and abs((Z.ROOT/('center.bmp' if r.axis=='C' else ('Theta/' if r.axis=='T' else 'Y/')+r.file)).stat().st_mtime-r.mtime)<1e-5 for r in manifest.itertuples())
    for protocol in ['stratified','session_out']:
        file=Z.HERE/f'modeling/unified_isolated_{protocol}_predictions.csv'
        d=pd.read_csv(file)
        checks[f'{protocol}_all_280_once']=len(d)==280 and d.names.nunique()==280
        recalculated=((d.pred_axis==d['axes'])&(np.sign(d.pred_value)==np.sign(d['values']))).mean()
        recorded=json.loads((Z.HERE/f'modeling/unified_isolated_{protocol}_metrics.json').read_text())
        checks[f'{protocol}_metrics_match_predictions']=abs(recalculated-recorded['joint_accuracy'])<1e-12
    center=Z.ROOT/'center.bmp'
    results=[]
    for name in ['T+0.01.bmp','T-0.05.bmp','T+1.00.bmp','Y+0.05.bmp','Y-2.00.bmp']:
        p=Z.ROOT/('Theta' if name[0]=='T' else 'Y')/name
        for mode in ['calibrated-runs','general']:
            a=run(center,p,mode);checks[f'{mode}_{name}_finite']=a['status']=='prediction' and np.isfinite(a['value'])
            results.append(dict(file=name,mode=mode,axis=a.get('axis'),value=a.get('value'),valid_points=a.get('valid_points')))
    # Arbitrary file names must not influence inference.
    alias=out/'arbitrary_unlabelled_input.bmp'
    try:
        source=Z.ROOT/'Theta'/'T-0.05.bmp';alias.write_bytes(source.read_bytes())
        for mode in ['calibrated-runs','general']:
            a=run(center,source,mode);b=run(center,alias,mode)
            checks[f'{mode}_filename_invariance']=a['axis']==b['axis'] and abs(a['value']-b['value'])<1e-12
            checks[f'{mode}_json_serializable']=json.loads(json.dumps(clean(a),allow_nan=False))['axis']==a['axis']
    finally:
        if alias.exists():alias.unlink()
    checks['identical_reference_returns_zero']=run(center,center)['value']==0.
    # Compare calibrated runtime against feature-cache inference, never its label.
    params=json.loads((Z.HERE/'calibrated_runs/model.json').read_text())
    m,X,_,_=Z.load('gn17_spread14');i=m.index[m.file=='T-0.05.bmp'][0]
    cached=solve_calibrated(X[i],params);runtime=run(center,Z.ROOT/'Theta'/'T-0.05.bmp')
    checks['calibrated_batch_runtime_equivalent']=cached['axis']==runtime['axis'] and abs(cached['value']-runtime['value'])<2e-6
    checks={k:bool(v) for k,v in checks.items()}
    (out/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    pd.DataFrame(results).to_csv(out/'runtime_examples.csv',index=False)
    assert all(checks.values()),checks
    print('All',len(checks),'pipeline checks passed.')

if __name__=='__main__':main()
