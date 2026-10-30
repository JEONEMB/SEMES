"""Smoke-check frozen experimental inference on original images with neutral names."""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
from predict import run

HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def main():
    ref=json.loads((BASE/'reference.json').read_text(encoding='utf8'))
    root=Path(ref['source_path']).parent
    inputs=HERE/'raw_checks'
    inputs.mkdir(exist_ok=True)
    center=inputs/'center.bmp';current=inputs/'current.bmp'
    shutil.copy2(root/'center.bmp',center)
    predictions=pd.read_csv(HERE/'predictions.csv')
    rows=[]
    for name in ['Y-0.80.bmp','T+0.01.bmp']:
        shutil.copy2(root/('Y' if name.startswith('Y') else 'Theta')/name,current)
        for method in ['baseline','ols_gls_lw','spatial_wls','spatial_gls_lw']:
            actual=run(center,current,method)
            expected=predictions[(predictions.method==method)&(predictions.protocol=='full280')&(predictions.file==name)].iloc[0]
            diff=abs(actual['value']-expected.pred)
            assert actual['axis']==expected.pred_axis and diff<1e-7
            rows.append(dict(file=name,method=method,pred_axis=actual['axis'],pred_value=actual['value'],
                cache_value=float(expected.pred),absolute_difference=diff,valid_points=actual['valid_points']))
    out=dict(cases=len(rows),input_filenames_unlabelled=True,all_passed=True,
             max_difference=max(r['absolute_difference'] for r in rows),results=rows)
    (HERE/'raw_inference_checks.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
