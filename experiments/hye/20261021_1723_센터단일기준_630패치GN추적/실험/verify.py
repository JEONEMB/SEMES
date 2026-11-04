from predict import *
import pandas as pd
expected=pd.read_csv(HERE/'predictions.csv');checks=[]
for n in ['Y+0.05.bmp','T-0.05.bmp']:
    path=Z.ROOT/('Y' if n[0]=='Y' else 'Theta')/n
    # Neutral filenames ensure the predictor cannot derive truth from names.
    renamed=HERE/'unlabeled_input.bmp';renamed.write_bytes(path.read_bytes())
    r=predict(Z.ROOT/'center.bmp',renamed)
    row=expected[(expected.method=='v2_patch630_quality')&(expected.protocol=='full280')&(expected.file==n)].iloc[0]
    assert r['axis']==row.pred_axis and abs(r['value']-row.pred)<1e-8
    checks.append(dict(source=n,result=r,batch_match=True))
renamed.unlink()
(HERE/'inference_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
