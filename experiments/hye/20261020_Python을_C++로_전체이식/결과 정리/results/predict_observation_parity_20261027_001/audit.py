"""Native raw-image prediction compared with the original API."""
import sys,json,subprocess,importlib.util,csv
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;final=out.parents[1];root=final.parent;source=root/'codex_experiments_20261014/joint_yt_visual_feedback_20261023';sys.path.insert(0,str(source))
spec=importlib.util.spec_from_file_location('observation_predict_oracle',source/'predict.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
manifest=list(csv.DictReader((final/'config/images341.csv').open(encoding='utf-8-sig')));records=[]
with threadpool_limits(1):
    for condition in ['qr','upper','lower','both']:
        predictor=mod.Predictor(condition)
        for i in [0,80,291,336]:
            image=(final/'config'/manifest[i]['path']).resolve();destination=out/f'{condition}_{i}';subprocess.run([str(final/'build/modern/bin/Release/predict_observation.exe'),str(final/f'models/observation_full341_frozen/{condition}.json'),str(image),str(destination)],check=True,capture_output=True,text=True);actual=json.loads((destination/'prediction.json').read_text());expected=predictor.predict_bytes(image.read_bytes());delta=max(abs(actual[k]-expected[k]) for k in ['Y','T']);records.append(dict(condition=condition,index=i,max_prediction_difference=delta,hash_equal=actual['image_sha256']==expected['image_sha256'],passed=delta<=1e-5 and actual['image_sha256']==expected['image_sha256']))
summary=dict(passed=all(r['passed'] for r in records),absolute_prediction_tolerance=1e-5,cases=records,scope='Exposed image API parity using frozen full-development models; no independent validation')
(out/'checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));sys.exit(0 if summary['passed'] else 1)
