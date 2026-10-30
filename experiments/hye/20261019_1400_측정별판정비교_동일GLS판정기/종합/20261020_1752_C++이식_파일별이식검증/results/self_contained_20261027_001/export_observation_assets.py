"""One-time conversion of frozen regression assets; not native training evidence."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,joblib
sys.dont_write_bytecode=True
final=Path(__file__).resolve().parents[2];root=final.parent;source=root/'codex_experiments_20261014/joint_yt_visual_feedback_20261023';sys.path.insert(0,str(source))
destination=final/'models/observation_full341_frozen';destination.mkdir(exist_ok=False);records=[]
for condition in ['qr','upper','lower','both']:
    path=source/f'{condition}_model.joblib';bundle=joblib.load(path);observation=bundle['model'];f=observation.model
    model={k:np.asarray(getattr(f,k)).tolist() for k in ['origin','coef','std','projector','gram','initial','scale','lower','upper']};model.update(degree=f.degree,shrink=f.shrink)
    data=dict(condition=condition,scope=observation.config['scope'],model=model,training_indices=observation.training_indices.tolist(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),notice='Frozen full-development model format conversion; no new training or independent validation claim')
    output=destination/f'{condition}.json';output.write_text(json.dumps(data,indent=2));reloaded=json.loads(output.read_text());assert all(np.array_equal(np.asarray(model[k]),np.asarray(reloaded['model'][k])) for k in model)
    records.append(dict(condition=condition,source_sha256=data['source_sha256'],json_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),numeric_roundtrip_exact=True))
(destination/'conversion_manifest.json').write_text(json.dumps(dict(passed=True,models=records),indent=2));print('Four observation regression models converted with exact numeric round trips')
