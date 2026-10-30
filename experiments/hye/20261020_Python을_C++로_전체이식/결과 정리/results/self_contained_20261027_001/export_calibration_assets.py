"""One-time model format conversion, not training or a Python runtime dependency."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,joblib
sys.dont_write_bytecode=True
final=Path(__file__).resolve().parents[2];root=final.parent
source=root/'codex_experiments_20261014/joint_yt_visual_feedback_20261023/control';sys.path.insert(0,str(source))
destination=final/'models/feedback_full341_frozen';destination.mkdir(exist_ok=True)
records=[]
for condition in ['qr','upper','lower','both']:
    path=source/f'full_calibration_{condition}.joblib';bundle=joblib.load(path);cal=bundle['calibration']
    data=dict(condition=condition,projection=cal.projection.tolist(),jacobian=cal.jacobian.tolist(),noise_rms=float(cal.noise_rms),training_indices=bundle['training_indices'].tolist(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Frozen original training model converted to JSON; not a native retraining result; no independent accuracy claim')
    output=destination/f'{condition}.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(data,indent=2),encoding='utf-8');loaded=json.loads(output.read_text())
    for field,value in [('projection',cal.projection),('jacobian',cal.jacobian)]:assert np.array_equal(np.array(loaded[field]),value)
    assert loaded['noise_rms']==float(cal.noise_rms)
    records.append(dict(condition=condition,source=str(path),source_sha256=data['source_sha256'],destination=str(output),destination_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),array_roundtrip_exact=True))
(destination/'conversion_manifest.json').write_text(json.dumps(dict(models=records,passed=True),indent=2))
print('Four frozen calibration assets converted with exact array round trips')
