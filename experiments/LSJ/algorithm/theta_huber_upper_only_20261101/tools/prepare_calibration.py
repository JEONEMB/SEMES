"""Upper-only ablation: same 40 training images; no test labels/angles in fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parents[1]
BASE=HERE.parent/'theta_huber_20261101'
ROOT=HERE.parents[1]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    if (HERE/'config/calibration.json').exists():
        raise RuntimeError('Preserve the existing frozen calibration instead of overwriting.')
    old=read(BASE/'config/calibration.json')
    split=read(BASE/'calibration/data_split.json')
    train_ids={r['id'] for r in split['calibration']}
    assert len(train_ids)==40 and train_ids.isdisjoint(r['id'] for r in split['held_out'])
    rows=[r for r in read(BASE/'calibration/training_angles.json') if r['target']=='upper_structure']
    assert len(rows)==40 and {r['image_id'] for r in rows}==train_ids
    X=np.array([[1.,r['theta_deg']] for r in rows])
    phi=np.array([r['angle_deg'] for r in rows])
    assert np.linalg.matrix_rank(X)==2 and np.all(np.isfinite(phi))
    b,a=np.linalg.lstsq(X,phi,rcond=None)[0]
    assert np.isfinite(b) and np.isfinite(a) and a!=0
    config={'schema_version':1,'calibration_id':'D2_repeat12_upper_only_20261101',
        'image':old['image'],'upper_region':old['upper_region'],'huber':old['huber'],
        'model':{'type':'affine_forward_upper_only','intercept_deg':float(b),'per_theta_deg':float(a),
                 'target_theta_deg':old['model']['target_theta_deg']}}
    write(HERE/'config/calibration.json',config)
    write(HERE/'calibration/training_angles.json',rows)
    (HERE/'calibration/data_split.json').write_bytes((BASE/'calibration/data_split.json').read_bytes())
    provenance={'method':'OLS forward fit phi_upper=b+a*theta, then theta=(phi_upper-b)/a.',
        'model_change':'Remove lower feature and two-axis inverse. Fit one-angle forward equation with no Y term; Y is not supplied or estimated.',
        'constant_components':'Same training 40 images, upper ROI/occlusion mask, signed vertical Sobel, ordinary Huber, target zero, continuous correction.',
        'limits':'Cannot separate unknown Y and theta from one scalar angle. This compares a reduced model, not pure lower-edge measurement causality.',
        'held_out_reuse':'The same 254 evaluation images were seen in the prior model test; this is a paired exploratory comparison, not untouched validation.',
        'sources':{str(p.relative_to(ROOT)):sha(p) for p in [BASE/'config/calibration.json',BASE/'calibration/data_split.json',BASE/'calibration/training_angles.json',Path(__file__)]},
        'train_ids':[r['image_id'] for r in rows],
        'calibration_config_sha256':sha(HERE/'config/calibration.json')}
    write(HERE/'calibration/provenance.json',provenance)
    print(json.dumps({'training_images':len(rows),'model':config['model']},indent=2))


if __name__=='__main__': main()
