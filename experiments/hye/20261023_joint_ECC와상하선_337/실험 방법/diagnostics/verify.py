from pathlib import Path
import json
import numpy as np
import pandas as pd

H=Path(__file__).resolve().parent
m=pd.read_csv(H.parent/'manifest.csv')
a=np.load(H.parent/'features.npz')
lines=np.load(H.parent/'lines/features.npz')
d=pd.read_csv(H/'fold_separation.csv')
j=pd.read_csv(H/'fold_response_coefficients.csv')
aud=json.loads((H/'training_fold_audit.json').read_text())
assert np.array_equal(a['X_upper'][:,:54],a['X_qr'])
assert np.array_equal(a['X_lower'][:,:54],a['X_qr'])
assert np.array_equal(a['X_both'][:,:54],a['X_qr'])
assert np.array_equal(a['X_upper'][:,54:],lines['X_upper'])
assert np.array_equal(a['X_lower'][:,54:],lines['X_lower'])
assert np.array_equal(a['X_both'][:,54:78],lines['X_upper'])
assert np.array_equal(a['X_both'][:,78:],lines['X_lower'])
checks=[]
for fold in range(4):
    tr=np.array(aud[fold]['training_indices']); te=np.array(aud[fold]['excluded_indices'])
    assert not set(tr)&set(te)
    assert (m.iloc[tr].fold!=fold).all() and (m.iloc[tr].dataset==20261020).all()
    assert not set(m.iloc[tr].cv_group)&set(m.iloc[te].cv_group)
    for name in ['qr','upper','lower','both']:
        jj=j[(j.fold==fold)&(j.observation==name)].sort_values('feature')
        J=jj[['slope_Y_px_per_command','slope_T_px_per_command']].to_numpy().T
        row=d[(d.fold==fold)&(d.observation==name)&(d.metric=='raw_coordinate')].iloc[0]
        residual=.09*J[0]+row.Y_009_compensating_T*J[1]
        orth=float(residual@J[1])
        assert abs(orth)<1e-8
        expected_cond=np.sqrt((1+abs(row.cosine))/(1-abs(row.cosine)))
        assert np.isclose(expected_cond,row.normalized_jacobian_condition,rtol=1e-9)
        assert np.isclose(np.sqrt(np.mean(residual**2)),row.remaining_coordinate_RMS_px,rtol=1e-10)
        checks.append(dict(fold=fold,observation=name,orthogonality_dot=orth,
                           condition_formula_error=float(abs(expected_cond-row.normalized_jacobian_condition))))
result=dict(passed=True,checks=checks,
    feature_composition_exact=True,folds_group_disjoint=True,
    note='Independent algebra and artifact mapping checks; not validation of physical pose or new-photo performance.')
(H/'verification.json').write_text(json.dumps(result,indent=2))
print('Verified',len(checks),'fold/observation geometry cases and exact feature composition.')
