"""Original Python is used only as an independent audit oracle."""
import sys,json,subprocess,importlib.util
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
script=Path(__file__).resolve().parent;final=script.parents[1];root=final.parent
out=final/'results/calibration_parity_20261027_005';out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('calibration_oracle',root/'codex_experiments_20261014/joint_yt_visual_feedback_20261023/control/controller.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
archive=np.load(final/'data/reference/codex_experiments_20261014/joint_yt_visual_feedback_20261023/features.npz')
import csv
rows=list(csv.DictReader((final/'data/reference/codex_experiments_20261014/joint_yt_visual_feedback_20261023/manifest.csv').open(encoding='utf-8-sig')))
results=[]
for condition in ['qr','upper','lower','both']:
    X=archive['X_'+condition];good=archive['success_'+condition].astype(bool)&np.isfinite(X).all(1);ix=np.flatnonzero(good);X=X[ix];q=np.array([[float(rows[i]['Y']),float(rows[i]['T'])] for i in ix]);dates=[rows[i]['dataset'] for i in ix]
    path=out/(condition+'_training.json');path.write_text(json.dumps(dict(X=X.tolist(),commands=q.tolist(),dates=dates)))
    subprocess.run([str(final/'build/modern/bin/Release/calibrate.exe'),str(path),str(out/condition)],check=True)
    actual=json.loads((out/condition/'calibration.json').read_text());expected=mod.Calibration.fit(X,q,np.array(dates,dtype=float),ix)
    P=np.array(actual['projection']);E=expected.projection
    # An SVD basis may rotate/sign-flip. Compare its invariant induced metric.
    metric=float(np.max(abs(P@P.T-E@E.T)));relative=metric/max(float(np.max(abs(E@E.T))),1)
    noise=abs(actual['noise_rms']-expected.noise_rms)
    results.append(dict(condition=condition,metric_relative_difference=relative,noise_rms_difference=noise,passed=relative<=1e-6 and noise<=1e-8))
summary=dict(criteria=dict(metric_relative_tolerance=1e-6,noise_rms_tolerance=1e-8),conditions=results,passed=all(x['passed'] for x in results),scope='TRAIN artifact parity only; invariant metric permits SVD basis rotations')
(out/'checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
sys.exit(0 if summary['passed'] else 1)

