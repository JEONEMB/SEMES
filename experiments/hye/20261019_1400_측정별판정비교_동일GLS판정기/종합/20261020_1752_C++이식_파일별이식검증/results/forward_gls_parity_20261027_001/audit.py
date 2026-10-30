"""Original forward calibration oracle, using exposed development features."""
import sys,json,subprocess,csv,importlib.util
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;final=out.parents[1];root=final.parent
spec=importlib.util.spec_from_file_location('forward_gls_oracle',root/'codex_experiments_20261014/joint_yt_generalization_20261022/forward_model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
features=np.load(final/'data/reference/codex_experiments_20261014/joint_yt_visual_feedback_20261023/features.npz')['X_both']
rows=list(csv.DictReader((final/'config/images341.csv').open(encoding='utf-8-sig')));commands=np.array([[float(r['Y']),float(r['T'])] for r in rows]);train=np.array([int(r['fold'])!=0 for r in rows]);dates=np.array([r['dataset'] for r in rows])[train];weights=np.array([1/sum(dates==d) for d in dates]);results=[]
with threadpool_limits(1):
    for degree in [1,2,3]:
        for shrink in [.01,.1]:
            directory=out/f'degree{degree}_shrink{shrink}';directory.mkdir(exist_ok=False);request=dict(X=features[train].tolist(),commands=commands[train].tolist(),weights=weights.tolist(),test_X=features.tolist(),degree=degree,shrink=shrink);path=directory/'input.json';path.write_text(json.dumps(request));subprocess.run([str(final/'build/modern/bin/Release/forward_gls.exe'),str(path),str(directory/'native')],check=True,capture_output=True,text=True)
            actual=json.loads((directory/'native/model.json').read_text());expected=mod.ForwardGLS(degree,shrink).fit(features[train],commands[train],weights);prediction=expected.predict(features);delta=float(np.max(abs(np.array(actual['predictions'])-prediction)));origin=float(np.max(abs(np.array(actual['origin'])-expected.origin)));results.append(dict(degree=degree,shrink=shrink,max_prediction_difference=delta,max_origin_difference=origin,passed=delta<=1e-6 and origin<=1e-8))
summary=dict(passed=all(x['passed'] for x in results),cases=results,prediction_tolerance=1e-6,scope='Numerical algorithm parity on exposed development data with training fold0 excluded; no independent performance claim')
(out/'checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));sys.exit(0 if summary['passed'] else 1)
