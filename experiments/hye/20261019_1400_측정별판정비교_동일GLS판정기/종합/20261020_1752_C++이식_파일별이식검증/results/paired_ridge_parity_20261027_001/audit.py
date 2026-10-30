import sys,json,subprocess,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;final=out.parents[1];root=final.parent
spec=importlib.util.spec_from_file_location('paired_ridge_oracle',root/'codex_experiments_20261014/joint_yt_disentangle_20261022/paired_model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
bank=np.load(final/'results/generalization_feature_bank_native_20261027_001/feature_bank.npz');m=pd.read_csv(final/'results/generalization_data_native_20261027_001/manifest_with_folds.csv');y=m[['Y','T']].to_numpy();train=m.fold.to_numpy()!=0;dates=m.dataset.to_numpy();results=[]
with threadpool_limits(1):
    for feature in ['P12','raw1260','raw_shift']:
        X=bank[feature]
        for alpha,pair,repeat in [(.0001,0,0),(.01,1,1),(1.,10,10)]:
            config=dict(alpha=alpha,pair_weight=pair,repeat_weight=repeat);directory=out/f'{feature}_{alpha}';directory.mkdir(exist_ok=False);path=directory/'input.json';path.write_text(json.dumps(dict(X=X[train].tolist(),commands=y[train].tolist(),dates=[str(d) for d in dates[train]],test_X=X.tolist(),config=config)));run=subprocess.run([str(final/'build/modern/bin/Release/paired_ridge.exe'),str(path),str(directory/'native')],capture_output=True,text=True);assert run.returncode==0,run.stderr;actual=json.loads((directory/'native/model.json').read_text());model=mod.PairedRidge(config).fit(X[train],y[train],dates[train]);delta=float(np.max(abs(np.array(actual['prediction'])-model.predict(X))));results.append(dict(feature=feature,config=config,max_prediction_difference=delta,passed=delta<=1e-6))
summary=dict(passed=all(r['passed'] for r in results),prediction_tolerance=1e-6,cases=results,scope='Pair graphs and ridge fit use training metadata only; prediction receives features only')
(out/'checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));sys.exit(0 if summary['passed'] else 1)
