import sys,json,subprocess,csv
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;final=out.parents[1];root=final.parent;source=root/'codex_experiments_20261014/joint_yt_visual_feedback_20261023';sys.path.insert(0,str(source))
from models import ObservationModel,CONFIGS
X=np.load(final/'data/reference/codex_experiments_20261014/joint_yt_visual_feedback_20261023/features.npz')['X_both'];m=pd.read_csv(final/'config/images341.csv');y=m[['Y','T']].to_numpy();train=m.fold.to_numpy()!=0;records=[]
with threadpool_limits(1):
    for i,config in enumerate(CONFIGS):
        directory=out/str(i);directory.mkdir(exist_ok=False);input=directory/'input.json';input.write_text(json.dumps(dict(X=X[train].tolist(),commands=y[train].tolist(),dates=m.dataset[train].astype(str).tolist(),indices=m['index'][train].tolist(),test_X=X.tolist(),**config)));subprocess.run([str(final/'build/modern/bin/Release/fit_observation.exe'),str(input),str(directory/'native')],check=True,capture_output=True,text=True);actual=json.loads((directory/'native/model.json').read_text());expected=ObservationModel(config).fit(X[train],y[train],m[train]);error=float(np.max(abs(np.array(actual['predictions'])-expected.predict(X))));ids_equal=actual['training_indices']==expected.training_indices.tolist();records.append(dict(config=config,max_prediction_difference=error,training_indices_equal=ids_equal,passed=error<=1e-6 and ids_equal))
summary=dict(passed=all(r['passed'] for r in records),cases=records,prediction_tolerance=1e-6,scope='Static fit/predict only; sequence optimizer not yet declared ported')
(out/'checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));sys.exit(0 if summary['passed'] else 1)
