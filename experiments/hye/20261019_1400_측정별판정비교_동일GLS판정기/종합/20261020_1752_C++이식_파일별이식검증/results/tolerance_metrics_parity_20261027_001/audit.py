import sys,json,subprocess,importlib.util
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;final=out.parents[1];root=final.parent
spec=importlib.util.spec_from_file_location('tolerance_oracle',root/'codex_experiments_20261014/joint_yt_reboot_20261022/metrics.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rng=np.random.default_rng(1027);y=rng.normal(size=(341,2));p=y+rng.normal(scale=.1,size=y.shape)
cases=[('mixed',y,p),('all_exact',y,y),('boundaries',np.zeros((4,2)),np.array([[.09,-.09],[.09+5e-13,0],[.09+2e-12,0],[0,-.09-2e-12]])),('failed_predictions',np.zeros((3,2)),np.array([[0,0],[np.nan,0],[0,np.nan]])),('failed_truth',np.array([[np.nan,0],[0,np.nan],[0,0]]),np.zeros((3,2))),('all_failed',np.zeros((2,2)),np.full((2,2),np.nan))]
records=[]
def serial(a):return [[float(v) if np.isfinite(v) else None for v in row] for row in a]
for name,y,p in cases:
    path=out/(name+'.json');path.write_text(json.dumps(dict(truth=serial(y),prediction=serial(p))));destination=out/name;subprocess.run([str(final/'build/modern/bin/Release/tolerance_metrics.exe'),str(path),str(destination)],check=True,capture_output=True,text=True);actual=json.loads((destination/'metrics.json').read_text());expected=mod.metric(y,p)
    for key,value in expected.items():
        if value is None:assert actual[key] is None
        elif key in ['n','passed','failures']:assert actual[key]==value
        else:assert abs(actual[key]-value)<1e-12
    records.append(dict(case=name,passed=True,metrics=actual))
(out/'checks.json').write_text(json.dumps(dict(passed=True,cases=records,absolute_tolerance=1e-12),indent=2));print('Six tolerance/failure-denominator cases passed')
