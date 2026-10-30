import sys,json,importlib.util
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;final=out.parents[1];root=final.parent
spec=importlib.util.spec_from_file_location('generalization_bank_oracle',root/'codex_experiments_20261014/joint_yt_generalization_20261022/data.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
a=np.load(final/'data/reference/codex_experiments_20261014/joint_yt_generalization_20261022/measurements/features.npz');expected=mod.features(a);actual=np.load(out/'feature_bank.npz');records={}
for key in expected:
    error=float(np.nanmax(abs(actual[key]-expected[key])));records[key]=dict(shape_equal=actual[key].shape==expected[key].shape,max_difference=error,nonfinite_pattern_equal=bool(np.array_equal(np.isfinite(actual[key]),np.isfinite(expected[key]))),passed=actual[key].shape==expected[key].shape and error<=1e-8 and np.array_equal(np.isfinite(actual[key]),np.isfinite(expected[key])))
summary=dict(passed=all(r['passed'] for r in records.values()),absolute_tolerance=1e-8,arrays=records,scope='Feature transformation parity from frozen source measurements, not image measurement parity')
(out/'checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));sys.exit(0 if summary['passed'] else 1)
