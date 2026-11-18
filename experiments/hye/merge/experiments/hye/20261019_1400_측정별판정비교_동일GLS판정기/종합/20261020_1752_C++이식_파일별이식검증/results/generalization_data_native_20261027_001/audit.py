import sys,json,importlib.util
from pathlib import Path
import pandas as pd
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;final=out.parents[1];root=final.parent
spec=importlib.util.spec_from_file_location('general_data_loader_oracle',root/'codex_experiments_20261014/joint_yt_generalization_20261022/data.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
expected,_=mod.load();actual=pd.read_csv(out/'manifest_with_folds.csv');fields=['index','dataset','file','Y','T','relative','fold','cv_group'];checks={field:bool(expected[field].equals(actual[field])) for field in fields};splits=[dict(fold=f,train_indices=expected.index[expected.fold!=f].tolist(),test_indices=expected.index[expected.fold==f].tolist(),train=expected.relative[expected.fold!=f].tolist(),test=expected.relative[expected.fold==f].tolist()) for f in range(4)];checks['splits_exact']=splits==json.loads((out/'splits.json').read_text(encoding='utf-8'));checks['all_copied_paths_exist']=all((out/str(p)).is_file() for p in actual.path)
result=dict(passed=all(checks.values()),checks=checks,scope='Data loading and grouped split reconstruction only; portable image paths replace original absolute paths')
(out/'parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['passed']
