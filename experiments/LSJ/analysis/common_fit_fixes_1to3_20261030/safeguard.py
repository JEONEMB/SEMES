"""Snapshot source before items 1-3 edits; verify protected artifacts afterwards."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BASE=ROOT/'analysis/reference_line_methods_108_20261030'
EDITABLE=['cpp/include/common.hpp','cpp/src/line_fit.cpp','cpp/src/main.cpp',
          'cpp/tests/line_selftest.cpp','cpp/CMakeLists.txt',
          'tools/make_figures.py','tools/make_report.py']
ADDITIONS=['tools/fit_point_views.py','tools/test_fit_point_views.py']
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():
    roots=[BASE,ROOT/'analysis/reference_line_methods_selection_20261030']
    return {p.relative_to(ROOT).as_posix():digest(p) for base in roots for p in base.rglob('*')
            if p.is_file() and not any(x in p.parts for x in ('build','__pycache__','.git','qa'))}
def main():
    baseline=HERE/'before_manifest.json'
    if '--verify' not in sys.argv:
        if baseline.exists():raise SystemExit('Refusing to overwrite baseline')
        for rel in EDITABLE:
            dst=HERE/'before'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(BASE/rel,dst)
        baseline.write_text(json.dumps(inventory(),ensure_ascii=False,indent=2),encoding='utf-8')
        print('before snapshot saved');return
    old=json.loads(baseline.read_text(encoding='utf-8'));now=inventory()
    prefix=BASE.relative_to(ROOT).as_posix()+'/'
    allowed={prefix+r for r in EDITABLE+ADDITIONS}
    changed=sorted(k for k in old.keys()|now.keys() if old.get(k)!=now.get(k))
    unexpected=[k for k in changed if k not in allowed]
    result={'changed_files':changed,'unexpected_changes':unexpected,
            'configs_original_measurements_figures_logs_and_selection_unchanged':not unexpected,
            'allowed_scope':'Common fit items 1-3, result schema/rendering and regression checks only.'}
    (HERE/'preservation_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'changed':len(changed),'unexpected':unexpected},ensure_ascii=True))
    raise SystemExit(bool(unexpected))
if __name__=='__main__':main()
