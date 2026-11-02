"""Preserve the post-1--3 baseline before making requested fixes 4--6."""
import hashlib,json,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BASE=ROOT/'analysis/reference_line_methods_108_20261030'
EDITABLE=['cpp/src/main.cpp','cpp/src/methods_segments.cpp','cpp/src/methods_points.cpp','cpp/src/methods_qr.cpp',
          'cpp/CMakeLists.txt','configs/methods.json','tools/make_figures.py','tools/make_report.py']
ADDITIONS=['cpp/include/segment_support.hpp','cpp/src/segment_support.cpp',
           'cpp/include/effective_settings.hpp','cpp/src/effective_settings.cpp',
           'cpp/tests/pipeline_contract_test.cpp']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():
    roots=[BASE,ROOT/'analysis/reference_line_methods_selection_20261030']
    return {p.relative_to(ROOT).as_posix():sha(p) for base in roots for p in base.rglob('*')
            if p.is_file() and not any(x in p.parts for x in ('build','__pycache__','.git','qa'))}
if __name__=='__main__':
    baseline=HERE/'before_manifest.json'
    if '--verify' not in sys.argv:
        if baseline.exists():raise SystemExit('Refusing to replace baseline')
        for name in EDITABLE:
            dst=HERE/'before'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,dst)
        baseline.write_text(json.dumps(inventory(),ensure_ascii=False,indent=2),encoding='utf-8')
        print('baseline saved')
    else:
        old=json.loads(baseline.read_text(encoding='utf-8'));now=inventory()
        changed=sorted(k for k in old.keys()|now.keys() if old.get(k)!=now.get(k))
        allowed={BASE.relative_to(ROOT).as_posix()+'/'+k for k in EDITABLE+ADDITIONS}
        unexpected=[k for k in changed if k not in allowed]
        before=json.loads((HERE/'before/configs/methods.json').read_text(encoding='utf-8-sig'))
        after=json.loads((BASE/'configs/methods.json').read_text(encoding='utf-8-sig'))
        def controls(x):
            if isinstance(x,dict):return {k:controls(v) for k,v in x.items() if k not in ('rationale','note','version','segment_selection_rationale','method_key_note')}
            if isinstance(x,list):return [controls(v) for v in x]
            return x
        same_controls=controls(before)==controls(after)
        out={'changed_files':changed,'unexpected_changes':unexpected,'configured_control_values_unchanged':same_controls,
             'protected_artifacts_unchanged':not unexpected}
        (HERE/'preservation_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'changed':len(changed),'unexpected':unexpected,'controls_unchanged':same_controls}))
        raise SystemExit(bool(unexpected) or not same_controls)
