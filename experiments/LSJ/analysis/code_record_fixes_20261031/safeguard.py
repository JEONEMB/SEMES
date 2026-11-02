"""Preserve the post-1--6 source and all historical measurement artifacts."""
import hashlib,json,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BASE=ROOT/'analysis/reference_line_methods_108_20261030'
EDITABLE=['cpp/CMakeLists.txt','cpp/src/line_fit.cpp','cpp/src/methods_points.cpp','cpp/src/methods_hough.cpp',
          'cpp/src/methods_segments.cpp','cpp/src/methods_qr.cpp','cpp/src/main.cpp',
          'tools/run_pipeline.py','tools/make_figures.py','tools/make_report.py','tools/fit_point_views.py']
ADDITIONS=['cpp/include/zernike_location.hpp','cpp/src/zernike_location.cpp','cpp/tests/numeric_record_test.cpp',
           'tools/run_comparison.py','tools/generation_provenance.py','tools/test_code_records.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():
    return {p.relative_to(ROOT).as_posix():sha(p)
            for base in (BASE,ROOT/'analysis/reference_line_methods_selection_20261030') for p in base.rglob('*')
            if p.is_file() and not any(x in p.parts for x in ('build','__pycache__','.git','qa'))}
if __name__=='__main__':
    baseline=HERE/'before_manifest.json'
    if '--verify' not in sys.argv:
        if baseline.exists():raise SystemExit('Refusing to replace baseline')
        for name in EDITABLE:
            dest=HERE/'before'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,dest)
        baseline.write_text(json.dumps(inventory(),ensure_ascii=False,indent=2),encoding='utf-8')
        print('baseline saved')
    else:
        old=json.loads(baseline.read_text(encoding='utf-8'));now=inventory()
        changed=sorted(k for k in old.keys()|now.keys() if old.get(k)!=now.get(k))
        allowed={BASE.relative_to(ROOT).as_posix()+'/'+k for k in EDITABLE+ADDITIONS}
        unexpected=[k for k in changed if k not in allowed]
        config_paths=[k for k in old if '/configs/' in k]
        out={'changed_files':changed,'unexpected_changes':unexpected,
             'all_config_files_byte_identical':all(old[k]==now.get(k) for k in config_paths),
             'historical_artifacts_preserved':not unexpected}
        (HERE/'preservation_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(out,ensure_ascii=True))
        raise SystemExit(bool(unexpected) or not out['all_config_files_byte_identical'])
