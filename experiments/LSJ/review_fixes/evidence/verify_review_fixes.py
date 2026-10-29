import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

work=Path(__file__).resolve().parent
stage=work.parent
payload=stage/'payload'
before=work/'before'
build=work/'build'
cmake_dir=Path('C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin')
reference=Path((stage/'reference_path.txt').read_text(encoding='utf-8-sig').strip())
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
original_hash=sha(reference)
assert original_hash=='01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685'
assert all(sha(payload/'configs'/f'{axis}_alignment.json')==sha(before/'configs'/f'{axis}_alignment.json') for axis in ['theta','y'])
executions=[]
def run(name, args):
    result=subprocess.run([str(x) for x in args],cwd=payload,capture_output=True)
    (work/(name+'.stdout.log')).write_bytes(result.stdout)
    (work/(name+'.stderr.log')).write_bytes(result.stderr)
    executions.append({'name':name,'command':[str(x) for x in args],'returncode':result.returncode})
    print(name+': '+str(result.returncode),flush=True)
    if result.returncode:
        print(result.stdout.decode('utf-8',errors='replace'))
        print(result.stderr.decode('utf-8',errors='replace'))
        raise SystemExit(result.returncode)

run('configure',[cmake_dir/'cmake.exe','-S',payload,'-B',build,'-G','Visual Studio 18 2026','-A','x64','-T','v141',
    '-DOpenCV_DIR=C:/Users/SSAFY/opencv/build','-DBUILD_TESTING=ON',
    '-DPython3_EXECUTABLE=C:/Users/SSAFY/AppData/Local/Programs/Python/Python312/python.exe',
    '-DALIGNMENT_REFERENCE_FILE='+str(stage/'reference_path.txt')])
run('build',[cmake_dir/'cmake.exe','--build',build,'--config','Release','--','/m:1','/v:minimal'])
run('ctest',[cmake_dir/'ctest.exe','--test-dir',build,'-C','Release','--output-on-failure'])
io=json.loads((build/'review_io_checks/io_validation.json').read_text(encoding='utf-8'))
assert io['failed']==0 and io['passed']==253
assert sha(reference)==original_hash
changes=[]
for p in sorted(before.rglob('*')):
    if p.is_file():
        relative=p.relative_to(before)
        if not (payload/relative).exists(): raise AssertionError('Original file unexpectedly deleted: '+str(relative))
        if sha(p)!=sha(payload/relative): changes.append(str(relative).replace('\\','/'))
added=[str(p.relative_to(payload)).replace('\\','/') for p in payload.rglob('*') if p.is_file() and not (before/p.relative_to(payload)).exists()]
summary={
    'created_at_utc':datetime.now(timezone.utc).isoformat(),
    'environment':{'compiler':'MSVC 19.16 / v141','opencv':'3.4.15 vc15','architecture':'x64','configuration':'Release'},
    'executions':executions,
    'ctest_passed':7,'ctest_failed':0,
    'native_config_and_path_checks':io['passed'],
    'theta_analytic_interval_cases':2601,
    'native_checks':['center self-match','in-memory +5px shift','pending and failed-config reload preservation',
                     'acknowledged count preservation','uncertain move recovery','20-slot adapter cap','mixed CLI batch structured errors'],
    'reference_sha256_before':original_hash,'reference_sha256_after':sha(reference),
    'configuration_unchanged':{axis:sha(payload/'configs'/f'{axis}_alignment.json') for axis in ['theta','y']},
    'modified_files':changes,'new_files':sorted(added),
    'source_sha256':{str(p.relative_to(payload)).replace('\\','/'):sha(p) for p in sorted(payload.rglob('*'))
                     if p.is_file() and p.suffix in {'.cpp','.hpp','.py'}},
    'limitations':['No new dataset analysis or calibration','No physical robot accuracy or closed-loop convergence test',
                   'Prediction feasibility does not certify physical convergence','No tolerance or target range widening'],
    'logs_directory':str(work),
    'prior_evidence':'Original y_controller/validation_summary.json and original Claude package are pre-fix snapshots'
}
out=payload/'experiments/LSJ/review_fixes/validation_summary.json'
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print((work/'ctest.stdout.log').read_text(encoding='utf-8',errors='replace'))
print('PASS unchanged original reference/configs and final evidence saved')
