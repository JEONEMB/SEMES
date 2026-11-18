"""Build and run contract checks only. Never invokes captured-image analysis."""
import ast,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_line_methods_108_20261030'
BUILD=HERE/'build'
LOGS=HERE/'logs';LOGS.mkdir(exist_ok=True)
CMAKE=Path(r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe')
steps=[]
def run(name,args,cwd=BASE):
    start=time.time();result=subprocess.run(list(map(str,args)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (LOGS/(name+'.stdout.txt')).write_bytes(result.stdout);(LOGS/(name+'.stderr.txt')).write_bytes(result.stderr)
    steps.append({'name':name,'command':list(map(str,args)),'exit_code':result.returncode,'seconds':round(time.time()-start,2)})
    (HERE/'validation.json').write_text(json.dumps({'steps':steps,'scope':'software/geometry/config checks; no captured-image measurement or statistical validation'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(name+': exit '+str(result.returncode),flush=True)
    if result.returncode:
        print((result.stdout+result.stderr).decode('utf-8',errors='replace')[-6500:]);raise SystemExit(result.returncode)
    return result.stdout
for name in ('make_figures.py','make_report.py','fit_point_views.py'):
    ast.parse((BASE/'tools'/name).read_text(encoding='utf-8-sig'))
print('Python syntax passed',flush=True)
run('01_configure',[CMAKE,'-S',BASE/'cpp','-B',BUILD,'-G','Visual Studio 18 2026','-A','x64','-T','v141',
    '-DCMAKE_GENERATOR_INSTANCE=C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools','-DOpenCV_DIR=C:/Users/SSAFY/opencv/build'])
run('02_build',[CMAKE,'--build',BUILD,'--config','Release','--','-m:1','/nr:false','/v:normal'])
run('03_fit_regression',[BUILD/'Release/line_selftest.exe'])
run('04_geometry_settings',[BUILD/'Release/pipeline_contract_test.exe'])
raw=run('05_config_only',[BUILD/'Release/line_methods.exe','--root','.', '--check-config'])
resolved=json.loads(raw.decode('utf-8-sig'))
assert len(resolved)==54 and len({k.split('_')[0] for k in resolved})==21
(HERE/'resolved_settings.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2),encoding='utf-8')
run('06_report_regression',[sys.executable,'-m','unittest','discover','-s','tools','-p','test_fit_point_views.py','-v'])
run('07_display_contract',[sys.executable,HERE/'test_display_contract.py','-v'])
print('Passed. No captured images read or measured.',flush=True)
