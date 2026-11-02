"""Build/check code and temporary fixtures; never run captured-image pipelines or generators."""
import ast,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_line_methods_108_20261030'
BUILD=HERE/'build'
LOGS=HERE/'logs';LOGS.mkdir(exist_ok=True)
CMAKE=Path(r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe')
steps=[]
sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
def run(name,args,cwd=BASE):
    start=time.time();result=subprocess.run(list(map(str,args)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    (LOGS/(name+'.stdout.txt')).write_bytes(result.stdout);(LOGS/(name+'.stderr.txt')).write_bytes(result.stderr)
    steps.append({'name':name,'command':list(map(str,args)),'exit_code':result.returncode,'seconds':round(time.time()-start,2)})
    (HERE/'validation.json').write_text(json.dumps({'steps':steps,'scope':'software/known moments/coordinate serialization/temporary artifacts; no captured image or statistical analysis'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(name+': exit '+str(result.returncode),flush=True)
    if result.returncode:
        print((result.stdout+result.stderr).decode('utf-8',errors='replace')[-7000:]);raise SystemExit(result.returncode)
    return result.stdout
for path in (BASE/'tools').glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
print('Python syntax passed',flush=True)
if '--python-only' in sys.argv:
    steps=json.loads((HERE/'validation.json').read_text(encoding='utf-8'))['steps'][:6]
    assert len(steps)==6 and all(s['exit_code']==0 for s in steps)
else:
    run('01_configure',[CMAKE,'-S',BASE/'cpp','-B',BUILD,'-G','Visual Studio 18 2026','-A','x64','-T','v141',
        '-DCMAKE_GENERATOR_INSTANCE=C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools','-DOpenCV_DIR=C:/Users/SSAFY/opencv/build'])
    run('02_build',[CMAKE,'--build',BUILD,'--config','Release','--','-m:1','/nr:false','/v:normal'])
    run('03_fit_regression',[BUILD/'Release/line_selftest.exe'])
    run('04_pipeline_regression',[BUILD/'Release/pipeline_contract_test.exe'])
    run('05_numeric_records',[BUILD/'Release/numeric_record_test.exe'])
    raw=run('06_config_only',[BUILD/'Release/line_methods.exe','--root','.', '--check-config'])
    resolved=json.loads(raw.decode('utf-8-sig'));assert len(resolved)==54
    (HERE/'resolved_settings.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2),encoding='utf-8')
run('07_python_checks',[sys.executable,'-m','unittest','discover','-s','tools','-p','test_*.py','-v'])
run('08_display_regression',[sys.executable,HERE.parent/'common_pipeline_fixes_4to6_20261030/test_display_contract.py','-v'])
print('All checks passed; no captured image analysis or report/figure regeneration.',flush=True)
