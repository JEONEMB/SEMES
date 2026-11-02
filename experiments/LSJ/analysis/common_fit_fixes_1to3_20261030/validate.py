"""Compile code and test known-point/record cases. Never invokes the image pipeline."""
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_line_methods_108_20261030'
CMAKE=Path(r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe')
BUILD=HERE/'build'
LOGS=HERE/'logs';LOGS.mkdir(exist_ok=True)
steps=[]
def run(name,args,cwd=BASE):
    start=time.time()
    result=subprocess.run(list(map(str,args)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (LOGS/(name+'.stdout.txt')).write_bytes(result.stdout)
    (LOGS/(name+'.stderr.txt')).write_bytes(result.stderr)
    steps.append(dict(name=name,command=list(map(str,args)),exit_code=result.returncode,seconds=round(time.time()-start,2)))
    (HERE/'validation.json').write_text(json.dumps({'steps':steps,'scope':'Compile and known-input software checks only; no image measurement/statistical validation.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(name+': exit '+str(result.returncode),flush=True)
    if result.returncode:
        print((result.stdout+result.stderr).decode('utf-8',errors='replace')[-6000:])
        raise SystemExit(result.returncode)

for name in ['make_figures.py','make_report.py','fit_point_views.py','test_fit_point_views.py']:
    p=BASE/'tools'/name;ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
print('Python syntax: passed',flush=True)
run('01_configure',[CMAKE,'-S',BASE/'cpp','-B',BUILD,'-G','Visual Studio 18 2026','-A','x64','-T','v141',
                    '-DCMAKE_GENERATOR_INSTANCE=C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools',
                    '-DOpenCV_DIR=C:/Users/SSAFY/opencv/build'])
run('02_build',[CMAKE,'--build',BUILD,'--config','Release','--','-m:1','/nr:false','/v:normal'])
run('03_known_points',[BUILD/'Release/line_selftest.exe'])
run('04_report_records',[sys.executable,'-m','unittest','discover','-s','tools','-p','test_fit_point_views.py','-v'])
print('All scoped code checks passed. No image analysis run.',flush=True)
