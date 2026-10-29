"""Freeze the current model, select ALL existing images, and run the native estimator once each.
Raw images are referenced in a manifest, never copied, changed, synthesized or relabeled as ground truth.
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PAYLOAD=ROOT/'.tmp/theta_implementation_20261029/payload'
BUILD=ROOT/'.tmp/algorithm_benchmark_20261029/build'
CMAKE=Path('C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe')
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def write(name,data): (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def prepare():
    first=read(ROOT/'analysis/initial_error/capture_order.json')
    second=read(ROOT/'analysis/second_data_inspection/inventory.json')
    first_hashes={r['file']:r['sha256'] for r in read(ROOT/'analysis/initial_error/measurements.json')['measurements']}
    first_root=Path(first['summary']['data_root'])
    second_root=Path(second['summary']['extracted_root'])
    group_numbers={g['group']:i+1 for i,g in enumerate(first['chronological_groups'])}
    cases=[]
    for r in first['files_in_modification_order']:
        ref=r['group']=='center'
        axis='reference' if ref else ('theta' if r['group'].startswith('Theta') else 'y')
        cases.append({'id':f'first_{len(cases)+1:03d}','dataset':'first','path':str(first_root/r['file']),
            'file':r['file'],'group':r['group'],'capture_group':group_numbers[r['group']],
            'condition':r['file'],'axis':axis,'is_reference':ref,'label_y_mm':float(r['label']) if axis=='y' else 0.0,
            'label_theta_deg':float(r['label']) if axis=='theta' else 0.0,
            'time_metadata':r['modified_kst'],'expected_sha256':first_hashes[r['file']],
            'evaluation_role':'reference_control' if ref else 'calibration_data_replay'})
    for i,r in enumerate(sorted(second['images'],key=lambda r:(r['archive_modified_time_no_timezone'],r['file']))):
        cases.append({'id':f'second_{i+1:03d}','dataset':'second','path':str(second_root/r['file']),
            'file':r['file'],'group':r['condition'],'condition':r['condition'],'axis':'reference' if r['is_reference'] else 'mixed',
            'is_reference':r['is_reference'],'label_y_mm':r['label_y_mm'],'label_theta_deg':r['label_theta_degree'],
            'time_metadata':r['archive_modified_time_no_timezone'],'expected_sha256':r['sha256'],
            'evaluation_role':'separate_acquisition_previously_explored_not_blind_holdout'})
    assert len(cases)==341 and len({c['id'] for c in cases})==341
    for i,c in enumerate(cases):
        p=Path(c['path']); stat=p.stat()
        actual=sha(p)
        assert actual==c['expected_sha256'],('Raw image changed from existing inventory',c['id'])
        c['size_bytes']=stat.st_size; c['mtime_ns']=stat.st_mtime_ns
        if (i+1)%100==0: print(f'Inventory/hash checks: {i+1}/341',flush=True)
    fingerprints={str(p.relative_to(PAYLOAD)).replace('\\','/'):sha(p) for p in sorted(PAYLOAD.rglob('*'))
                  if p.is_file() and (p.suffix in ['.cpp','.hpp'] or p.parent==PAYLOAD/'configs')}
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'reference':str(first_root/'center.bmp'),'theta_config':str(PAYLOAD/'configs/theta_alignment.json'),
        'y_config':str(PAYLOAD/'configs/y_alignment.json'),
        'selection':'All 281 first + 60 second images; no random subset or train/test refit',
        'source_sha256':fingerprints,'processor':platform.processor(),'platform':platform.platform(),
        'scope':'Frozen model, independent images; no synthetic movement or closed-loop simulation',
        'label_limitation':'Labels are nominal relative positions; first and second targets are not the same coordinate origin',
        'cases':cases}
    write('test_manifest.json',manifest)
    print('Prepared 341 verified image entries; original images referenced in place',flush=True)
    return manifest

def command(name,args,stream=False):
    print(name,flush=True)
    if stream:
        with (HERE/(name+'.log')).open('wb') as log:
            p=subprocess.Popen([str(x) for x in args],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            for line in p.stdout:
                log.write(line); log.flush(); print(line.decode('utf-8',errors='replace').rstrip(),flush=True)
            rc=p.wait()
    else:
        p=subprocess.run([str(x) for x in args],cwd=HERE,capture_output=True)
        (HERE/(name+'.log')).write_bytes(p.stdout+p.stderr); rc=p.returncode
        if rc: print((p.stdout+p.stderr).decode('utf-8',errors='replace'))
    if rc: raise RuntimeError(f'{name} failed with exit {rc}')

if __name__=='__main__':
    manifest=prepare()
    command('configure',[CMAKE,'-S',HERE,'-B',BUILD,'-G','Visual Studio 18 2026','-A','x64','-T','v141',
        '-DOpenCV_DIR=C:/Users/SSAFY/opencv/build','-DALIGNMENT_SOURCE_DIR='+str(PAYLOAD)])
    command('build',[CMAKE,'--build',BUILD,'--config','Release','--target','alignment_benchmark','--','/m:1','/v:minimal'])
    command('native_run',[BUILD/'Release/alignment_benchmark.exe','test_manifest.json','native_results.json'],stream=True)
    rows=read(HERE/'native_results.json')['results']
    assert len(rows)==341 and {r['id'] for r in rows}=={c['id'] for c in manifest['cases']}
    for c in manifest['cases']:
        s=Path(c['path']).stat()
        assert (s.st_size,s.st_mtime_ns)==(c['size_bytes'],c['mtime_ns']),('Input metadata changed',c['id'])
    assert all(sha(PAYLOAD/rel)==v for rel,v in manifest['source_sha256'].items()),'Model changed during benchmark'
    write('integrity.json',{'all_341_input_hashes_match_saved_inventory':True,'input_size_and_mtime_unchanged_after_run':True,
        'model_source_and_config_hashes_unchanged':True,'outputs':len(rows),
        'model_source_sha256':manifest['source_sha256']})
    print('PASS full output coverage and frozen model/input integrity checks',flush=True)
