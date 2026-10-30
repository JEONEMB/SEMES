"""Finalize audit records after all seven ECC variants have finished."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from recheck_20261027 import ROOT, OUT, exe
from compare_recheck_20261027 import main as compare

def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def run(name,args):
    with (OUT/(name+'.log')).open('w',encoding='utf-8') as f:
        p=subprocess.run([str(x) for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    return dict(name=name,command=[str(x) for x in args],exit_code=p.returncode)

def main():
    if not (OUT/'ecc_tune_parallel/checks.json').is_file():raise RuntimeError('ECC aggregation not complete')
    args=[exe('compare_csv'),ROOT/'experiments/provenance_recovery_20261023_01/original_variants.csv',OUT/'ecc_tune_parallel/variants.csv','variant',OUT/'original_ecc_noise_parity','0.00005','blur','mask_expansion_px','ecc_filter','T_noise']
    parity=run('original_ecc_noise_parity',args)
    logs=json.loads((OUT/'original_parity_commands.json').read_text(encoding='utf-8'))
    logs.append(parity);save('original_parity_commands.json',logs)
    preservation=run('preservation_final',[exe('verify_manifest'),'..','experiments/inventory_20261023_01/files.csv','config/port_progress.json',OUT/'preservation_final'])
    save('final_preservation_command.json',preservation)
    compare()
    progress=json.loads((ROOT/'config/port_progress.json').read_text(encoding='utf-8-sig'))
    integrity=[]
    for name,item in progress.items():
        source=ROOT.parent/name
        digest=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        integrity.append(dict(source=name,complete=item.get('cpp_complete',False),source_exists=source.is_file(),source_hash_matches=digest==item.get('source_sha256'),destination_exists=(ROOT/item['destination']).is_file(),missing_evidence=[x for x in item.get('evidence',[]) if not (ROOT/x).is_file()]))
    save('migration_integrity.json',integrity)
    references=[]
    for base in ('src','include','tool','experiments'):
        for p in (ROOT/base).rglob('*'):
            if p.suffix not in ('.cpp','.hpp','.h') or 'build' in p.relative_to(ROOT).parts:continue
            for n,line in enumerate(p.read_text(encoding='utf-8-sig',errors='replace').splitlines(),1):
                if re.search(r'(analysis_v[0-9]|analysis_agent|codex_experiments_|hw-analysis|motion_speed_|motion_web_|joint_center_|[A-Z]:[/\\])',line):
                    references.append(dict(file=str(p.relative_to(ROOT)),line=n,text=line.strip()))
    save('external_path_references.json',references)
    drift=json.loads((OUT/'inventory_drift.json').read_text(encoding='utf-8'))
    with (ROOT/'experiments/inventory_20261023_01/source_timeline.csv').open(encoding='utf-8-sig',newline='') as f:
        old={x['path'] for x in csv.DictReader(f)}
    paths=sorted((old|set(drift['added_since_baseline']))-set(drift['missing_since_baseline']))
    with (OUT/'archive_readiness.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['source','cpp_complete_in_registry','cpp_destination','evidence','archive_approved_by_audit','reason'])
        for s in paths:
            p=progress.get(s,{})
            w.writerow([s,p.get('cpp_complete',False),p.get('destination',''),' | '.join(p.get('evidence',[])),False,'Whole-project replacement gate not met; preserve runtime dependencies' if p.get('cpp_complete') else 'No completed replacement recorded'])
    reproduction=json.loads((OUT/'reproduction_comparison.json').read_text(encoding='utf-8'))
    artifacts=json.loads((OUT/'artifact_integrity.json').read_text(encoding='utf-8'))
    summary=dict(archive_allowed=False,archive_reason=['incomplete_replacement_coverage','runtime_inputs_in_original_folders','original_python_numeric_parity_failures'],current_original_sources=len(paths),completed_registry_sources=sum(x['complete'] for x in integrity),reproduction_cases=len(reproduction),reproduction_failed_cases=[x['run'] for x in reproduction if not x['passed']],compared_files=sum(len(x['files']) for x in reproduction),artifact_checks=len(artifacts),artifact_failures=sum(not x['passed'] for x in artifacts),ecc_variants=7,ecc_measurement_rows=476,original_parity_exit_codes={x['name']:x['exit_code'] for x in logs},preservation_exit_code=preservation['exit_code'],intentionally_interrupted='sequential ECC run replaced by seven complete independent variant runs',legacy_moves=0)
    media=json.loads((OUT/'decoded_media_comparison.json').read_text(encoding='utf-8'))
    summary['decoded_media_failures']=sum(not x['passed'] for x in media)
    cli=[]
    for name in ('main_cli_commands.json','eda_fix_commands.json'):
        cli.extend(json.loads((OUT/name).read_text(encoding='utf-8')))
    summary['cli_expected_outcomes_passed']=all((x['exit_code']==0)==x.get('expected_success',True) for x in cli)
    save('audit_summary.json',summary)
    sources={}
    for base in ('src','include','tool','experiments','models','config'):
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and p.suffix in ('.cpp','.hpp','.h','.json','.npz'):
                sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    save('audited_source_and_asset_hashes.json',sources)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
