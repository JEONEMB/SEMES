"""Reference self-match and missing-file CLI regression. No new dataset analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--bin',type=Path,required=True)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--reference-file',type=Path,required=True)
a=p.parse_args()
payload=Path(__file__).resolve().parents[1]
reference=Path(a.reference_file.read_text(encoding='utf-8-sig').strip())
digest=lambda: hashlib.sha256(reference.read_bytes()).hexdigest()
before=digest()
a.work.mkdir(parents=True,exist_ok=True)
output=a.work/'results.json'
missing=a.work/'does_not_exist.bmp'
assert not missing.exists()
run=subprocess.run([str(a.bin/'main.exe'),'--reference',str(reference),
    '--config',str(payload/'configs/theta_alignment.json'),
    '--y-config',str(payload/'configs/y_alignment.json'),
    '--output',str(output),str(reference),str(missing)],capture_output=True)
assert run.returncode==1,(run.returncode,run.stdout,run.stderr)
rows=json.loads(output.read_text(encoding='utf-8'))['results']
assert len(rows)==2
assert rows[0]['status']=='aligned' and rows[0]['aligned']==1
assert rows[0]['command_y_mm']==0 and rows[0]['command_theta_deg']==0
assert rows[0]['error_code']=='none'
assert rows[1]['valid']==0 and rows[1]['error_code']=='io_failure'
assert digest()==before,'Reference was modified'
print('PASS native CLI reference aligned, missing-file structured failure, mixed-batch exit and unchanged reference')
