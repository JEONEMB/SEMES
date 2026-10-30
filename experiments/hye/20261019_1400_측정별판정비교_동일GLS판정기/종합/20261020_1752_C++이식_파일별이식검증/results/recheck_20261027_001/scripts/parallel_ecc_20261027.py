"""Run the unchanged seven ECC settings over all 68 selected images per setting."""
import concurrent.futures
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from recheck_20261027 import ROOT, OUT, exe, DATA

def main():
    destination=Path(sys.argv[1]).resolve()
    destination.mkdir(exist_ok=False)
    binary=exe('historical_ecc_tune')
    def run(i):
        target=OUT/f'ecc_variant_{i}'
        args=[str(binary),'..',DATA,str(target),str(i)]
        start=time.monotonic()
        with (OUT/f'ecc_variant_{i}.log').open('w',encoding='utf-8') as f:
            p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        return dict(name=f'ecc_variant_{i}',command=args,exit_code=p.returncode,seconds=time.monotonic()-start)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,i) for i in range(7)]
        for f in concurrent.futures.as_completed(futures):
            r=f.result();records.append(r)
            (OUT/'ecc_parallel_commands.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
            print(r['name'],r['exit_code'],flush=True)
    if any(r['exit_code'] for r in records):raise SystemExit(1)
    for filename in ('variants.csv','homographies.csv','failures.csv'):
        rows=[];header=None
        for i in range(7):
            with (OUT/f'ecc_variant_{i}'/filename).open(encoding='utf-8',newline='') as f:
                part=list(csv.reader(f))
            if header is None:header=part[0]
            if header!=part[0]:raise RuntimeError('Variant header mismatch')
            rows.extend(part[1:])
        with (destination/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerow(header);writer.writerows(rows)
    checks=json.loads((OUT/'ecc_variant_0/checks.json').read_text(encoding='utf-8'))
    checks.pop('selected_variant');checks['variants']=7
    checks['failures']=sum(json.loads((OUT/f'ecc_variant_{i}/checks.json').read_text(encoding='utf-8'))['failures'] for i in range(7))
    (destination/'checks.json').write_text(json.dumps(checks,ensure_ascii=False),encoding='utf-8')
    shutil.copyfile(OUT/'ecc_variant_0/qr_polygons.json',destination/'qr_polygons.json')
    records.append(dict(name='ecc_tune_parallel',command=[sys.executable,str(Path(__file__).resolve()),str(destination)],exit_code=0))
    (OUT/'ecc_parallel_commands.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
