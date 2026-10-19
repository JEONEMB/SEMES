"""Run the C++ collector in independent processes and restore global frame IDs."""
import argparse,subprocess,time,concurrent.futures,json
from pathlib import Path
import pandas as pd
parser=argparse.ArgumentParser();parser.add_argument('executable');parser.add_argument('image_list');parser.add_argument('output');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
exe=str(Path(args.executable).resolve());lines=Path(args.image_list).read_text(encoding='utf-8').splitlines();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
workers=max(1,min(args.workers,len(lines)));size=(len(lines)+workers-1)//workers;started=time.perf_counter()
def work(offset):
 chunk=out/f'worker_{offset//size}';chunk.mkdir(exist_ok=True);lst=chunk/'images.txt';lst.write_text('\n'.join(lines[offset:offset+size]),encoding='utf-8')
 with (chunk/'stdout.log').open('w') as log:subprocess.run([exe,str(lst),str(chunk)],stdout=log,stderr=log,check=True)
 return offset,chunk
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(work,range(0,len(lines),size)))
for name in ['frames.csv','qr_status.csv','points.csv']:
 tables=[]
 for offset,chunk in sorted(results):
  t=pd.read_csv(chunk/name);t['frame']=t.frame+offset;tables.append(t)
 pd.concat(tables,ignore_index=True).to_csv(out/name,index=False)
metadata={'workers':workers,'images':len(lines),'wall_seconds':time.perf_counter()-started}
(out/'parallel_run.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))
