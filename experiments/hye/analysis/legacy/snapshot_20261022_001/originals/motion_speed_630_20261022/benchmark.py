"""Paired single-image timings, no concurrent jobs and no result cache."""
import json,time
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from engine import Engine,ROOT,HERE
if __name__=='__main__':
 engines={name:Engine(name) for name in ['baseline','compiled']};rows=[]
 with threadpool_limits(1):
  threads=threadpool_info()
  for name in ['Y/Y+0.50.bmp','Y/Y+0.05.bmp','Theta/T-0.05.bmp']:
   raw=(ROOT/name).read_bytes()
   for variant in ['baseline','compiled']:
    r=engines[variant].predict(raw);r.update(file=name);rows.append(r);print(variant,name,r['timings'],flush=True)
    (HERE/'results/paired_timings.json').write_text(json.dumps(dict(rows=rows,threads=threads,comparison='sequential, one image at a time; BLAS1, compiled corner threads4; no inference result cache',init_seconds={k:v.init_seconds for k,v in engines.items()}),indent=2))
 print('done',flush=True)
