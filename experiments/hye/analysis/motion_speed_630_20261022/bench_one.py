import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from engine import Engine,ROOT,HERE
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--variant',default='baseline');ap.add_argument('--file',default='Y/Y+0.50.bmp');ap.add_argument('--threads',type=int,default=1);a=ap.parse_args()
 e=Engine(a.variant)
 with threadpool_limits(a.threads):
  raw=(ROOT/a.file).read_bytes();r=e.predict(raw,True);m=r.pop('measurement');r.update(file=a.file,init_seconds=e.init_seconds,threads=threadpool_info())
  name=a.variant+'_'+Path(a.file).stem
  np.savez(HERE/'results'/f'{name}.npz',coords=m['current_points'],valid=m['valid'],quality=m['quality'],ids=m['ids'])
  (HERE/'results'/f'{name}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
