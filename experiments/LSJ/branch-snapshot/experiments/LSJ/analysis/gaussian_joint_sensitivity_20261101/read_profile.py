"""Read existing raw results, without running an analysis.
Example: python read_profile.py R01 upper_structure 95 21
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('ref');p.add_argument('target',choices=['upper_structure','lower_structure'])
p.add_argument('x',type=int);p.add_argument('n',type=int);p.add_argument('--out',type=Path)
args=p.parse_args();m=json.loads((HERE/'metadata.json').read_text())
g=next(g for g in m['groups'] if (g['ref'],g['target'])==(args.ref,args.target))
pi=g['x'].index(args.x);ni=m['sample_counts'].index(args.n)
d=np.memmap(HERE/g['file'],mode='r',dtype='<f8',shape=tuple(g['shape']))
file=args.out.open('w',encoding='utf-8-sig',newline='') if args.out else sys.stdout
w=csv.writer(file);w.writerow(['ref','target','x','coarse_y','n','initialization','stopping']+m['fields'])
for si,start in enumerate(m['initializations']):
 for ci,stop in enumerate(m['stopping']):
  w.writerow([args.ref,args.target,args.x,g['coarse_y'][pi],args.n,start,stop['id']]+d[pi,ni,si,ci].tolist())
if args.out:file.close()
