"""Deterministic subset of heldout images, not selected by estimation error."""
import argparse,concurrent.futures,json
from pathlib import Path
import numpy as np,pandas as pd
from photometric import gray,register

def work(job):
    model,repo,frame=job
    folder='Y' if frame['axis']=='Y' else 'Theta'
    result=register(model,gray(repo/'data/SSAFY Test Image Set/center.bmp'),
        gray(repo/'data/SSAFY Test Image Set'/folder/(frame['name']+'.bmp')),frame['group'],[frame['y_mm'],frame['theta_deg']])
    truth=[frame['nominal'],0] if frame['axis']=='Y' else [0,frame['nominal']]
    return dict(name=frame['name'],axis=frame['axis'],group=frame['group'],nominal=frame['nominal'],
                y_error=result['y_mm']-truth[0],theta_error=result['theta_deg']-truth[1],**result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    model=json.loads((a.output/'homography_physical_pivot_4285_interleaved.json').read_text())
    frames=pd.read_csv(a.output/'homography_physical_pivot_4285_interleaved_holdout.csv');selected=[]
    for _,g in frames.groupby('group'):
        g=g.sort_values('nominal');selected.extend(g.iloc[sorted(set([0,len(g)//2,len(g)-1]))].to_dict('records'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(work,[(model,a.repo.resolve(),f) for f in selected]))
    r=pd.DataFrame(rows);r.to_csv(a.output/'photometric_holdout_subset.csv',index=False)
    stats={axis:{'n':len(g),'y_rmse':float(np.sqrt(np.mean(g.y_error**2))),
        'theta_rmse':float(np.sqrt(np.mean(g.theta_error**2))),
        'joint_within_tolerance':int(((abs(g.y_error)<=.09)&(abs(g.theta_error)<=.09)).sum()),
        'converged':int(g.converged.sum())} for axis,g in r.groupby('axis')}
    (a.output/'photometric_summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
