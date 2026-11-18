"""Run the already selected statsmodels ordinary Huber settings on every point set."""
from pathlib import Path
import hashlib
import json
import math
import time
import warnings
import numpy as np
import statsmodels
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT

HERE=Path(__file__).resolve().parent

def read(name): return json.loads((HERE/name).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(name,data): (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def number(v): return float(v) if np.isfinite(v) else None

def main():
    assert statsmodels.__version__=='0.15.0'
    prov=read('provenance.json')
    assert sha(HERE/'points.npz')==prov['points_sha256']
    old_prov=json.loads((HERE.parent/'huber_convergence_20261101/provenance.json').read_text(encoding='utf-8'))
    for s in old_prov['source_snapshots']:
        assert sha(s['installed_file'])==s['sha256']
    points=np.load(HERE/'points.npz',allow_pickle=False)
    rows=[]; histories=[]; start=time.perf_counter()
    for i,meta in enumerate(read('extraction.json'),1):
        key=meta['id']+'_'+meta['target'];xy=points[key].astype(np.float64)
        x,y=xy[:,0],xy[:,1];X=np.column_stack([np.ones(len(x)),x])
        row=dict(meta)
        try:
            model=RLM(y,X,M=HuberT(t=1.345),missing='raise')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit=model.fit(maxiter=50,tol=1e-8,scale_est='mad',update_scale=True,conv='dev',cov='H1')
            b,m=map(float,fit.params);h=fit.fit_history
            d=np.asarray(h['deviance'][1:],float);delta=float(abs(d[-1]-d[-2]))
            finite=bool(np.isfinite(fit.params).all() and np.isfinite(fit.scale) and fit.scale>0)
            tol_met=bool(finite and math.isfinite(delta) and delta<=1e-8)
            count=int(h['iteration'])
            reason='tolerance' if tol_met else 'iteration_cap' if count>=50 else 'other_or_nonfinite'
            residual=y-(b+m*x)
            row.update(slope=number(m),intercept_px=number(b),image_angle_deg=number(np.degrees(np.arctan(m))),
                       scale_px=number(fit.scale),tau_px=number(1.345*fit.scale),
                       line_rmse_px=number(np.sqrt(np.mean(residual**2))),
                       max_abs_line_residual_px=number(np.max(abs(residual))),
                       downweighted_points=int(np.count_nonzero(np.asarray(fit.weights)<1)),
                       iteration=count,deviance_delta=number(delta),finite=finite,tol_met=tol_met,
                       stop_reason=reason,warnings=[str(w.message) for w in caught])
            histories.append({'id':meta['id'],'target':meta['target'],
                              'params_b_m':[[number(v) for v in a] for a in h['params'][1:]],
                              'deviance':[number(v) for v in d]})
        except Exception as exc:
            row.update(slope=None,intercept_px=None,image_angle_deg=None,finite=False,tol_met=False,
                       stop_reason='exception',exception=repr(exc))
        rows.append(row)
        if i%100==0 or i==1150: print(f'Fitted {i}/1150 lines; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    dump('measurements.json',rows);dump('histories.json',histories)
    old=json.loads((HERE.parent/'huber_convergence_20261101/results.json').read_text(encoding='utf-8'))
    old={r['input']:r for r in old if r['config']=='A'}
    checks=[];bykey={r['id']+'_'+r['target']:r for r in rows}
    for ref in prov['reference_crosschecks']:
        r=bykey[ref['id']+'_'+ref['target']];o=old[ref['reference']+'_'+ref['target']]
        checks.append({'id':ref['id'],'target':ref['target'],'reference':ref['reference'],
                       'slope_delta':r['slope']-o['slope'],
                       'intercept_delta_px':r['intercept_px']-o['intercept_px'],
                       'angle_delta_deg':r['image_angle_deg']-o['image_angle_deg']})
    summary={'statsmodels_version':statsmodels.__version__,'numpy_version':np.__version__,
             'lines':len(rows),'finite':sum(r['finite'] for r in rows),
             'tolerance_met':sum(r['tol_met'] for r in rows),
             'cap_only':sum(r['stop_reason']=='iteration_cap' for r in rows),
             'exceptions':sum(r['stop_reason']=='exception' for r in rows),
             'reference_checks':checks,'elapsed_seconds':time.perf_counter()-start,
             'settings_unchanged':True,'measurement_sha256':sha(HERE/'measurements.json'),
             'nonconverged_handling':'Retained and flagged; not silently filtered or rerun with different settings.'}
    dump('fit_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='reference_checks'},ensure_ascii=True),flush=True)

if __name__=='__main__': main()
