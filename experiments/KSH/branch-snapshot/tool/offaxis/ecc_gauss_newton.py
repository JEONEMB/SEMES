"""Normalized-correlation residual Gauss-Newton experiment; no input mutation.

Run from repository root. Uses fixed interior pixels, exact bilinear derivatives,
mean/norm derivatives, diagonal damping and monotone backtracking. This is not
the closed-form photometric update in the original ECC paper.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from refit_model import fit, predict


def load(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return image


def normalized(v):
    v = v - v.mean()
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError('Zero contrast')
    return v / norm, norm


def sample(image, x, y):
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    if np.any(ix < 0) or np.any(iy < 0) or np.any(ix+1 >= image.shape[1]) or np.any(iy+1 >= image.shape[0]):
        raise ValueError('Warp leaves valid image support')
    a, b = x-ix, y-iy
    v00, v10 = image[iy, ix].astype(float), image[iy, ix+1].astype(float)
    v01, v11 = image[iy+1, ix].astype(float), image[iy+1, ix+1].astype(float)
    value = (1-b)*((1-a)*v00+a*v10) + b*((1-a)*v01+a*v11)
    return value, (1-b)*(v10-v00)+b*(v11-v01), (1-a)*(v01-v00)+a*(v11-v10)


def residual(image, x, y, basis, target, p, jac=True):
    value, gx, gy = sample(image, x+basis@p[:3], y+basis@p[3:])
    v, norm = normalized(value)
    r = v-target
    if not jac:
        return r
    raw = np.column_stack([gx[:, None]*basis, gy[:, None]*basis])
    centered = raw-raw.mean(axis=0)
    j = (centered-v[:, None]*(v@centered)[None, :])/norm
    return r, j


def optimize(image, x, y, basis, target, initial, max_iter=80):
    p = initial.copy()
    status = 'max_iterations'
    trace = []
    for iteration in range(max_iter):
        r, j = residual(image, x, y, basis, target, p)
        cost = float(r@r/2)
        trace.append(cost)
        h, g = j.T@j, j.T@r
        if np.max(abs(g)) < 1e-10:
            status = 'gradient_converged'
            break
        accepted = False
        for damping in [1e-6, 1e-4, .01, 1.]:
            step = np.linalg.solve(h+damping*np.diag(np.maximum(np.diag(h), 1e-12)), -g)
            for alpha in [1., .5, .25, .125, .0625, .015625]:
                candidate = p+alpha*step
                try:
                    rr = residual(image, x, y, basis, target, candidate, False)
                    new_cost = float(rr@rr/2)
                except ValueError:
                    continue
                if new_cost < cost:
                    accepted = True
                    p = candidate
                    break
            if accepted:
                break
        if not accepted:
            status = 'stalled'
            break
        if np.max(abs(alpha*step)) < 1e-5 or cost-new_cost < 1e-12:
            status = 'step_converged'
            break
    final = residual(image, x, y, basis, target, p, False)
    return p, dict(status=status, iterations=iteration+1, correlation=1-float(final@final/2),
                   initial_correlation=1-trace[0], condition=float(np.linalg.cond(h)),
                   monotone=bool(np.all(np.diff(trace) <= 0)))


def self_check():
    rng = np.random.default_rng(42)
    image = cv2.GaussianBlur(rng.normal(120, 30, (90, 100)), (7, 7), 1.2)
    yy, xx = np.mgrid[15:70, 15:80]
    x, y = xx.ravel().astype(float), yy.ravel().astype(float)
    basis = np.column_stack([np.ones(len(x)), (x-47)/32, (y-42)/27])
    truth = np.array([.37, .12, -.08, -.28, .07, .09])
    target = normalized(1.3*sample(image, x+basis@truth[:3], y+basis@truth[3:])[0]+17)[0]
    p = truth + np.array([.13, .01, .02, -.11, .01, .02])
    r, j = residual(image, x, y, basis, target, p)
    numeric = np.column_stack([(residual(image,x,y,basis,target,p+np.eye(6)[k]*1e-6,False)-
                                residual(image,x,y,basis,target,p-np.eye(6)[k]*1e-6,False))/2e-6 for k in range(6)])
    relative = float(np.linalg.norm(j-numeric)/np.linalg.norm(numeric))
    estimate, info = optimize(image,x,y,basis,target,np.zeros(6))
    error = float(np.max(abs(estimate-truth)))
    assert relative < 1e-6, relative
    assert error < 1e-4, error
    assert info['monotone']
    return dict(jacobian_relative_error=relative, synthetic_max_parameter_error_px=error,
                synthetic_gain=1.3, synthetic_offset=17, seed=42)


def measure(image, base, qr):
    x0,y0,w,h = qr['bounds']
    template = base[y0:y0+h,x0:x0+w]
    left,top = max(0,x0-300),max(0,y0-120)
    search = image[top:y0+h+120,left:x0+w+300]
    scores = cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
    _, coarse, _, loc = cv2.minMaxLoc(scores)
    dx,dy = left+loc[0]-x0,top+loc[1]-y0
    yy,xx = np.mgrid[y0+5:y0+h-5,x0+5:x0+w-5]
    x,y = xx.ravel().astype(float),yy.ravel().astype(float)
    cx,cy = qr['center']
    basis = np.column_stack([np.ones(len(x)),(x-cx)/(w/2),(y-cy)/(h/2)])
    target = normalized(base[yy,xx].ravel().astype(float))[0]
    initial = np.array([dx,0,0,dy,0,0],float)
    start = time.perf_counter()
    p, info = optimize(image,x,y,basis,target,initial)
    gn_seconds=time.perf_counter()-start
    # Reproduce the repository's ECC baseline with today's library version.
    patch=image[y0+dy:y0+dy+h,x0+dx:x0+dx+w]
    begin=time.perf_counter()
    corr,warp=cv2.findTransformECC(template,patch,np.eye(2,3,dtype=np.float32),
                                  cv2.MOTION_AFFINE,
                                  (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-7),None,1)
    center=np.array([cx-x0,cy-y0])
    delta=warp[:,:2]@center+warp[:,2]+[dx,dy]-center
    # Re-score both warps on the same fixed interior using exact bilinear sampling.
    global_x=warp[0,0]*(x-x0)+warp[0,1]*(y-y0)+warp[0,2]+x0+dx
    global_y=warp[1,0]*(x-x0)+warp[1,1]*(y-y0)+warp[1,2]+y0+dy
    same_support=float(normalized(sample(image,global_x,global_y)[0])[0]@target)
    return dict(dx=p[0],dy=p[3],coarse_correlation=coarse,seconds=gn_seconds,
                ecc_dx=delta[0],ecc_dy=delta[1],ecc_reported_correlation=corr,
                ecc_fixed_support_correlation=same_support,ecc_seconds=time.perf_counter()-begin,
                **info,**{f'p{k}':float(v) for k,v in enumerate(p)})


def evaluate(methods, meta, output):
    mixed = meta[meta.category.eq('mixed')].sort_index()
    y = mixed[['y_mm','theta_deg']].to_numpy()
    coords = np.unique(y,axis=0)
    assert len(mixed)==57 and len(coords)==19
    rows, predictions, fitted = [], [], {}
    for method, table in methods.items():
        wide = table.pivot(index='path',columns='qr',values=['dx','dy'])
        cols = [(d,s) for s in 'LCR' for d in ['dx','dy']]
        wide = wide[cols]
        wide = wide-wide.loc['기준 위치/center1.bmp']
        x = wide.loc[mixed.index].to_numpy()
        if not np.isfinite(x).all():
            raise ValueError(f'Missing measurements: {method}')
        for kind in ['mean_affine','three_qr_affine']:
            xx = x if kind=='three_qr_affine' else x.reshape(-1,3,2).mean(1)
            held = np.zeros_like(y)
            for coord in coords:
                test = np.all(y==coord,axis=1)
                held[test] = predict(fit(xx[~test],y[~test],kind),xx[test])
            error = held-y
            rows.append(dict(method=method,model=kind,y_rmse_mm=np.sqrt(np.mean(error[:,0]**2)),
                             theta_rmse_deg=np.sqrt(np.mean(error[:,1]**2)),
                             y_max_abs_mm=abs(error[:,0]).max(),theta_max_abs_deg=abs(error[:,1]).max()))
            for i,path in enumerate(mixed.index):
                predictions.append(dict(method=method,model=kind,path=path,y_mm=y[i,0],theta_deg=y[i,1],
                                        predicted_y_mm=held[i,0],predicted_theta_deg=held[i,1],
                                        error_y_mm=error[i,0],error_theta_deg=error[i,1]))
            model = fit(xx,y,kind)
            refs = wide.loc[meta[meta.category.eq('new-reference')].index].to_numpy()
            if kind=='mean_affine': refs=refs.reshape(-1,3,2).mean(1)
            model['reference_predictions'] = predict(model,refs).tolist()
            model['reference_paths'] = meta[meta.category.eq('new-reference')].index.tolist()
            fitted[method+'_'+kind] = model
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'coordinate_holdout.csv',index=False)
    (output/'models.json').write_text(json.dumps(fitted,indent=2),encoding='utf-8')
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric,label in zip(axes,['y_rmse_mm','theta_rmse_deg'],['Y RMSE (mm)','Theta RMSE (degree)']):
        pivot=scores.pivot(index='method',columns='model',values=metric)
        pivot.plot.bar(ax=ax,rot=0)
        ax.set_ylabel(label);ax.set_xlabel('');ax.grid(axis='y',alpha=.3)
    fig.suptitle('Leave-one-coordinate-out: 19 coordinates / 57 images')
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/SSAFY Test Image Set'))
    parser.add_argument('--source',type=Path,default=Path('docs/offaxis-eda'))
    parser.add_argument('--output',type=Path,default=Path('docs/offaxis-eda/ecc-gauss-newton'))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    checks=self_check();print('Checks:',checks,flush=True)
    if args.check_only:return
    args.output.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(args.source/'ppt-method-v2/frames.csv').set_index('path')
    selected=meta[meta.category.isin(['mixed','new-reference'])].sort_index()
    assert len(selected)==60
    models=json.loads((args.source/'raw/reference.json').read_text())
    base=load(args.root/'center.bmp')
    rows=[];start=time.perf_counter(); hashes={}
    for i,path in enumerate(selected.index):
        image=load(args.root/path)
        hashes[path]=hashlib.sha256((args.root/path).read_bytes()).hexdigest()
        for qr in models:
            rows.append(dict(path=path,qr=qr['slot'],**measure(image,base,qr)))
        if (i+1)%5==0:print(f'{i+1}/60 images; {time.perf_counter()-start:.1f}s',flush=True)
    measurements=pd.DataFrame(rows)
    measurements.to_csv(args.output/'measurements.csv',index=False)
    fresh=measurements[['path','qr','ecc_dx','ecc_dy']].rename(columns={'ecc_dx':'dx','ecc_dy':'dy'})
    methods={'GN':measurements,'ECC_fresh':fresh,'ECC_saved':pd.read_csv(args.source/'raw/ecc.csv'),
             'Corners630':pd.read_csv(args.source/'ppt-method-v2/qr_displacements.csv')}
    scores=evaluate(methods,meta,args.output)
    manifest=dict(checks=checks,seconds=time.perf_counter()-start,
                  status_counts=measurements.status.value_counts().to_dict(),
                  min_correlation=float(measurements.correlation.min()),
                  input_sha256=hashes,versions={k:v for k,v in [('numpy',np.__version__),('opencv',cv2.__version__),('pandas',pd.__version__)]},
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__),Path(__file__).with_name('refit_model.py'),args.root/'center.bmp',
                                  args.source/'raw/reference.json',args.source/'raw/ecc.csv',
                                  args.source/'ppt-method-v2/qr_displacements.csv',args.source/'ppt-method-v2/frames.csv']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(scores.to_string(index=False),flush=True)
    print(manifest['status_counts'],flush=True)


if __name__=='__main__':main()
