"""Collect direct warps in independent processes, calibrate on training frames only."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from ytheta_model import physical, gauss_newton, estimate


def collect(repo, output, workers):
    paths = sorted((repo/'data/SSAFY Test Image Set').rglob('*.bmp'))
    if not paths: raise ValueError('no BMP input')
    reference = repo/'data/SSAFY Test Image Set/center.bmp'
    jobs = []
    for method in ['affine', 'homography']:
        for worker in range(workers):
            chunk = output/f'{method}_{worker}'; chunk.mkdir(parents=True, exist_ok=True)
            listing = chunk/'images.txt'; listing.write_text('\n'.join(map(str, paths[worker::workers])), encoding='utf-8')
            jobs.append((method, chunk, listing))
    def work(job):
        method, chunk, listing = job
        with (chunk/'log.txt').open('w') as log:
            subprocess.run([str(repo/'build/Release/ytheta_collect.exe'), str(reference),
                str(listing), str(chunk/'measurements.csv'), method], stdout=log, stderr=log, check=True)
        return method, chunk
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(work, jobs))
    for method in ['affine', 'homography']:
        pd.concat([pd.read_csv(chunk/'measurements.csv') for m, chunk in result if m==method],
                  ignore_index=True).to_csv(output/f'{method}.csv', index=False)
    manifest = {'images': len(paths), 'workers': workers, 'opencv_threads_per_worker': 1,
                'seconds': time.perf_counter()-started, 'sources': {}}
    for relative in ['src/direct_registration.cpp','tool/ytheta_collect.cpp','include/alignment.hpp',
                     'src/qr_features.cpp','src/qr_decode.cpp','src/qr_model.cpp','src/plate_detection.cpp']:
        manifest['sources'][relative] = hashlib.sha256((repo/relative).read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))


def load_frames(csv):
    raw = pd.read_csv(csv); frames = []
    for path, points in raw.groupby('path'):
        name = Path(path).stem
        if name=='center' or not (points.status=='ok').all(): continue
        points = points.sort_values('id'); axis = name[0]; value = float(name[1:])
        group = axis+('+' if value>0 else '-')+(str(int(round(abs(value)*100))%10) if axis=='Y' else '')
        frames.append(dict(path=path, name=name, axis=axis, value=value, group=group,
                           observed=points[['u','v']].values,
                           reference=points[['u0','v0']].values, ecc=float(points.ecc.iloc[0])))
    if not frames: raise ValueError('no successful non-reference frames')
    return raw, frames


def calibrate(frames, split, kind, fx):
    groups = sorted({r['group'] for r in frames})
    train = []; test = []
    for group in groups:
        subset = sorted([r for r in frames if r['group']==group], key=lambda r:r['value'])
        for j, frame in enumerate(subset):
            hold = j%3==0 if split=='interleaved' else len(subset)//3 <= j < 2*len(subset)//3
            (test if hold else train).append(frame)
    q = train[0]['reference']; observed = np.array([r['observed'] for r in train])
    if any(not np.allclose(r['reference'],q) for r in frames): raise ValueError('reference grid changed')
    poses = np.array([[r['value'],0] if r['axis']=='Y' else [0,r['value']] for r in train])
    indices = np.array([groups.index(r['group']) for r in train])
    if kind=='physical':
        def residual(v):
            prediction = physical(q, poses, v[:3], fx=fx)
            return (prediction+v[3:].reshape(-1,2)[indices,None,:]-observed).ravel()
        initial = np.r_[np.log(307.8),0.,0.,np.zeros(2*len(groups))]
        fit = gauss_newton(residual, initial,
            np.r_[np.log(150),-1.,-1.,np.full(2*len(groups),-20.)],
            np.r_[np.log(600),1.,1.,np.full(2*len(groups),20.)], iterations=70)
        model = dict(kind=kind, reference=q.tolist(), parameters=fit['x'][:3].tolist(),
                     fx=fx, fy=4285.714285714,
                     offsets={g:fit['x'][3+2*j:5+2*j].tolist() for j,g in enumerate(groups)},
                     training_residual_px=fit['rms'], calibration_converged=fit['converged'])
    else:
        X = np.array([[float(g==r['group']) for g in groups]+[y,t,y*y,t*t]
                      for r,(y,t) in zip(train,poses)])
        coefficients = np.linalg.lstsq(X,(observed-q).reshape(len(train),-1),rcond=None)[0]
        # Group intercept is pointwise here to match prior feature baseline; report capacity difference.
        offsets = coefficients[:len(groups)].reshape(len(groups),len(q),2)
        model = dict(kind=kind, reference=q.tolist(), coefficients=coefficients[-4:].reshape(4,len(q),2).tolist(),
                     offsets={g:offsets[j].tolist() for j,g in enumerate(groups)},
                     training_residual_px=float(np.sqrt(np.mean((X@coefficients-(observed-q).reshape(len(train),-1))**2))))
    model.update(split=split, calibration_reference='center image', bounds={'y_mm':[-2,2],'theta_deg':[-1,1]},
                 unobserved_mixed_term=True, real_mixed_validation=False,
                 assumptions=['fixed wall, camera parallel translation and yaw',
                              'physical: pivot co-located, D omitted, fixed nominal K',
                              'session label known during heldout evaluation; offsets learned only from train'])
    rows=[]
    for frame in test:
        result=estimate(model, frame['observed'],frame['group'])
        truth=[frame['value'],0] if frame['axis']=='Y' else [0,frame['value']]
        rows.append(dict(name=frame['name'],axis=frame['axis'],group=frame['group'],
                         nominal=frame['value'], y_error=result['y_mm']-truth[0],
                         theta_error=result['theta_deg']-truth[1], **result))
    return model, pd.DataFrame(rows)


def analyze(output):
    report={}
    for method in ['affine','homography']:
        raw,frames=load_frames(output/f'{method}.csv')
        statuses=raw.groupby('path').first().status.value_counts().to_dict()
        report[method]={'statuses':statuses,'models':{}}
        for split in ['interleaved','blocked']:
            for kind,fx in [('physical',4017.857142857),('physical',4285.714285714),('direct',0.)]:
                label=f'{kind}_{int(fx)}_{split}'
                model,pred=calibrate(frames,split,kind,fx)
                (output/f'{method}_{label}.json').write_text(json.dumps(model,indent=2))
                pred.to_csv(output/f'{method}_{label}_holdout.csv',index=False)
                stats={'training_residual_px':model['training_residual_px'],'holdout':{}}
                for axis,g in pred.groupby('axis'):
                    stats['holdout'][axis]={'n':len(g),'y_rmse':float(np.sqrt(np.mean(g.y_error**2))),
                        'theta_rmse':float(np.sqrt(np.mean(g.theta_error**2))),
                        'joint_within_tolerance':int(((abs(g.y_error)<=.09)&(abs(g.theta_error)<=.09)).sum()),
                        'converged':int(g.converged.sum()),'boundary':int(g.at_boundary.sum()),
                        'median_condition':float(g.condition.median())}
                if kind=='physical':stats['depth_mm']=float(np.exp(model['parameters'][0]));stats['plane_slopes']=model['parameters'][1:]
                report[method]['models'][label]=stats
                print(method,label,json.dumps(stats),flush=True)
    (output/'evaluation_summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--collect',action='store_true');args=parser.parse_args()
    if not 1<=args.workers<=16:parser.error('workers must be in [1,16]')
    args.repo=args.repo.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.collect:collect(args.repo,args.output,args.workers)
    analyze(args.output)
