"""Train-only feasible GLS ablations; original deployment is never overwritten.

All methods retain the QR205 measurements, six observed origin modes, axis-shared
quadratic response and the original sign-constrained inference optimizer.
The full-280 fit is resubstitution. Repeated CV is development sensitivity, not
an independent new acquisition. Every covariance and spatial projection is
fitted inside each training split. No held-out command or group enters inference.
"""
from pathlib import Path
import hashlib, json, shutil, sys, time
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import zero_experiment as Z

CONFIGS = {
    'baseline': ('ols', 'legacy'),
    'ols_euclidean': ('ols', 'identity'),
    'ols_wls': ('ols', 'diagonal'),
    'ols_gls_lw': ('ols', 'ledoit_wolf'),
    'spatial_wls': ('diagonal', 'legacy'),
    'spatial_gls_lw': ('ledoit_wolf', 'legacy'),
    'spatial_gls_lw_metric_lw': ('ledoit_wolf', 'ledoit_wolf'),
}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def snapshot():
    files = [
        'infer.py', 'zero_experiment.py', 'validate_calibrated_runs.py',
        'calibrated_runs/model.json', 'calibrated_runs/stress_metrics.csv',
        'calibrated_runs/stress_predictions.csv', 'reference.json', 'manifest.csv',
        'measurement/gn17_spread14_features.npz', 'measurement/gn17_spread14.npz',
        'measurement/spread.py', 'measurement/refine.py', 'measurement/measure.py',
        'full280/summary.json', 'full280/predictions.csv', 'full280/evaluate_all.py',
    ]
    ledger = HERE / 'baseline_snapshot' / 'hashes.json'
    if ledger.exists():
        records = json.loads(ledger.read_text(encoding='utf8'))
        for f, digest in records.items():
            assert sha(BASE/f) == digest and sha(ledger.parent/f) == digest
        return records
    records = {}
    for f in files:
        src, dst = BASE/f, ledger.parent/f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records[f] = sha(src)
        assert sha(dst) == records[f]
    ledger.write_text(json.dumps(records, indent=2), encoding='utf8')
    return records

def fit_curves(m, X, tr):
    branches, residual, rank = {}, [], 0
    for axis in ['Y','T']:
        ii = tr[m.axis.to_numpy()[tr] == axis]
        v = m.value.to_numpy()[ii]
        ss = m.session.to_numpy()[ii]
        groups = np.unique(ss)
        A = np.c_[v, v*v, (ss[:,None] == groups).astype(float)]
        co = np.linalg.lstsq(A, X[ii], rcond=None)[0]
        residual.append(X[ii]-A@co)
        rank += np.linalg.matrix_rank(A)
        for j, group in enumerate(groups):
            branches[group] = (axis, co[j+2], co[0], co[1])
    return branches, np.vstack(residual), int(rank)

def covariance(R, rank, kind):
    S = R.T@R/max(len(R)-rank, 1)
    sd = np.sqrt(np.maximum(np.diag(S), 1e-14))
    if kind == 'identity':
        return np.eye(R.shape[1]), 1.
    if kind == 'diagonal':
        return np.diag(sd*sd), 1.
    if kind == 'legacy':
        C = S/sd[:,None]/sd[None,:]
        C = .999*C+.001*np.eye(len(C))
        alpha = .001
    elif kind == 'ledoit_wolf':
        lw = LedoitWolf(assume_centered=True).fit(R/sd[None,:])
        C, alpha = lw.covariance_, float(lw.shrinkage_)
    else:
        raise ValueError(kind)
    return C*sd[:,None]*sd[None,:], alpha

def precision_root(S):
    # W.T @ W = inv(S), matching the original convention.
    return np.linalg.cholesky(np.linalg.inv(S)).T

def fit_variant(m, D, X_ols, H, tr, spatial, metric, raw=None):
    projection, spa_alpha, spa_condition = None, None, None
    X = X_ols
    if spatial != 'ols':
        if raw is None:
            _, Rraw, raw_rank = fit_curves(m, D, tr)
        else:
            Rraw, raw_rank = raw
        Sigma, spa_alpha = covariance(Rraw, raw_rank, spatial)
        # GLS: (H' Sigma^-1 H)^-1 H' Sigma^-1 d, computed without inv(Sigma).
        precision_H = np.linalg.solve(Sigma, H)
        projection = np.linalg.solve(H.T@precision_H, precision_H.T)
        X = D@projection.T
        spa_condition = float(np.linalg.cond(Sigma))
    branches, R, rank = fit_curves(m, X, tr)
    Sigma, alpha = covariance(R, rank, metric)
    if metric == 'legacy':
        # Exact same operation order as the deployed model for reproducibility.
        S = R.T@R/max(len(R)-rank,1)
        std = np.sqrt(np.maximum(np.diag(S),1e-14))
        C = S/std[:,None]/std[None,:]
        C = .999*C+.001*np.eye(len(C))
        W = np.linalg.cholesky(np.linalg.inv(C)).T/std[None,:]
    else:
        W = precision_root(Sigma)
    info = dict(spatial_shrinkage=spa_alpha, metric_shrinkage=alpha,
                spatial_condition=spa_condition, metric_condition=float(np.linalg.cond(Sigma)))
    return (branches, W), X, projection, info

def score(df):
    r = Z.score(df)
    correct = (df.axis == df.pred_axis) & (np.sign(df.value) == np.sign(df.pred))
    truth = np.c_[np.where(df.axis=='Y',df.value,0),np.where(df.axis=='T',df.value,0)]
    pred = np.c_[np.where(df.pred_axis=='Y',df.pred,0),np.where(df.pred_axis=='T',df.pred,0)]
    e = np.abs(pred-truth)
    r.update(correct=int(correct.sum()), failed=int((~correct | (e>.09).any(1)).sum()),
             Y_correct=int(correct[df.axis=='Y'].sum()), T_correct=int(correct[df.axis=='T'].sum()),
             Y_vector_mae=float(e[:,0].mean()),T_vector_mae=float(e[:,1].mean()))
    return r

def protocols(m):
    ids = np.flatnonzero(m.axis!='C')
    yield 'full280', [('all',ids,ids)]
    for kind in ['interleaved', 'stratified5']:
        yield kind, list(Z.splits(m,kind))
    blocks = []
    for k in range(5):
        te = []
        for s in m.session[m.axis!='C'].unique():
            ii = m.index[m.session==s].to_numpy()
            ii = ii[np.argsort(m.value.iloc[ii])]
            te.extend(np.array_split(ii,5)[k])
        te = np.sort(te)
        blocks.append((str(k),np.setdiff1d(ids,te),te))
    yield 'blocked_within_session5', blocks
    for seed in range(20):
        folds = [(str(k),ids[a],ids[b]) for k,(a,b) in enumerate(
            StratifiedKFold(5,shuffle=True,random_state=7000+seed).split(ids,m.session.iloc[ids]))]
        yield f'repeated5_seed{seed}', folds
    yield 'session_out_missing_origin', list(Z.splits(m,'session_out'))

def main():
    start = time.perf_counter()
    hashes = snapshot()
    (HERE/'experiment_plan.json').write_text(json.dumps(dict(
        methods=CONFIGS, seeds=list(range(7000,7020)), point_count=205,
        same_measurements=True, same_origin_branches=True, same_quadratic_curves=True,
        covariance_training_only=True, no_posthoc_hyperparameter_search=True,
        full280='resubstitution on development data',
        cv='same acquisition groups; all 280 held out once per protocol; development only',
        session_out='diagnostic: absent origin branch, T sign can be unsupported',
        scope='Feasible GLS across point-coordinate errors; no independently known sensor covariance.'
    ),indent=2),encoding='utf8')
    m,X_ols,_,_ = Z.load('gn17_spread14')
    z = np.load(BASE/'measurement/gn17_spread14.npz')
    d = z['D'].astype(float)
    d -= d[z['names']=='center.bmp']
    D = d.reshape(len(d),-1)
    p = z['points'].astype(float)
    x,y = (p[:,0]-1906)/486,(p[:,1]-1070)/122
    B = np.array([x*0+1,x,y,x*x,x*y,y*y]).T
    H = np.kron(B,np.eye(2))
    assert np.allclose(D@np.linalg.pinv(H).T,X_ols,atol=1e-10)
    assert z['valid'].all() and D.shape[1] == 410
    records, stats, diagnostics, split_records = [],[],[],[]
    full_models = HERE/'models'
    full_models.mkdir(exist_ok=True)
    for protocol, folds in protocols(m):
        grouped = {name:[] for name in CONFIGS}
        for fold,tr,te in folds:
            if protocol!='full280': assert not np.intersect1d(tr,te).size
            split_records.append(dict(protocol=protocol,fold=fold,train=m.file.iloc[tr].tolist(),test=m.file.iloc[te].tolist()))
            _,Rraw,rank = fit_curves(m,D,tr)
            raw = (Rraw,rank)
            for name,(spatial,metric) in CONFIGS.items():
                model,X,P,info = fit_variant(m,D,X_ols,H,tr,spatial,metric,raw)
                out = Z.predict(m,X,te,model,'inferred_session').assign(method=name,protocol=protocol,fold=fold)
                grouped[name].append(out)
                diagnostics.append(dict(method=name,protocol=protocol,fold=fold,**info))
                if protocol=='full280':
                    params = dict(method=name,spatial=spatial,metric=metric,
                        projection=None if P is None else P.tolist(),
                        W=model[1].tolist(), branches={k:[a,b.tolist(),s.tolist(),c.tolist()] for k,(a,b,s,c) in model[0].items()},
                        point_feature_path='../../measurement/gn17_spread14_features.npz',
                        scope='Existing six origin modes; trained on all 280; new acquisitions unvalidated')
                    (full_models/(name+'.json')).write_text(json.dumps(params,indent=2),encoding='utf8')
        for name, parts in grouped.items():
            out = pd.concat(parts,ignore_index=True)
            assert len(out)==280 and out.file.nunique()==280
            records.append(out)
            stats.append(dict(method=name,protocol=protocol,**score(out)))
        pd.DataFrame(stats).to_csv(HERE/'metrics.csv',index=False)
        print(protocol, {r['method']:r['failed'] for r in stats[-len(CONFIGS):]}, round(time.perf_counter()-start,2), flush=True)
    pred = pd.concat(records,ignore_index=True)
    pred['correct'] = (pred.axis==pred.pred_axis)&(np.sign(pred.value)==np.sign(pred.pred))
    for axis in ['Y','T']:
        pred['error_'+axis] = np.abs(np.where(pred.axis==axis,pred.value,0)-np.where(pred.pred_axis==axis,pred.pred,0))
    pred['pass09'] = pred.correct & (pred.error_Y<=.09)&(pred.error_T<=.09)
    pred.to_csv(HERE/'predictions.csv',index=False,encoding='utf-8-sig')
    pred[~pred.pass09].to_csv(HERE/'failures.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(HERE/'covariance_diagnostics.csv',index=False)
    (HERE/'splits.json').write_text(json.dumps(split_records),encoding='utf8')
    # Verify original repeat results and deploy fit; ensure nothing was overwritten.
    baseline = pd.DataFrame(stats).query("method=='baseline'")
    old = pd.read_csv(BASE/'calibrated_runs/stress_metrics.csv')
    merged = baseline.merge(old,left_on='protocol',right_on='cv',suffixes=('_new','_old'))
    assert len(merged)==23
    for key in ['axis_sign_accuracy','vector_Y_max','vector_T_max']:
        assert np.allclose(merged[key+'_new'],merged[key+'_old'],atol=1e-8),key
    oldfull = pd.read_csv(BASE/'full280/predictions.csv')
    newfull = pred.query("method=='baseline' and protocol=='full280'").merge(oldfull,on='file')
    assert (newfull.pred_axis_x==newfull.pred_axis_y).all()
    full_diff = float(np.max(np.abs(newfull.pred-newfull.pred_value)))
    assert full_diff<1e-7
    # Held-out changes must not alter training-fit projection/curves/precision.
    _,tr,te = next(Z.splits(m,'stratified5'))
    first = fit_variant(m,D,X_ols,H,tr,'ledoit_wolf','legacy')
    m2=m.copy();m2.loc[te,'value']+=999;m2.loc[te,'session']='UNUSED';m2.loc[te,'axis']='T'
    D2=D.copy();D2[te]+=1234;X2=X_ols.copy();X2[te]+=1234
    second = fit_variant(m2,D2,X2,H,tr,'ledoit_wolf','legacy')
    assert np.array_equal(first[2],second[2]) and np.array_equal(first[0][1],second[0][1])
    assert all(np.array_equal(np.asarray(first[0][0][k][1:]),np.asarray(second[0][0][k][1:])) for k in first[0][0])
    unchanged = all(sha(BASE/f)==digest for f,digest in hashes.items())
    assert unchanged
    checks=dict(original_files_unchanged=unchanged, all_protocols_each_have_280_unique=True,
        legacy_23_protocols_reproduced=True, baseline_full_prediction_max_difference=full_diff,
        heldout_perturbation_does_not_change_fitted_gls=True,
        spatial_identity_matches_ols=True, measurement_source='fixed cached raw-image GN17 QR205 tracks',
        independent_test=False, runtime_seconds=time.perf_counter()-start)
    (HERE/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':
    main()
