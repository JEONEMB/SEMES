"""Fixed small candidate screen: repeated-corpus development OOF, not new test."""
from pathlib import Path
import sys, json, time, hashlib
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

H = Path(__file__).resolve().parent
PREV = H.parents[1] / 'joint_yt_generalization_20261022'
sys.path.insert(0, str(PREV))
from data import load, metric
from robust_model import RobustBaggedForward, CONFIGS, REPRESENTATIVE_CONFIGS


def main():
    m, arrays = load()
    D = arrays['D']
    y = m[['Y', 'T']].to_numpy()
    protocol = dict(
        candidate_configs=CONFIGS,
        a_priori_nested_representatives=REPRESENTATIVE_CONFIGS,
        measurement_source=str(PREV / 'measurements/features.npz'),
        source_sha256=hashlib.sha256((PREV / 'measurements/features.npz').read_bytes()).hexdigest(),
        inference='predict(D) only; metadata and labels are used solely during fit',
        partition='Existing fixed four folds; same new command repeats grouped; all new centers in fold0',
        interpretation='Repeated-corpus development OOF, not an untouched independent test; prior failures informed method design',
        fixed_points=630,
        distinction='No target snapping or special handling of the prior ten failed photos',
        model_source_sha256=hashlib.sha256((H / 'robust_model.py').read_bytes()).hexdigest(),
        folds=[dict(fold=f, train_indices=m.loc[m.fold != f, 'index'].tolist(),
                    test_indices=m.loc[m.fold == f, 'index'].tolist()) for f in range(4)])
    (H / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    predictions, summaries, audits = [], [], []
    for k, config in enumerate(CONFIGS):
        start = time.monotonic()
        pred = np.full(y.shape, np.nan)
        for fold in range(4):
            tr = m.fold.to_numpy() != fold
            te = ~tr
            assert not set(m.loc[tr, 'cv_group']) & set(m.loc[te, 'cv_group'])
            model = RobustBaggedForward(config).fit(D[tr], y[tr], m.loc[tr])
            pred[te] = model.predict(D[te])
            single = model.predict(D[np.flatnonzero(te)[0]])[0]
            assert np.allclose(single, pred[te][0], atol=1e-9)
            audits.append(dict(candidate=k, fold=fold, **model.audit))
            joblib.dump(dict(model=model, train_indices=m.loc[tr, 'index'].to_numpy(), config=config), H / f'candidate{k}_fold{fold}.joblib')
        assert np.isfinite(pred).all()
        p = m[['index', 'dataset', 'file', 'relative', 'Y', 'T', 'scope', 'condition', 'cv_group', 'fold']].copy()
        p['candidate'] = k
        p['predY'], p['predT'] = pred.T
        p['eY'], p['eT'] = np.abs(pred-y).T
        p['passed'] = (p.eY <= .09+1e-12) & (p.eT <= .09+1e-12)
        p['protocol'] = 'fixed4_development_oof'
        predictions.append(p)
        for dataset in ['all', 20261014, 20261020]:
            mask = np.ones(len(m), bool) if dataset == 'all' else m.dataset.to_numpy() == dataset
            summaries.append(dict(candidate=k, dataset=dataset, **config, **metric(y[mask], pred[mask])))
        pd.concat(predictions).to_csv(H / 'predictions.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(summaries).to_csv(H / 'summary.csv', index=False, encoding='utf-8-sig')
        print(k, config, summaries[-1], 'seconds', round(time.monotonic()-start, 1), flush=True)
    (H / 'fit_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    (H / 'verification.json').write_text(json.dumps(dict(
        status='passed', candidates=len(CONFIGS), rows=len(CONFIGS)*len(m),
        finite_predictions=True, same_condition_group_overlap=False,
        batch_and_single_consistent=True,
        qualification='These checks do not remove adaptive model-development bias from reusing the corpus.'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
