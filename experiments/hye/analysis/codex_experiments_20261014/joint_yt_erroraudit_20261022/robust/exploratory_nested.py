"""Exploratory nested16 after screen: exactly reuse prior inner indices.

This selection is additional adaptive development on a reused corpus, not an
untouched test. Within each outer split, however, candidate ranking uses only
that split's original inner validation rows.
"""
from pathlib import Path
from datetime import datetime, timezone
import sys, json, time, hashlib
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

H = Path(__file__).resolve().parent
PREV = H.parents[1] / 'joint_yt_generalization_20261022'
PRIOR = H.parents[1] / 'joint_yt_disentangle_20261022'
sys.path.insert(0, str(PREV))
from data import load, metric
from robust_model import RobustBaggedForward, CONFIGS


def select(m, D, y, audit):
    scores = []
    new = m.dataset.to_numpy() == 20261020
    for k, cfg in enumerate(CONFIGS):
        validation_predictions, validation_targets = [], []
        for part in audit['inner_splits']:
            it = np.asarray(part['train'], dtype=int)
            iv = np.asarray(part['validation'], dtype=int)
            iv = iv[new[iv]]
            assert not set(m.iloc[it].cv_group) & set(m.iloc[iv].cv_group)
            model = RobustBaggedForward(cfg).fit(D[it], y[it], m.iloc[it])
            validation_predictions.append(model.predict(D[iv]))
            validation_targets.append(y[iv])
        errors = np.abs(np.vstack(validation_predictions) - np.vstack(validation_targets))
        scores.append(dict(candidate=k, config=cfg,
                           failures=int(np.any(errors > .09+1e-12, axis=1).sum()),
                           maximum=float(errors.max()), mean=float(errors.mean())))
    chosen = min(scores, key=lambda r:(r['failures'], r['maximum'], r['mean']))['candidate']
    return chosen, dict(seed=audit['seed'], inner_splits=audit['inner_splits'],
                        scores=scores, selected=chosen)


def rows(m, ix, pred, candidate, protocol):
    r = m.iloc[ix][['index', 'dataset', 'file', 'relative', 'Y', 'T', 'scope', 'condition', 'cv_group', 'fold']].copy()
    r['candidate'] = candidate
    r['predY'], r['predT'] = pred.T
    r['eY'], r['eT'] = abs(pred-r[['Y', 'T']].to_numpy()).T
    r['passed'] = (r.eY <= .09+1e-12) & (r.eT <= .09+1e-12)
    r['protocol'] = protocol
    return r


def main():
    m, arrays = load()
    D = arrays['D']
    y = m[['Y', 'T']].to_numpy()
    previous = json.loads((PRIOR / 'selection.json').read_text())
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        note='16개 screen 결과를 본 뒤 추가한 탐색적 nested 비교; 같은341장 반복개발이며 독립 테스트가 아니다.',
        prior_inner_splits=str(PRIOR / 'selection.json'),
        prior_inner_splits_sha256=hashlib.sha256((PRIOR / 'selection.json').read_bytes()).hexdigest(),
        candidates=CONFIGS,
        rule='Fewest new-date inner-validation images exceeding either0.09; tie by worst axis error then mean axis error',
        input='predict(D) only; no target/date/filename in inference',
        a_priori_primary_comparison='Separate root evaluation with originally declared four representative candidates is preserved',
        independent_test=False)
    (H / 'exploratory_nested_protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    predictions, selection = [], []
    for f in range(4):
        start = time.monotonic()
        tr = np.flatnonzero(m.fold.to_numpy() != f)
        te = np.flatnonzero(m.fold.to_numpy() == f)
        audit = next(x for x in previous if x.get('fold') == f)
        for s in audit['inner_splits']:
            assert set(s['train']) <= set(tr)
            assert set(s['validation']) <= set(tr)
        k, chosen = select(m, D, y, audit)
        model = RobustBaggedForward(CONFIGS[k]).fit(D[tr], y[tr], m.iloc[tr])
        p = model.predict(D[te])
        predictions.append(rows(m, te, p, k, 'exploratory_nested16_oof'))
        selection.append(dict(fold=f, **chosen))
        joblib.dump(dict(model=model, config=CONFIGS[k], train_indices=tr,
                         points=arrays['points'], ids=arrays['ids']), H / f'exploratory_nested_fold{f}.joblib')
        pd.concat(predictions).to_csv(H / 'exploratory_nested_predictions.csv', index=False, encoding='utf-8-sig')
        (H / 'exploratory_nested_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
        mask = m.iloc[te].dataset.to_numpy() == 20261020
        print('fold', f, 'candidate', k, metric(y[te][mask], p[mask]), 'seconds', round(time.monotonic()-start, 1), flush=True)
    full_audit = next(x['full_selection'] for x in previous if 'full_selection' in x)
    k, chosen = select(m, D, y, full_audit)
    model = RobustBaggedForward(CONFIGS[k]).fit(D, y, m)
    joblib.dump(dict(model=model, config=CONFIGS[k], train_indices=np.arange(len(m)),
                     points=arrays['points'], ids=arrays['ids']), H / 'exploratory_nested_model.joblib')
    predictions.append(rows(m, np.arange(len(m)), model.predict(D), k, 'resub341'))
    selection.append(dict(full_selection=chosen))
    p = pd.concat(predictions)
    p.to_csv(H / 'exploratory_nested_predictions.csv', index=False, encoding='utf-8-sig')
    (H / 'exploratory_nested_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    summaries=[]
    for kind,g in p.groupby('protocol'):
        for dataset in ['all',20261014,20261020]:
            z = g if dataset == 'all' else g[g.dataset == dataset]
            summaries.append(dict(protocol=kind,dataset=dataset,**metric(z[['Y','T']],z[['predY','predT']])))
    pd.DataFrame(summaries).to_csv(H / 'exploratory_nested_summary.csv',index=False,encoding='utf-8-sig')
    old = pd.read_csv(PRIOR / 'predictions.csv')
    old = old[(old.protocol == 'nested_group_oof') & (old.method == 'pass_first')]
    compare = old[['index','predY','predT','eY','eT','passed']].merge(p[p.protocol == 'exploratory_nested16_oof'],on='index',suffixes=('_prior','_candidate'))
    compare['transition'] = np.select([compare.passed_prior & compare.passed_candidate,~compare.passed_prior & compare.passed_candidate,compare.passed_prior & ~compare.passed_candidate],['kept_pass','recovered','new_failure'],default='still_failure')
    compare.to_csv(H / 'exploratory_nested_comparison.csv',index=False,encoding='utf-8-sig')
    analysis=dict(new60_transitions=compare[compare.dataset==20261020].transition.value_counts().to_dict(),old281_transitions=compare[compare.dataset==20261014].transition.value_counts().to_dict(),selected=[s['selected'] for s in selection if 'fold' in s],full_selected=k,independent_test=False)
    (H / 'exploratory_nested_analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(json.dumps(analysis),flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):main()
