"""Independent read-only recheck; writes exclusively to this audit directory."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import joblib, numpy as np, pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
OLD = EXPERIMENT.parent / 'joint_yt_disentangle_20261022'
DATA = EXPERIMENT.parent / 'joint_yt_generalization_20261022'
for folder in [DATA, OLD, EXPERIMENT / 'robust']:
    sys.path.insert(0, str(folder))
from data import load
from learned_model import LearnedSeparation
from robust_model import RobustBaggedForward

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def metric(frame):
    error = np.abs(frame[['predY', 'predT']].to_numpy() - frame[['Y', 'T']].to_numpy())
    valid = np.isfinite(error).all(axis=1)
    return dict(n=len(frame), failures=int((~valid).sum()), passed=int((valid & (error <= .09 + 1e-12).all(1)).sum()), Y_MAE=float(error[valid, 0].mean()), Y_max=float(error[valid, 0].max()), T_MAE=float(error[valid, 1].mean()), T_max=float(error[valid, 1].max()))

def main():
    m, a = load()
    previous = json.loads((OLD / 'verification.json').read_text())
    source_checks = {name: sha(OLD / name) == expected for name, expected in previous['source_hashes'].items()}
    assert all(source_checks.values()), 'Previously audited source changed; static audit must be repeated.'
    original = json.loads((OLD.parent / 'joint_yt_improved_20261022/splits.json').read_text())
    assert len(m) == 341 and a['D'].shape == (341, 630, 2)
    assert m.groupby('cv_group').fold.nunique().eq(1).all()
    centers = m[(m.dataset == 20261020) & (m.scope == 'center')]
    assert len(centers) == 3 and set(centers.fold) == {0}
    checks, metrics = [], []
    max_difference = 0.
    for folder, methods, oof, resub in [(OLD, ['pass_first', 'max_first'], 'nested_group_oof', 'resub341'), (EXPERIMENT, ['extended', 'robust_only'], 'nested_development_oof', 'training_reapplication')]:
        predictions = pd.read_csv(folder / 'predictions.csv')
        summary = pd.read_csv(folder / 'summary.csv', dtype={'dataset': str})
        selection = json.loads((folder / 'selection.json').read_text())
        for fold in range(4):
            tr = np.flatnonzero(m.fold.to_numpy() != fold)
            te = np.flatnonzero(m.fold.to_numpy() == fold)
            assert not set(m.iloc[tr].cv_group) & set(m.iloc[te].cv_group)
            assert set(m.loc[(m.dataset == 20261014) & (m.scope == 'single') & (m.fold == fold), 'file']) == set(original[fold]['test'])
            seen = []
            for split in selection[fold]['inner_splits']:
                it, iv = set(split['train']), set(split['validation'])
                assert it | iv == set(tr) and not it & iv and not (it | iv) & set(te)
                assert not set(m.iloc[list(it)].cv_group) & set(m.iloc[list(iv)].cv_group)
                seen.extend(iv)
            assert sorted(seen) == sorted(tr)
            if folder == EXPERIMENT:
                earlier = json.loads((OLD / 'selection.json').read_text())[fold]
                assert selection[fold]['inner_splits'] == earlier['inner_splits']
                assert selection[fold]['scores'][:15] == earlier['scores']
                rank = lambda row: (row['failures'], row['maximum'], row['mean'])
                assert selection[fold]['chosen']['extended'] == min(selection[fold]['scores'], key=rank)['candidate']
                assert selection[fold]['chosen']['robust_only'] == min(selection[fold]['scores'][15:], key=rank)['candidate']
            for method in methods:
                bundle = joblib.load(folder / f'{method}_fold{fold}.joblib')
                assert np.array_equal(bundle['train_indices'], tr)
                if hasattr(bundle['model'], 'training_indices'):
                    assert np.array_equal(bundle['model'].training_indices, m.iloc[tr]['index'])
                else:
                    assert bundle['model'].audit['training_rows'] == len(tr)
                assert np.array_equal(bundle['points'], a['points']) and np.array_equal(bundle['ids'], a['ids'])
                p = bundle['model'].predict(a['D'][te])
                saved = predictions[(predictions.method == method) & (predictions.protocol == oof) & (predictions.fold == fold)].set_index('index').loc[te]
                diff = float(np.max(abs(p - saved[['predY', 'predT']].to_numpy())))
                assert diff < 1e-10
                max_difference = max(max_difference, diff)
            checks.append(dict(experiment=folder.name, fold=fold, train=len(tr), test=len(te), direct_group_overlap=False, saved_train_indices_verified=True))
        for method in methods:
            bundle = joblib.load(folder / f'{method}_model.joblib')
            assert np.array_equal(bundle['train_indices'], np.arange(341))
            if hasattr(bundle['model'], 'training_indices'):
                assert np.array_equal(bundle['model'].training_indices, m['index'])
            else:
                assert bundle['model'].audit['training_rows'] == len(m)
            p = bundle['model'].predict(a['D'])
            saved = predictions[(predictions.method == method) & (predictions.protocol == resub)].set_index('index').loc[np.arange(341)]
            diff = float(np.max(abs(p - saved[['predY', 'predT']].to_numpy())))
            assert diff < 1e-10
            max_difference = max(max_difference, diff)
        for (method, protocol), g in predictions.groupby(['method', 'protocol']):
            assert len(g) == 341 and set(g['index']) == set(range(341)) and g['index'].is_unique
            assert np.allclose(g.eY, abs(g.predY - g.Y)) and np.allclose(g.eT, abs(g.predT - g['T']))
            assert np.allclose(g.correctionY, -g.predY) and np.allclose(g.correctionT, -g.predT)
            for dataset in ['all', '20261014', '20261020']:
                z = g if dataset == 'all' else g[g.dataset == int(dataset)]
                values = metric(z)
                row = summary[(summary.method == method) & (summary.protocol == protocol) & (summary.dataset == dataset)].iloc[0]
                assert all(np.isclose(row[key], value, rtol=1e-10, atol=1e-12) for key, value in values.items())
                metrics.append(dict(experiment=folder.name, method=method, protocol=protocol, dataset=dataset, **values))
    preserved = []
    for record in previous['preserved_models']:
        assert sha(record['file']) == record['sha256']
        preserved.append(record)
    for name, digest in previous['current_model_hashes'].items():
        assert sha(OLD / name) == digest
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), status='checks_passed_with_material_evaluation_limitation', direct_fold_leakage_found=False, filename_or_test_label_in_predict_found=False, adaptive_test_contamination=True, all_341_images_are_development_data=True, a_resplit_is_pristine_independent_test=False, independent_capture_available=False, statement='No direct outer-test fitting defect was found in the audited saved nested runs. However, all 341 images have influenced iterative feature/model development through repeated OOF and error inspection. Their present and future resplits are development estimates, not independent new-photo performance. The earlier phrase no material data leakage is limited to direct fold implementation and must not be read as absence of adaptive evaluation contamination.', original_source_hash_checks=source_checks, fold_checks=checks, all_saved_predictions_reproduced=True, maximum_prediction_difference=max_difference, arithmetic_recomputed=True, metrics=metrics, preserved_earlier_models=preserved, disentangle_models_preserved=True, source_inventory='source_inventory.json', exposure_registry='exposed_images.csv', references=[dict(title='Cawley & Talbot (2010), On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation', url='https://jmlr.org/papers/v11/cawley10a.html', relevance='Repeated optimization of finite-sample validation criteria can bias reported performance.'), dict(title='Dwork et al. (2015), Generalization in Adaptive Data Analysis and Holdout Reuse', url='https://proceedings.neurips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html', relevance='Analyses chosen in response to earlier holdout results can overfit the reused holdout.'), dict(title='scikit-learn, Common pitfalls and recommended practices', url='https://scikit-learn.org/stable/common_pitfalls.html', relevance='Test information must not guide model choices; preprocessing and fitting use training data only.')], limits=['341-image full-training reapplication is resubstitution, including the reported new60/60.', 'No new real capture was identified by the project inventory; a different file hash alone cannot prove independent acquisition or historical non-use.', 'Image errors are against filename command targets, not independently measured robot pose.', 'No physical correction or closed-loop return-to-center test was executed.'])
    (HERE / 'evaluation_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'direct_fold_leakage_found', 'adaptive_test_contamination', 'all_341_images_are_development_data', 'maximum_prediction_difference']}, indent=2))

if __name__ == '__main__':
    with threadpool_limits(1): main()
