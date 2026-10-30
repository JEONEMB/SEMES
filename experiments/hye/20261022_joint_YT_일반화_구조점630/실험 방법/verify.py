"""Independent, read-only verification of saved grouped evaluation results.

This script writes verification.json only; it does not fit or modify models.
The source-code review assertions are documented separately from numerical checks.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import joblib
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
sys.path.insert(0, str(H))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(frame):
    error = np.abs(frame[['predY', 'predT']].to_numpy() - frame[['Y', 'T']].to_numpy())
    valid = np.isfinite(error).all(axis=1)
    e = error[valid]
    return dict(n=len(frame), failures=int((~valid).sum()),
                passed=int(np.all(e <= .09 + 1e-12, axis=1).sum()),
                Y_MAE=float(e[:, 0].mean()), Y_max=float(e[:, 0].max()),
                T_MAE=float(e[:, 1].mean()), T_max=float(e[:, 1].max()))


def main():
    m = pd.read_csv(H / 'manifest_with_folds.csv')
    p = pd.read_csv(H / 'predictions.csv')
    summary = pd.read_csv(H / 'summary.csv', dtype={'dataset': str})
    splits = json.loads((H / 'splits.json').read_text(encoding='utf-8'))
    selection = json.loads((H / 'selection.json').read_text(encoding='utf-8'))
    original = json.loads((H.parent / 'joint_yt_improved_20261022/splits.json').read_text(encoding='utf-8'))
    arrays = np.load(H / 'measurements/features.npz')
    assert len(m) == 341 and m['index'].is_unique
    assert np.array_equal(m['index'].to_numpy(), arrays['index'])
    assert np.array_equal(m['dataset'].to_numpy(), arrays['dataset'])
    assert arrays['D'].shape == (341, 630, 2) and np.isfinite(arrays['D']).all()
    assert np.allclose(m[['Y', 'T']].to_numpy(), arrays['Y'])
    assert m.groupby('cv_group').fold.nunique().eq(1).all()
    assert m.success.all()
    centers = m[(m.dataset == 20261020) & (m.scope == 'center')]
    assert len(centers) == 3 and centers.cv_group.nunique() == 1
    assert set(centers.fold) == {0}

    fold_checks = []
    model_hashes_before = {path.name: sha(path) for path in H.glob('*model.joblib')}
    for f, split in enumerate(splits):
        tr, te = set(split['train_indices']), set(split['test_indices'])
        assert not tr & te and tr | te == set(range(341))
        assert not set(m.iloc[list(tr)].cv_group) & set(m.iloc[list(te)].cv_group)
        assert set(m.loc[(m.dataset == 20261014) & (m.scope == 'single') & (m.fold == f), 'file']) == set(original[f]['test'])
        seen = []
        for inner in selection[f]['inner_splits']:
            it, iv = set(inner['train']), set(inner['validation'])
            assert not it & iv and it | iv == tr and not iv & te
            assert not set(m.iloc[list(it)].cv_group) & set(m.iloc[list(iv)].cv_group)
            seen.extend(iv)
        assert sorted(seen) == sorted(tr)
        best = min(selection[f]['scores'], key=lambda item: item['score'])['config']
        assert selection[f]['selected'] == best
        bundle = joblib.load(H / f'fold{f}_model.joblib')
        assert np.array_equal(bundle['train_indices'], split['train_indices'])
        assert np.array_equal(bundle['ids'], arrays['ids'])
        assert np.array_equal(bundle['points'], arrays['points'])
        model = bundle['model']
        assert np.allclose(model.shifts, arrays['D'][bundle['train_indices'], :, 0].mean(axis=1))
        if model.config['family'] != 'global':
            assert np.array_equal(model.y, m.iloc[bundle['train_indices']][['Y', 'T']].to_numpy())
            assert len(model.X) == len(tr)
        fold_checks.append(dict(fold=f, train=len(tr), test=len(te),
                                test_new=int((m.iloc[list(te)].dataset == 20261020).sum()),
                                test_new_centers=int(((m.iloc[list(te)].dataset == 20261020) & (m.iloc[list(te)].scope == 'center')).sum()),
                                outer_group_disjoint=True, inner_group_disjoint=True,
                                inner_validation_partitions_outer_training=True,
                                original_280_test_files_unchanged=True,
                                saved_model_training_indices_match=True,
                                selected_config_has_best_inner_score=True))

    calculated = []
    for (method, protocol), group in p.groupby(['method', 'protocol']):
        assert not group['index'].duplicated().any()
        assert np.isfinite(group[['predY', 'predT']].to_numpy()).all()
        assert np.allclose(group.eY, abs(group.predY - group.Y))
        assert np.allclose(group.eT, abs(group.predT - group['T']))
        assert np.allclose(group.correctionY, -group.predY)
        assert np.allclose(group.correctionT, -group.predT)
        assert np.array_equal(group.both_within09, (group.eY <= .09 + 1e-12) & (group.eT <= .09 + 1e-12))
        for dataset in ['all', '20261014', '20261020']:
            g = group if dataset == 'all' else group[group.dataset == int(dataset)]
            if not len(g):
                continue
            values = numbers(g)
            row = summary[(summary.method == method) & (summary.protocol == protocol) & (summary.dataset == dataset)]
            assert len(row) == 1
            for key, value in values.items():
                assert np.isclose(row.iloc[0][key], value, rtol=1e-10, atol=1e-12), (method, protocol, dataset, key)
            calculated.append(dict(method=method, protocol=protocol, dataset=dataset, **values))

    oof = p[p.protocol == 'nested_group_oof']
    assert len(oof) == 341 and oof['index'].is_unique
    assert np.array_equal(oof.fold, oof['index'].map(m.set_index('index').fold))
    main_new = oof[oof.dataset == 20261020]
    assert len(main_new) == 60 and int(main_new.both_within09.sum()) == 52
    assert int(((main_new.eY > .09) & (main_new.eT <= .09)).sum()) == 8

    old_audit = json.loads((H.parent / 'evaluate_archives_20261022/audit.json').read_text(encoding='utf-8'))
    preserved = []
    for filename, previous in old_audit['model_hashes'].items():
        current = sha(H.parent / 'joint_yt_improved_20261022' / filename)
        assert current == previous
        preserved.append(dict(file=filename, previous_sha256=previous, current_sha256=current, unchanged=True))
    assert model_hashes_before == {path.name: sha(path) for path in H.glob('*model.joblib')}

    code_files = ['data.py', 'joint_model.py', 'forward_model.py', 'evaluate.py']
    result = dict(
        status='passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        verifier='independent diagnose_y agent', models_refit=False,
        input_images=341, original_motion_images=280, new_date_images=60,
        all_measurements_finite=True, fixed_point_count=630,
        dataframe_array_order_and_targets_match=True,
        all_reported_summaries_recomputed=True,
        corrections_equal_negative_predictions=True,
        centers=dict(new_count=3, group=centers.cv_group.iloc[0], test_fold=0,
                     all_excluded_from_fold0_training_and_inner_selection=True),
        folds=fold_checks, metrics=calculated,
        preserved_original_models=preserved,
        newly_created_models_unchanged_by_verifier=model_hashes_before,
        source_files={name: sha(H / name) for name in code_files},
        code_review=dict(
            target_filename_or_trial_in_inference_features=False,
            inference_features='D and functions of D: raw coordinates and mean-dx interaction',
            target_use='training forward regression; split grouping and scoring only',
            local_weight_use='query image mean dx weights stored training examples only',
            train_dependent_fit='regression, scaling, residual covariance fitted inside training folds',
            assessment='No material leakage or metric defect found in reviewed source and saved results.'),
        limitations=[
            '20261020 training conditions are used in calibration; nested grouped OOF is not old-date-only transfer.',
            'Model families and features were developed using this corpus; this is not an untouched future-date validation.',
            'The nested primary result is52/60, not the55/60 fixed-configuration development comparison.',
            'Eight new-date OOF images fail Y tolerance; maximumY error is0.3281988459245895.',
            'New-date resubstitution is59/60 with maximumY error0.10005088055205817; it is not generalization.',
            'Original-date OOF is197/281; internal selection targets new-date performance and can reduce old-date performance.',
            'Filename command values are evaluation targets, not independently measured robot pose.',
            'Correction vectors were calculated but not executed on a robot.',
            'Preservation check covers the two original model files with prior recorded hashes.'])
    (H / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], new_date_oof=numbers(main_new), folds_verified=len(fold_checks), preserved_models=len(preserved)), indent=2))


if __name__ == '__main__':
    main()
