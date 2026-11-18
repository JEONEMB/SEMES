"""Train on all three sessions and evaluate position blocks and whole sessions."""
import csv
import json
from pathlib import Path

import numpy as np

from train import export_model, features, fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation'
DATASETS = ('rig_1st', 'rig_grid', 'rig_grid_rev')


def scores(error):
    absolute = np.abs(error)
    return {
        'count': len(error),
        'rmse': np.sqrt(np.mean(error**2, axis=0)).tolist(),
        'bias': error.mean(axis=0).tolist(),
        'p95_absolute': np.quantile(absolute, .95, axis=0).tolist(),
        'theta_pass': int((absolute[:, 1] <= .01).sum()),
        'joint_pass_y_0045_theta_001': int((absolute <= [.045, .01]).all(axis=1).sum()),
        'joint_pass_y_009_theta_001': int((absolute <= [.09, .01]).all(axis=1).sum()),
    }


def main():
    captures = []
    for dataset in DATASETS:
        with np.load(OUT/(dataset+'.npz')) as d:
            assert d['names'][0] == 'center'
            captures.append(dict(d))
    assert [len(d['names'])-1 for d in captures] == [280, 1680, 1680]
    labels = np.concatenate([d['labels'][1:] for d in captures])
    inputs = np.concatenate([features(d['shifts'][1:, :, :2]-d['shifts'][0, :, :2])
                             for d in captures])
    sessions = np.repeat(DATASETS, [len(d['names'])-1 for d in captures])
    names = np.concatenate([d['names'][1:] for d in captures])
    folds = {
        'theta_blocks': np.clip(np.floor((labels[:, 1]+1+1e-8)/.4), 0, 4).astype(int),
        'y_blocks': np.clip(np.floor((labels[:, 0]+2+1e-8)/.8), 0, 4).astype(int),
        'leave_one_session_out': sessions,
        'leave_one_day_out': np.where(sessions == 'rig_grid_rev', '11-13', '11-12'),
    }
    summary = {'training_samples': dict(zip(DATASETS, [280, 1680, 1680])), 'splits': {}}
    output = OUT/'pooled-qr-validation.csv'
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split','fold','capture','file','y_mm','theta_deg',
                         'estimated_y_mm','estimated_theta_deg','error_y_mm','error_theta_deg'])
        for split, groups in folds.items():
            prediction = np.empty_like(labels)
            by_fold = {}
            for group in np.unique(groups):
                test = groups == group
                prediction[test] = inputs[test] @ fit(inputs[~test], labels[~test])
                by_fold[str(group)] = scores(prediction[test]-labels[test])
            error = prediction-labels
            summary['splits'][split] = {
                'all': scores(error), 'by_fold': by_fold,
                'by_capture': {name: scores(error[sessions == name]) for name in DATASETS},
            }
            for i in range(len(labels)):
                writer.writerow([split, groups[i], sessions[i], names[i], *labels[i],
                                 *prediction[i], *error[i]])
    # Compare training sets on identical reverse-grid test rows and exclusions.
    reverse = sessions == 'rig_grid_rev'
    groups = folds['theta_blocks']
    ablations = {}
    for name, available in {
        'reverse_only': reverse,
        'both_grids': sessions != 'rig_1st',
        'all_three': np.ones(len(labels), dtype=bool),
    }.items():
        predicted = np.empty_like(labels)
        for group in np.unique(groups):
            test = reverse & (groups == group)
            train = available & (groups != group)
            predicted[test] = inputs[test] @ fit(inputs[train], labels[train])
        ablations[name] = scores(predicted[reverse]-labels[reverse])
    summary['same_reverse_theta_holdout'] = ablations
    coefficients = fit(inputs, labels)
    desktop = Path.home()/'Desktop'
    for name, capture in zip(DATASETS, captures):
        path = ROOT/'config'/('qr_pose.yml' if name == 'rig_grid_rev' else f'qr_pose_{name}.yml')
        export_model(path, desktop/(name+'.zip'), capture['points'],
                     capture['shifts'][0, :, :2], coefficients)
    (OUT/'pooled-qr-metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for split, result in summary['splits'].items():
        print(split, result['all'])
        for name, s in result['by_capture'].items():
            print(' ', name, s['rmse'], s['joint_pass_y_0045_theta_001'], s['count'])
    for name, s in ablations.items():
        print('ablation', name, s['rmse'], s['joint_pass_y_0045_theta_001'], s['count'])


if __name__ == '__main__':
    main()
