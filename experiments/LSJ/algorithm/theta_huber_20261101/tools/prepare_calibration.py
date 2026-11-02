"""Freeze the user-selected split and fit ONLY second-data repetitions 1/2.

This fits calibration coefficients, not a held-out performance evaluation.
Uses already validated point/line measurements without re-selecting/filtering points.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
SOURCE = WORKSPACE / 'analysis/huber_axis_sensitivity_20261101'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def repeat(entry):
    if entry['repeat'] is not None:
        return entry['repeat']
    if entry['dataset'] == 2 and entry['is_reference']:
        match = re.fullmatch(r'center([123])\.bmp', Path(entry['file']).name)
        if match:
            return int(match[1])
    raise ValueError(f"cannot determine repetition: {entry['id']}")


def main():
    manifest = read(SOURCE/'manifest.json')
    settings = read(SOURCE/'settings.json')
    train = [r for r in manifest if r['dataset'] == 2 and repeat(r) in (1, 2)]
    test = [r for r in manifest if r['dataset'] == 3 or (r['dataset'] == 2 and repeat(r) == 3)]
    unused = [r for r in manifest if r['dataset'] == 1]
    assert (len(train), len(test), len(unused)) == (40, 254, 281)
    assert {r['id'] for r in train}.isdisjoint(r['id'] for r in test)
    assert {r['expected_sha256'] for r in train}.isdisjoint(r['expected_sha256'] for r in test)
    split = {
        'decision': 'User: second-data repetition 3 and all third data held out.',
        'reference_rule': 'center1/center2 = calibration; center3 = held out.',
        'calibration': train, 'held_out': test, 'not_used_for_calibration': unused,
        'counts': {'calibration': 40, 'held_out_second': 20, 'held_out_third': 234, 'unused_first': 281},
        'evaluation_status': 'Not executed by this preparation script.',
        'prior_exposure': 'All datasets were previously explored. This is a fixed internal holdout for the new model, not a never-seen independent validation set.'
    }
    train_ids = {r['id'] for r in train}
    # Only these selected angle measurements enter the model fit.
    measured = {(r['id'], r['target']): r for r in read(SOURCE/'measurements.json') if r['id'] in train_ids}
    design = np.array([[1., r['y_mm'], r['theta_deg']] for r in train])
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError('calibration design is not full rank')
    model = {'type': 'affine_forward_two_angles', 'target_theta_deg': 0.0}
    selected_rows = []
    for short in ('upper', 'lower'):
        target = short + '_structure'
        rows = [measured[r['id'], target] for r in train]
        assert all(r['finite'] and r['tol_met'] for r in rows)
        angles = np.array([r['image_angle_deg'] for r in rows])
        assert np.all(np.isfinite(angles))
        beta = np.linalg.lstsq(design, angles, rcond=None)[0]
        model[short] = dict(zip(('intercept_deg', 'per_y_mm', 'per_theta_deg'), beta.tolist()))
        for image, measurement in zip(train, rows):
            assert image['expected_sha256'] == measurement['sha256']
            selected_rows.append({'image_id': image['id'], 'target': target,
                                  'angle_deg': measurement['image_angle_deg'],
                                  'y_mm': image['y_mm'], 'theta_deg': image['theta_deg']})
    sensitivity = np.array([[model[s]['per_y_mm'], model[s]['per_theta_deg']] for s in ('upper', 'lower')])
    det = float(np.linalg.det(sensitivity))
    if det == 0 or not np.isfinite(det):
        raise ValueError('singular sensitivity matrix')
    config = {
        'schema_version': 1, 'calibration_id': 'D2_repeat12_affine_20261101',
        'image': {'width': 3840, 'height': 2160},
        'huber': {'t': settings['huber']['t'], 'max_iterations': settings['huber']['maxiter'],
                  'tolerance': settings['huber']['tol']},
        'model': model,
    }
    for short in ('upper', 'lower'):
        r = next(t for t in settings['targets'] if t['id'] == short+'_structure')
        config[short+'_region'] = {'x_first': r['x_range'][0], 'x_last': r['x_range'][1],
            'y_first': r['band_y'][0], 'y_last': r['band_y'][1], 'excluded_x': r['exclude_x']}
    provenance = {
        'model': 'Two OLS forward equations: phi = b + aY*Y + aTheta*theta; invert the 2x2 sensitivity matrix.',
        'new_assumption': 'One additive affine response across the second-data calibration range. No claim of accuracy before held-out testing.',
        'target': 'Nominal second-data theta=0. The intercept is an OLS estimate, not a measured mechanical truth.',
        'labels': 'Nominal corrected relative movement labels, not absolute encoder positions.',
        'calibration_images': [r['id'] for r in train],
        'sources': {str(p.relative_to(WORKSPACE)): digest(p) for p in
                    (SOURCE/'manifest.json', SOURCE/'measurements.json', SOURCE/'settings.json', Path(__file__))},
        'sensitivity_determinant': det,
        'condition_number_in_mm_degree_coordinates': float(np.linalg.cond(sensitivity)),
        'condition_number_note': 'Depends on the chosen mm/degree units; diagnostic only, no acceptance cutoff.',
        'theta_deg_per_angle_deg': np.linalg.inv(sensitivity)[1].tolist(),
        'data_range': {'y_mm': [float(design[:,1].min()), float(design[:,1].max())],
                       'theta_deg': [float(design[:,2].min()), float(design[:,2].max())]},
        'held_out_evaluation': False,
        'gaussian': False, 'extra_point_rejection': False, 'coefficient_regularization': False,
    }
    write(HERE/'config/calibration.json', config)
    write(HERE/'calibration/data_split.json', split)
    write(HERE/'calibration/provenance.json', provenance)
    write(HERE/'calibration/training_angles.json', selected_rows)
    print(json.dumps({'counts':split['counts'], 'model':model, 'determinant':det}, indent=2))


if __name__ == '__main__':
    main()
