"""Per-line hypothesis checks. Uses acquired labels, not absolute physical truth."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'zero_equivalence'))
from check_zero_equivalence import hac_covariance

TOL = .005
VARIANTS = ['angle_deg', 'l2_angle_deg', 'trimmed_angle_deg', 'sigma2_angle_deg']


def linear(q, phi):
    q, phi = np.asarray(q), np.asarray(phi)
    X = np.column_stack([np.ones(len(q)), q])
    beta = np.linalg.lstsq(X, phi, rcond=None)[0]
    residual = phi-X@beta
    bread = np.linalg.inv(X.T@X)
    s2 = float(residual@residual/(len(q)-2))
    lag = int(np.floor(4*(len(q)/100)**(2/9)))
    hac = hac_covariance(X, residual, bread, lag)
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {'intercept_image_deg': float(beta[0]), 'slope_image_deg_per_label_unit': float(beta[1]),
            'rms_image_deg': rmse, 'max_abs_residual_image_deg': float(max(abs(residual))),
            'r2': float(1-residual@residual/np.sum((phi-phi.mean())**2)),
            'rms_equivalent_label_unit': rmse/abs(float(beta[1])),
            'hac_covariance_b_a': hac.tolist(), 'hac_lags': lag,
            'hac_slope_approx95': [float(beta[1]-1.96*np.sqrt(hac[1, 1])), float(beta[1]+1.96*np.sqrt(hac[1, 1]))],
            'hac_intercept_approx95': [float(beta[0]-1.96*np.sqrt(hac[0, 0])), float(beta[0]+1.96*np.sqrt(hac[0, 0]))],
            'phi_min_max': [float(phi.min()), float(phi.max())], 'label_min_max': [float(q.min()), float(q.max())]}


def metrics(errors):
    errors = np.asarray(errors)
    return {'n': len(errors), 'mae_deg': float(np.mean(abs(errors))),
            'rmse_deg': float(np.sqrt(np.mean(errors**2))), 'max_abs_deg': float(max(abs(errors))),
            'within_0_005_count': int(np.sum(abs(errors) <= TOL)),
            'within_0_005_fraction': float(np.mean(abs(errors) <= TOL))}


def first_cv(rows, feature, variant):
    q = np.array([r['label'] for r in rows])
    phi = np.array([r['features'][feature][variant] for r in rows])
    predicted = np.zeros(len(q))
    folds = []
    # Five consecutive 20-image blocks. Folds 1/5 also assess endpoint extrapolation.
    for indices in np.array_split(np.arange(len(q)), 5):
        mask = np.ones(len(q), bool); mask[indices] = False
        fit = linear(q[mask], phi[mask])
        predicted[indices] = (phi[indices]-fit['intercept_image_deg'])/fit['slope_image_deg_per_label_unit']
        folds.append({'test_labels': q[indices].tolist(), 'train_n': int(mask.sum()), 'test_n': len(indices)})
    return {'metrics': metrics(predicted-q), 'predicted_theta': predicted.tolist(), 'actual_label_theta': q.tolist(), 'folds': folds}


def second_cv(rows, feature, variant):
    # One complete mixed Y/theta condition (all 3 repeats) is withheld each fold.
    # Each line independently predicts theta; Y is deliberately not an input.
    predictions = []
    for condition in sorted({r['condition'] for r in rows}):
        train = [r for r in rows if r['condition'] != condition]
        test = [r for r in rows if r['condition'] == condition]
        q = np.array([r['Theta'] for r in train])
        phi = np.array([r['features'][feature][variant] for r in train])
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(q)), q]), phi, rcond=None)[0]
        for row in test:
            predicted = (row['features'][feature][variant]-beta[0])/beta[1]
            predictions.append({'file': row['file'], 'condition': condition, 'Y': row['Y'], 'Theta': row['Theta'],
                                'predicted_theta': float(predicted), 'error_deg': float(predicted-row['Theta'])})
    return {'metrics': metrics([r['error_deg'] for r in predictions]), 'predictions': predictions}


def main():
    path = HERE/'line_measurements.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = data['measurements']
    first = [r for r in rows if r['dataset'] == 'first']
    second = [r for r in rows if r['dataset'] == 'second']
    conditions = sorted({r['condition'] for r in second})
    assert len(first) == 281 and len(second) == 60 and len(conditions) == 20
    results = {}
    for feature in data['features']:
        fits, variants = {}, {}
        for group in range(2, 8):
            rr = [r for r in first if r['group'] == group]
            fits[str(group)] = linear([r['label'] for r in rr], [r['features'][feature]['angle_deg'] for r in rr])
            fits[str(group)]['axis'] = rr[0]['axis']
            fits[str(group)]['n'] = len(rr)
        for variant in VARIANTS:
            variants[variant] = {'theta_fits': {}, 'first_block_cv': {}}
            for group in [4, 5]:
                rr = [r for r in first if r['group'] == group]
                variants[variant]['theta_fits'][str(group)] = linear([r['label'] for r in rr], [r['features'][feature][variant] for r in rr])
                variants[variant]['first_block_cv'][str(group)] = first_cv(rr, feature, variant)
            mixed = [r for r in second if not r['is_reference']]
            variants[variant]['second_condition_cv'] = second_cv(mixed, feature, variant)

        # Group-specific slopes retained; no averaging first-data runs or lines.
        slopes = {str(g): fits[str(g)]['slope_image_deg_per_label_unit'] for g in [4, 5]}
        repeated = []
        for condition in conditions:
            rr = [r for r in second if r['condition'] == condition]
            phi = np.array([r['features'][feature]['angle_deg'] for r in rr])
            assert len(phi) == 3
            repeated.append({'condition': condition, 'Y': rr[0]['Y'], 'Theta': rr[0]['Theta'],
                             'angles_image_deg': phi.tolist(), 'range_image_deg': float(np.ptp(phi)),
                             'equivalent_theta_range_by_first_group': {g: float(np.ptp(phi)/abs(a)) for g, a in slopes.items()}})
        y_effect = []
        for theta in [.01, .1, 1., 2.]:
            rr = [r for r in repeated if r['Theta'] == theta]
            means = np.array([np.mean(r['angles_image_deg']) for r in rr])
            y_effect.append({'theta': theta, 'Y_values': [r['Y'] for r in rr], 'condition_mean_angles': means.tolist(),
                             'range_of_condition_means_image_deg': float(np.ptp(means)),
                             'equivalent_theta_range_by_first_group': {g: float(np.ptp(means)/abs(a)) for g, a in slopes.items()}})
        minimum_steps = []
        for low, high in [(.01, .02), (.02, .03)]:
            l = next(r for r in repeated if r['Y'] == .05 and r['Theta'] == low)
            h = next(r for r in repeated if r['Y'] == .05 and r['Theta'] == high)
            la, ha = np.array(l['angles_image_deg']), np.array(h['angles_image_deg'])
            overlap = float(min(la.max(), ha.max())-max(la.min(), ha.min()))
            minimum_steps.append({'theta_low': low, 'theta_high': high, 'Y': .05,
                                  'low_range_image_deg': [float(la.min()), float(la.max())],
                                  'high_range_image_deg': [float(ha.min()), float(ha.max())],
                                  'observed_ranges_overlap': overlap >= 0, 'signed_overlap_image_deg': overlap,
                                  'mean_angle_difference_deg': float(ha.mean()-la.mean()),
                                  'equivalent_theta_step_by_first_group': {g: float((ha.mean()-la.mean())/a) for g, a in slopes.items()}})
        zero = []
        for group in [4, 5]:
            fit = fits[str(group)]
            b, a = fit['intercept_image_deg'], fit['slope_image_deg_per_label_unit']
            cov = np.array(fit['hac_covariance_b_a'])
            qq = np.linspace(-TOL, TOL, 101)
            X = np.column_stack([np.ones(len(qq)), qq])
            u = 1.96*np.sqrt(np.einsum('ij,jk,ik->i', X, cov, X))
            lo, hi = float(np.min(b+a*qq-u)), float(np.max(b+a*qq+u))
            zero.append({'first_group': group, 'image_angle_at_nominal_theta0': b,
                         'theta_at_horizontal_linear_root': -b/a,
                         'root_within_measured_theta_range': fit['label_min_max'][0] <= -b/a <= fit['label_min_max'][1],
                         'image_angle_envelope_near_theta0_approx_hac': [lo, hi],
                         'horizontal_0_in_envelope': lo <= 0 <= hi})
        baseline = next(r for r in second if r['is_reference'] and Path(r['file']).stem == 'center1')['features'][feature]['angle_deg']
        transfer = {}
        for group, a in slopes.items():
            evaluated = [r for r in second if Path(r['file']).stem != 'center1']
            errors = [(r['features'][feature]['angle_deg']-baseline)/a-r['Theta'] for r in evaluated]
            transfer[group] = metrics(errors)
        quality = {k: [float(min(r['features'][feature][k] for r in rows)), float(max(r['features'][feature][k] for r in rows))]
                   for k in ['edge_rms_px', 'valid_fraction', 'median_gradient']}
        results[feature] = {'first_fits': fits, 'variant_checks': variants,
                            'second_repeats': repeated, 'second_fixed_theta_y_effect': y_effect,
                            'second_minimum_steps': minimum_steps, 'horizontal_target_check': zero,
                            'first_slope_second_reference_transfer': transfer, 'edge_quality_ranges': quality}

    result = {
        'scope': 'Per-line feasibility of theta-from-line-angle followed by Y correction; not a controller test',
        'images': {'first': 281, 'second': 60}, 'features_n': len(results), 'theta_tolerance_deg': TOL,
        'groups': {'2': 'Y +0.10..+2.00', '3': 'Y -0.10..-2.00', '4': 'Theta +0.01..+1.00',
                   '5': 'Theta -0.01..-1.00', '6': 'Y +0.05..+1.95', '7': 'Y -1.95..-0.05'},
        'notes': [
            'Group 4 is retained here for its within-run angle response; prior exclusion applied only to the zero-equivalence subset, not automatic deletion from this different hypothesis check.',
            'No first/second absolute-zero pooling. First-data fits are per-run. Second-data transfer uses center1 as its own angle origin.',
            'Repeated-condition means are used only to summarize Y effects. All individual images are retained in grouped CV; all 3 of a condition stay in the same test fold.',
            'Approximate HAC intervals are pointwise sensitivity summaries, not simultaneous small-sample confidence guarantees. No equivalence acceptance is based on nonsignificance.',
            'First CV is five contiguous acquisition blocks per theta run; second CV withholds one complete mixed condition, predicting theta without Y.',
            'Displayed per-line best scores are exploratory, not independent validation of a selected feature. No averaging or feature fusion was introduced.',
            'An observed 3-image range is not a population noise bound. Pixel-edge samples are spatially dependent and are not treated as independent repeated images.',
            'Horizontal roots outside the measured theta range are extrapolations and must not be commanded to a robot.',
            'Out-of-sample observed point errors already above tolerance prevent an accuracy-pass claim; adding uncertainty cannot establish that the observed violations did not occur.',
        ],
        'features': results,
        'measurement_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE/'hypothesis_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for f, r in results.items():
        print(f, 'second CV', r['variant_checks']['angle_deg']['second_condition_cv']['metrics'], flush=True)


if __name__ == '__main__':
    main()
