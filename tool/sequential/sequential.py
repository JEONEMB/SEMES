"""Theta from mean-removed shape ONLY, then Y from remaining mean motion.

Observed and modeled means always use the same accepted point identities.
Units: nominal mm, degrees, original image pixels. No K or D required.
"""
import json
import numpy as np


def centered(x):
    x = np.asarray(x, float)
    return x-x.mean(axis=0)


def theta_shape(b, c, theta):
    return centered(b)*theta+centered(c)*theta*theta


def solve_theta(shape, b, c):
    """Scalar damped Gauss-Newton, plus exact stationary candidates for validation.

    The objective is a quartic; all real stationary roots and bounds are compared,
    so a local GN minimum cannot hide the best solution of this scalar model.
    """
    shape, b, c = centered(shape), centered(b), centered(c)
    bb, cc, bc = np.sum(b*b), np.sum(c*c), np.sum(b*c)
    if bb+cc < 1e-12: raise ValueError('theta has no mean-removed shape sensitivity')
    sb, sc = np.sum(shape*b), np.sum(shape*c)
    polynomial = np.trim_zeros(np.array([2*cc, 3*bc, bb-2*sc, -sb]), 'f')
    candidates = [-1., 0., 1.]
    if len(polynomial)>1:
        candidates += [float(z.real) for z in np.roots(polynomial)
                       if abs(z.imag)<1e-8 and -1<=z.real<=1]
    for initial in [-.8, 0., .8]:
        t = initial
        for _ in range(30):
            residual = b*t+c*t*t-shape
            derivative = b+2*c*t
            delta = -np.sum(residual*derivative)/(np.sum(derivative**2)+1e-10)
            old = np.sum(residual**2)
            for step in [1., .5, .25, .125, .01]:
                trial = float(np.clip(t+step*delta,-1,1))
                loss = np.sum((b*trial+c*trial*trial-shape)**2)
                if loss<=old: break
            if loss>old or abs(trial-t)<1e-10: break
            t = trial
        candidates.append(t)
    costs = [float(np.sum((b*t+c*t*t-shape)**2)) for t in candidates]
    order = np.argsort(costs)
    best = candidates[order[0]]
    return best, float(np.sqrt(costs[order[0]]/shape.size))


def estimate(model, observed, session=None, oracle_theta=None):
    keys = sorted(set(observed)&set(model['points']))
    if len(keys)<24 or len({k.split(':')[0] for k in keys})<3:
        raise ValueError('at least 24 accepted correspondences spanning all three QR required')
    if session is not None and session not in model['groups']:
        raise ValueError('unknown calibrated session')
    q = np.array([model['points'][k]['reference'] for k in keys])
    coefficients = np.array([model['points'][k]['coefficients'] for k in keys])
    a,b,c = coefficients[:,-3],coefficients[:,-2],coefficients[:,-1]
    offsets = np.zeros_like(q) if session is None else coefficients[:,model['groups'].index(session)]
    d = np.array([observed[k] for k in keys],float)-q-offsets
    if not np.isfinite(d).all(): raise ValueError('nonfinite coordinates')
    # Critically: d.mean() never enters the theta objective.
    theta, shape_residual = solve_theta(centered(d),b,c)
    if oracle_theta is not None: theta = float(oracle_theta)  # evaluation-only diagnostic
    theta_full = b*theta+c*theta*theta
    remaining_mean = (d-theta_full).mean(axis=0)
    y_direction = a.mean(axis=0)
    if np.linalg.norm(y_direction)<1e-8: raise ValueError('unobservable Y mean sensitivity')
    y = float(remaining_mean@y_direction/(y_direction@y_direction))
    sensitivity = float(np.sqrt(np.mean(np.sum(centered(b+2*c*theta)**2,axis=1))))
    return dict(y_mm=y,theta_deg=theta,points=len(keys),shape_residual_px=shape_residual,
                theta_shape_sensitivity_px_per_deg=sensitivity,
                observed_mean_px=d.mean(axis=0).tolist(),
                theta_mean_px=theta_full.mean(axis=0).tolist(),
                remaining_y_mean_px=remaining_mean.tolist(),
                theta_at_boundary=bool(abs(theta)>=1-1e-6),
                session_offset_applied=session,absolute_machine_zero_verified=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('observation',help='JSON payload:id -> [u,v] for accepted original-pixel features');p.add_argument('--session');a=p.parse_args()
    print(json.dumps(estimate(json.load(open(a.model,encoding='utf-8')),
        json.load(open(a.observation,encoding='utf-8')),a.session),indent=2))
