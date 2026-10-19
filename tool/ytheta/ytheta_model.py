"""Joint Y/yaw estimation. Units: nominal mm, degrees, original image pixels.

Only numpy is required. A successful numerical fit is not a metrology certificate.
Virtual grid samples from a common ECC warp are correlated, not independent points.
"""
import json
import numpy as np


def jacobian(fun, x, step=1e-5):
    x = np.asarray(x, float)
    columns = []
    for j in range(len(x)):
        h = step * max(1., abs(x[j]))
        d = np.zeros_like(x); d[j] = h
        columns.append((np.asarray(fun(x+d)).ravel()-np.asarray(fun(x-d)).ravel())/(2*h))
    return np.column_stack(columns)


def gauss_newton(fun, initial, lower=None, upper=None, iterations=60):
    """Damped GN with scaled columns and backtracking; never inverts J.T@J."""
    x = np.asarray(initial, float).copy()
    low = np.full_like(x, -np.inf) if lower is None else np.asarray(lower, float)
    high = np.full_like(x, np.inf) if upper is None else np.asarray(upper, float)
    x = np.clip(x, low, high)
    damping = 1e-4; converged = False
    r = np.asarray(fun(x)).ravel()
    if not np.isfinite(r).all(): raise ValueError('nonfinite initial residual')
    for k in range(iterations):
        J = jacobian(fun, x)
        scale = np.maximum(np.linalg.norm(J, axis=0), 1e-10)
        A = J/scale
        delta = np.linalg.lstsq(np.vstack([A, np.sqrt(damping)*np.eye(len(x))]),
                                np.r_[-r, np.zeros(len(x))], rcond=None)[0]/scale
        accepted = False
        for alpha in [1., .5, .25, .125, .0625, .01]:
            trial = np.clip(x+alpha*delta, low, high)
            rr = np.asarray(fun(trial)).ravel()
            if np.isfinite(rr).all() and rr@rr <= r@r:
                change = np.linalg.norm(trial-x)
                improvement = r@r-rr@rr
                x, r = trial, rr; accepted = True
                damping = max(1e-9, damping/3)
                if change < 1e-8 or improvement < 1e-12*max(1., r@r): converged = True
                break
        if converged: break
        if not accepted: damping *= 10
        if damping > 1e10: break
    return {'x': x, 'rms': float(np.sqrt(np.mean(r*r))),
            'converged': converged, 'iterations': k+1}


def physical(points, pose, parameters, fx=4017.857142857, fy=4285.714285714):
    """Fixed wall; camera translates parallel to reference image x and yaws.

    Positive nominal motion signs follow this dataset's observed image motion.
    Camera/pivot co-location is a baseline hypothesis; optional lever x/z is fitted.
    parameters=(log depth, plane slope x, plane slope y[, lever x, lever z]). K principal point fixed.
    D is omitted in this baseline; residuals must justify any extension.
    """
    p = np.asarray(points, float); pose = np.asarray(pose, float)
    rx = (p[:, 0]-1920.)/fx; ry = (p[:, 1]-1080.)/fy
    denominator = 1.+parameters[1]*rx+parameters[2]*ry
    z = np.exp(parameters[0])/denominator
    X = rx*z + pose[..., 0, None]; Y = ry*z
    angle = np.deg2rad(pose[..., 1, None]); c = np.cos(angle); s = np.sin(angle)
    lx, lz = parameters[3:5] if len(parameters)>=5 else (0.,0.)
    xx = c*X+s*z+(c-1)*lx+s*lz; zz = -s*X+c*z-s*lx+(c-1)*lz
    if np.any(zz <= 0) or np.any(denominator <= 0):
        raise ValueError('points behind camera or invalid plane')
    return np.stack([fx*xx/zz+1920., fy*Y/zz+1080.], axis=-1)


def physical_pose_jacobian(points, pose, parameters, fx=4017.857142857, fy=4285.714285714):
    p = np.asarray(points, float)
    rx = (p[:, 0]-1920.)/fx; ry = (p[:, 1]-1080.)/fy
    z = np.exp(parameters[0])/(1.+parameters[1]*rx+parameters[2]*ry)
    X = rx*z+pose[0]; Y = ry*z
    t = np.deg2rad(pose[1]); c = np.cos(t); s = np.sin(t)
    lx, lz = parameters[3:5] if len(parameters)>=5 else (0.,0.)
    xx = c*X+s*z+(c-1)*lx+s*lz; zz = -s*X+c*z-s*lx+(c-1)*lz
    dy = np.stack([fx*(c*zz+s*xx)/zz**2, fy*Y*s/zz**2], axis=-1)
    dt = np.stack([fx*((zz+lz)*zz+xx*(xx+lx))/zz**2, fy*Y*(xx+lx)/zz**2], axis=-1)*np.pi/180
    return np.stack([dy, dt], axis=-1).reshape(-1, 2)


def predict(model, pose):
    q = np.asarray(model['reference'], float)
    if model['kind'] == 'physical':
        return physical(q, pose, model['parameters'], model['fx'], model['fy'])
    y, t = np.asarray(pose, float)
    return q+np.einsum('k,kij->ij', [y, t, y*y, t*t], np.asarray(model['coefficients']))


def estimate(model, observed, session=None):
    observed = np.asarray(observed, float)
    q = np.asarray(model['reference'], float)
    if observed.shape != q.shape or not np.isfinite(observed).all():
        raise ValueError('observation must match the saved fixed reference grid')
    offset = np.zeros(2)
    if session is not None:
        if session not in model['offsets']: raise ValueError('unknown calibrated session')
        offset = np.asarray(model['offsets'][session])
    target = observed-offset
    fun = lambda pose: (predict(model, pose)-target).ravel()
    solutions = [gauss_newton(fun, [0, t], [-2., -1.], [2., 1.]) for t in [-.8, 0., .8]]
    best = min(solutions, key=lambda r: r['rms'])
    J = jacobian(lambda x: predict(model, x), best['x'])
    # Target-scaled singular values; units are pixels per allowed-error unit.
    singular = np.linalg.svd(J*np.array([.09, .09]), compute_uv=False)
    lengths = np.linalg.norm(J, axis=0)
    cosine = float(J[:, 0]@J[:, 1]/max(np.prod(lengths), 1e-30))
    return {'y_mm': float(best['x'][0]), 'theta_deg': float(best['x'][1]),
            'residual_px': best['rms'], 'converged': best['converged'],
            'at_boundary': bool(np.any(np.isclose(np.abs(best['x']), [2., 1.], atol=1e-5))),
            'target_scaled_singular_values': singular.tolist(), 'column_cosine': cosine,
            'condition': float(singular[0]/max(singular[-1], 1e-30)),
            'session_offset_applied': session,
            'absolute_machine_zero_verified': False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model'); parser.add_argument('observation', help='JSON Nx2 original pixel coordinates')
    parser.add_argument('--session', help='optional calibrated session; omitted means model reference zero')
    args = parser.parse_args()
    print(json.dumps(estimate(json.load(open(args.model, encoding='utf-8')),
                              json.load(open(args.observation, encoding='utf-8')), args.session), indent=2))
