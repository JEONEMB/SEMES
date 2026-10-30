"""Condition-balanced robust forward calibration; predict accepts only coordinates.

Training metadata defines condition weights and domains. No metadata or target is
consulted at prediction. Bags use positive exponential condition weights, so all
fixed structural points and all training conditions remain represented.
"""
import numpy as np


def feature_map(D):
    d = np.asarray(D, dtype=float)
    d = d[None] if d.ndim == 2 else d
    raw = d.reshape(len(d), -1)
    h = d[:, :, 0].mean(axis=1, keepdims=True) / 100
    return np.c_[raw, raw * h]


def gls_decoder(slope, residual, weights, shrink):
    weights = np.asarray(weights) / np.sum(weights)
    R = residual * np.sqrt(weights[:, None])
    std = np.maximum(np.sqrt(np.sum(R * R, axis=0)), 1e-6)
    Z = R / std
    V = (slope / std).T
    a = float(shrink)
    b = 1 - a
    Q = (V - b * Z.T @ np.linalg.solve(a * np.eye(len(Z)) + b * Z @ Z.T, Z @ V)) / a
    return Q @ np.linalg.pinv((slope / std) @ Q) / std[:, None]


class RobustBaggedForward:
    def __init__(self, config):
        self.config = dict(config)

    def fit(self, D, y, metadata):
        X = feature_map(D)
        y = np.asarray(y, dtype=float)
        dates = metadata.dataset.to_numpy()
        # Train-only command grouping does not become an inference lookup.
        keys = [(int(d), float(q[0]), float(q[1])) for d, q in zip(dates, y)]
        names = sorted(set(keys))
        gi = np.array([names.index(k) for k in keys])
        ng = len(names)
        gx = np.array([X[gi == j].mean(axis=0) for j in range(ng)])
        gy = np.array([y[gi == j].mean(axis=0) for j in range(ng)])
        gd = np.array([k[0] for k in names])
        counts = np.bincount(gi)
        A = np.c_[np.ones(ng), gy]
        AX = np.c_[np.ones(len(y)), y]
        selected = np.ones(ng, dtype=bool)
        if self.config['fit_mode'] == 'new_condition_means':
            selected = gd == 20261020
        if selected.sum() < 3 or np.linalg.matrix_rank(A[selected]) < 3:
            raise ValueError('Training conditions do not identify affine Y/T forward response')
        domain_weight = np.array([1 / np.sum(gd == d) for d in gd])
        rng = np.random.default_rng(self.config.get('seed', 20261022))
        betas, biases, audit = [], [], []
        for bag in range(int(self.config.get('bags', 1))):
            bootstrap = np.ones(ng) if int(self.config.get('bags', 1)) == 1 else np.maximum(rng.exponential(size=ng), 0.05)
            group_base = domain_weight * bootstrap
            fw = group_base * selected
            robust = np.ones(ng)
            huber = self.config.get('huber')
            iterations = 1 if huber is None else 5
            for _ in range(iterations):
                w = fw * robust
                sw = np.sqrt(w)
                coef = np.linalg.lstsq(A * sw[:, None], gx * sw[:, None], rcond=None)[0]
                rr = gx - A @ coef
                scale = np.maximum(np.sqrt(np.sum(w[:, None] * rr * rr, axis=0) / np.sum(w)), 1e-6)
                distance = np.sqrt(np.mean((rr / scale) ** 2, axis=1))
                cutoff = float(huber or 1) * max(float(np.median(distance[selected])), 1e-9)
                robust = np.maximum(np.minimum(1, cutoff / np.maximum(distance, 1e-9)), 0.05)
            # Equal total weight per date, equal condition weight within date;
            # repeats split their condition weight instead of dominating by count.
            rw = group_base[gi] / counts[gi]
            residual = X - AX @ coef
            if huber is not None:
                scale = np.maximum(np.sqrt(np.sum(rw[:, None] * residual**2, axis=0) / np.sum(rw)), 1e-6)
                dist = np.sqrt(np.mean((residual / scale)**2, axis=1))
                cutoff = float(huber) * max(float(np.median(dist)), 1e-9)
                rw *= np.maximum(np.minimum(1, cutoff / np.maximum(dist, 1e-9)), 0.05)
            beta = gls_decoder(coef[1:], residual, rw, self.config['shrink'])
            betas.append(beta)
            biases.append(-coef[0] @ beta)
            audit.append(dict(min_group_weight=float((fw * robust)[selected].min()),
                              covariance_effective_n=float(rw.sum()**2 / (rw @ rw))))
        self.beta = np.mean(betas, axis=0)
        self.bias = np.mean(biases, axis=0)
        self.audit = dict(training_rows=len(y), training_conditions=ng,
                          forward_conditions=int(selected.sum()), bags=audit,
                          input_features=X.shape[1], points=np.asarray(D).shape[1])
        return self

    def predict(self, D):
        return feature_map(D) @ self.beta + self.bias


REPRESENTATIVE_CONFIGS = [
    dict(fit_mode='new_condition_means', shrink=.1, huber=h, bags=b, seed=20261022)
    for h, b in [(None, 1), (1.5, 1), (None, 12), (1.5, 12)]
]

CONFIGS = [
    dict(fit_mode=f, shrink=s, huber=h, bags=b, seed=20261022)
    for f in ['new_condition_means', 'all_condition_means']
    for s in [.01, .1]
    for h, b in [(None, 1), (1.5, 1), (None, 12), (1.5, 12)]
]
