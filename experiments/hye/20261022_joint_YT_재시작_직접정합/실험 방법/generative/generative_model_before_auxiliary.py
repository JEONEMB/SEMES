"""Empirical projective forward models, inverted by two-coordinate optimization.

No camera intrinsics, metric plane size, or physical T unit is assumed. This is
not a 3D pose recovery implementation. Commands supervise the FORWARD geometry
model in fit; predict accepts only observed coordinates or measured homographies.
"""
import time
import cv2
import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import least_squares

CONFIGS = [
    dict(family='rational', interaction=False, shrink=.1),
    dict(family='rational', interaction=True, shrink=.1),
    dict(family='lie_YT', interaction=False, shrink=.1),
    dict(family='lie_TY', interaction=False, shrink=.1),
]
SL_BASIS = np.zeros((8, 3, 3))
for i in range(8): SL_BASIS[i].flat[i] = 1
SL_BASIS[0, 2, 2] = SL_BASIS[4, 2, 2] = -1


def project(H, points):
    q = np.asarray(H) @ np.c_[points, np.ones(len(points))].T
    denominator = q[..., 2, :]
    if np.any(np.abs(denominator) < 1e-8): raise ValueError('Projective denominator near zero.')
    return (q[..., :2, :] / denominator[..., None, :]).swapaxes(-1, -2)


def algebra(v): return np.einsum('...k,kij->...ij', v, SL_BASIS)


class GenerativeProjective:
    def __init__(self, config, points, regions):
        self.config = dict(config)
        self.points = np.asarray(points, float)
        self.regions = np.asarray(regions)
        # Existing region IDs subdivide QR by quadrant; their hundreds identify QR.
        if len(np.unique(self.regions)) == 3:
            self.qr_ids = self.regions
        else:
            # Structural reference points occur in three disjoint fixed x intervals.
            boundaries = (1770., 2030.)
            self.qr_ids = np.digitize(self.points[:, 0], boundaries)
        self.qr_values = np.unique(self.qr_ids)
        if len(self.qr_values) != 3: raise ValueError('Expected three fixed reference QR regions.')
        self.normalizers, self.anchors = [], []
        for qr in self.qr_values:
            p = self.points[self.qr_ids == qr]
            center = p.mean(0)
            N = np.array([[.01, 0., -center[0]/100], [0., .01, -center[1]/100], [0., 0., 1.]])
            self.normalizers.append(N)
            local = project(N, p)
            x = np.linspace(local[:,0].min(), local[:,0].max(), 3)
            y = np.linspace(local[:,1].min(), local[:,1].max(), 3)
            xx, yy = np.meshgrid(x, y)
            self.anchors.append(np.c_[xx.ravel(), yy.ravel()])
        self.normalizers = np.asarray(self.normalizers)
        self.anchors = np.asarray(self.anchors)

    def measure_homographies(self, D):
        """All fixed correspondences contribute; method=0 performs no RANSAC trim."""
        D = np.asarray(D, float)
        D = D[None] if D.ndim == 2 else D
        result = np.full((len(D), 3, 3, 3), np.nan)
        for i, d in enumerate(D):
            if not np.isfinite(d).all(): continue
            for j, qr in enumerate(self.qr_values):
                mask = self.qr_ids == qr
                H, _ = cv2.findHomography(self.points[mask], self.points[mask]+d[mask], method=0)
                if H is not None and np.isfinite(H).all(): result[i,j] = H / H[2,2]
        return result

    def normalize_homographies(self, H):
        H = np.asarray(H, float)
        H = H[None] if H.ndim == 3 else H
        return self.normalizers[None] @ H @ np.linalg.inv(self.normalizers)[None]

    def observed_geometry(self, normalized_H):
        return np.stack([project(normalized_H[:,j], self.anchors[j]) for j in range(3)], axis=1).reshape(len(normalized_H), -1) * 100

    def _design(self, y):
        y = np.asarray(y).reshape(-1, 2)
        Z = np.c_[np.ones(len(y)), y]
        return np.c_[Z, y[:,0]*y[:,1]] if self.config['interaction'] else Z

    def _matrices_one_qr(self, coef, y):
        y = np.asarray(y).reshape(-1, 2)
        if self.config['family'] == 'rational':
            raw = self._design(y) @ coef.reshape(-1,8)
            H = np.zeros((len(y),3,3))
            H.reshape(len(y),9)[:,:8] = raw
            H[:,2,2] = 1
            return H
        O, A, B = algebra(coef.reshape(3,8))
        h0 = expm(O)
        ey = expm(y[:,0,None,None]*A)
        et = expm(y[:,1,None,None]*B)
        return h0 @ ey @ et if self.config['family'] == 'lie_YT' else h0 @ et @ ey

    def generate_homographies(self, y, global_pixels=False):
        H = np.stack([self._matrices_one_qr(coef, y) for coef in self.coefs], axis=1)
        if global_pixels: H = np.linalg.inv(self.normalizers)[None] @ H @ self.normalizers[None]
        return H

    def generate_geometry(self, y): return self.observed_geometry(self.generate_homographies(y))

    def fit(self, D, y, metadata):
        return self.fit_homographies(self.measure_homographies(D), y, metadata)

    def fit_homographies(self, H, y, metadata):
        """H maps original center GLOBAL PIXELS to current GLOBAL PIXELS."""
        start = time.monotonic()
        H = self.normalize_homographies(H)
        y = np.asarray(y, float)
        usable = np.isfinite(H).all(axis=(1,2,3)) & np.isfinite(y).all(1)
        self.training_indices = metadata['index'].to_numpy().copy()
        self.usable_training_indices = metadata.loc[usable, 'index'].to_numpy().copy()
        H, y = H[usable], y[usable]
        dates = metadata.loc[usable, 'dataset'].to_numpy()
        observation = self.observed_geometry(H)
        keys = [(int(date), float(v[0]), float(v[1])) for date,v in zip(dates,y)]
        names = sorted(set(keys))
        gi = np.array([names.index(k) for k in keys])
        gy = np.array([y[gi==j].mean(0) for j in range(len(names))])
        gH = np.array([H[gi==j].mean(0) for j in range(len(names))])
        gF = np.array([observation[gi==j].mean(0) for j in range(len(names))])
        gd = np.array([k[0] for k in names])
        # Each date has equal total weight, each command condition equal within date.
        weights = np.array([1/np.sum(gd==d) for d in gd])
        weights /= weights.sum()
        sw = np.sqrt(weights)[:,None,None]
        self.coefs, calibration = [], []
        for qr in range(3):
            matrices = gH[:,qr] / gH[:,qr,2:3,2:3]
            if self.config['family'] == 'rational':
                design = self._design(gy)
                target = matrices.reshape(len(matrices),9)[:,:8]
            else:
                design = self._design(gy)
                logs = []
                for mat in matrices:
                    det = np.linalg.det(mat)
                    if det <= 0: raise ValueError('Observed homography is outside the local SL(3) chart.')
                    L = logm(mat / np.cbrt(det))
                    if np.max(np.abs(L.imag)) > 1e-7: raise ValueError('Homography logarithm is not real.')
                    logs.append(L.real)
                target = np.array(logs).reshape(len(matrices),9)[:,:8]
            initial = np.linalg.lstsq(design*np.sqrt(weights)[:,None], target*np.sqrt(weights)[:,None], rcond=None)[0].ravel()
            goal = gF.reshape(len(gF),3,9,2)[:,qr] / 100
            def residual(params):
                predicted = project(self._matrices_one_qr(params, gy), self.anchors[qr])
                return ((predicted-goal)*sw).ravel()
            # Forward calibration, with numerical derivative of a small 24/32 parameter map.
            result = least_squares(residual, initial, method='trf', max_nfev=30, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            self.coefs.append(result.x)
            calibration.append(dict(qr=int(qr), cost=float(result.cost), success=bool(result.success), nfev=int(result.nfev), optimality=float(result.optimality)))
        generated = self.generate_geometry(y)
        residual = observation-generated
        repeat = observation-np.array([gF[g] for g in gi])
        counts = np.bincount(gi)
        row_weight = weights[gi]/counts[gi]
        covariance = .5*((residual*row_weight[:,None]).T@residual + (repeat*row_weight[:,None]).T@repeat)
        std = np.maximum(np.sqrt(np.diag(covariance)), .01)
        normalized = covariance / std[:,None] / std[None,:]
        shrink = float(self.config['shrink'])
        covariance = ((1-shrink)*normalized + shrink*np.eye(len(std))) * std[:,None]*std[None,:]
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        self.whitener = (eigenvector/np.sqrt(np.maximum(eigenvalue,1e-10))) @ eigenvector.T
        self.origin_geometry = self.generate_geometry([[0.,0.]])[0]
        step = 1e-3
        derivative = np.array([(self.generate_geometry([direction*step])[0]-self.generate_geometry([-direction*step])[0])/(2*step) for direction in np.eye(2)]).T
        self.initial_inverse = np.linalg.pinv(self.whitener@derivative)
        span = np.maximum(np.ptp(y,axis=0),1.)
        self.bounds = (y.min(0)-.5*span, y.max(0)+.5*span)
        self.audit = dict(training_rows=len(metadata), usable_training_rows=len(y), failed_training_measurements=int((~usable).sum()), training_conditions=len(names), calibrations=calibration, fit_seconds=time.monotonic()-start, forward_training_RMSE_px=float(np.sqrt(np.mean(residual**2))), parameter_count=int(sum(len(c) for c in self.coefs)), observation='3 per-QR homographies summarized at a fixed3x3 anchor grid each; all630 input points contribute to measurement when coordinate observation is used', inversion='Two continuous coordinates minimize whitened forward geometry residual; no target lookup, rounding, axis forcing or test metadata.', interpretation='Empirical projective response in command units; not metric 3D or calibrated physical rotation.', origin_jacobian_singular_values=np.linalg.svd(self.whitener@derivative,compute_uv=False).tolist())
        return self

    def predict_homographies(self, H):
        H = self.normalize_homographies(H)
        predictions = np.full((len(H),2), np.nan)
        self.prediction_diagnostics = []
        for i, mat in enumerate(H):
            try:
                if not np.isfinite(mat).all(): raise ValueError('Observed homography failed.')
                geometry = self.observed_geometry(mat[None])[0]
                def residual(q): return self.whitener@(self.generate_geometry([q])[0]-geometry)
                initial = self.initial_inverse @ (self.whitener@(geometry-self.origin_geometry))
                lower, upper = self.bounds
                initial = np.clip(initial,lower+1e-5,upper-1e-5)
                solutions = [least_squares(residual,start,bounds=self.bounds,max_nfev=50,ftol=1e-9,xtol=1e-9,gtol=1e-8) for start in [initial,np.zeros(2)]]
                best = min(solutions,key=lambda r:r.cost)
                on_boundary = np.any(best.x-lower<1e-6) or np.any(upper-best.x<1e-6)
                if not best.success or not np.isfinite(best.x).all() or on_boundary: raise ValueError('Inverse optimization failed or reached broad numerical bounds.')
                predictions[i] = best.x
                self.prediction_diagnostics.append(dict(index=i,success=True,cost=float(best.cost),nfev=int(best.nfev)))
            except Exception as exc: self.prediction_diagnostics.append(dict(index=i,success=False,error=str(exc)))
        return predictions

    def predict(self, D): return self.predict_homographies(self.measure_homographies(D))
