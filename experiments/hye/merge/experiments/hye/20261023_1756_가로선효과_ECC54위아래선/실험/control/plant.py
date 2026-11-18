"""Evaluator-owned empirical plant; NEVER supply this object to the controller."""
import numpy as np
from scipy.spatial import ConvexHull
from controller import polynomial,grouped_rows

class HeldoutPlant:
    def __init__(self,X,q,dates,indices):
        means,commands,residual=grouped_rows(X,q,dates)
        Z=polynomial(commands)
        penalty=np.diag([0.,1e-8,1e-8,1e-4,1e-4,1e-4])
        self.B=np.linalg.solve(Z.T@Z+penalty,Z.T@means)
        self.residual=residual;self.indices=np.array(indices)
        self.low=np.minimum(commands.min(0),0.);self.high=np.maximum(commands.max(0),0.)
        self.hull=ConvexHull(np.vstack([commands,np.zeros((1,2))])).equations
        error=Z@self.B-means
        self.diagnostic=dict(heldout_images=len(X),heldout_conditions=len(means),design_rank=int(np.linalg.matrix_rank(Z)),design_columns=6,
                             condition_fit_RMS=float(np.sqrt(np.mean(error**2))),intercept_feature_RMS=float(np.sqrt(np.mean(self.B[0]**2))),
                             repeat_residual_rows=len(residual),repeat_residual_RMS=float(np.sqrt(np.mean(residual**2))),
                             coverage_low=self.low.tolist(),coverage_high=self.high.tolist(),source='Heldout labels and features used only in SIMULATED PLANT')
    def value(self,q,offset):
        value=polynomial(q)[0]@self.B
        return value if offset else value-self.B[0]
    def outside_coverage(self,q):return bool(np.any(self.hull[:,:2]@q+self.hull[:,2]>1e-10))
