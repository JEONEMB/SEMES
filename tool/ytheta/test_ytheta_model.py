"""Independent analytic projection and derivative checks; no image interpolation involved."""
import unittest
import numpy as np
from ytheta_model import physical, physical_pose_jacobian, jacobian, estimate


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.q = np.array([[u, v] for u in [1450, 1700, 1900, 2150, 2400]
                           for v in [960, 1080, 1200]], float)
        self.p = [np.log(300.), .1, -.05]
        self.model = dict(kind='physical', reference=self.q.tolist(), parameters=self.p,
                          fx=4017.857142857, fy=4285.714285714, offsets={})

    def test_zero_and_independent_projection(self):
        np.testing.assert_allclose(physical(self.q, [0, 0], self.p), self.q, atol=1e-10)
        pose = [.4, -.32]
        rays = np.c_[(self.q[:, 0]-1920)/self.model['fx'],
                      (self.q[:, 1]-1080)/self.model['fy'], np.ones(len(self.q))]
        xyz = rays*(300/(rays@np.array([.1, -.05, 1.])))[:, None]
        a = np.deg2rad(pose[1]); rotation = np.array([[np.cos(a), 0, np.sin(a)],
            [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        xyz[:, 0] += pose[0]; transformed = xyz@rotation.T
        expected = transformed[:, :2]/transformed[:, 2, None]*[self.model['fx'], self.model['fy']]+[1920,1080]
        np.testing.assert_allclose(physical(self.q, pose, self.p), expected, atol=1e-9)

    def test_jacobian(self):
        for pose in [[0, 0], [.8, -.7], [-1.4, .9]]:
            numeric = jacobian(lambda x: physical(self.q, x, self.p), pose)
            np.testing.assert_allclose(numeric, physical_pose_jacobian(self.q, pose, self.p), rtol=1e-5, atol=1e-6)

    def test_mixed_pose_recovery(self):
        for y in [-1.2, -.05, .4, 1.5]:
            for t in [-.8, -.01, .03, .7]:
                observed = physical(self.q, [y, t], self.p)
                result = estimate(self.model, observed)
                self.assertLess(abs(result['y_mm']-y), 1e-4)
                self.assertLess(abs(result['theta_deg']-t), 1e-4)

    def test_reject_bad_observations(self):
        with self.assertRaises(ValueError): estimate(self.model, np.zeros((2, 2)))
        with self.assertRaises(ValueError): estimate(self.model, self.q, 'unknown')


if __name__ == '__main__': unittest.main()
