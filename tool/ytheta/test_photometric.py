import unittest
import numpy as np
from photometric import sample,cubic_weights
class InterpolationTests(unittest.TestCase):
    def test_weights_and_linear_ramp(self):
        t=np.linspace(0,1,21);np.testing.assert_allclose(cubic_weights(t).sum(1),1,atol=1e-14)
        y,x=np.mgrid[:40,:60];im=2*x+3*y+7
        p=np.array([[10.2,12.7],[30.8,20.1],[15,15]])
        np.testing.assert_allclose(sample(im,p),2*p[:,0]+3*p[:,1]+7,atol=1e-10)
    def test_bounds(self):
        with self.assertRaises(ValueError):sample(np.ones((10,10)),np.array([[0,0]]))
if __name__=='__main__':unittest.main()
