import unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares as real_least_squares
from scipy.spatial.transform import Rotation
from extract import affine, fit_corner
from inventory import coordinates
from physics import predict, geometry


class AnalysisChecks(unittest.TestCase):
    def test_coordinates_require_folder_filename_agreement(self):
        self.assertEqual(coordinates(Path('Y+0.10 T+0.20/Y+0.1 T+0.2 (3).bmp')),(.1,.2,'mixed'))
        self.assertEqual(coordinates(Path('Y+0.10 T+0.20/Y+0.1 T+0.3 (3).bmp'))[2],'unresolved')
        self.assertEqual(coordinates(Path('arbitrary/center.bmp'))[2],'unresolved')

    def test_brightness_model_jacobian_against_central_difference(self):
        rng=np.random.default_rng(42)
        image=rng.integers(20,230,(40,40),dtype=np.uint8)
        def checked(fun,x0,jac,**kwargs):
            p=np.array([.17,-.12,.9,180.,240.])
            numerical=np.column_stack([(fun(p+np.eye(5)[i]*1e-5)-fun(p-np.eye(5)[i]*1e-5))/2e-5 for i in range(5)])
            np.testing.assert_allclose(jac(p),numerical,atol=2e-7,rtol=2e-6)
            return real_least_squares(fun,x0,jac=jac,**kwargs)
        with patch('extract.least_squares',checked):
            fit_corner(image,[20.2,20.1],[[0,1],[0,0]],[[1,.03],[.02,1]],4)

    def test_ecc_warp_sign_and_global_origin(self):
        rng=np.random.default_rng(7)
        blocks=rng.integers(0,2,(60,80),dtype=np.uint8)*220+20
        base=cv2.resize(blocks,(800,600),interpolation=cv2.INTER_NEAREST)
        truth=np.array([[1.0001,.001,.37],[-.001,1.0002,-.42]],np.float32)
        image=cv2.warpAffine(base,truth,(800,600))
        qr={'bounds':[290,190,204,216],'center':[392,298]}
        warp,summary=affine(image,base,qr)
        center=np.array([392.,298.])
        expected=truth[:,:2]@center+truth[:,2]-center
        np.testing.assert_allclose([summary['dx'],summary['dy']],expected,atol=.06)
        np.testing.assert_allclose(warp[:,:2],truth[:,:2],atol=.001)

    def test_plane_homography_matches_independent_3d_projection(self):
        p=np.array([.2,-.1,.3,-.003,.2,-.7]);f=4500.
        ref=np.array([[1650.,1000.],[1900.,1100.],[2160.,1150.]])
        y=np.array([.05,.5,1.]);theta=np.array([.01,.1,2.])
        n,a,pivot=geometry(p)
        direct=[]
        for uv,yi,ti in zip(ref,y,theta):
            ray=np.array([(uv[0]-1920)/f,(uv[1]-1080)/f,1.])
            X=ray/(n@ray)
            R=Rotation.from_rotvec(np.deg2rad(ti)*n).as_matrix()
            C=a*yi+(np.eye(3)-R)@pivot
            camera=R.T@(X-C)
            direct.append(f*camera[:2]/camera[2]+[1920,1080]-uv)
            np.testing.assert_allclose((np.eye(3)-R)@pivot,(np.eye(3)-R)@(pivot-2*n),atol=1e-14)
        np.testing.assert_allclose(predict(p,ref,y,theta,f),direct,atol=1e-10)


if __name__=='__main__':unittest.main()
