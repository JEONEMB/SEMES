"""Projection, map reuse, failure handling, and reference-version regression checks."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from demo_web.camera import Camera
from demo_web.rectification import CALIBRATION_CROP, Rectifier, reference_signature


class RectificationTests(unittest.TestCase):
    def test_full_sensor_intrinsics_and_scaled_crop(self):
        rectifier = Rectifier()
        np.testing.assert_array_equal(rectifier.camera_matrix((4056,3040), CALIBRATION_CROP), rectifier.matrix)
        scaled = rectifier.camera_matrix((2028,1520), CALIBRATION_CROP)
        np.testing.assert_allclose(scaled[:2], rectifier.matrix[:2]/2)
        cropped = rectifier.camera_matrix((960,720), (10,16,4053,3040))
        self.assertAlmostEqual(cropped[0,2], (rectifier.matrix[0,2]-2)*960/4053)
        for crop in ((0,0,4056,3040),(8,16,0,3040),(8,16,5000,3040)):
            with self.assertRaises(ValueError):
                rectifier.camera_matrix((960,720),crop)

    def test_maps_invert_distortion_and_preserve_color_shape(self):
        rectifier = Rectifier()
        frame = np.zeros((720,960,3),dtype=np.uint8)
        frame[:,:,0] = 25
        frame[:,:,1] = 100
        frame[:,:,2] = 200
        crop = (10,16,4053,3040)
        corrected = rectifier.apply(frame,crop)
        self.assertEqual(corrected.shape,frame.shape)
        np.testing.assert_array_equal(corrected[360,480], [25,100,200])
        mx,my = cv2.convertMaps(*rectifier.maps,cv2.CV_32FC1)
        # Sample corrected grid points -> distorted source -> independent inverse.
        points = np.array([(x,y) for x in (150,480,800) for y in (150,360,600)],dtype=np.float32)
        sources = np.array([(mx[int(y),int(x)],my[int(y),int(x)]) for x,y in points],dtype=np.float32)
        k = rectifier.camera_matrix((960,720),crop)
        recovered = cv2.undistortPointsIter(sources[:,None,:],k,rectifier.distortion,None,k,
                                           (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-9))[:,0]
        np.testing.assert_allclose(recovered,points,atol=.05)
        maps = rectifier.maps
        rectifier.apply(frame,crop)
        self.assertIs(rectifier.maps,maps)
        rectifier.apply(frame,(8,16,4056,3040))
        self.assertIsNot(rectifier.maps,maps)

    def test_bad_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.yml'
            path.write_text('%YAML:1.0\n---\nimage_width: 960\nimage_height: 720\n')
            with self.assertRaises(ValueError):
                Rectifier(path)

    def test_camera_callback_fails_closed_and_can_be_reconfigured(self):
        camera=Camera(True)
        camera.write(b'old-jpeg')
        request=SimpleNamespace(get_metadata=lambda: {})
        fake_picamera=SimpleNamespace(MappedArray=object)
        with patch.dict('sys.modules',{'picamera2':fake_picamera}), self.assertLogs(level='ERROR'):
            camera._rectify_request(request)
        camera.write(b'uncorrected-jpeg')
        self.assertFalse(camera.health()['connected'])
        self.assertIsNone(camera.frame)
        self.assertIn('보정 실패',camera.health()['message'])
        camera.configure('full',False,False)
        self.assertFalse(camera.processing_failed)
        self.assertFalse(camera.health()['undistort'])
        camera.stop()

    def test_callback_corrects_before_encoding(self):
        camera=Camera(True)
        camera.rectifier=Rectifier()
        array=np.full((72,96,3),100,dtype=np.uint8)
        class Mapped:
            def __init__(self,*_): self.array=array
            def __enter__(self): return self
            def __exit__(self,*_): pass
        request=SimpleNamespace(get_metadata=lambda: {'ScalerCrop':CALIBRATION_CROP})
        with patch.dict('sys.modules',{'picamera2':SimpleNamespace(MappedArray=Mapped)}):
            camera._rectify_request(request)
        self.assertFalse(camera.processing_failed)
        self.assertEqual(camera.rectifier.key,((96,72),CALIBRATION_CROP))
        camera.stop()

    def test_signature_tracks_calibration_only_when_enabled(self):
        before=reference_signature('full',True)
        disabled=reference_signature('full',False)
        with patch('demo_web.rectification.CALIBRATION_PATH') as path:
            path.read_bytes.return_value=b'updated calibration'
            self.assertNotEqual(before,reference_signature('full',True))
            self.assertEqual(disabled,reference_signature('full',False))
        self.assertNotEqual(before,reference_signature('medium',True))


if __name__ == '__main__':
    unittest.main()
