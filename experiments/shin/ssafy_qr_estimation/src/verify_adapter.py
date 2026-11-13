"""Algorithm object for verify_web; the runner owns all stage movement."""
from time import perf_counter

from .qr_observer import qr_observer
from .sequential import THETA_PX_PER_DEG


class QRSequentialAlgorithm:
    kind = 'qr-sequential'
    output = 'offset'

    def __init__(self):
        self._measure = None
        self._zero = None

    def set_reference(self, image):
        measure = qr_observer(image)
        self._zero = measure(image)
        self._measure = measure
        return dict(y=0.0, t=0.0, rois=[])

    def infer(self, image, truth=None):
        if self._measure is None:
            raise RuntimeError('Register a center image before QR measurement')
        if truth is None or abs(truth[0]) > .005:
            raise RuntimeError('QR theta measurement requires Y at the center (STATUS ±0.005 mm)')
        start = perf_counter()
        dx = self._measure(image)-self._zero
        return dict(y=truth[0], t=-dx/THETA_PX_PER_DEG, qr_dx_px=dx,
                    ms=(perf_counter()-start)*1000, rois=[])

    def status(self):
        return dict(kind=self.kind, output=self.output, ready=self._measure is not None,
                    detail='Y anchor via STATUS, theta via QR')

    def close(self):
        self._measure = self._zero = None
