"""QR-only horizontal displacement shared by local and verify_web runners."""
import cv2 as cv
import numpy as np


def qr_observer(reference):
    if reference.shape != (2160, 3840) or reference.dtype != np.uint8:
        raise ValueError('Expected a rectified 3840x2160 grayscale reference')
    points = ((1650, 1130), (1907, 1130), (2166, 1132))
    templates = [reference[y-50:y+51, x-50:x+51].copy() for x,y in points]

    def measure(frame):
        if frame.shape != reference.shape or frame.dtype != np.uint8:
            raise ValueError('Unexpected rectified camera frame')
        shifts = []
        for (x,y), template in zip(points, templates):
            window = frame[y-72:y+73, x-190:x+191]
            score = cv.matchTemplate(window, template, cv.TM_CCOEFF_NORMED)
            _, quality, _, (i,j) = cv.minMaxLoc(score)
            if quality < .90 or not (0 < i < score.shape[1]-1 and 0 < j < score.shape[0]-1):
                raise ValueError('QR matching failed')
            a, b, c = map(float, score[j, i-1:i+2])
            if a+c-2*b >= -1e-9:
                raise ValueError('Ambiguous QR match')
            shifts.append(i-140+.5*(a-c)/(a+c-2*b))
        return float(np.median(shifts))
    return measure
