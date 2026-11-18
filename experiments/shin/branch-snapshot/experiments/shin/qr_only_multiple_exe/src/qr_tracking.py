"""Shared QR-only tracking for the fitted runtime and calibration tools."""
import cv2 as cv
import numpy as np


def patch(frame, x, y, radius):
    return frame[y-radius:y+radius+1, x-radius:x+radius+1].copy()


def dense_measure(frame, refs, points, coarse_offsets, centers=((1650,1130),(1907,1130),(2166,1132))):
    shifts = []
    for (x, y), reference, qr_points, coarse in zip(centers, refs, points, coarse_offsets):
        dx, dy = np.rint(coarse).astype(int)
        target = patch(frame, x + dx, y + dy, 101)
        found, valid, _ = cv.calcOpticalFlowPyrLK(
            reference, target, qr_points, None, winSize=(19, 19), maxLevel=1,
            criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 35, .001))
        if found is None:
            raise ValueError('Dense QR tracking failed')
        local = (found - qr_points).reshape(-1, 2)
        keep = valid.ravel().astype(bool) & np.isfinite(local).all(axis=1)
        keep &= np.linalg.norm(local, axis=1) <= 8
        coordinates = qr_points.reshape(-1, 2)
        design = np.column_stack((np.ones(len(local)), coordinates - coordinates.mean(axis=0)))
        for _ in range(2):
            if keep.sum() < .75 * len(keep):
                raise ValueError(f'Unreliable dense QR tracking: {keep.sum()}/{len(keep)}')
            affine = design @ np.linalg.lstsq(design[keep], local[keep], rcond=None)[0]
            keep &= np.linalg.norm(local - affine, axis=1) <= 1
        if keep.sum() < .75 * len(keep):
            raise ValueError(f'Unreliable dense QR tracking: {keep.sum()}/{len(keep)}')
        local[~keep] = affine[~keep]
        shifts.append(local + [dx, dy])
    return np.concatenate(shifts)



def initial_offset(frame, template, x, y):
    radius, sx, sy = 75, 240, 45
    window = frame[y-radius-sy:y+radius+sy+1,
                   x-radius-sx:x+radius+sx+1].copy()
    scores = cv.matchTemplate(window,template,cv.TM_CCOEFF_NORMED)
    _, quality, _, (u,v) = cv.minMaxLoc(scores)
    if quality < .85 or not (0 < u < 2*sx and 0 < v < 2*sy):
        raise ValueError('Coarse QR match below threshold or at ROI edge')
    center = scores[v,u]
    def delta(left,right):
        curvature = left+right-2*center
        if curvature >= -1e-9:
            raise ValueError('Ambiguous QR correlation peak')
        return .5*(left-right)/curvature
    return (u-sx+delta(scores[v,u-1],scores[v,u+1]),
            v-sy+delta(scores[v-1,u],scores[v+1,u]))
