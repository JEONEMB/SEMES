import struct

import cv2 as cv
import numpy as np


cv.setNumThreads(1)


def image(archive, name):
    data = archive.read(name)
    start = struct.unpack_from('<I', data, 10)[0]
    width, height, planes, depth, compression = struct.unpack_from('<iiHHI', data, 18)
    assert (width, height, planes, depth, compression) == (3840, 2160, 1, 8, 0)
    return np.frombuffer(data, np.uint8, offset=start).reshape(height, width)[::-1]


def patch(im, x, y, radius):
    return im[y-radius:y+radius+1, x-radius:x+radius+1].copy()


def offset(im, template, x, y, radius, search_x, search_y, strict=True):
    window = im[y-radius-search_y:y+radius+search_y+1,
                x-radius-search_x:x+radius+search_x+1].copy()
    corr = cv.matchTemplate(window, template, cv.TM_CCOEFF_NORMED)
    _, score, _, (u, v) = cv.minMaxLoc(corr)
    if not (0 < u < 2*search_x and 0 < v < 2*search_y):
        if not strict:
            return [np.nan, np.nan, score]
        raise ValueError(f'Template reached search edge at {(x, y)}: {(u, v, score)}')
    center = corr[v, u]
    def subpixel(a, b):
        return 0.5 * (a-b) / (a+b-2*center)
    return [u-search_x+subpixel(corr[v, u-1], corr[v, u+1]),
            v-search_y+subpixel(corr[v-1, u], corr[v+1, u]), score]


def qr_templates(ref):
    points = []
    for x, y in [(1650, 1130), (1907, 1130), (2166, 1132)]:
        corners = cv.goodFeaturesToTrack(patch(ref, x, y, 79), 14, 0.02, 22)
        points.extend((round(float(u))+x-79, round(float(v))+y-79)
                      for u, v in corners.reshape(-1, 2))
    return points, [patch(ref, x, y, 17) for x, y in points], patch(ref, 1907, 1130, 75)


def track_qr(im, points, templates, coarse):
    dx, dy, _ = offset(im, coarse, 1907, 1130, 75, 140, 22)
    ix, iy = round(dx), round(dy)
    shifts = np.asarray([offset(im, tpl, x+ix, y+iy, 17, 9, 7)
                         for (x, y), tpl in zip(points, templates)])
    shifts[:, :2] += [ix, iy]
    return shifts
