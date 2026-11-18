"""Four reference images only: line through two upper QR finder centers.

Otsu/nested-contour detection initializes outer finder corners; cornerSubPix
refines those corners on raw grayscale pixels. Center = diagonal intersection.
No outer QR border fitting, image warp, motion calibration, or controller change.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REFS = BASE/'reference_slope_check_20261029/reference_slope_results.json'
# Original 3840x2160 image coordinates; fixed from the existing reference QR positions.
ROIS = {'L': [1535, 948, 1760, 1190], 'C': [1794, 948, 2018, 1190], 'R': [2054, 948, 2278, 1190]}
WINDOW = (3, 3)  # 7x7 samples: smaller than the separation to inner finder-ring corners.
CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-5)


def quad_order(points):
    points = np.asarray(points, dtype=float).reshape(4, 2)
    center = points.mean(axis=0)
    ordered = points[np.argsort(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))]
    ordered = np.roll(ordered, -np.argmin(ordered.sum(axis=1)), axis=0)
    return ordered


def diagonal_intersection(quad):
    p0, p1, p2, p3 = np.asarray(quad, dtype=float)
    system = np.column_stack([p2-p0, -(p3-p1)])
    t, s = np.linalg.solve(system, p1-p0)
    assert 0 < t < 1 and 0 < s < 1
    return p0+t*(p2-p0)


def find_patterns(image, roi):
    x0, y0, x1, y1 = roi
    patch = image[y0:y1, x0:x1].copy()
    threshold, mask = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert hierarchy is not None
    hierarchy = hierarchy[0]
    candidates = []
    for index, contour in enumerate(contours):
        child = int(hierarchy[index, 2])
        if child < 0:
            continue
        grandchild = int(hierarchy[child, 2])
        if grandchild < 0:
            continue
        outer_area = abs(cv2.contourArea(contour))
        if not 200 < outer_area < .2*patch.size:
            continue
        child_area = abs(cv2.contourArea(contours[child]))
        inner_area = abs(cv2.contourArea(contours[grandchild]))
        if not (.25 < child_area/outer_area < .85 and .1 < inner_area/child_area < .8):
            continue
        perimeter = cv2.arcLength(contour, True)
        quad = cv2.approxPolyDP(contour, .03*perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        initial = quad_order(quad)
        sides = np.linalg.norm(np.roll(initial, -1, axis=0)-initial, axis=1)
        if min(sides)/max(sides) < .65:
            continue
        refined_local = initial.astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(patch, refined_local, WINDOW, (-1, -1), CRITERIA)
        refined = refined_local.reshape(4, 2).astype(float)+[x0, y0]
        initial += [x0, y0]
        center = diagonal_intersection(refined)
        candidates.append({'center_xy_px': center.tolist(), 'outer_corners_xy_px': refined.tolist(),
            'initial_corners_xy_px': initial.tolist(),
            'max_corner_refinement_px': float(np.max(np.linalg.norm(refined-initial, axis=1))),
            'outer_area_px2': outer_area, 'white_to_outer_area_ratio': child_area/outer_area,
            'inner_to_white_area_ratio': inner_area/child_area})
    assert len(candidates) == 3, (roi, 'Expected three nested square patterns', candidates)
    by_height = sorted(candidates, key=lambda c: c['center_xy_px'][1])
    upper = sorted(by_height[:2], key=lambda c: c['center_xy_px'][0])
    patterns = {'A_top_left': upper[0], 'B_top_right': upper[1], 'C_bottom_left': by_height[2]}
    a = np.array(patterns['A_top_left']['center_xy_px'])
    b = np.array(patterns['B_top_right']['center_xy_px'])
    c = np.array(patterns['C_bottom_left']['center_xy_px'])
    assert b[0] > a[0] and c[1] > a[1] and abs(c[0]-a[0]) < .3*np.linalg.norm(b-a)
    delta = b-a
    slope = delta[1]/delta[0]
    intercept = a[1]-slope*a[0]
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    return {'roi_xyxy': roi, 'otsu_threshold': threshold, 'patterns': patterns,
        'slope_dy_dx': float(slope), 'intercept_y_px': float(intercept),
        'image_angle_deg': float(angle), 'baseline_length_px': float(np.linalg.norm(delta)),
        'baseline_endpoints_xy_px': [a.tolist(), b.tolist()]}


def main():
    cv2.setNumThreads(2)
    refs = json.loads(REFS.read_text(encoding='utf-8'))['rows']
    assert len(refs) == 4
    records, images = [], []
    for ref in refs:
        source = Path(ref['path'])
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == ref['sha256']
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        assert image.shape == (2160, 3840)
        record = {k: ref[k] for k in ['dataset', 'file', 'path', 'sha256']}
        record['qrs'] = {name: find_patterns(image, roi) for name, roi in ROIS.items()}
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        records.append(record)
        images.append(image)
        print(ref['file'], {key: round(value['image_angle_deg'], 9) for key, value in record['qrs'].items()}, flush=True)
    for record in records:
        for name, qr in record['qrs'].items():
            qr['delta_from_first_center_image_deg'] = qr['image_angle_deg']-records[0]['qrs'][name]['image_angle_deg']
    ranges = {name: float(np.ptp([r['qrs'][name]['image_angle_deg'] for r in records])) for name in ROIS}
    result = {'scope': 'Exactly four saved references; one AB finder-center line per QR; 12 lines total.',
        'method': {'detection': 'Per-QR Otsu inverse threshold; outer/white/inner nested contours; convex quadrilateral outer-ring corners.',
            'selection_parameters': {'min_outer_area_px2': 200, 'max_outer_area_roi_fraction': .2,
                'white_outer_area_ratio': [.25, .85], 'inner_white_area_ratio': [.1, .8],
                'polygon_epsilon_perimeter_fraction': .03, 'min_side_length_ratio': .65},
            'corner_refinement': 'OpenCV cornerSubPix on raw grayscale QR crop; no image warp or smoothing added.',
            'subpix_half_window_xy': WINDOW, 'subpix_zero_zone': [-1, -1],
            'subpix_max_iterations': 100, 'subpix_epsilon_px': 1e-5,
            'center': 'Intersection of the two diagonals of the refined OUTER finder quadrilateral.',
            'line': 'Unique line through A(top-left finder center) and B(top-right finder center). C identifies layout only.',
            'units': 'Original image x right/y down; atan2(dy,dx) in degrees, NOT robot theta.',
            'precision_note': 'Detection/iteration settings and decimal output are not physical-accuracy thresholds.',
            'aggregation': 'No averaging of QR, references, nested rings or outer-border slopes.'},
        'opencv_version': cv2.__version__, 'all_reference_hashes_unchanged': True,
        'rows': records, 'max_minus_min_image_deg_by_qr': ranges,
        'limitations': ['Finder-center baseline is a new feature, not the same line as the former QR outer top border.',
            'Single image per saved reference; no uncertainty/equivalence or motion-response claim.',
            'Two estimated centers always determine a line; zero residual to those two points does not prove accurate detection.',
            'Illumination, printed pattern geometry, sampling, blur and lens distortion can bias corners.',
            'No first/second/third movement images, new theta calibration or controller changes were processed.']}
    (HERE/'finder_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    fig, axes = plt.subplots(4, 3, figsize=(13, 14), layout='constrained')
    for row, (record, image) in enumerate(zip(records, images)):
        for col, (name, qr) in enumerate(record['qrs'].items()):
            ax = axes[row, col]
            x0, y0, x1, y1 = qr['roi_xyxy']
            ax.imshow(image[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255,
                extent=(x0-.5, x1-.5, y1-.5, y0-.5), interpolation='nearest')
            for key, pattern in qr['patterns'].items():
                quad = np.array(pattern['outer_corners_xy_px']); closed = np.vstack([quad, quad[0]])
                ax.plot(closed[:, 0], closed[:, 1], color='#00ddff', lw=.8)
                ax.scatter(quad[:, 0], quad[:, 1], s=9, color='#00ddff')
                p = np.array(pattern['center_xy_px'])
                ax.scatter(p[0], p[1], s=35, color='#ffec32', marker='+', linewidths=1.5)
                ax.text(p[0]+7, p[1]-8, key[0], color='#ffec32', fontsize=10,
                    bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none', 'pad': 1})
            ends = np.array(qr['baseline_endpoints_xy_px'])
            ax.plot(ends[:, 0], ends[:, 1], color='#ff5a31', lw=1.5)
            ax.set_title(f"{record['file']} | QR {name}\nangle={qr['image_angle_deg']:+.6f} deg; m={qr['slope_dy_dx']:+.8f}", fontsize=10)
            ax.set_xlabel('Original image x (px)'); ax.set_ylabel('Original image y (px)')
    fig.suptitle('Finder-center reference lines: cyan = refined outer corners; yellow = diagonal intersections; orange = A-B', fontsize=12)
    fig.savefig(HERE/'finder_lines_4references.png', dpi=150)
    plt.close(fig)
    print('Max-min image degrees:', ranges)


if __name__ == '__main__':
    main()
