"""Measure line angles directly in original first/second-data BMP pixels.

Saved registrations only locate QR patches (translation); no image is warped.
Feature ROIs are selected on center.bmp, before looking at label responses.
Angles are image angles in degrees, positive downward to the right.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.special import erf

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

# Physical-looking straight segments from QR outer borders and mentor-marked rails.
# [x0, x1, expected_y, search_halfheight, gradient_polarity, tracking_marker]
FEATURES = {
    'L_top': [1560, 1725, 967, 5, 1, 'L'],
    'L_bottom': [1560, 1725, 1167, 5, -1, 'L'],
    'C_top': [1817, 1984, 969, 5, 1, 'C'],
    'C_bottom': [1817, 1984, 1168, 5, -1, 'C'],
    'R_top': [2077, 2243, 970, 5, 1, 'R'],
    'R_bottom': [2077, 2243, 1170, 5, -1, 'R'],
    'rail_UL_edge': [200, 1100, 711, 10, 1, 'C'],
    'rail_UL_ridge': [200, 1100, 747, 9, 1, 'C'],
    'rail_UR_edge': [2700, 3550, 722, 10, 1, 'C'],
    'rail_UR_ridge': [2700, 3550, 759, 9, 1, 'C'],
    'rail_BL_edge': [400, 1300, 1877, 22, 1, 'C'],
    'rail_BR_edge': [2500, 3400, 1877, 22, 1, 'C'],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_line(x, y, robust=True):
    # Center the coordinates for numerical conditioning of float32 fitLine.
    cx, cy = float(np.mean(x)), float(np.mean(y))
    pts = np.column_stack([x-cx, y-cy]).astype(np.float32)
    vx, vy, px, py = cv2.fitLine(pts, cv2.DIST_HUBER if robust else cv2.DIST_L2,
                               0, 1e-6, 1e-8).ravel().astype(float)
    slope = vy/vx
    intercept = cy+py-slope*(cx+px)
    angle = float(np.degrees(np.arctan(slope)))
    residual = (y-slope*x-intercept)/np.sqrt(1+slope*slope)
    return angle, slope, intercept, residual


def step_centers(observed, ys, initial):
    """Batched least squares of B+A*normal_CDF((y-center)/blur_sigma).

    Fitting a continuous blurred step avoids the pixel-phase bias of a quadratic
    approximation to sampled gradient peaks. This remains a local edge model,
    not a guarantee that a textured/specular real boundary is an ideal step.
    """
    count = observed.shape[1]
    params = np.column_stack([observed[0], observed[-1]-observed[0], initial,
                              np.full(count, 1.5)]).astype(float)
    values = observed.T.astype(float)
    coordinates = ys.T.astype(float)
    for _ in range(30):
        b, a, center, width = params.T
        t = (coordinates-center[:, None])/width[:, None]
        cdf = .5*(1+erf(t/np.sqrt(2)))
        density = np.exp(-.5*t*t)/np.sqrt(2*np.pi)
        residual = b[:, None]+a[:, None]*cdf-values
        jac = np.stack([np.ones_like(cdf), cdf,
                        -a[:, None]*density/width[:, None],
                        -a[:, None]*density*t/width[:, None]], axis=2)
        normal = np.einsum('nki,nkj->nij', jac, jac)
        rhs = np.einsum('nki,nk->ni', jac, residual)
        normal += np.eye(4)[None]*1e-6
        delta = np.linalg.solve(normal, rhs[..., None])[..., 0]
        # Active bound: refit the other parameters with the bound parameter fixed.
        # Clipping a coupled four-parameter update alone can bias broad edges.
        active_center = ((center >= initial+2-1e-8) & (delta[:, 2] < 0)) | ((center <= initial-2+1e-8) & (delta[:, 2] > 0))
        active_width = ((width >= 4-1e-8) & (delta[:, 3] < 0)) | ((width <= .6+1e-8) & (delta[:, 3] > 0))
        if np.any(active_center | active_width):
            jac[active_center, :, 2] = 0
            jac[active_width, :, 3] = 0
            normal = np.einsum('nki,nkj->nij', jac, jac)+np.eye(4)[None]*1e-6
            rhs = np.einsum('nki,nk->ni', jac, residual)
            delta = np.linalg.solve(normal, rhs[..., None])[..., 0]
        delta[:, 2:] = np.clip(delta[:, 2:], -.5, .5)
        params -= delta
        params[:, 2] = np.clip(params[:, 2], initial-2, initial+2)
        params[:, 3] = np.clip(params[:, 3], .6, 4.)
    return params[:, 2], params[:, 3]


def extract_line(image, spec, shift=(0, 0), sigma=1.0):
    x0, x1, yref, half, polarity, _ = spec
    dx, dy = shift
    x0, x1 = int(round(x0+dx)), int(round(x1+dx))
    yc = int(round(yref+dy))
    y0, y1 = yc-half-8, yc+half+9
    assert 0 <= x0 < x1 < image.shape[1] and 0 <= y0 < y1 < image.shape[0]
    patch = image[y0:y1, x0:x1+1].astype(np.float32)
    blurred = cv2.GaussianBlur(patch, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)
    gradient = polarity*np.gradient(blurred, axis=0)
    columns = np.arange(2, patch.shape[1]-2, 1 if x1-x0 < 300 else 2)
    peak = np.argmax(gradient[8:-8, columns], axis=0)+8
    ga, gb, gc = gradient[peak-1, columns], gradient[peak, columns], gradient[peak+1, columns]
    denominator = ga-2*gb+gc
    offset = np.divide(.5*(ga-gc), denominator, out=np.zeros_like(gb), where=denominator < -1e-8)
    offset = np.clip(offset, -.5, .5)
    valid = (peak > 8) & (peak < gradient.shape[0]-9) & (gb > 0)
    x = (x0+columns[valid]).astype(float)
    quad_y = (y0+peak[valid]+offset[valid]).astype(float)
    sample_y = peak[valid][None, :]+np.arange(-5, 6)[:, None]
    observed = blurred[sample_y, columns[valid][None, :]]
    refined, widths = step_centers(observed, sample_y, peak[valid]+offset[valid])
    y = y0+refined
    if len(x) < 20:
        raise ValueError('Insufficient valid edge columns')
    angle, slope, intercept, residual = fit_line(x, y)
    l2_angle, _, _, _ = fit_line(x, y, robust=False)
    quad_angle, _, _, _ = fit_line(x, quad_y)
    # A separate trimming diagnostic detects curved edges or influence of corners.
    middle = (x >= x.min()+.1*np.ptp(x)) & (x <= x.max()-.1*np.ptp(x))
    trimmed_angle, _, _, _ = fit_line(x[middle], y[middle])
    return {
        'angle_deg': angle, 'slope_dy_dx': slope, 'intercept_y_px': intercept,
        'l2_angle_deg': l2_angle, 'trimmed_angle_deg': trimmed_angle,
        'quadratic_peak_angle_deg': quad_angle,
        'median_fitted_blur_sigma_px': float(np.median(widths)),
        'step_center_at_bound_fraction': float(np.mean(abs(refined-(peak[valid]+offset[valid])) >= 1.999)),
        'edge_rms_px': float(np.sqrt(np.mean(residual**2))),
        'edge_mad_px': float(np.median(abs(residual-np.median(residual)))),
        'valid_fraction': float(np.mean(valid)), 'n_points': len(x),
        'median_gradient': float(np.median(gb[valid])),
        'max_peak_offset_from_expected_px': float(np.max(abs(y-yc))),
        'endpoints_xy': [[float(x.min()), float(slope*x.min()+intercept)],
                         [float(x.max()), float(slope*x.max()+intercept)]],
        'sample_points_xy': np.column_stack([x, y])[::max(1, len(x)//35)].tolist(),
        'shift_xy_px': [float(dx), float(dy)],
    }


def source_rows():
    first = json.loads((BASE/'initial_error/measurements.json').read_text(encoding='utf-8'))
    second = json.loads((BASE/'increment_consistency/second_measurements.json').read_text(encoding='utf-8'))
    metadata = json.loads((BASE/'initial_error/capture_order.json').read_text(encoding='utf-8'))
    lookup = {r['file']: r for r in first['measurements']}
    groups = {g['group']: i+1 for i, g in enumerate(metadata['chronological_groups'])}
    rows = []
    for m in metadata['files_in_modification_order']:
        row = lookup[m['file']]
        rows.append({'dataset': 'first', 'file': row['file'], 'path': str(Path(first['data_root'])/row['file']),
                     'sha256': row['sha256'], 'group': groups[m['group']], 'axis': row['axis'],
                     'label': row['label'], 'order': m['modification_order'],
                     'modified': m['modified_kst'], 'regions': row['regions']})
    for row in second['measurements']:
        rows.append({'dataset': 'second', 'file': row['file'], 'path': str(Path(second['data_root'])/row['file']),
                     'sha256': row['sha256'], 'condition': row['condition'],
                     'is_reference': row['is_reference'], 'Y': row['label_y_mm'], 'Theta': row['label_theta_degree'],
                     'modified': row['archive_modified_time_no_timezone'], 'regions': row['regions']})
    return rows


def annotate(image, row, path):
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    palette = [(20, 20, 230), (10, 150, 10), (220, 100, 20), (180, 30, 150)]
    for i, (name, data) in enumerate(row['features'].items()):
        color = palette[i//3 % len(palette)]
        a, b = np.rint(data['endpoints_xy']).astype(int)
        cv2.line(canvas, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
        # Numeric labels; descriptive legend is in the report/plots.
        pos = (int(a[0]), int(a[1]-12 if 'bottom' not in name else a[1]+24))
        cv2.putText(canvas, str(i+1)+': '+name, pos, cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.png', cv2.resize(canvas, (1920, 1080)))
    assert ok
    encoded.tofile(path)


def main():
    cv2.setNumThreads(2)
    result = []
    sources = source_rows()
    assert len(sources) == 341
    qa_names = {'center.bmp', 'Theta\\T+1.00.bmp', 'Theta\\T-1.00.bmp',
                'Y\\Y+2.00.bmp', 'Y\\Y-2.00.bmp'}
    for i, source in enumerate(sources):
        raw = np.fromfile(source['path'], dtype=np.uint8)
        assert hashlib.sha256(raw).hexdigest() == source['sha256'], source['path']
        image = cv2.imdecode(raw, 0)
        row = {k: v for k, v in source.items() if k not in ('regions', 'path')}
        features = {}
        for name, spec in FEATURES.items():
            reg = source['regions'][spec[-1]]['affine']
            # Second measurements use their own center1; <1 px reference offset
            # stays well inside the edge search band. Final angles use raw pixels.
            shift = (reg['dx'], reg['dy'])
            features[name] = extract_line(image, spec, shift)
            features[name]['sigma2_angle_deg'] = extract_line(image, spec, shift, sigma=2.0)['angle_deg']
        row['features'] = features
        result.append(row)
        if source['dataset'] == 'first' and source['file'] in qa_names:
            annotate(image, row, HERE/('overlay_first_'+Path(source['file']).stem+'.png'))
        if source['dataset'] == 'second' and (source['is_reference'] or (source['Y'] == 1 and source['Theta'] == 2)):
            annotate(image, row, HERE/('overlay_second_'+Path(source['file']).stem+'.png'))
        if (i+1) % 40 == 0 or i+1 == len(sources):
            print(f'Measured {i+1}/{len(sources)} images', flush=True)
    document = {
        'features': FEATURES, 'images': len(result), 'measurements': result,
        'method': {
            'angle': 'atan(dy/dx) in image degrees, positive downward to right; not robot theta',
            'edge': 'Gaussian sigma=1 px; signed vertical gradient for initialization; continuous Gaussian-blurred step B+A*CDF((y-center)/sigma) fitted to 11 samples per column, 30 active-bound Gauss-Newton iterations',
            'fit': 'cv2.fitLine DIST_HUBER, param=0 (OpenCV default Huber scale); centered point coordinates; reps=1e-6, aeps=1e-8',
            'diagnostics': 'Initial quadratic-peak result, L2 fit, central-80%-segment Huber angle, and Gaussian sigma=2 px angle also saved; no accuracy acceptance threshold implied by solver settings',
            'location': 'Previously saved affine displacement at patch center used only for translating search ROI; no affine resampling',
            'scope': 'Per-line descriptive response and label-based feasibility; no hardware control or mechanical cause attribution',
            'selection': '12 lines fixed from center.bmp and mentor-marked structural regions before label-response evaluation',
        },
        'source_hashes': {str(p.relative_to(BASE)): sha(p) for p in [
            BASE/'initial_error/measurements.json', BASE/'initial_error/capture_order.json',
            BASE/'increment_consistency/second_measurements.json']},
        'script_sha256': sha(Path(__file__)), 'opencv_version': cv2.__version__,
    }
    (HERE/'line_measurements.json').write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved line_measurements.json', flush=True)


if __name__ == '__main__':
    main()
