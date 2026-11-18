"""Known-input numerical checks; these are not physical accuracy validation."""
import json
from pathlib import Path

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.special import erf

from measure_lines import extract_line, fit_line, source_rows, FEATURES, step_centers

HERE = Path(__file__).resolve().parent


def main():
    checks = []
    for angle in [-.3, 0., .25, .4]:
        x = np.arange(1500., 1750.)
        y = 1000+np.tan(np.radians(angle))*(x-1600)
        estimate = fit_line(x, y)[0]
        assert abs(estimate-angle) < 1e-6
        checks.append({'angle': angle, 'estimated': estimate})
    yy, xx = np.mgrid[0:100, 0:1200]
    synthetic = []
    for length in [166, 900]:
        for angle in [.2, .25, .3]:
            for phase in [0., .25, .5, .75]:
                edge_y = 50+phase+np.tan(np.radians(angle))*(xx-600)
                continuous = 60+140*.5*(1+erf((yy-edge_y)/(np.sqrt(2)*1.2)))
                for quantized in [False, True]:
                    image = np.rint(continuous).astype(np.uint8) if quantized else continuous.astype(np.float32)
                    spec = [600-length//2, 600+length//2, 50, 10, 1, 'C']
                    estimate = extract_line(image, spec)['angle_deg']
                    synthetic.append({'length_px': length, 'true_angle_image_deg': angle,
                                      'vertical_phase_px': phase, 'quantized_uint8': quantized,
                                      'estimated_angle_image_deg': estimate, 'error_image_deg': estimate-angle})
    summary = []
    for length in [166, 900]:
        for quantized in [False, True]:
            errors = [r['error_image_deg'] for r in synthetic if r['length_px'] == length and r['quantized_uint8'] == quantized]
            summary.append({'length_px': length, 'quantized_uint8': quantized,
                            'max_abs_error_image_deg': max(map(abs, errors)),
                            'rms_error_image_deg': float(np.sqrt(np.mean(np.square(errors))))})
    result = {'direct_line_checks': checks, 'synthetic_edges': synthetic, 'summary': summary,
              'scope': 'Noiseless ideal blurred step edges with known angle and pixel phase; numerical sampling sensitivity only. Does not model real texture, distortion, illumination, or physical camera angles.'}
    (HERE/'numerical_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    source = source_rows()[0]
    image = cv2.imdecode(np.fromfile(source['path'], np.uint8), 0)
    crosschecks = []
    for name in ['L_top', 'C_bottom', 'rail_UL_edge', 'rail_UR_edge', 'rail_BL_edge']:
        x0, x1, yc, half, polarity, _ = FEATURES[name]
        y0 = yc-half-8
        patch = cv2.GaussianBlur(image[y0:yc+half+9, x0:x1+1].astype(np.float32), (0, 0), 1)
        gradient = polarity*np.gradient(patch, axis=0)
        for column in np.linspace(10, x1-x0-10, 3).astype(int):
            peak = np.argmax(gradient[8:-8, column])+8
            ga, gb, gc = gradient[peak-1:peak+2, column]
            initial = np.array([peak+.5*(ga-gc)/(ga-2*gb+gc)])
            yy = np.arange(peak-5, peak+6)[:, None]
            observed = patch[yy, np.array([[column]])]
            center, width = step_centers(observed, yy, initial)
            def objective(p):
                return p[0]+p[1]*.5*(1+erf((yy[:, 0]-p[2])/(np.sqrt(2)*p[3])))-observed[:, 0]
            fit = least_squares(objective, [observed[0, 0], observed[-1, 0]-observed[0, 0], initial[0], 1.5],
                                bounds=([-np.inf, -np.inf, initial[0]-2, .6], [np.inf, np.inf, initial[0]+2, 4]),
                                ftol=1e-12, xtol=1e-12, gtol=1e-12)
            difference = float(center[0]-fit.x[2])
            assert abs(difference) < 1e-4, (name, difference)
            crosschecks.append({'feature': name, 'column': int(column), 'center_delta_px': difference,
                                'batch_width': float(width[0]), 'scipy_width': float(fit.x[3])})
    (HERE/'solver_crosscheck.json').write_text(json.dumps(crosschecks, indent=2), encoding='utf-8')
    print('Independent solver crosscheck: max center difference', max(abs(r['center_delta_px']) for r in crosschecks))


if __name__ == '__main__':
    main()
