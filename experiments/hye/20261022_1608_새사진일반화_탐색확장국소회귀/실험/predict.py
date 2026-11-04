"""Image-bytes inference for the separate joint Y/T experiment.

The current image's path is used only to read bytes. No filename, command value,
capture date, or trial ID is passed into measurement or prediction.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import cv2
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, HERE / 'measurements'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from extractor_scipy import StructuralExtractor

CENTER_PATH = ROOT / 'hw-analysis/data/hw/bmp/20261014/center.bmp'
CENTER_SHA256 = '01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685'


def _load_bundle(model_path=None):
    path = Path(model_path) if model_path else HERE / 'model.joblib'
    bundle = joblib.load(path)
    if not {'model', 'config', 'ids', 'points'}.issubset(bundle):
        raise ValueError('The selected file is not a joint Y/T model bundle.')
    return bundle, hashlib.sha256(path.read_bytes()).hexdigest()


def _measure_bytes(image_bytes, bundle):
    center_bytes = CENTER_PATH.read_bytes()
    if hashlib.sha256(center_bytes).hexdigest() != CENTER_SHA256:
        raise ValueError('The original calibrated center image has changed.')
    center = cv2.imdecode(np.frombuffer(center_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    current = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if current is None or center is None:
        raise ValueError('The input could not be decoded as an image.')
    if current.shape != center.shape:
        raise ValueError(f'Expected image dimensions {center.shape}, received {current.shape}.')
    extractor = StructuralExtractor(center)
    if not np.array_equal(extractor.ids, bundle['ids']):
        raise ValueError('Fixed structural IDs do not match the calibrated model.')
    if not np.array_equal(extractor.points, bundle['points']):
        raise ValueError('Fixed reference coordinates do not match the calibrated model.')
    measured = extractor.extract(current)
    if measured['D'].shape != (630, 2) or not np.isfinite(measured['D']).all():
        raise ValueError('Measurement did not produce all630 finite structural displacements.')
    return measured


def predict_bytes(image_bytes, model_path=None):
    """Return a numerical estimate and its negative, without robot actuation."""
    started = time.perf_counter()
    bundle, model_digest = _load_bundle(model_path)
    measured = _measure_bytes(image_bytes, bundle)
    # Thread limits keep the local GLS solve consistent with model evaluation.
    with threadpool_limits(1):
        estimate = np.asarray(bundle['model'].predict(measured['D'][None]))[0]
    if estimate.shape != (2,) or not np.isfinite(estimate).all():
        raise ValueError('The model did not produce finite Y and T estimates.')
    return {
        'status': 'ok',
        'Y': float(estimate[0]),
        'T': float(estimate[1]),
        'correction_Y': float(-estimate[0]),
        'correction_T': float(-estimate[1]),
        'actuation_executed': False,
        'model': 'joint_yt_generalization_experimental',
        'config': bundle['config'],
        'model_sha256': model_digest,
        'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
        'reference_sha256': CENTER_SHA256,
        'measurement': 'fixed630_blurred_boundary_scipy_expanded_ncc',
        'point_count': 630,
        'diagnostic_valid_count': int(np.sum(measured['valid'])),
        'diagnostic_valid_fraction': float(np.mean(measured['valid'])),
        'ncc_min': float(np.min(measured['coarse_score'])),
        'measurement_seconds': float(measured['runtime_seconds']),
        'total_seconds': time.perf_counter() - started,
        'evaluation': 'No ground truth supplied; error and tolerance pass are not assessed.',
    }


def predict_image(image_path, model_path=None):
    """Read an image file, then pass only its bytes to the estimator."""
    return predict_bytes(Path(image_path).read_bytes(), model_path)


def main():
    parser = argparse.ArgumentParser(description='Estimate Y and T from one image and the fixed calibrated center.')
    parser.add_argument('image', help='Current image path; the filename is not an inference feature.')
    parser.add_argument('--model', help='Optional joint model bundle; defaults to model.joblib beside this script.')
    parser.add_argument('--output', help='Optional JSON result path.')
    args = parser.parse_args()
    try:
        result = predict_image(args.image, args.model)
        exit_code = 0
    except Exception as exc:
        result = {'status': 'error', 'error': str(exc), 'actuation_executed': False}
        exit_code = 1
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding='utf-8')
    print(text)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
