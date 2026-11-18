"""Fit the runtime calibration from one rig capture ZIP and its tracked QR NPZ."""

import argparse
import json
import zipfile
from pathlib import Path

import cv2 as cv
import numpy as np


def features(shifts):
    mean = shifts.mean(axis=1)
    travel = mean[:, 0]/110.0
    relative = shifts-mean[:, None, :]
    return np.column_stack((shifts.reshape(len(shifts), -1), 10*travel**2,
                            (travel[:, None, None]*relative).reshape(len(shifts), -1)))


def fit(values, labels):
    # Zero displacement maps to the supplied center by construction.
    regularization = 0.1*np.eye(values.shape[1])
    return np.linalg.solve(values.T @ values+regularization, values.T @ labels)


def evaluate(values, labels):
    result = {}
    for axis, scale, name in [(1, 20, 'theta_blocks'), (0, 10, 'y_blocks')]:
        folds = (np.rint(labels[:, axis]*scale).astype(int)+20)//4 % 5
        prediction = np.empty_like(labels)
        for fold in np.unique(folds):
            test = folds == fold
            prediction[test] = values[test] @ fit(values[~test], labels[~test])
        error = prediction-labels
        result[name] = {
            'count': len(error),
            'rmse': np.sqrt(np.mean(error**2, axis=0)).tolist(),
            'axis_pass_y_009_theta_001': (np.abs(error)<=[.09,.01]).sum(axis=0).tolist(),
            'joint_pass_y_009_theta_001': int((np.abs(error)<=[.09,.01]).all(axis=1).sum()),
            'joint_pass_verify_web_defaults': int((np.abs(error)<=[.045,.01]).all(axis=1).sum()),
        }
    return result


def export_model(output, capture, points, zero_shifts, coefficients):
    with zipfile.ZipFile(capture) as archive:
        data = archive.read(capture.stem+'/center.bmp')
    reference = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_GRAYSCALE)
    if reference is None or reference.shape != (2160, 3840):
        raise ValueError('Expected a rectified 3840×2160 center BMP')
    output.parent.mkdir(parents=True, exist_ok=True)
    storage = cv.FileStorage(str(output), cv.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f'Cannot write {output}')
    try:
        storage.write('version', 1)
        storage.write('training_capture', capture.stem)
        storage.write('points', points)
        storage.write('zero_shifts', zero_shifts)
        storage.write('coarse', reference[1055:1206, 1832:1983].copy())
        storage.write('coefficients', coefficients)
        storage.startWriteStruct('patches', cv.FileNode_SEQ)
        for x, y in points:
            storage.write('', reference[y-17:y+18, x-17:x+18].copy())
        storage.endWriteStruct()
    finally:
        storage.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('tracks', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with np.load(args.tracks) as tracks:
        grid = np.char.startswith(tracks['names'], 'Grid/')
        points = tracks['points'].astype(np.int32)
        shifts = tracks['shifts'][grid, :, :2]
        zero_shifts = tracks['shifts'][0, :, :2].copy()
        labels = tracks['labels'][grid]
    if points.shape != (42, 2) or len(labels) != 1680 or not np.isfinite(shifts).all():
        parser.error('Expected 42 QR points and 1,680 valid grid observations')

    values = features(shifts-zero_shifts)
    export_model(args.output, args.capture, points, zero_shifts, fit(values, labels))
    print(json.dumps(evaluate(values, labels), indent=2))


if __name__ == '__main__':
    main()
