"""Port Hye's condition-balanced forward calibration and GLS to Rig QR tracks."""
import argparse
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from train import features, fit
from qr_tracking import dense_measure, patch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'evaluation'
CAPTURES = ('rig_grid', 'rig_grid_rev')
TOLERANCE = np.array([.045, .01])
QR_CENTERS = ((1650, 1130), (1907, 1130), (2166, 1132))


def coarse_match(frame, template, x, y, search_x, search_y):
    height, width = template.shape
    top, left = y - height//2 - search_y, x - width//2 - search_x
    window = frame[top:top+height+2*search_y, left:left+width+2*search_x]
    match = cv.matchTemplate(window, template, cv.TM_CCOEFF_NORMED)
    _, score, _, (u, v) = cv.minMaxLoc(match)
    if score < .90 or not (0 < u < 2*search_x and 0 < v < 2*search_y):
        raise ValueError('Unreliable coarse QR match')
    middle = match[v, u]
    a, b = match[v, u-1], match[v, u+1]
    dx = .5*(a-b)/(a+b-2*middle)
    a, b = match[v-1, u], match[v+1, u]
    dy = .5*(a-b)/(a+b-2*middle)
    return np.array([u-search_x+dx, v-search_y+dy])


def dense_reference(frame):
    refs = np.array([patch(frame, x, y, 101) for x, y in QR_CENTERS])
    points = []
    for reference in refs:
        mask = np.zeros_like(reference)
        mask[24:-24, 24:-24] = 255
        corners = cv.goodFeaturesToTrack(reference, 210, .008, 4, mask=mask)
        if corners is None or len(corners) < 100:
            raise ValueError('Insufficient QR corners for dense tracking')
        points.append(corners)
    return refs, points


def extract_dense():
    import zipfile
    from track_sessions import DESKTOP, image

    captures = archive_data()
    # Fixed identities across both sessions. Transfer the reverse reference's
    # corners to the forward center, then track against each session's center.
    with zipfile.ZipFile(DESKTOP / 'rig_grid_rev.zip') as archive:
        reference = image(archive, 'rig_grid_rev/center.bmp')
    canonical, points = dense_reference(reference)
    for name, data in zip(CAPTURES, captures):
        with zipfile.ZipFile(DESKTOP / f'{name}.zip') as archive:
            ref = image(archive, f'{name}/center.bmp')
            refs = np.array([patch(ref, x, y, 101) for x, y in QR_CENTERS])
            mapped = []
            for first, second, group in zip(canonical, refs, points):
                found, valid, _ = cv.calcOpticalFlowPyrLK(first, second, group, None,
                                                          winSize=(25, 25), maxLevel=2)
                if found is None or not valid.all():
                    raise ValueError(f'Cannot transfer fixed QR corner identities to {name}')
                mapped.append(found)
            tracks = []
            for i, filename in enumerate(data['names']):
                frame = image(archive, f'{name}/{filename}.bmp')
                coarse = np.median(data['shifts'][i, :, :2].reshape(3, 14, 2), axis=1)
                tracks.append(dense_measure(frame, refs, mapped, coarse))
                if i % 400 == 0:
                    print(name, i, len(data['names']), flush=True)
        np.savez_compressed(RESULTS / f'{name}-hye-dense-tracks.npz', shifts=np.asarray(tracks),
                            refs=refs, points=np.concatenate(mapped),
                            group_lengths=[len(group) for group in mapped])


def hye_features(shifts):
    raw = shifts.reshape(len(shifts), -1)
    horizontal = shifts[:, :, 0].mean(axis=1, keepdims=True) / 100.
    return np.column_stack((raw, raw * horizontal))


def fit_hye(values, labels, capture, shrink=.01, capture_weights=None):
    """Fit image response from pose, then invert using shrinkage GLS."""
    _, group, counts = np.unique(np.column_stack((capture, labels)), axis=0,
                                 return_inverse=True, return_counts=True)
    domain_count = np.bincount(capture[np.unique(group, return_index=True)[1]],
                               minlength=int(capture.max()) + 1)
    weights = 1. / (counts[group] * domain_count[capture])
    if capture_weights is not None:
        weights *= capture_weights[capture]
    weights /= weights.sum()
    design = np.column_stack((np.ones(len(labels)), labels))
    root = np.sqrt(weights[:, None])
    forward = np.linalg.lstsq(design * root, values * root, rcond=None)[0]
    error = (values - design @ forward) * root
    scale = np.maximum(np.linalg.norm(error, axis=0), 1e-6)
    normalized = error / scale
    covariance = (1. - shrink) * (normalized.T @ normalized)
    covariance.flat[::len(scale) + 1] += shrink
    response = forward[1:] / scale
    weighted_response = np.linalg.solve(covariance, response.T)
    decoder = (weighted_response @ np.linalg.pinv(response @ weighted_response)) / scale[:, None]
    return decoder, -forward[0] @ decoder


def archive_data():
    captures = []
    for name in CAPTURES:
        with np.load(RESULTS / f'{name}.npz') as data:
            captures.append(dict(data))
    return captures


def stats(prediction, label):
    from train_pooled import scores
    error = prediction - label
    result = scores(error)
    result['mae'] = np.abs(error).mean(axis=0).tolist()
    result['std'] = error.std(axis=0).tolist()
    result['closer'] = int(np.sum(np.linalg.norm(error / TOLERANCE, axis=1)
                                 < np.linalg.norm(label / TOLERANCE, axis=1)))
    result['error_correlation'] = float(np.corrcoef(error.T)[0, 1])
    return result


def evaluate():
    captures = archive_data()
    first_path = RESULTS / 'rig_1st.npz'
    with np.load(first_path) as first:
        old = dict(first)
    campaign = captures + [old]
    shifts = np.concatenate([d['shifts'][1:, :, :2] - d['shifts'][0, :, :2]
                             for d in campaign])
    labels = np.concatenate([d['labels'][1:] for d in campaign])
    sessions = np.repeat(np.arange(len(campaign)), [len(d['labels']) - 1 for d in campaign])
    names = np.concatenate([d['names'][1:] for d in campaign])
    baseline, hye = features(shifts), hye_features(shifts)
    dense_data = []
    for name in CAPTURES:
        with np.load(RESULTS / f'{name}-hye-dense-tracks.npz') as data:
            dense_data.append(dict(data))
    dense = hye_features(np.concatenate([data['shifts'][1:] - data['shifts'][0]
                                        for data in dense_data]))
    blocks = {
        'theta_blocks': np.clip(np.floor((labels[:, 1] + 1 + 1e-8) / .4), 0, 4),
        'y_blocks': np.clip(np.floor((labels[:, 0] + 2 + 1e-8) / .8), 0, 4),
        'day_holdout': np.where(sessions == 1, 1, 0),
    }
    metrics = {}
    with (RESULTS / 'rig-hye-holdout.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split','capture','file','y_mm','theta_deg','baseline_y','baseline_theta',
                         'hye_y','hye_theta','dense_y','dense_theta',
                         'same_training_y','same_training_theta'])
        for split, folds in blocks.items():
            predictions = np.zeros((len(labels), 4, 2))
            for fold in np.unique(folds):
                test = (folds == fold) & (sessions < 2)
                previous_train = folds != fold
                train = previous_train & (sessions < 2)
                predictions[test, 0] = baseline[test] @ fit(baseline[previous_train], labels[previous_train])
                beta, bias = fit_hye(hye[train], labels[train], sessions[train])
                predictions[test, 1] = hye[test] @ beta + bias
                active = sessions < 2
                beta, bias = fit_hye(dense[train[active]], labels[train], sessions[train])
                predictions[test, 2] = dense[test[active]] @ beta + bias
                predictions[test, 3] = baseline[test] @ fit(baseline[train], labels[train])
            current = sessions < 2
            metrics[split] = {key: stats(predictions[current, i], labels[current])
                              for i, key in enumerate(('baseline','hye','hye_dense','baseline_same_training'))}
            for i in np.flatnonzero(current):
                writer.writerow([split, CAPTURES[sessions[i]], names[i], *labels[i],
                                 *predictions[i].ravel()])
    use = sessions < 2
    beta, bias = fit_hye(dense, labels[use], sessions[use])
    # The existing calibration YAML carries the actual QR templates for each capture.
    for name, data in zip(CAPTURES, dense_data):
        filename = 'qr_pose.yml' if name == 'rig_grid_rev' else 'qr_pose_rig_grid.yml'
        source = cv.FileStorage(str(ROOT / 'config' / filename), cv.FILE_STORAGE_READ)
        try:
            if not source.isOpened():
                raise ValueError(f'Cannot read calibration {filename}')
            templates = source.getNode('patches')
            np.savez_compressed(RESULTS / f'{name}-hye-model.npz', beta=beta, bias=bias,
                                points=source.getNode('points').mat(),
                                coarse=source.getNode('coarse').mat(),
                                dense_refs=data['refs'], dense_points=data['points'],
                                dense_groups=data['group_lengths'], dense_zero=data['shifts'][0],
                                templates=np.stack([templates.at(i).mat() for i in range(templates.size())]))
        finally:
            source.release()
    (RESULTS / 'rig-hye-holdout.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    for split, methods in metrics.items():
        for method, s in methods.items():
            print(split, method, s['joint_pass_y_0045_theta_001'], s['rmse'], flush=True)


def load_estimator(path):
    with np.load(path) as model:
        beta, bias = model['beta'], model['bias']
        points, templates = model['points'], model['templates']
        coarse = model['coarse']
        dense_refs, dense_zero = model['dense_refs'], model['dense_zero']
        dense_points = np.split(model['dense_points'], np.cumsum(model['dense_groups'])[:-1])

    def estimate(frame):
        if frame.shape != (2160,3840) or frame.dtype != np.uint8:
            raise ValueError('Expected a rectified 3840x2160 grayscale frame')
        global_offset = np.rint(coarse_match(frame, coarse, 1907, 1130, 140, 22)).astype(int)
        tracked = [coarse_match(frame, template, x, y, 9, 7) + global_offset
                   for (x, y), template in zip(points + global_offset, templates)]
        offsets = np.median(np.asarray(tracked).reshape(3, 14, 2), axis=1)
        dense = dense_measure(frame, dense_refs, dense_points, offsets)
        return (hye_features((dense - dense_zero)[None]) @ beta + bias)[0]
    return estimate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict', type=Path)
    parser.add_argument('--extract-dense', action='store_true')
    parser.add_argument('--model', type=Path, default=RESULTS / 'rig_grid_rev-hye-model.npz')
    args = parser.parse_args()
    if args.predict:
        frame = cv.imdecode(np.frombuffer(args.predict.read_bytes(), np.uint8), cv.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f'Cannot decode {args.predict}')
        print(json.dumps(dict(zip(('y_mm','theta_deg'), load_estimator(args.model)(frame)))))
    elif args.extract_dense:
        extract_dense()
    else:
        evaluate()
