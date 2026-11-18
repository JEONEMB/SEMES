"""Standalone single-frame QR pose estimation for the calibrated SSAFY setup."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter_ns

if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'

import cv2 as cv
import numpy as np

from qr_tracking import dense_measure, initial_offset


TOLERANCE = np.array([.045,.01])
SIZE = (2160,3840)


def read_frame(path):
    image = cv.imdecode(np.frombuffer(Path(path).read_bytes(),np.uint8), cv.IMREAD_GRAYSCALE)
    if image is None or image.shape != SIZE:
        raise ValueError('Expected a rectified 3840x2160 image')
    return image


def load_configuration(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict):
        raise ValueError('Configuration must be a JSON object')
    values = [data.get(key) for key in ('y_mm','theta_deg')]
    if any(type(value) not in (int,float) for value in values) or not np.isfinite(values).all():
        raise ValueError('Set finite numeric y_mm and theta_deg in the configuration')
    return data, np.array(values,dtype=float)


def describe(shifts, ends, polynomials):
    groups = np.split(shifts,ends[:-1],axis=1)
    field = np.stack([np.einsum('kp,npd->nkd',b,g) for b,g in zip(polynomials,groups)],axis=1)
    common = shifts.mean(axis=1)
    field[:,:,0] -= common[:,None]
    return common,field.reshape(len(shifts),-1)


def embedding(common, structure, raw, kind):
    coefficients = {'track':(.5,.1,.5,.5),'shape':(1.,.2,1.,0.),
                    'soft_shape':(1.,.03,.1,0.),'motion':(1.,.08,0.,0.)}
    x,v,g,r = np.sqrt(coefficients[kind])
    n = np.sqrt(structure.shape[1])
    blocks = [x*common[:,:1]/10.,v*common[:,1:]/.12]
    if g > 0:
        blocks.append(g*structure/(.05*n))
    if r > 0:
        blocks.append(r*(raw-common[:,None]).reshape(len(raw),-1)/(.12*n))
    return np.column_stack(blocks)


def blend_kernel(distance, width, wide_weight):
    return ((1-wide_weight)*np.exp(-.5*distance/width**2)
            + wide_weight*np.exp(-.5*distance/36.))


def origin_kernel(vectors, width, wide_weight):
    return blend_kernel(np.einsum('ij,ij->i',vectors,vectors),width,wide_weight)


class QRPose:
    """Estimate Y and theta relative to the calibration center or a configured target."""

    def __init__(self, model_path, target=(0.,0.)):
        cv.setNumThreads(1)
        self.target = np.asarray(target,dtype=float)
        if self.target.shape != (2,) or not np.isfinite(self.target).all():
            raise ValueError('Target must contain finite Y and theta')
        with np.load(model_path,allow_pickle=False) as model:
            if not bool(model['fast_observer']) or len(model['reference_ids']) != 1:
                raise ValueError('This runtime expects the single-reference fast QR model')
            self.ends = model['ends']
            self.polynomials = [model[f'poly_{i}'] for i in range(len(self.ends))]
            self.centers, self.patches = model['0_centers'],model['0_patches']
            self.points = np.split(model['0_points'],self.ends[:-1])
            self.inverse, self.zero = model['0_inverse'],model['0_zero']
            self.coarse = model['0_sparse_1']
            self.width, self.kind = float(model['width']),str(model['kind'])
            self.beta, self.anchors = model['beta'],model['anchors']
            self.rotate, self.baseline = model['rotate'],model['horizontal_fit']
            self.wide_weight = float(model['wide_weight'])
            self.anchored = bool(model['anchor_origin'])
        self.anchor_norm = np.einsum('ij,ij->i',self.anchors,self.anchors)
        self.zero_kernel = origin_kernel(self.anchors,self.width,self.wide_weight)

    def predict(self, frame):
        if frame.shape != SIZE or frame.dtype != np.uint8:
            raise ValueError('Expected a rectified 3840x2160 uint8 grayscale frame')
        dx,dy = initial_offset(frame,self.coarse,*self.centers[1])
        displacement = dense_measure(frame,self.patches,self.points,[(dx,dy)]*3,self.centers)
        groups = np.split(displacement,self.ends[:-1])
        shifts = (np.concatenate([g@matrix for g,matrix in zip(groups,self.inverse)])-self.zero)[None]
        common, shape = describe(shifts,self.ends,self.polynomials)
        v = embedding(common,shape,shifts,self.kind)[0]
        d = np.maximum(0., self.anchor_norm + v@v - 2.*np.einsum('ij,j->i',self.anchors,v))
        kernel = blend_kernel(d,self.width,self.wide_weight)
        if self.anchored:
            kernel -= origin_kernel(v[None],self.width,self.wide_weight)[0]*self.zero_kernel
        z = kernel@self.beta
        x = common[0,0]
        z[0] += np.array([1.,x,x*x/100.])@self.baseline
        return (z@self.rotate.T)*TOLERANCE-self.target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images',type=Path,nargs='+')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1] / 'config' / 'ssafy.json')
    args = parser.parse_args()
    try:
        config, target = load_configuration(args.config)
        model = Path(config.get('model','ssafy_qr_pose.npz'))
        predictor = QRPose(args.config.parent/model,target)
        for path in args.images:
            frame = read_frame(path)
            start = perf_counter_ns()
            y, theta = predictor.predict(frame)
            print(json.dumps({'image':str(path),'y_mm':float(y),'theta_deg':float(theta),
                              'inference_ms':(perf_counter_ns()-start)/1e6}))
    except (ValueError,OSError,KeyError,TypeError,cv.error) as exc:
        parser.exit(1,f'QR pose: {exc}\n')


if __name__ == '__main__':
    main()
