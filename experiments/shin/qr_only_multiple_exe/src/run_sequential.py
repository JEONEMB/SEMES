"""Run the QR sequential controller on the rig's Linux camera/stage stack."""
import argparse
import json
import os
import sys
from time import perf_counter_ns
from pathlib import Path

if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'

import cv2 as cv
import numpy as np

from sequential import Settings, converge
from qr_observer import qr_observer


def read_reference(path):
    frame = cv.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv.IMREAD_GRAYSCALE)
    if frame is None or frame.shape != (2160, 3840):
        raise ValueError('Reference must be a rectified 3840x2160 center image')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Local rig checkout on the Pi')
    parser.add_argument('--calib', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Verified center image')
    parser.add_argument('--center-y-mm', type=float, required=True)
    parser.add_argument('--center-theta-deg', type=float, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--hye-model', type=Path,
                        help='Optional Rig Hye model calibrated for the supplied center image')
    parser.add_argument('--pose-model',type=Path,
                        help='Rig QR model to measure in shadow mode before the first move')
    parser.add_argument('--apply-pose',action='store_true',
                        help='Use the shadow estimate for the optional first move')
    parser.add_argument('--transport',choices=('driver','serial'),default='driver')
    parser.add_argument('--zoom',type=float,default=1.)
    args = parser.parse_args()
    if args.apply_pose and not args.pose_model:
        parser.error('--apply-pose requires --pose-model')
    if args.hye_model and args.pose_model:
        parser.error('Choose one optional mixed estimator')
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    # Use the rig's actual driver, rectifier and capture implementation.
    from demo_web.stage import make_stage
    from verify_web.camera import PiCamera
    from verify_web.geometry import Calibration, Rectifier, ViewMapper

    cv.setNumThreads(1)
    reference = read_reference(args.reference)
    measure = qr_observer(reference)
    zero = measure(reference)
    estimate = None
    shadow = None
    if args.pose_model:
        from ssafy_qr_runtime import QRPose
        shadow = QRPose(args.pose_model)
        if not all(np.array_equal(reference[y-101:y+102,x-101:x+102],roi)
                   for (x,y),roi in zip(shadow.centers,shadow.patches)):
            raise ValueError('The supplied reference does not match the Rig pose model')
    if args.hye_model:
        from rig_hye import load_estimator
        estimate = load_estimator(args.hye_model)
        with np.load(args.hye_model) as model:
            from rig_hye import QR_CENTERS, patch
            if not all(np.array_equal(patch(reference, x, y, 101), roi)
                       for (x, y), roi in zip(QR_CENTERS, model['dense_refs'])):
                raise ValueError('Hye model and --reference must use the same calibration center')
    calib = Calibration.load(args.calib)
    rectifier = Rectifier(calib,args.zoom)
    stage = make_stage('hardware',args.transport)
    settings = Settings(round(args.center_y_mm*1000), round(args.center_theta_deg*1000))
    try:
        camera = PiCamera(ViewMapper(calib,args.zoom), exposure_us=8333, gain=3.0, fps=10)
        try:
            return run(stage,camera,rectifier,settings,measure,zero,args.log,estimate,shadow,args.apply_pose)
        finally:
            camera.close()
    finally:
        stage.close()


def run(stage,camera,rectifier,settings,measure,zero,log,estimate=None,shadow=None,apply_pose=False):
    with log.open('x',encoding='utf-8') as output:
        def record(entry):
            line = json.dumps(entry)
            print(line, flush=True)
            output.write(line+'\n')
            output.flush()

        def current_frame():
            settled, _, _ = camera.wait_settled(.25, timeout=6)
            if not settled:
                raise RuntimeError('Stage image did not settle')
            frame, _ = camera.grab(4)
            return rectifier.apply(frame)

        def observe():
            frame = current_frame()
            start = perf_counter_ns()
            value = measure(frame)-zero
            record({'kind':'qr_frame','inference_ms':(perf_counter_ns()-start)/1e6})
            return value

        camera.start()
        if shadow is not None:
            frame = current_frame()
            start = perf_counter_ns()
            value = shadow.predict(frame)
            record({'kind':'rig_pose_shadow','y_mm':float(value[0]),'theta_deg':float(value[1]),
                    'inference_ms':(perf_counter_ns()-start)/1e6,'applied':apply_pose})
            if apply_pose:
                estimate = lambda _: value
        result = converge(stage, observe, settings, record,
                          estimate_mixed=(lambda: estimate(current_frame())) if estimate else None)
        record({'kind': 'result', **result})
        return 0 if (result['result'] == 'qr_aligned' and result['status_y_within']
                     and result['status_theta_within']) else 1


if __name__ == '__main__':
    sys.exit(main())
