"""Pixel-access invariance and reference behavior; no Y/T target checks."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np,pandas as pd
from scipy.fft import fft2,ifft2
from scipy.ndimage import fourier_shift
from extractor import LineExtractor,PROTOCOL

H=Path(__file__).resolve().parent

def encoded(image):
    ok,data=cv2.imencode('.png',image);assert ok;return data.tobytes()

def main():
    cv2.setNumThreads(1);ex=LineExtractor();m=pd.read_csv(H/'manifest.csv');a=np.load(H/'features.npz');records=[];rng=np.random.default_rng(20261023)
    for i in [0,281,336]:
        raw=Path(m.iloc[i].path).read_bytes();im=cv2.imdecode(np.frombuffer(raw,np.uint8),0);base=ex.extract_bytes(raw)
        for mode in ['outside_zero','outside_random']:
            changed=np.zeros_like(im) if mode=='outside_zero' else rng.integers(0,256,im.shape,dtype=np.uint8);changed[ex.read_mask]=im[ex.read_mask];new=ex.extract_bytes(encoded(changed));keys=['measured_shifts','valid_blocks','rmse','correlation','gain','offset','contrast','X_upper','X_lower'];diff={k:float(np.max(abs(base[k].astype(float)-new[k].astype(float)))) if k!='valid_blocks' else int(np.sum(base[k]!=new[k])) for k in keys}
            for k in keys:assert np.array_equal(base[k],new[k],equal_nan=True)
            records.append(dict(index=i,alteration=mode,all_measurements_bit_identical=True,max_differences=diff,changed_pixels=int((~ex.read_mask).sum())))
            if i==336 and mode=='outside_zero':cv2.imwrite(str(H/'allowed_pixels_only_example.png'),changed)
    # Fourier vertical translation probes normal direction and subpixel behavior.
    # It does not verify robot motion or horizontal observability.
    spectrum=fft2(ex.center.astype(np.float32));synthetic=[]
    for delta in [.37,-.23]:
        image=np.clip(np.rint(ifft2(fourier_shift(spectrum,(delta,0))).real),0,255).astype(np.uint8);r=ex.extract_bytes(encoded(image));error=r['measured_shifts']-delta;synthetic.append(dict(expected_vertical_shift=delta,upper_error_rms=float(np.sqrt(np.mean(error[:24]**2))),lower_error_rms=float(np.sqrt(np.mean(error[24:]**2))),max_absolute_error=float(np.max(abs(error))),success_upper=r['success_upper'],success_lower=r['success_lower']))
    report=dict(n=341,X_upper_shape=list(a['X_upper'].shape),X_lower_shape=list(a['X_lower'].shape),read_pixels=int(ex.read_mask.sum()),total_image_pixels=int(ex.center.size),mask_fraction=float(ex.read_mask.mean()),center_max_abs_shift=float(np.max(abs(a['measured_shifts'][0]))),invariance=records,synthetic_vertical_checks=synthetic,forbidden_external_measurement_files_read=False,limits=['Only normal/vertical displacement is observed. The horizontal aperture ambiguity remains.','Fixed geometric line identity differs from tracking a unique material point along an infinitely long line.','Mask invariance tests prove dependence is restricted to the specified pixel union for these executions, not physical pose correctness.','Fourier-generated images share a source image with the reference and do not emulate real mechanical motion, shadow change or independent scenes.'])
    (H/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
