"""Compare repeat variability at equal geometric capacity; no model selection."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,cv2
from scipy.fft import fft2,ifft2
from scipy.ndimage import fourier_shift
from extractor import DirectExtractor,project,CANDIDATES,CENTER_PATH

H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'codex_experiments_20261014/joint_yt_generalization_20261022'))
from data import load

def main():
    cv2.setNumThreads(1);m,old=load();a=np.load(H/'features.npz');points=a['points'];regions=a['regions'];oldD=old['D'];smoothH=np.full(oldD.shape,np.nan);smoothA=np.full(oldD.shape,np.nan)
    for i in range(len(m)):
        for qr in range(3):
            rr=regions==qr;dst=points[rr]+oldD[i,rr];HH,_=cv2.findHomography(points[rr],dst,method=0);smoothH[i,rr]=project(HH,points[rr])-points[rr]
            X=np.c_[points[rr],np.ones(rr.sum())];coef=np.linalg.lstsq(X,dst,rcond=None)[0];smoothA[i,rr]=X@coef-points[rr]
    methods={'old_boundary630':oldD,'old_boundary630_projected_H':smoothH,'old_boundary630_projected_affine':smoothA,**{n:a['D_'+n] for n,_,_ in CANDIDATES}}
    rows=[]
    for name,D in methods.items():
        for i in np.flatnonzero(m.dataset.to_numpy()==20261020):
            cond=np.flatnonzero((m.dataset.to_numpy()==20261020)&(m.condition.to_numpy()==m.condition.iloc[i]));delta=D[i]-D[cond].mean(0);ds=delta-delta.mean(0)
            qrvar=[]
            for qr in range(3):
                rr=regions==qr;qrvar.extend((delta[rr]-delta[rr].mean(0)).ravel())
            rows.append(dict(method=name,index=int(i),file=m.file.iloc[i],condition=m.condition.iloc[i],repeat_coordinate_rms=float(np.sqrt(np.mean(delta**2))),repeat_global_centered_rms=float(np.sqrt(np.mean(ds**2))),repeat_within_qr_centered_rms=float(np.sqrt(np.mean(np.array(qrvar)**2)))))
    df=pd.DataFrame(rows);df.to_csv(H/'repeat_diagnostics.csv',index=False,encoding='utf-8-sig');summary=df.groupby('method')[['repeat_coordinate_rms','repeat_global_centered_rms','repeat_within_qr_centered_rms']].agg(['median','mean','max']);summary.to_csv(H/'repeat_summary.csv',encoding='utf-8-sig')
    # Fourier translation avoids generating data with the ECC bilinear sampler.
    ex=DirectExtractor();sanity=[];spectrum=fft2(ex.center.astype(np.float32))
    cases=[('fourier_translation_1',np.array([.13,.37])),('fourier_translation_2',np.array([.73,-.21]))]
    for label,shift in cases:
        test=np.clip(np.rint(ifft2(fourier_shift(spectrum,(shift[1],shift[0]))).real),0,255).astype(np.uint8);ok,encoded=cv2.imencode('.png',test);assert ok
        result=ex.extract_bytes(encoded.tobytes());metrics={n:dict(success=c['success'],mean_measured_displacement=c['D'].mean(0).tolist(),mean_error=(c['D'].mean(0)-shift).tolist(),coordinate_error_rms=float(np.sqrt(np.mean((c['D']-shift)**2))),coordinate_error_max=float(np.max(abs(c['D']-shift)))) for n,c in result['candidates'].items()};sanity.append(dict(name=label,expected_shift=shift.tolist(),generator='Fourier phase shift, clipped and quantized uint8',candidates=metrics))
    A=np.array([[1.0002,.0003],[-.0002,.99985]]);pivot=points.mean(0);t=pivot-A@pivot+np.array([.37,-.23]);M=np.c_[A,t];expected=points@A.T+t-points
    test=cv2.warpAffine(ex.center,M,ex.center.shape[::-1],flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101);test=np.clip(np.rint(test.astype(float)*.95+4),0,255).astype(np.uint8);ok,encoded=cv2.imencode('.png',test);assert ok
    result=ex.extract_bytes(encoded.tobytes());metrics={n:dict(success=c['success'],mean_measured_displacement=c['D'].mean(0).tolist(),mean_error=(c['D']-expected).mean(0).tolist(),coordinate_error_rms=float(np.sqrt(np.mean((c['D']-expected)**2))),coordinate_error_max=float(np.max(abs(c['D']-expected)))) for n,c in result['candidates'].items()};sanity.append(dict(name='small_affine_plus_brightness',expected_affine=M.tolist(),generator='OpenCV cubic warp, gain0.95 offset4, uint8 quantization',candidates=metrics))
    centers={n:float(np.max(abs(a['D_'+n][0]))) for n,_,_ in CANDIDATES}
    verify=dict(n=341,index_equal=bool(np.array_equal(a['index'],m.index)),reference_points_equal=bool(np.array_equal(a['points'],old['points'])),fixed_center_max_displacement=centers,all_candidate_finite={n:bool(np.isfinite(a['D_'+n]).all()) for n,_,_ in CANDIDATES},synthetic_test=sanity,limits=['All synthetic images originate from the same fixed center. Fourier generation differs from ECC bilinear sampling but does not test real optics, mechanical pose errors, independent images or unseen QR patterns.','Fourier shift assumes periodic band-limited images; clipping/quantization and QR edges introduce discrepancies. Cubic affine generation also shares ideal warp assumptions with the estimator.','Synthetic measurement bias is a pixel registration diagnostic, not Y/T physical pose accuracy.','Old boundary homography/affine projections are post-hoc comparison controls only; they were never used by the new pixel extractor.','Repeat variability includes any real machine repeat-pose difference and systematic image change; lower variance is not proof of greater pose accuracy.'])
    assert verify['index_equal'] and verify['reference_points_equal']
    (H/'verification.json').write_text(json.dumps(verify,indent=2),encoding='utf-8');print(summary.to_string());print(json.dumps(verify,indent=2))

if __name__=='__main__':main()
