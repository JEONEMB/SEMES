"""Read-only measurement checks and development repeat-variance comparisons."""
from pathlib import Path
import json,sys
import cv2,numpy as np,pandas as pd
from scipy.fft import fft2,ifft2
from scipy.ndimage import fourier_shift
from shared_extractor import SharedExtractor,CANDIDATES,base

H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'codex_experiments_20261014/joint_yt_generalization_20261022'))
from data import load

def main():
    cv2.setNumThreads(1);m,_=load();a=np.load(H/'features.npz');old=np.load(H.parent/'features.npz');ex=SharedExtractor();rows=[]
    for origin,bank,names in [('per_qr',old,list(old['candidate_names'])),('shared',a,list(a['candidate_names']))]:
        for name in names:
            D=bank['D_'+name]
            for i in np.flatnonzero(m.dataset.to_numpy()==20261020):
                condition=(m.dataset.to_numpy()==20261020)&(m.condition.to_numpy()==m.condition.iloc[i]);delta=D[i]-D[condition].mean(0);ds=delta-delta.mean(0);qr=[]
                for k in range(3):
                    rr=ex.regions==k;qr.extend((delta[rr]-delta[rr].mean(0)).ravel())
                rows.append(dict(family=origin,method=name,index=int(i),repeat_coordinate_rms=float(np.sqrt(np.mean(delta**2))),repeat_global_centered_rms=float(np.sqrt(np.mean(ds**2))),repeat_within_qr_centered_rms=float(np.sqrt(np.mean(np.array(qr)**2)))))
    df=pd.DataFrame(rows);df.to_csv(H/'repeat_diagnostics.csv',index=False,encoding='utf-8-sig');summary=df.groupby('method')[['repeat_coordinate_rms','repeat_global_centered_rms','repeat_within_qr_centered_rms']].agg(['median','mean','max']);summary.to_csv(H/'repeat_summary.csv',encoding='utf-8-sig')
    # Geometry sanity uses no pose target or old boundary displacement.
    spectrum=fft2(ex.center.astype(np.float32));synthetic=[]
    for shift in [np.array([.13,.37]),np.array([.73,-.21])]:
        test=np.clip(np.rint(ifft2(fourier_shift(spectrum,(shift[1],shift[0]))).real),0,255).astype(np.uint8);ok,enc=cv2.imencode('.png',test);assert ok
        r=ex.extract_bytes(enc.tobytes());synthetic.append(dict(expected_shift=shift.tolist(),generator='Fourier shift, clip and uint8 quantization',metrics={n:dict(success=c['success'],mean_error=(c['D'].mean(0)-shift).tolist(),coordinate_rms=float(np.sqrt(np.mean((c['D']-shift)**2)))) for n,c in r['candidates'].items()}))
    repeated=[];raw=Path(m.iloc[336].path).read_bytes()
    for _ in range(2):repeated.append(ex.extract_bytes(raw))
    full_geometry={}
    for name,_,_ in CANDIDATES:
        maximum=0.
        for i in range(len(m)):
            pred=base.project(a['H_single_'+name][i],ex.points)-ex.points;maximum=max(maximum,float(np.nanmax(abs(pred-a['D_'+name][i]))))
        full_geometry[name]=maximum
    result=dict(n=len(m),all_finite={n:bool(np.isfinite(a['D_'+n]).all()) for n,_,_ in CANDIDATES},index_equal=bool(np.array_equal(a['index'],m.index)),reference_equal=bool(np.array_equal(a['points'],old['points'])),center_max_abs={n:float(np.max(abs(a['D_'+n][0]))) for n,_,_ in CANDIDATES},projected_coordinates_max_discrepancy=full_geometry,repeat_bytes_max_abs_difference={n:float(np.max(abs(repeated[0]['candidates'][n]['D']-repeated[1]['candidates'][n]['D']))) for n,_,_ in CANDIDATES},synthetic_translation_checks=synthetic,mean_photometric_rmse={n:float(np.mean(a['rmse_'+n])) for n,_,_ in CANDIDATES},limits=['Synthetic Fourier displacement is a measurement direction/bias check. It is not real Y/T motion or independent camera validation.','All shared transforms assume coherent common-plane geometry. Lower repeat variance may remove true differences as well as noise.','No method selected by this script; Y/T evaluation belongs to separate training-fold evaluation.'])
    assert result['index_equal'] and result['reference_equal']
    (H/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(summary.to_string());print(json.dumps(result,indent=2))

if __name__=='__main__':main()
