"""Saved-fold image-model consistency diagnostics; never changes predictions."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib

H=Path(__file__).resolve().parent;R=H.parent;OUT=H/'qa';sys.path.insert(0,str(R))
from predict import qr_grid
from parallax.extractor import NAMES

def main():
    a=np.load(H/'features.npz');q=np.load(R/'direct/features.npz');p=pd.read_csv(R/'parallax_predictions.csv');p=p[(p.method=='qr_external')&(p.protocol=='development_nested_oof')].set_index('index');regions=['QR_left','QR_center','QR_right',*NAMES];records=[]
    for fold in range(4):
        b=joblib.load(R/f'qr_external_fold{fold}.joblib');decoder=b['model'];f=decoder.model;assert b['config']=={'measurement':'affine_g5','head':'linear'}
        for i,row in p[p.fold==fold].iterrows():
            D=np.concatenate([qr_grid(q['H_homography_g5'][i]),a['D_affine_g5'][i]]);X=decoder.features(D)[0];pred=row[['predY','predT']].to_numpy(float);expect=f.origin+(pred/f.scale)@f.coef;r=(X-expect)[:D.size].reshape(D.shape);z=(r.ravel()/f.std[:D.size]).reshape(D.shape)
            for k,name in enumerate(regions):
                rs=r[k*9:(k+1)*9];zs=z[k*9:(k+1)*9];records.append(dict(index=int(i),fold=fold,file=row['file'],region=name,raw_pixel_rms=float(np.sqrt(np.mean(rs**2))),standardized_rms=float(np.sqrt(np.mean(zs**2))),mean_dx=float(rs[:,0].mean()),mean_dy=float(rs[:,1].mean()),Y_absolute_error=float(row.eY),T_absolute_error=float(row.eT)))
    df=pd.DataFrame(records);df.to_csv(OUT/'all341_oof_region_residuals.csv',index=False,encoding='utf-8-sig');summary=df.groupby('region')[['raw_pixel_rms','standardized_rms']].agg(['median','mean','max']);summary.to_csv(OUT/'region_residual_summary.csv',encoding='utf-8-sig')
    selected=pd.read_csv(OUT/'reviewed_images.csv')['index'];print(df[df['index'].isin(selected)].groupby('index')[['raw_pixel_rms','standardized_rms']].max().to_string());print(summary.to_string())
    note={'diagnostic':'Forward observation residual at the saved OOF estimate, using that fold training-derived origin/coefficient/std only. No refit, target-conditioned adjustment or gating.','reference':'The scalar standardized RMS is a per-coordinate training residual normalization, not a calibrated probability or independence test. Coordinates from each fitted ROI are correlated.','rigidity_assumptions':['The five external patches are fixed physical structures in this installation. Their relation to the QR plate must remain stable enough for the learned image-motion mapping.','They need not lie on the QR plane; differing image-motion sensitivities are the proposed source of additional information. No depth or physical parallax is measured.','If a bracket, handle, wall panel or camera mount changes relative to the calibrated setup, a previously learned relationship can fail. A high correlation between the same visible bolt and its template would not detect all such physical changes.','An isolated large region residual can indicate disagreement with the learned multi-region motion, but can also arise from real geometry, photometric changes or model approximation. No rejection threshold is validated here.','No independently measured mechanical rigidity or external component motion is available. The current image review cannot prove that the hardware is rigid across future captures.']}
    (OUT/'rigidity_and_residual_notes.json').write_text(json.dumps(note,indent=2),encoding='utf-8')

if __name__=='__main__':main()
