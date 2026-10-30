"""Read-only image-correspondence and saved inference QA after model selection."""
from pathlib import Path
import json,sys,hashlib,importlib.util
import cv2,numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits

H=Path(__file__).resolve().parent;REBOOT=H.parent;OUT=H/'qa';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(REBOOT))
from predict import Predictor,qr_grid
from parallax.extractor import BOXES,NAMES

def main():
    cv2.setNumThreads(1);m=pd.read_csv(H/'manifest.csv').set_index('index');a=np.load(H/'features.npz');q=np.load(REBOOT/'direct/features.npz');allp=pd.read_csv(REBOOT/'parallax_predictions.csv');p=allp[(allp.method=='qr_external')&(allp.protocol=='development_nested_oof')].set_index('index');resub=allp[(allp.method=='qr_external')&(allp.protocol=='training_reapplication')].set_index('index')
    selected=[(int(p.eY.idxmax()),'largest absolute Y error'),(int(p.eT.idxmax()),'largest absolute T error'),(int(p.Y.idxmax()),'maximum commanded Y'),(int(p.Y.idxmin()),'minimum commanded Y'),(100,'old positive T extreme'),(int(p['T'].idxmin()),'negative T extreme'),(int(p['T'].idxmax()),'maximum commanded T'),(336,'previous model worst mixed image')]
    selected=list(dict(selected).items());report=[]
    for i,reason in selected:
        row=p.loc[i].to_dict();row.update(index=i,review_reason=reason);report.append(row)
    pd.DataFrame(report).to_csv(OUT/'reviewed_images.csv',index=False,encoding='utf-8-sig')
    # Every crop is untouched uint8 source pixels at the NCC integer offset.
    # No ECC warping, intensity normalization, resizing or training transform.
    for page,start in enumerate(range(0,len(selected),4),1):
        cases=[(0,'fixed reference')]+selected[start:start+4];w=1130;rowheight=205;canvas=np.full((75+rowheight*len(cases),w,3),255,np.uint8)
        cv2.putText(canvas,'Original-pixel correspondence QA: 1 source pixel = 1 output pixel',(12,22),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(canvas,'Current crops use fixed ROI + integer NCC shift only. No ECC warp / brightness changes.',(12,44),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,0,0),1,cv2.LINE_AA)
        for row,(i,reason) in enumerate(cases):
            im=cv2.imdecode(np.fromfile(str(m.loc[i,'path']),np.uint8),0);y=75+row*rowheight;x=245
            cv2.putText(canvas,f'index {i}: {m.loc[i,"file"]}',(10,y+22),cv2.FONT_HERSHEY_SIMPLEX,.37,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(canvas,reason,(10,y+45),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1,cv2.LINE_AA)
            if i:
                cv2.putText(canvas,f'eY {p.loc[i,"eY"]:.5f} eT {p.loc[i,"eT"]:.5f}',(10,y+68),cv2.FONT_HERSHEY_SIMPLEX,.37,(0,0,0),1,cv2.LINE_AA)
            for k,((x0,y0,x1,y1),name) in enumerate(zip(BOXES,NAMES)):
                dx,dy=a['ncc_shifts'][i,k];crop=im[y0+dy:y1+dy,x0+dx:x1+dx];hh,ww=crop.shape;canvas[y+20:y+20+hh,x:x+ww]=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
                cv2.putText(canvas,f'ROI{k+1}',(x,y+12),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1,cv2.LINE_AA)
                if i:cv2.putText(canvas,f'NCC {a["ncc_scores"][i,k]:.3f} ECC {a["ecc_affine_g5"][i,k]:.3f}',(x,y+190),cv2.FONT_HERSHEY_SIMPLEX,.29,(0,0,0),1,cv2.LINE_AA)
                x+=ww+18
        cv2.imwrite(str(OUT/f'contact_sheet_{page}_1x.png'),canvas)
    # All341 ROI quality and fit-family differences accompany selected-image QA.
    perROI=[]
    for i in range(len(m)):
        for k,name in enumerate(NAMES):
            rr=a['regions']==k;delta=a['D_affine_g5'][i,rr]-a['D_translation_g5'][i,rr]
            perROI.append(dict(index=i,roi=name,ncc=float(a['ncc_scores'][i,k]),ecc_affine=float(a['ecc_affine_g5'][i,k]),rmse_affine=float(a['rmse_affine_g5'][i,k]),rmse_translation=float(a['rmse_translation_g5'][i,k]),affine_vs_translation_grid_rms=float(np.sqrt(np.mean(delta**2))),affine_determinant=float(np.linalg.det(a['H_affine_g5'][i,k,:2,:2])),dx=float(a['centroidD_affine_g5'][i,k,0]),dy=float(a['centroidD_affine_g5'][i,k,1])))
    pd.DataFrame(perROI).to_csv(OUT/'all341_roi_quality.csv',index=False,encoding='utf-8-sig')
    # Live raw-image API vs cached features and stored predictions, full-fit and OOF.
    pred=Predictor('qr_external');saved_bundle=pred.bundle;checks=[];paths=[REBOOT/'qr_external_model.joblib']+[REBOOT/f'qr_external_fold{f}.joblib' for f in range(4)];hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    for i in [0,int(p.eY.idxmax()),int(p.eT.idxmax()),336]:
        raw=Path(m.loc[i,'path']).read_bytes();fresh=pred.predict_bytes(raw);cacheD=np.concatenate([qr_grid(q['H_homography_g5'][i]),a['D_affine_g5'][i]])
        cached=saved_bundle['model'].predict(cacheD[None])[0];actual=np.array([fresh['Y'],fresh['T']]);stored=resub.loc[i,['predY','predT']].to_numpy(float)
        fold=int(p.loc[i,'fold']);pred.bundle=joblib.load(REBOOT/f'qr_external_fold{fold}.joblib');foldfresh=pred.predict_bytes(raw);fp=np.array([foldfresh['Y'],foldfresh['T']]);foldstored=p.loc[i,['predY','predT']].to_numpy(float);pred.bundle=saved_bundle
        check=dict(index=i,file=m.loc[i,'file'],fullfit_actual_api=actual.tolist(),fullfit_cached_feature_prediction=cached.tolist(),fullfit_saved_prediction=stored.tolist(),fullfit_api_vs_cache_max_difference=float(np.max(abs(actual-cached))),fullfit_api_vs_saved_max_difference=float(np.max(abs(actual-stored))),fold=fold,fold_actual_api=fp.tolist(),fold_saved_prediction=foldstored.tolist(),fold_api_vs_saved_max_difference=float(np.max(abs(fp-foldstored))),image_sha256=fresh['image_sha256'],hash_matches_manifest=fresh['image_sha256']==m.loc[i,'sha256'])
        assert max(check['fullfit_api_vs_cache_max_difference'],check['fullfit_api_vs_saved_max_difference'],check['fold_api_vs_saved_max_difference'])<1e-8;assert check['hash_matches_manifest'];checks.append(check)
    for f,digest in hashes.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest
    (OUT/'api_verification.json').write_text(json.dumps(dict(samples=checks,model_hashes_before_after_equal=True,model_hashes=hashes,no_model_or_ROI_changed=True,limits=['Fresh raw-image inference is compared against existing development predictions; it is not a new independent accuracy test.','The full-fit and corresponding outer-fold models are explicitly separated. The API selects no model using a sample label.','Raw bytes are the only prediction input; file names are used in this QA report for identification only.','No robot actuation is performed.']),indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2));print(pd.DataFrame(perROI).groupby('roi')[['ncc','ecc_affine','rmse_affine','affine_vs_translation_grid_rms','affine_determinant']].agg(['min','median','max']).to_string())

if __name__=='__main__':
    with threadpool_limits(1):main()
