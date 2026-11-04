"""Prespecified raw intensity families on the existing grouped development split."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;GENERAL=H.parent/'joint_yt_generalization_20261022'
sys.path.insert(0,str(GENERAL))
from data import load
from metrics import metric
from pixel_models import PixelPLS,PixelCNN

def error_score(errors):
    e=np.asarray(errors);finite=np.isfinite(e).all(axis=1);ev=e[finite]
    return dict(failures=int((~finite).sum()+np.any(ev>.09+1e-12,axis=1).sum()),maximum=float(ev.max()) if len(ev) else 1e100,mean=float(ev.mean()) if len(ev) else 1e100)

def main():
    m,_=load();a=np.load(H/'pixel_features.npz');pixels=a['pixels'];shift=a['shift'];y=m[['Y','T']].to_numpy();good=a['success'];oldselect=json.loads((H.parent/'joint_yt_disentangle_20261022/selection.json').read_text());out=[];audit=[]
    configs=[4,12,32]
    protocol=dict(scope='ADAPTIVE DEVELOPMENT only; new image observations, not new photos',families=['rawpixel_PLS_inner_choice_4_12_32','rawpixel_shared_CNN_fixed350epochs'],architecture='3 shared QR intensity-difference branches;6x6 spatial grid retained;6 coarse coordinates kept; supervised2-output regression',selection='PLS components inner3groupCV failures,max,mean; CNN fixed architecture/epochs/seed no early stopping',augmentation='none; no invented mixed-pose labels',all_preprocessing='train-only feature centering/scaling and PLS/NN fit',data_failure_policy='Nonfinite measurements remain failures in full denominator')
    (H/'pixel_model_protocol.json').write_text(json.dumps(protocol,indent=2))
    for fold in range(5):
        tr=np.flatnonzero((m.fold.to_numpy()!=fold)&good) if fold<4 else np.flatnonzero(good);te=np.flatnonzero(m.fold.to_numpy()==fold) if fold<4 else np.arange(len(m));inner=oldselect[fold]['inner_splits'] if fold<4 else oldselect[-1]['full_selection']['inner_splits'];scores=[]
        for comp in configs:
            errors=[]
            for split in inner:
                it=np.array(split['train']);it=it[good[it]];iv=np.array(split['validation']);iv=iv[m.iloc[iv].dataset.to_numpy()==20261020]
                assert not set(m.iloc[it].cv_group)&set(m.iloc[iv].cv_group)
                model=PixelPLS(comp).fit(pixels[it],shift[it],y[it],m.iloc[it]);p=np.full((len(iv),2),np.nan);p[good[iv]]=model.predict(pixels[iv[good[iv]]],shift[iv[good[iv]]]);errors.extend(abs(p-y[iv]))
            scores.append(dict(components=comp,**error_score(errors)))
        chosen=min(scores,key=lambda r:(r['failures'],r['maximum'],r['mean']))['components'];audit.append(dict(fold=fold,inner_splits=inner,scores=scores,chosen=chosen))
        for name,model in [('pixel_pls',PixelPLS(chosen)),('pixel_cnn',PixelCNN())]:
            start=time.perf_counter();model.fit(pixels[tr],shift[tr],y[tr],m.iloc[tr]);valid=te[good[te]];pred=np.full((len(te),2),np.nan);pred[good[te]]=model.predict(pixels[valid],shift[valid]);r=m.iloc[te].copy();r['predY']=pred[:,0];r['predT']=pred[:,1];r['method']=name;r['protocol']='development_group_oof' if fold<4 else 'training_reapplication';r['eY']=abs(r.predY-r.Y);r['eT']=abs(r.predT-r['T']);r['passed']=(r.eY<=.09+1e-12)&(r.eT<=.09+1e-12);out.append(r)
            joblib.dump(dict(model=model,train_indices=tr,protocol=protocol),H/(f'{name}_fold{fold}.joblib' if fold<4 else f'{name}_model.joblib'))
            n=m.iloc[te].dataset.to_numpy()==20261020;print(fold,name,metric(y[te][n],pred[n]),'seconds',time.perf_counter()-start,flush=True)
        pd.concat(out).to_csv(H/'pixel_predictions_progress.csv',index=False)
    allp=pd.concat(out);allp.to_csv(H/'pixel_predictions.csv',index=False);(H/'pixel_selection.json').write_text(json.dumps(audit,indent=2))
    summary=[]
    for (method,protocol),g in allp.groupby(['method','protocol']):
        for ds in ['all',20261014,20261020]:
            z=g if ds=='all' else g[g.dataset==ds];summary.append(dict(method=method,protocol=protocol,dataset=ds,**metric(z[['Y','T']],z[['predY','predT']])))
    pd.DataFrame(summary).to_csv(H/'pixel_summary.csv',index=False);print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
