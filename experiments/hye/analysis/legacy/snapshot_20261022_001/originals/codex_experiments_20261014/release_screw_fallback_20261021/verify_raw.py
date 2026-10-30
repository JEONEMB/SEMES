from pipeline import *
import shutil,time,concurrent.futures,pandas as pd
import zero_experiment as Z
sys.path.insert(0,str(B/'multiref_4fold_20261021'));from experiment import fit,enrich,stats
source=B/'multiref_4fold_20261021/screw_fallback';before=sha(source/'c12_model.json')
shutil.copy2(source/'c12_model.json',H/'model.json');shutil.copy2(B/'center_patch_v2_20261021/features.npz',H/'qr_features.npz')
f=np.load(B/'measurement/features.npz');sel=f['regions']>=3;np.savez(H/'screw_features.npz',points=f['points'][sel],regions=f['regions'][sel],strengths=f['strengths'][sel])
(H/'reference.json').write_text(json.dumps(dict(sha256=sha(Z.ROOT/'center.bmp'),rule='same fixed calibration center',feature_scope='QR630+39 points on2 screws'),indent=2))
names=np.load(B/'center_patch_v2_20261021/tracks.npz')['names'];m=Z.manifest(names);ids=np.flatnonzero(m.axis!='C');ix={n:i for i,n in enumerate(names)};ext=Predictor(Z.ROOT/'center.bmp');paths=[Z.ROOT/('center.bmp' if n=='center.bmp' else ('Y/' if n[0]=='Y' else 'Theta/')+n) for n in names];start=time.time();records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for i,result in enumerate(pool.map(ext.extract,paths)):
        records.append(result)
        if i%40==0:print('raw',i,round(time.time()-start,1),flush=True)
X=np.stack([r[0] for r in records]);np.savez(H/'raw_features.npz',names=names,X=X,qr_valid=np.array([r[1] for r in records]),screw_valid=np.array([r[2] for r in records]))
oldX=np.load(source/'gn17_screws_mean_features.npy');assert np.isfinite(X).all()
folds=json.loads((source/'splits.json').read_text());out=[]
for fold in folds:
    tr=np.array([ix[n] for n in fold['train']]);te=np.array([ix[n] for n in fold['test']]);model=fit(oldX,m,tr,'inferred_session',True,.001)
    branches,W=model;(H/f'fold{fold["fold"]}_model.json').write_text(json.dumps(dict(W=W.tolist(),branches={k:[v[0],*[a.tolist() for a in v[1:]]] for k,v in branches.items()})))
    out.append(enrich(Z.predict(m,X,te,model,'inferred_session')).assign(protocol='raw_oof4',fold=fold['fold']))
model=( {k:(v[0],*[np.array(a) for a in v[1:]]) for k,v in ext.model['branches'].items()},np.array(ext.model['W']))
out.append(enrich(Z.predict(m,X,ids,model,'inferred_session')).assign(protocol='raw_resub280',fold=-1));P=pd.concat(out);P.to_csv(H/'raw_predictions.csv',index=False)
S=pd.DataFrame([dict(protocol=k,**stats(g)) for k,g in P.groupby('protocol')]);S.to_csv(H/'raw_summary.csv',index=False)
pd.DataFrame([dict(protocol=k,fold=f,**stats(g)) for (k,f),g in P.groupby(['protocol','fold'])]).to_csv(H/'raw_fold_results.csv',index=False)
neutral=H/'unlabeled.bmp';shutil.copy2(Z.ROOT/'Theta/T-0.05.bmp',neutral);r=ext.predict(neutral);neutral.unlink();truth=P[(P.protocol=='raw_resub280')&(P.file=='T-0.05.bmp')].iloc[0];assert r['axis']==truth.pred_axis and abs(r['value']-truth.pred)<1e-9
audit=dict(source_model_preserved=before==sha(source/'c12_model.json'),max_cached_vs_raw_feature_difference=float(abs(X-oldX).max()),all_qr_valid=all(q==630 for _,q,s in records),all_screw_valid=all(s==39 for _,q,s in records),raw_seconds=time.time()-start,neutral_filename_inference_matches=True,neutral_example=r,scope='post-selection fixed setting; independent new-run generalization NOT established; QR-only goal remains unmet')
(H/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(S.to_string(index=False));print(json.dumps(audit,ensure_ascii=False))
assert all(S.failures==0)
