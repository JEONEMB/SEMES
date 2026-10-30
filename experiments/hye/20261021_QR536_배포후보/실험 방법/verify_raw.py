from pipeline import *
import shutil,time,concurrent.futures,pandas as pd
import zero_experiment as Z
sys.path.insert(0,str(B/'multiref_4fold_20261021'));from experiment import fit,enrich,stats
source=B/'multiref_4fold_20261021/structural_regions';before=sha(source/'c45_model.json')
shutil.copy2(source/'c45_model.json',H/'model.json');shutil.copy2(B/'center_patch_v2_20261021/features.npz',H/'features630.npz')
f=np.load(H/'features630.npz');a=json.loads((source/'selection_audit.json').read_text());used=next(r['ids'] for r in a['regions'] if r['region']=='interior');mask=np.isin(f['ids'],used);assert mask.sum()==536;np.save(H/'mask536.npy',mask)
(H/'reference.json').write_text(json.dumps(dict(sha256=sha(Z.ROOT/'center.bmp'),rule='actual decoded grid interior 0<grid_x<21 and 0<grid_y<21 after diagonal-touch exclusion',used_ids=used,points=536,excluded_outer_boundary=94),indent=2))
names=np.load(B/'center_patch_v2_20261021/tracks.npz')['names'];m=Z.manifest(names);ids=np.flatnonzero(m.axis!='C');ix={n:i for i,n in enumerate(names)};ext=Predictor(Z.ROOT/'center.bmp');paths=[Z.ROOT/('center.bmp' if n=='center.bmp' else ('Y/' if n[0]=='Y' else 'Theta/')+n) for n in names];start=time.time();records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for i,result in enumerate(pool.map(ext.extract,paths)):
        records.append(result)
        if i%40==0:print('raw QR-only',i,round(time.time()-start,1),flush=True)
X=np.stack([r[0] for r in records]);np.savez(H/'raw_features.npz',names=names,X=X,valid=np.array([r[1] for r in records]));oldX=np.load(source/'interior_features.npy');assert np.isfinite(X).all()
folds=json.loads((source/'splits.json').read_text());shutil.copy2(source/'splits.json',H/'splits.json');out=[]
for fold in folds:
    tr=np.array([ix[n] for n in fold['train']]);te=np.array([ix[n] for n in fold['test']]);assert len(tr)==210 and len(te)==70 and len(set(tr)&set(te))==0;assert sum(m.axis.iloc[te]=='Y')==20 and sum(m.axis.iloc[te]=='T')==50
    model=fit(oldX,m,tr,'inferred_session',True,.05);branches,W=model;(H/f'fold{fold["fold"]}_model.json').write_text(json.dumps(dict(W=W.tolist(),branches={k:[v[0],*[a.tolist() for a in v[1:]]] for k,v in branches.items()})))
    out.append(enrich(Z.predict(m,X,te,model,'inferred_session')).assign(protocol='raw_oof4',fold=fold['fold']))
model=({k:(v[0],*[np.array(a) for a in v[1:]]) for k,v in ext.model['branches'].items()},np.array(ext.model['W']))
out.append(enrich(Z.predict(m,X,ids,model,'inferred_session')).assign(protocol='raw_resub280',fold=-1));P=pd.concat(out);P.to_csv(H/'raw_predictions.csv',index=False)
S=pd.DataFrame([dict(protocol=k,**stats(g)) for k,g in P.groupby('protocol')]);S.to_csv(H/'raw_summary.csv',index=False);pd.DataFrame([dict(protocol=k,fold=f,**stats(g)) for (k,f),g in P.groupby(['protocol','fold'])]).to_csv(H/'raw_fold_results.csv',index=False)
neutral=H/'unlabeled.bmp';shutil.copy2(Z.ROOT/'Theta/T-0.05.bmp',neutral);r=ext.predict(neutral);neutral.unlink();truth=P[(P.protocol=='raw_resub280')&(P.file=='T-0.05.bmp')].iloc[0];assert r['axis']==truth.pred_axis and abs(r['value']-truth.pred)<1e-9
audit=dict(source_model_preserved=before==sha(source/'c45_model.json'),max_cached_vs_raw_feature_difference=float(abs(X-oldX).max()),all_points_valid=all(v==536 for _,v in records),raw_seconds=time.time()-start,neutral_filename_matches=True,neutral_example=r,scope='QR-only fixed setting after broad search; internal method selection in structural-region experiment279/280; not independent new-capture proof')
(H/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(S.to_string(index=False));print(json.dumps(audit,ensure_ascii=False));assert all(S.failures==0);assert P[P.protocol=='raw_oof4'].file.nunique()==280
