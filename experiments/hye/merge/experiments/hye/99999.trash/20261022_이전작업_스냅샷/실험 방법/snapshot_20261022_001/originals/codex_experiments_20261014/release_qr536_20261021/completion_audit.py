from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B),str(B/'multiref_4fold_20261021')]
import zero_experiment as Z
from experiment import fit
P=pd.read_csv(H/'raw_predictions.csv');rows=[]
for protocol,g in P.groupby('protocol'):
    assert len(g)==g.file.nunique()==280
    ok=(g.axis==g.pred_axis)&(np.sign(g.value)==np.sign(g.pred));ey=abs(np.where(g.axis=='Y',g.value,0)-np.where(g.pred_axis=='Y',g.pred,0));et=abs(np.where(g.axis=='T',g.value,0)-np.where(g.pred_axis=='T',g.pred,0))
    assert ok.all() and (ey<=.09).all() and (et<=.09).all()
    rows.append(dict(protocol=protocol,unique_images=280,axis_and_sign_correct=int(ok.sum()),Y_max=float(ey.max()),T_max=float(et.max())))
folds=json.loads((H/'splits.json').read_text());seen=[]
for f in folds:
    assert len(f['train'])==210 and len(f['test'])==70 and not set(f['train'])&set(f['test'])
    assert sum(n.startswith('Y') for n in f['train'])==60 and sum(n.startswith('T') for n in f['train'])==150
    assert sum(n.startswith('Y') for n in f['test'])==20 and sum(n.startswith('T') for n in f['test'])==50;seen+=f['test']
assert len(seen)==len(set(seen))==280
F=np.load(B/'center_patch_v2_20261021/tracks.npz');m=Z.manifest(F['names']);ix={n:i for i,n in enumerate(F['names'])};X=np.load(B/'multiref_4fold_20261021/structural_regions/interior_features.npy');tr=np.array([ix[n] for n in folds[0]['train']]);te=np.array([ix[n] for n in folds[0]['test']])
one=fit(X,m,tr,'inferred_session',True,.05);m2=m.copy();m2.loc[te,'value']=1234;m2.loc[te,'axis']='Y';m2.loc[te,'session']='dummy';X2=X.copy();X2[te]+=999
two=fit(X2,m2,tr,'inferred_session',True,.05);assert np.allclose(one[1],two[1])
assert all(one[0][k][0]==two[0][k][0] and all(np.allclose(a,b) for a,b in zip(one[0][k][1:],two[0][k][1:])) for k in one[0])
f=np.load(H/'features630.npz');C=pd.read_csv(B.parent/'analysis_grid_corners_20261019/out/corners_280.csv');ref=C[C.file=='center.bmp'].set_index('id').loc[f['ids']];mask=(ref.grid_x>0)&(ref.grid_x<21)&(ref.grid_y>0)&(ref.grid_y<21);assert np.array_equal(mask,np.load(H/'mask536.npy'));assert mask.sum()==536 and (ref.black_neighbors!=2).all()
for p,digest in json.loads((H/'source_manifest.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
report=(H/'report.html').read_text(encoding='utf-8');assert '279/280' in report and '과적합' in report
audit=dict(requirements=[dict(item='fixed4-fold and full280 resub numeric criteria',evidence=rows),dict(item='Y:T preserved, no overlap, each image held out once',passed=True),dict(item='center-only structural mask, no screw features',passed=True,used=536),dict(item='heldout label and feature perturbation cannot change fit',passed=True),dict(item='original image remeasurement and filename-independent inference',evidence=json.loads((H/'audit.json').read_text())),dict(item='documentation, source hashes, saved model and fold models',passed=True)],limitations=['Fixed setting selected after broad development search; not unbiased final-test accuracy','Internal method-selection OOF279/280','Independent new acquisition and new QR site validation deferred','Ground truth is commanded filename value, not externally measured pose','No claim of overfitting-free universal100%'],goal_scope='Existing280-image requested numerical gate and documentation; not a guarantee on new images')
(H/'completion_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print('All current-data completion gates passed; limitations retained.')
