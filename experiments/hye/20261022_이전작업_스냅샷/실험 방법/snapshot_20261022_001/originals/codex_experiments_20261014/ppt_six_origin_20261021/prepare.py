from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent;BASE=OUT.parent
sys.path.insert(0,str(BASE));import infer
M=OUT/'restored_model';M.mkdir(exist_ok=True)
for src,dst in [(BASE/'calibrated_runs/model.json',M/'model.json'),(BASE/'measurement/gn17_spread14_features.npz',M/'features.npz'),(BASE/'reference.json',M/'reference.json')]:shutil.copy2(src,dst)
params=json.loads((M/'model.json').read_text(encoding='utf-8'))
z=np.load(BASE/'measurement/gn17_spread14.npz');pts=z['points'];D=z['D'];names=z['names'].astype(str);assert z['valid'].all()
u=(pts[:,0]-1906)/486;v=(pts[:,1]-1070)/122;B=np.array([u*0+1,u,v,u*u,u*v,v*v]).T
rows=[];allcosts={}
for i,n in enumerate(names):
    if n=='center.bmp':continue
    X=np.linalg.lstsq(B.astype(float),D[i].astype(float),rcond=None)[0].ravel()-params['zero_feature']
    pred=infer.solve_calibrated(X,params);a=n[0];val=float(n[1:-4]);err=abs(val-pred['value'])
    rows.append(dict(file=n,axis=a,value=val,pred_axis=pred['axis'],pred=pred['value'],correct=a==pred['axis'] and np.sign(val)==np.sign(pred['value']),error=err))
    if n in ['Y+0.05.bmp','T-0.05.bmp','T+0.01.bmp']:allcosts[n]=infer.clean(pred)
df=pd.DataFrame(rows);df.to_csv(OUT/'restored_280_predictions.csv',index=False,encoding='utf-8-sig')
assert df.correct.all()
# Use the exact stored cross-validation predictions, not newly selected splits.
cv=pd.read_csv(BASE/'center_fixed_20261015/predictions.csv')
rep=cv[cv.protocol.str.startswith('repeated5_')]
repeated={}
for method,g in rep.groupby('method'):
    stats={}
    for a in ['Y','T']:
        sub=g[g.axis==a];stats[a]=dict(correct=int(sub.axis_sign_correct.sum()),n=len(sub),accuracy=float(sub.axis_sign_correct.mean()))
    stats.update(failures=int((~g['pass']).sum()),Y_max=float(g.Y_abs_error.max()),T_max=float(g.T_abs_error.max()))
    repeated[method]=stats
full={a:dict(n=int((df.axis==a).sum()),correct=int(df[df.axis==a].correct.sum()),max_error=float(df[df.axis==a].error.max()),mae=float(df[df.axis==a].error.mean())) for a in ['Y','T']}
data=dict(full=full,repeated=repeated,cases=allcosts,branches=[dict(name=k,axis=b[0],base_x=b[1][0],base_y=b[1][1]) for k,b in params['branches'].items()],model_sha256=hashlib.sha256((M/'model.json').read_bytes()).hexdigest(),features_sha256=hashlib.sha256((M/'features.npz').read_bytes()).hexdigest(),zero_feature=params['zero_feature'],scope=params['scope'])
(OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(full=full,repeated=repeated,case=allcosts['Y+0.05.bmp']),ensure_ascii=False,indent=2),flush=True)
