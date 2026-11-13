"""Image-only extension: QR/lines plus quality-controlled A patches; frozen v3 convention."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,argparse,ctypes
import numpy as np
import pandas as pd
from scipy.optimize import minimize,linprog

P=Path(__file__).resolve().parent;ROOT=P.parents[2]
AUDIT=ROOT/'analysis/frame_link_audit_20261103'
SCENE=ROOT/'analysis/full_scene_origin_20261102'
EXP=ROOT/'qrwithbar-lanczos-src/experiments/hye/20261102_173432_t_six_methods'
# Fixed before this extension's OOF run. Candidate selection is inside each outer train set.
CONFIGS=[('base',1.,0.),('base',100.,0.),('base',1.,1.),
         ('scene',1.,0.),('scene',100.,0.),('scene',1.,.2),('scene',1.,1.),('scene',1.,5.),('all',100.,0.)]

def feature_input(raw,natural,valid,ids,mode,train):
    count=72 if mode=='base' else 10 if mode=='scene' else 104
    columns=list(range(count))
    if mode!='base':
        for j,fid in enumerate(ids):
            if valid[train,j].mean()>=.85:columns.extend([104+2*j,105+2*j])
    x=np.c_[raw,np.where(valid[:,:,None],natural,np.nan).reshape(len(raw),-1)]
    return x[:,columns],np.asarray(columns,int)

def basis(z,centers,gamma):
    if not len(centers):return np.c_[np.ones(len(z)),z]
    dist=np.maximum(0,(z*z).sum(1)[:,None]+(centers*centers).sum(1)[None,:]-2*z@centers.T)/z.shape[1]
    return np.c_[np.ones(len(z)),z,np.exp(-gamma*dist)]

def fit(raw,natural,valid,ids,train,target,weights,cfg):
    mode,alpha,gamma=cfg;x,columns=feature_input(raw,natural,valid,ids,mode,train)
    v=x[train].copy();w=weights[train]
    mean=np.nansum(v*w[:,None],axis=0)/np.maximum(np.sum(np.isfinite(v)*w[:,None],axis=0),1e-12)
    mean=np.nan_to_num(mean);v=np.where(np.isfinite(v),v,mean)
    scale=np.maximum(np.sqrt(np.sum(w[:,None]*(v-mean)**2,axis=0)/w.sum()),1e-8)
    z=(v-mean)/scale;centers=np.zeros((0,z.shape[1]))
    if gamma:
        chosen=[0];distance=np.full(len(z),np.inf)
        for k in range(31):
            distance=np.minimum(distance,((z-z[chosen[-1]])**2).sum(1));chosen.append(int(distance.argmax()))
        centers=z[chosen]
    a=basis(z,centers,gamma);reg=np.eye(a.shape[1])*alpha;reg[0,0]=0
    coef=np.linalg.solve(a.T@(w[:,None]*a)+reg,a.T@(w*target[train]*100))
    return {'columns':columns,'mean':mean,'scale':scale,'centers':centers,'gamma':gamma,'coef':coef,'cfg':list(cfg)}

def predict(m,raw,natural,valid):
    x=np.c_[raw,np.where(valid[:,:,None],natural,np.nan).reshape(len(raw),-1)][:,m['columns']]
    x=np.where(np.isfinite(x),x,m['mean']);z=(x-m['mean'])/m['scale']
    return basis(z,m['centers'],m['gamma'])@m['coef']/100

def mixture(z,y):
    # Convex minimax on inner held-out predictions. No outer test labels enter this optimization.
    n,k=z.shape;z=z*100;y=y*100
    aub=np.r_[np.c_[z,-np.ones(n)],np.c_[-z,-np.ones(n)]];bub=np.r_[y,-y]
    result=linprog(np.r_[np.zeros(k),1.],A_ub=aub,b_ub=bub,A_eq=np.array([np.r_[np.ones(k),0.]]),b_eq=[1.],bounds=[(0,1)]*k+[(0,None)],method='highs')
    if not result.success:raise RuntimeError(result.message)
    def obj(v):
        e=z@v[:-1]-y;rmse=np.sqrt(np.mean(e*e)+1e-12)
        return v[-1]+.1*rmse,np.r_[.1*z.T@e/(n*rmse),1.]
    refined=minimize(obj,result.x,jac=True,bounds=[(0,1)]*k+[(0,None)],constraints=[
        {'type':'eq','fun':lambda v:v[:-1].sum()-1,'jac':lambda v:np.r_[np.ones(k),0.]},
        {'type':'ineq','fun':lambda v:bub-aub@v,'jac':lambda v:-aub}],method='SLSQP',options={'maxiter':200,'ftol':1e-9})
    w=(refined.x if refined.success else result.x)[:-1];w=np.maximum(w,0);return w/w.sum()

def data():
    manifest=pd.read_csv(P/'manifest.csv');raw=pd.read_csv(EXP/'observations.csv')
    spec=json.loads((AUDIT/'standard_origin_v3.json').read_text())
    ids=[int(s['feature_id']) for s in spec['origin_blocks'] if s['feature_id'] is not None and s['importance']>0]
    xy=pd.read_csv(SCENE/'feature_definitions.csv').set_index('feature_id').loc[ids,['reference_x','reference_y']].to_numpy()
    array=np.load(SCENE/'measurements.npz')['values'];valid=np.load(SCENE/'quality.npz')['valid']
    assert manifest.path.tolist()==raw.path.tolist()
    return manifest,raw[[f'f{i}' for i in range(104)]].to_numpy(),array[:,ids,:2]-xy,valid[:,ids],ids,xy

def dataset(root,manifest):
    d=pd.read_csv(root/'observations.csv');w=pd.read_csv(root/'weights.csv');available=pd.read_csv(root/'target_available.csv').available.to_numpy(bool)
    assert d.path.tolist()==manifest.path.tolist()==w.path.tolist()
    return d['T'].to_numpy(),w.weight.to_numpy(),available

def save_model(out,models,weights,ids,xy):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x))
    obj={'schema':'t_standard_scene.v1','coordinate_frame':'standard_origin_v3_assumed','physical_accuracy_certified':False,
        'tick_deg':.01,'natural_feature_ids':ids,'natural_reference_xy':xy,'weights':weights,'models':models,
        'feature_policy':'A patches only; training valid fraction >=85%; per-image strict-quality mask; training-mean imputation; no date/session input.'}
    out.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert),encoding='utf-8')

def train_variant(name):
    manifest,raw,natural,valid,ids,xy=data();root=P/name
    target,w,available=dataset(root,manifest)
    allids=np.flatnonzero(manifest.cohort.to_numpy()!='reference');assigned=manifest.fold.to_numpy()
    records=[];weightlog=[];foldweights=[]
    out=root/'scene_results';out.mkdir(exist_ok=True)
    for outer in range(5):
        split=root/f'outer{outer}';split=split if split.exists() else root
        outertarget,outerw,outerok=dataset(split,manifest)
        train=allids[assigned[allids]!=outer];test=allids[assigned[allids]==outer]
        z=np.zeros((len(manifest),len(CONFIGS)));innerY=np.zeros(len(manifest));innerOK=np.zeros(len(manifest),bool)
        for inner in range(5):
            if inner==outer:continue
            nested=root/f'outer{outer}_inner{inner}';nested=nested if nested.exists() else split
            localY,localw,localOK=dataset(nested,manifest)
            fitting=train[assigned[train]!=inner];validation=train[assigned[train]==inner]
            for k,cfg in enumerate(CONFIGS):
                m=fit(raw,natural,valid,ids,fitting,localY,localw,cfg);z[validation,k]=predict(m,raw[validation],natural[validation],valid[validation])
            innerY[validation]=localY[validation];innerOK[validation]=localOK[validation]
        tuning=train[innerOK[train]];chosen=mixture(z[tuning],innerY[tuning]);foldweights.append(chosen)
        models=[fit(raw,natural,valid,ids,train,outertarget,outerw,cfg) for cfg in CONFIGS]
        p=sum(weight*predict(m,raw[test],natural[test],valid[test]) for weight,m in zip(chosen,models))
        for j,i in enumerate(test):records.append({'path':manifest.iloc[i].path,'fold':outer,'cohort':manifest.iloc[i].cohort,
            'T_target':outertarget[i],'T_pred':p[j],'error_ticks':(p[j]-outertarget[i])*100,'target_available':int(outerok[i]),'status':0})
        weightlog.extend({'fold':outer,'candidate':k,'configuration':str(cfg),'weight':chosen[k]} for k,cfg in enumerate(CONFIGS))
        save_model(out/f'fold{outer}.json',models,chosen,ids,xy)
        print('SCENE',name,'fold',outer,'max',round(float(np.max(np.abs((p-outertarget[test])*100))),4),flush=True)
    pd.DataFrame(records).to_csv(out/'predictions.csv',index=False);pd.DataFrame(weightlog).to_csv(out/'weights.csv',index=False)
    models=[fit(raw,natural,valid,ids,allids,target,w,cfg) for cfg in CONFIGS]
    chosen=np.mean(foldweights,axis=0);save_model(out/'deployment.json',models,chosen,ids,xy)
    p=sum(weight*predict(m,raw,natural,valid) for weight,m in zip(chosen,models))
    pd.DataFrame({'path':manifest.path,'T_pred':p}).to_csv(out/'deployment_predictions.csv',index=False)

def restore(m):
    return {k:np.asarray(v,int if k=='columns' else float) if k in ['columns','mean','scale','centers','coef'] else v for k,v in m.items()}

def infer(image_path,model_path):
    import cv2
    cv2.setNumThreads(1)
    obj=json.loads(Path(model_path).read_text(encoding='utf-8'))
    if obj['schema']!='t_standard_scene.v1':raise ValueError('Incompatible scene model')
    assets=Path(model_path).resolve().parent
    runtime=assets/'runtime' if (assets/'runtime').exists() else SCENE/'build/Release'
    dll_handle=os.add_dll_directory(str(runtime)) if hasattr(os,'add_dll_directory') else None
    lib=ctypes.CDLL(str(runtime/'natural_lanczos.dll'))
    qlib=ctypes.CDLL(str(runtime/'natural_quality.dll'))
    lib.fit_batch.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    qlib.quality_batch.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    def read(path):return cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_GRAYSCALE)
    def patch_fit(ref,im,xy,seed):
        xy=np.ascontiguousarray(xy,float);seed=np.ascontiguousarray(seed,float);out=np.empty((len(xy),8),float)
        lib.fit_batch(ref.ctypes.data,im.ctypes.data,im.shape[1],im.shape[0],xy.ctypes.data,seed.ctypes.data,len(xy),out.ctypes.data)
        return out
    def quality(ref,im,xy,current,rmse):
        xy=np.ascontiguousarray(xy,float);current=np.ascontiguousarray(current,float);rmse=np.ascontiguousarray(rmse,float);out=np.empty((len(xy),8),float)
        qlib.quality_batch(ref.ctypes.data,im.ctypes.data,im.shape[1],im.shape[0],xy.ctypes.data,current.ctypes.data,rmse.ctypes.data,len(xy),out.ctypes.data)
        return out
    def initial(ref,im,xy):
        rh=cv2.resize(ref,None,fx=.5,fy=.5,interpolation=cv2.INTER_LANCZOS4);ih=cv2.resize(im,None,fx=.5,fy=.5,interpolation=cv2.INTER_LANCZOS4)
        seeds=[];scores=[]
        for x,y in xy.astype(int):
            hx,hy=x//2,y//2;templ=rh[hy-10:hy+11,hx-10:hx+11]
            left=max(0,hx-160);right=min(ih.shape[1],hx+161);top=max(0,hy-70);bottom=min(ih.shape[0],hy+71)
            result=cv2.matchTemplate(ih[top:bottom,left:right],templ,cv2.TM_CCOEFF_NORMED);_,_,_,loc=cv2.minMaxLoc(result)
            cx=(left+loc[0]+10)*2;cy=(top+loc[1]+10)*2;template=ref[y-20:y+21,x-20:x+21]
            l=max(0,cx-26);t=max(0,cy-26);r=min(im.shape[1],cx+27);b=min(im.shape[0],cy+27)
            result=cv2.matchTemplate(im[t:b,l:r],template,cv2.TM_CCOEFF_NORMED);_,fine,_,loc=cv2.minMaxLoc(result)
            seeds.append([l+loc[0]+20,t+loc[1]+20]);scores.append(fine)
        return np.asarray(seeds,float),np.asarray(scores)
    refpath=assets/'reference_scene.png'
    if not refpath.exists():refpath=Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set\center.bmp')
    image=read(Path(image_path));ref=read(refpath)
    if image.shape!=(2160,3840):raise ValueError('Expected 3840 x 2160 image')
    xy=np.asarray(obj['natural_reference_xy'],float);seed,ncc=initial(ref,image,xy)
    v=patch_fit(ref,image,xy,seed);back=patch_fit(image,ref,v[:,:2],xy)
    cycle=np.linalg.norm(back[:,:2]-xy,axis=1);q=quality(ref,image,xy,v[:,:2],v[:,2])
    valid=(v[:,6]>0)&(ncc>.8)&(cycle<.1)&(v[:,2]/np.maximum(v[:,3],1)<.2)&(v[:,7]<3.9)&np.isfinite(q[:,5])&(q[:,4]<50)&(q[:,5]<.10)
    feature_dll=runtime/'v3_feature_extract.dll'
    if not feature_dll.exists():feature_dll=P/'build/Release/v3_feature_extract.dll'
    library=ctypes.CDLL(str(feature_dll))
    library.v3_feature_extract.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_wchar_p,ctypes.c_void_p]
    library.v3_feature_extract.restype=ctypes.c_int
    raw=np.empty(104,float)
    qrassets=assets/'qr_assets'
    if not qrassets.exists():qrassets=ROOT/'qrwithbar-lanczos-src/modules/qrwithbar_lanczos/assets'
    status=library.v3_feature_extract(image.ctypes.data,image.shape[0],image.shape[1],str(qrassets),raw.ctypes.data)
    if status:raise RuntimeError('QR/line extraction failed: '+str(status))
    models=[restore(m) for m in obj['models']];natural=(v[:,:2]-xy)[None,:,:]
    result=sum(weight*predict(m,raw[None,:],natural,valid[None,:])[0] for weight,m in zip(obj['weights'],models))
    return {'coordinate_frame':obj['coordinate_frame'],'t_deg':float(result),'t_ticks':float(result*100),'tick_deg':.01,
        'strict_valid_A_features':int(valid.sum()),'total_A_features':len(valid),'physical_accuracy_certified':False},raw,natural[0],valid

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',nargs='+');parser.add_argument('--image');parser.add_argument('--model');args=parser.parse_args()
    if args.train:
        for name in args.train:train_variant(name)
    elif args.image and args.model:print(json.dumps(infer(args.image,args.model)[0],ensure_ascii=False,indent=2))
    else:parser.error('Use --train VARIANTS or --image IMAGE --model MODEL')
