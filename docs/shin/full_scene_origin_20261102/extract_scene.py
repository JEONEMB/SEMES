from pathlib import Path
import ctypes,json,time,sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
RAW=Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set')
OLD=ROOT/'analysis/session_eda_20261102'
cv2.setNumThreads(1)
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
def read(path):return cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_GRAYSCALE)
lib=ctypes.CDLL(str(P/'build/Release/natural_lanczos.dll'))
arr=ctypes.c_void_p
lib.fit_batch.argtypes=[arr,arr,ctypes.c_int,ctypes.c_int,arr,arr,ctypes.c_int,arr]
lib.fit_batch.restype=ctypes.c_int
assert lib.check_kernel()==0
def fit(ref,im,xy,seed):
    xy=np.ascontiguousarray(xy,np.float64);seed=np.ascontiguousarray(seed,np.float64)
    output=np.empty((len(xy),8),np.float64)
    lib.fit_batch(ref.ctypes.data,im.ctypes.data,im.shape[1],im.shape[0],xy.ctypes.data,seed.ctypes.data,len(xy),output.ctypes.data)
    return output

def choose(ref):
    half=cv2.resize(ref,None,fx=.5,fy=.5,interpolation=cv2.INTER_LANCZOS4)
    mask=np.full(half.shape,255,np.uint8);mask[475:600,760:1140]=0
    mask[:35]=0;mask[-35:]=0;mask[:,:35]=0;mask[:,-35:]=0
    sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.015,edgeThreshold=15)
    kp=sift.detect(half,mask)
    # Manual candidates cover visible holes/fasteners even where SIFT has no center keypoint.
    manual=[(1492,1072),(2313,1075),(1632,550),(2175,550),(2490,365),(2485,625),
            (964,1360),(2868,1360),(1010,1684),(2820,1684),(1634,1780),(2085,1780),(3590,1800),(100,1680)]
    points=[];kinds=[]
    for xy in manual:points.append(xy);kinds.append('manual_hole_or_fastener')
    counts={}
    for k in sorted(kp,key=lambda v:v.response,reverse=True):
        x,y=np.round(np.array(k.pt)*2/2)*2;cell=(int(x)//240,int(y)//240)
        if counts.get(cell,0)>=3:continue
        if any(np.linalg.norm(np.array([x,y])-p)<45 for p in points):continue
        patch=ref[int(y)-20:int(y)+21,int(x)-20:int(x)+21]
        if patch.shape!=(41,41) or patch.std()<7:continue
        points.append((x,y));kinds.append('automatic_texture');counts[cell]=counts.get(cell,0)+1
        if len(points)>=154:break
    xy=np.round(np.array(points)/2)*2
    table=pd.DataFrame({'feature_id':np.arange(len(xy)),'reference_x':xy[:,0],'reference_y':xy[:,1],'candidate_type':kinds})
    table.to_csv(P/'feature_definitions.csv',index=False)
    fig,ax=plt.subplots(figsize=(16,9),layout='constrained');ax.imshow(ref,cmap='gray',vmin=0,vmax=255)
    for kind,color in [('manual_hole_or_fastener','#ffbd24'),('automatic_texture','#4bd4e5')]:
        q=table[table.candidate_type==kind];ax.scatter(q.reference_x,q.reference_y,s=27,facecolors='none',edgecolors=color,label=kind)
    for q in table[table.candidate_type=='manual_hole_or_fastener'].itertuples():ax.text(q.reference_x+10,q.reference_y,str(q.feature_id),color='#ffbd24',fontsize=9)
    ax.set(title='표준 원점 확장 후보: 나사·구멍·지그·판 외곽·배경 텍스처',xlim=(0,ref.shape[1]),ylim=(ref.shape[0],0));ax.legend()
    fig.savefig(P/'01_natural_feature_map.png',dpi=150);plt.close(fig)
    return xy

def initial(ref,im,xy):
    rh=cv2.resize(ref,None,fx=.5,fy=.5,interpolation=cv2.INTER_LANCZOS4)
    ih=cv2.resize(im,None,fx=.5,fy=.5,interpolation=cv2.INTER_LANCZOS4)
    seeds=[];scores=[]
    for x,y in xy.astype(int):
        hx,hy=x//2,y//2;templ=rh[hy-10:hy+11,hx-10:hx+11]
        left=max(0,hx-150-10);right=min(ih.shape[1],hx+150+11)
        top=max(0,hy-60-10);bottom=min(ih.shape[0],hy+60+11)
        result=cv2.matchTemplate(ih[top:bottom,left:right],templ,cv2.TM_CCOEFF_NORMED)
        _,score,_,loc=cv2.minMaxLoc(result)
        cx=(left+loc[0]+10)*2;cy=(top+loc[1]+10)*2
        template=ref[y-20:y+21,x-20:x+21]
        l=max(0,cx-6-20);t=max(0,cy-6-20);r=min(im.shape[1],cx+6+21);b=min(im.shape[0],cy+6+21)
        result=cv2.matchTemplate(im[t:b,l:r],template,cv2.TM_CCOEFF_NORMED)
        _,fine,_,loc=cv2.minMaxLoc(result)
        seeds.append([l+loc[0]+20,t+loc[1]+20]);scores.append(fine)
    return np.asarray(seeds,float),np.asarray(scores)

def main():
    inv=pd.read_csv(OLD/'image_inventory.csv');ref=read(RAW/'center.bmp')
    xy=choose(ref) if not (P/'feature_definitions.csv').exists() else pd.read_csv(P/'feature_definitions.csv')[['reference_x','reference_y']].to_numpy()
    n=len(inv);f=len(xy)
    values=np.full((n,f,8),np.nan);cycle=np.full((n,f),np.nan);ncc=np.full((n,f),np.nan);complete=np.zeros(n,bool)
    if (P/'measurements.npz').exists():
        old=np.load(P/'measurements.npz');values=old['values'];cycle=old['cycle'];ncc=old['ncc'];complete=old['complete']
    indices=np.arange(n)
    active=np.arange(f)
    if '--pilot' not in sys.argv and (P/'pilot_feature_quality.csv').exists():
        locked=pd.read_csv(P/'pilot_feature_quality.csv');active=locked.loc[locked.active,'feature_id'].to_numpy(int)
    elif '--pilot' not in sys.argv and (P/'natural_measurements.csv').exists():
        pilot=pd.read_csv(P/'natural_measurements.csv')
        quality=pilot.groupby('feature_id').valid.mean()
        active=quality[quality>=.7].index.to_numpy(int)
        pd.DataFrame({'feature_id':np.arange(f),'pilot_valid_fraction':quality.reindex(np.arange(f)).to_numpy(),
                      'active':np.isin(np.arange(f),active)}).to_csv(P/'pilot_feature_quality.csv',index=False)
    if '--pilot' in sys.argv:
        indices=np.unique(np.r_[inv.index[inv.cohort=='reference'],inv.Y.idxmin(),inv.Y.idxmax(),inv['T'].idxmin(),inv['T'].idxmax(),0,50,100,200,300,400])
    start=time.time()
    for ii,i in enumerate(indices):
        if complete[i]:continue
        image=read(RAW/inv.iloc[i].path)
        seed,score=initial(ref,image,xy[active])
        forward=fit(ref,image,xy[active],seed)
        backward=fit(image,ref,forward[:,:2],xy[active])
        values[i,active]=forward;cycle[i,active]=np.linalg.norm(backward[:,:2]-xy[active],axis=1);ncc[i,active]=score
        complete[i]=True
        if (ii+1)%10==0 or ii==len(indices)-1:
            np.savez_compressed(P/'measurements.npz',values=values,cycle=cycle,ncc=ncc,complete=complete)
            valid=(values[i,:,6]>0)&(ncc[i]>.75)&(cycle[i]<.15)&(values[i,:,2]/np.maximum(values[i,:,3],1)<.22)&(values[i,:,7]<3.9)
            print('MEASURED',int(complete.sum()),'/',n,'current valid',int(valid.sum()),'/',f,'elapsed',round(time.time()-start,1),flush=True)
    np.savez_compressed(P/'measurements.npz',values=values,cycle=cycle,ncc=ncc,complete=complete)
    rows=[]
    for i in np.flatnonzero(complete):
        for k in range(f):
            v=values[i,k];valid=(v[6]>0)&(ncc[i,k]>.75)&(cycle[i,k]<.15)&(v[2]/max(v[3],1)<.22)&(v[7]<3.9)
            rows.append({'image_index':int(i),'path':inv.iloc[i].path,'feature_id':k,'x':v[0],'y':v[1],'dx':v[0]-xy[k,0],'dy':v[1]-xy[k,1],
                'RMSE':v[2],'reference_contrast':v[3],'gain':v[4],'offset':v[5],'optimizer_ok':int(v[6]) if np.isfinite(v[6]) else 0,'NCC':ncc[i,k],'cycle_error_px':cycle[i,k],'valid':bool(valid)})
    pd.DataFrame(rows).to_csv(P/'natural_measurements.csv',index=False)
    print('DONE',int(complete.sum()),'features',f,flush=True)
if __name__=='__main__':main()
