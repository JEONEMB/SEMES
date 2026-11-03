from extract_scene import P,ROOT,RAW,OLD,read
import ctypes,json
import numpy as np
import pandas as pd
from scipy.special import erf
import cv2

lib=ctypes.CDLL(str(P/'build/Release/natural_quality.dll'))
lib.quality_batch.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
lib.quality_batch.restype=ctypes.c_int
lib.fit_batch.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
assert lib.check_kernel()==0
def quality(ref,im,xy,current,rmse):
    xy=np.ascontiguousarray(xy,float);current=np.ascontiguousarray(current,float);rmse=np.ascontiguousarray(rmse,float)
    out=np.empty((len(xy),8),float)
    lib.quality_batch(ref.ctypes.data,im.ctypes.data,im.shape[1],im.shape[0],xy.ctypes.data,current.ctypes.data,rmse.ctypes.data,len(xy),out.ctypes.data)
    return out

def tests():
    y,x=np.mgrid[:128,:128];edge=np.ascontiguousarray(np.clip(125+100*erf((x-64)/2),0,255).astype(np.uint8))
    uniform=np.full((128,128),128,np.uint8);corner=np.ascontiguousarray(np.clip(128+85*erf((x-64)/2)*erf((y-64)/2),0,255).astype(np.uint8))
    center=np.array([[64.,64.]])
    a=quality(edge,edge,center,center,np.array([1.]));b=quality(corner,corner,center,center,np.array([1.]));c=quality(uniform,uniform,center,center,np.array([1.]))
    assert not np.isfinite(a[0,5]) and np.isfinite(b[0,5]) and not np.isfinite(c[0,5])
    recovered=[]
    def scene(xx,yy):return np.ascontiguousarray(np.clip(125+40*np.sin(xx*.16)+35*np.cos(yy*.19)+25*np.cos((xx+yy)*.11),0,255).astype(np.uint8))
    ref=scene(x,y)
    for dx,dy in [(.27,-.36),(-.42,.18),(.11,.41)]:
        im=scene(x-dx,y-dy);out=np.empty((1,8),float);seed=center.copy()
        lib.fit_batch(ref.ctypes.data,im.ctypes.data,128,128,center.ctypes.data,seed.ctypes.data,1,out.ctypes.data)
        error=float(np.linalg.norm(out[0,:2]-center[0]-[dx,dy]));assert error<.035,error
        recovered.append({'dx':dx,'dy':dy,'error_px':error})
    (P/'measurement_tests.json').write_text(json.dumps({'kernel_value_gradient':'PASS','edge_rejected_as_2D_point':True,'uniform_rejected':True,'corner_rank2':True,
        'analytic_scene_subpixel_tests':recovered,'uncertainty_note':'Schur-complement local covariance is conditional on patch/noise model, not a physical coverage guarantee.'},indent=2))

def main():
    tests()
    inv=pd.read_csv(OLD/'image_inventory.csv');features=pd.read_csv(P/'feature_definitions.csv')
    xy=features[['reference_x','reference_y']].to_numpy();m=np.load(P/'measurements.npz')
    v=m['values'];complete=m['complete'];assert complete.all()
    active=pd.read_csv(P/'pilot_feature_quality.csv').active.to_numpy(bool)
    q=np.full_like(v,np.nan);ref=read(RAW/'center.bmp')
    for i in range(len(inv)):
        ids=np.flatnonzero(active&np.isfinite(v[i,:,0])&(v[i,:,6]>0))
        image=read(RAW/inv.iloc[i].path)
        if len(ids):q[i,ids]=quality(ref,image,xy[ids],v[i,ids,:2],v[i,ids,2])
        if (i+1)%100==0:print('DIRECTIONAL_QUALITY',i+1,flush=True)
    valid=active[None,:]&(v[:,:,6]>0)&(m['ncc']>.8)&(m['cycle']<.1)&(v[:,:,2]/np.maximum(v[:,:,3],1)<.2)&(v[:,:,7]<3.9)&np.isfinite(q[:,:,5])&(q[:,:,4]<50)&(q[:,:,5]<.10)
    np.savez_compressed(P/'quality.npz',quality=q,valid=valid)
    summary=features.copy();summary['old_valid_fraction']=np.mean((v[:,:,6]>0)&(m['ncc']>.75)&(m['cycle']<.15),axis=0)
    summary['strict_valid_fraction']=valid.mean(0)
    summary['median_condition']=np.nanmedian(q[:,:,4],axis=0);summary['median_major_sigma_px']=np.nanmedian(q[:,:,5],axis=0)
    summary['measurement_role']=np.where(summary.strict_valid_fraction>=.9,'eligible_2D_patch',np.where(summary.median_condition>=50,'aperture_or_weak_2D','insufficient_consistency'))
    summary.to_csv(P/'directional_feature_quality.csv',index=False)
    # Baseline lines must also have two independently measured parameters: position + slope.
    masks=[];line_details=[]
    for i,row in inv.iterrows():
        fs=cv2.FileStorage(str(OLD/'remeasured/cache'/f'frame{int(row["index"])}.yml.gz'),cv2.FILE_STORAGE_READ)
        e=fs.getNode('edges').mat();fs.release();flags=[]
        for side in range(2):
            a=e[side*60:(side+1)*60];good=(a[:,5]>0)&(a[:,3]>10)&np.isfinite(a[:,1])
            z=(a[good,0]-1920)/1920
            flags.append(bool(good.sum()>=6 and len(z)>0 and np.ptp(z)>.6))
            line_details.append({'image_index':i,'line':side,'good_samples':int(good.sum()),'normalized_x_span':float(np.ptp(z)) if len(z) else 0,'valid_position_and_slope':flags[-1]})
        masks.append(flags)
    pd.DataFrame(line_details).to_csv(P/'baseline_line_quality.csv',index=False)
    np.save(P/'line_valid.npy',np.array(masks))
    print('STRICT_2D_FEATURES >=90%',int((summary.strict_valid_fraction>=.9).sum()),'TOTAL',len(summary),flush=True)
if __name__=='__main__':main()
