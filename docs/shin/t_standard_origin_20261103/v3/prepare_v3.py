"""Read-only upstream inputs; fold-specific origin fits and deployment targets."""
from pathlib import Path
import os,sys,json,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from scipy.optimize import minimize

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
AUDIT=ROOT/'analysis/frame_link_audit_20261103'
EXP=ROOT/'qrwithbar-lanczos-src/experiments/hye/20261102_173432_t_six_methods'
sys.path.insert(0,str(AUDIT))
from refine_origin import load,audited_data,RefinedModel,training_repeat_weights

class SensitivityModel(RefinedModel):
    def fit(self,start=None,maxiter=450,bound=3):
        u=np.zeros(self.npar) if start is None else start.copy()
        robust=np.ones_like(self.wq);logs=[]
        for _ in range(4):
            r=minimize(lambda v:self.objective_scene(v,robust),u,jac=True,method='L-BFGS-B',bounds=[(-bound,bound)]*self.npar,
                options={'maxiter':maxiter,'ftol':1e-9,'gtol':1e-5,'maxls':35})
            u=r.x;logs.append({'success':bool(r.success),'iterations':int(r.nit),'message':str(r.message)})
            self.objective_scene(u,robust,True)
            radius=np.sqrt(np.sum(self.residual_blocks**2,axis=2)/np.array(self.dims)[:,None])
            robust=np.minimum(1,2.5/np.maximum(radius,1e-9))
        self.u=u;self.objective_scene(u,robust,True);self.optimization=logs
        return self

def save_json(path,x):
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if isinstance(o,np.ndarray) else str(o)),encoding='utf-8')

def fold(y,t):
    h=2166136261
    # C++ lround uses half away from zero; these are integer centi-units.
    for c in f'{int(np.floor(abs(y*100)+.5)*np.sign(y))}:{int(np.floor(abs(t*100)+.5)*np.sign(t))}'.encode():
        h=((h^c)*16777619)&0xffffffff
    return h%5

def write_dataset(out,raw,d,target,weights=None,available=None):
    out.mkdir(parents=True,exist_ok=True)
    base=raw.copy();base['Y']=d.Y.to_numpy();base['T']=d['T'].to_numpy()
    base.to_csv(out/'original_observations.csv',index=False,float_format='%.17g')
    z=raw.copy();z['Y']=target[:,0];z['T']=target[:,1]
    z.to_csv(out/'observations.csv',index=False,float_format='%.17g')
    if weights is None:weights={}
    pd.DataFrame({'path':raw.path,'weight':[weights.get(int(i),1.) for i in d.row_id]}).to_csv(out/'weights.csv',index=False,float_format='%.17g')
    if available is None:available=np.ones(len(d),bool)
    pd.DataFrame({'path':raw.path,'available':np.asarray(available,int)}).to_csv(out/'target_available.csv',index=False)

def model_export(out,m,d):
    sid=d.group.to_numpy(int)-1
    target=d[['Y','T']].to_numpy()+m.B[sid]
    known=m.present[sid]
    wm,_=training_repeat_weights(m.train,F,A)
    write_dataset(out,RAW,d,target,wm,known)
    save_json(out/'origin_fit.json',{'B':m.B,'D':m.D,'present':m.present,'train_row_ids':m.train.row_id.tolist(),
        'sigma_y':m.sy,'sigma_t':m.st,'bound_hits':int((np.abs(m.u)>2.999).sum()),'optimization':m.optimization,
        'origin_QR':m.Cblocks[0,0,:6]@np.linalg.inv(m.specs[0]['L'])})

def main():
    global RAW,F,A
    original,F,A=load();d=audited_data(original,True)
    RAW=pd.read_csv(EXP/'observations.csv')
    assert len(d)==575 and RAW.path.tolist()==d.path.tolist()
    coords=pd.read_csv(AUDIT/'coordinates_standard_v3.csv')
    assert coords.path.tolist()==d.path.tolist()
    assigned=np.array([fold(y,t) for y,t in d[['Y','T']].to_numpy()])
    refs=d.cohort.to_numpy()=='reference'
    manifest=coords.copy();manifest['fold']=assigned
    manifest.to_csv(P/'manifest.csv',index=False)
    for name,target in [('original',np.c_[d.Y,d.T_original]),('corrected',d[['Y','T']].to_numpy()),('fixed_v3',coords[['Y_standard','T_standard']].to_numpy())]:
        weights=dict(zip(d.row_id,coords.training_image_weight)) if name=='fixed_v3' else {}
        write_dataset(P/name,RAW,d,target,weights)
    write_dataset(P/'crossfit',RAW,d,coords[['Y_standard','T_standard']].to_numpy(),dict(zip(d.row_id,coords.training_image_weight)))
    # No fixed calibration poses are added to test folds: only four zero anchors remain in calibration.
    # The origin family and label audit were selected earlier on this development collection.
    for outer in range(5):
        cases=[(f'outer{outer}',(assigned!=outer)|refs)]
        cases += [(f'outer{outer}_inner{inner}',((assigned!=outer)&(assigned!=inner))|refs) for inner in range(5) if inner!=outer]
        for name,mask in cases:
            out=P/'crossfit'/name
            if (out/'origin_fit.json').exists():
                print('RESUME',name,flush=True);continue
            train=d[mask].copy()
            m=RefinedModel(train,F,A,True,False,True).fit(maxiter=450)
            model_export(out,m,d)
            print('ORIGIN',name,'train',len(train),'bounds',int((np.abs(m.u)>2.999).sum()),flush=True)
    sensitivity=[]
    for name,scale,bound in [('prior_half',.5,3),('prior_double',2.,3),('bound_double',1.,6)]:
        out=P/name
        if not (out/'origin_fit.json').exists():
            m=SensitivityModel(d,F,A,True,False,True)
            m.sy*=scale;m.st*=scale
            m.fit(bound=bound,maxiter=600)
            model_export(out,m,d)
            data=json.loads((out/'origin_fit.json').read_text());data['raw_bound']=bound;data['bound_hits']=int((np.abs(m.u)>bound-.001).sum())
            save_json(out/'origin_fit.json',data)
        info=json.loads((out/'origin_fit.json').read_text())
        z=pd.read_csv(out/'observations.csv');delta=(z['T'].to_numpy()-coords.T_standard.to_numpy())*100
        sensitivity.append({'variant':name,'sigma_Y':.05*scale,'sigma_T':.01*scale,'bound':bound,'bound_hits':info['bound_hits'],
            'target_delta_RMS_ticks':float(np.sqrt(np.mean(delta[~refs]**2))),'target_delta_max_ticks':float(np.max(np.abs(delta[~refs])))})
        print('SENSITIVITY',name,sensitivity[-1],flush=True)
    pd.DataFrame(sensitivity).to_csv(P/'origin_sensitivity.csv',index=False)
    sources=[AUDIT/'standard_origin_v3.json',AUDIT/'coordinates_standard_v3.csv',EXP/'observations.csv']
    save_json(P/'input_hashes.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    print('PREPARED',len(d),'images,',int((~refs).sum()),'evaluation images',flush=True)

if __name__=='__main__':main()
