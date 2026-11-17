"""QR photometric features explicitly separate from coordinate-only candidates."""
from screen import *
sys.path.insert(0,str(H.parent))
from data import load,features

def bank_from(a):
 raw=a['D'].reshape(len(a['D']),-1);shift=a['D'].mean(1)[:,0:1]/100;q=a['quality'];regs=a['regions'];core={'raw':raw,'shift':np.c_[raw,raw*shift]};b={}
 for kind,ix in [('all',[0,1,2,3,4]),('photo',[2,3,4]),('contrast',[2]),('diagnostic',[0,1])]:
  pp=np.concatenate([np.quantile(q[:,regs==r][:,:,ix],[.1,.5,.9],axis=1).transpose(1,0,2).reshape(len(q),-1) for r in range(3)],axis=1)
  full=q[:,:,ix].reshape(len(q),-1)
  for f,X in core.items():
   b[f+'_qrsummary_'+kind]=np.c_[X,pp]
   if kind!='diagnostic':b[f+'_qrfull_'+kind]=np.c_[X,full]
 return b

def run(m,a,suffix):
 bank=bank_from(a);y=m[['Y','T']].to_numpy();new=m.dataset.to_numpy()==20261020;fold=m.fold.to_numpy();cfg=[dict(feature=f,family='gls',alpha=v) for f in bank for v in [.003,.01,.03,.1]];recs=[];preds=[]
 (H/f'quality_protocol_{suffix}.json').write_text(json.dumps(dict(configs=cfg,scope='development grouped4fold; QRappearancefeatures are distinct from coordinates; sameQRpattern only',source=suffix),indent=2))
 for k,c in enumerate(cfg):
  X=bank[c['feature']];pred=np.full(y.shape,np.nan)
  for f in range(4):
   tr=fold!=f;te=(fold==f)&new;pred[te]=make(c).fit(X[tr],y[tr]).predict(X[te])
  scores=metrics(y[new],pred[new]);recs.append(dict(candidate=k,**c,**scores));pp=m.loc[new,['index','file','Y','T','scope','condition']].copy();pp['candidate']=k;pp['fold']=fold[new];pp['predY']=pred[new,0];pp['predT']=pred[new,1];preds.append(pp)
  if k%10==0:print(suffix,k,scores,flush=True)
 df=pd.DataFrame(recs);df.to_csv(H/f'quality_summary_{suffix}.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(H/f'quality_predictions_{suffix}.csv',index=False)
 print('BEST',suffix,df.sort_values(['passed','Y_max'],ascending=[False,True]).head(12).to_string(index=False),flush=True)

def main():
 m,a=load();run(m,a,'strict329_expanded12')
 old=np.load(ROOT/'joint_center_630_20261022/results/measurements.npz');ix=m.expanded_cache_index.to_numpy();mapped={k:(old[k][ix] if old[k].ndim and len(old[k])==341 else old[k]) for k in old.files}
 run(m,mapped,'uniform_expanded341')

if __name__=='__main__':
 with threadpool_limits(1):main()
