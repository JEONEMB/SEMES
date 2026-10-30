"""Follow-up coordinate feature screen on verified strict329 + recovered12."""
from screen import *
from engine import extract

def load():
 h=H.parent;m=pd.read_csv(h/'measurements/manifest.csv');a=np.load(h/'measurements/features.npz');old=pd.read_csv(ROOT/'joint_center_630_20261022/results/manifest.csv');splits=json.loads((ROOT/'joint_center_630_20261022/results/expanded_features/splits.json').read_text());oldfold=np.full(len(old),-1)
 for f,s in enumerate(splits):oldfold[old.relative.isin(s['test'])]=f
 fold=oldfold[m.expanded_cache_index.to_numpy()];new=m.dataset.to_numpy()==20261020;fold[new&(m.scope=='center').to_numpy()]=0
 return m,a,fold,new

def build(a):
 D=a['D'];points=a['points'];q=a['quality'];raw=D.reshape(len(D),-1);shift=D.mean(1)/100;centered=(D-D.mean(1)[:,None,:]).reshape(len(D),-1)
 b={'raw':raw,'raw_xshift':np.c_[raw,raw*shift[:,0:1]],'raw_xshift2':np.c_[raw,raw*shift[:,0:1],raw*shift[:,0:1]**2], 'raw_yshift':np.c_[raw,raw*shift[:,1:2]],'raw_xyshift':np.c_[raw,raw*shift[:,0:1],raw*shift[:,1:2]],'centered':np.c_[shift,centered],'centered_xshift':np.c_[shift,centered,centered*shift[:,0:1]]}
 for f in ['grid4','grid8']:
  xx=np.stack([field_features(points,d,qq,f) for d,qq in zip(D,q)]);b[f]=xx;b[f+'_xshift']=np.c_[xx,xx*shift[:,0:1]]
 for agg in ['uniform','photo','stability','contrast','huber','soft_l1','cauchy']:
  for feat in ['global2','global3']:
   b[agg+'_'+feat]=np.stack([extract(points,d,qq,agg,feat) for d,qq in zip(D,q)])
 return b

def main():
 m,a,fold,new=load();y=m[['Y','T']].to_numpy();bank=build(a);np.savez_compressed(H/'strict_features.npz',**bank)
 cfg=[dict(feature=f,family='gls',alpha=alpha) for f in bank for alpha in [.001,.003,.01,.03,.1,.3,.5,1.]]
 (H/'strict_protocol.json').write_text(json.dumps(dict(configs=cfg,fold=fold.tolist(),center3_together=True,measurement='strict329+expanded12',scope='development screen only'),indent=2));recs=[];preds=[]
 for regime in ['combined','new_only']:
  for k,c in enumerate(cfg):
   X=bank[c['feature']];pred=np.full(y.shape,np.nan)
   for f in range(4):
    tr=(fold!=f)&(new if regime=='new_only' else np.ones(len(m),bool));te=(fold==f)&new
    pred[te]=make(c).fit(X[tr],y[tr]).predict(X[te])
   scores=metrics(y[new],pred[new]);recs.append(dict(regime=regime,candidate=k,**c,**scores));pp=m.loc[new,['index','file','Y','T','scope','condition']].copy();pp['regime']=regime;pp['candidate']=k;pp['fold']=fold[new];pp['predY']=pred[new,0];pp['predT']=pred[new,1];preds.append(pp)
   if k%25==0:print(regime,k,scores,flush=True)
  df=pd.DataFrame(recs);df.to_csv(H/'strict_summary.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(H/'strict_predictions.csv',index=False)
  print('BEST',regime,df[df.regime==regime].sort_values(['passed','Y_max'],ascending=[False,True]).head(12).to_string(index=False),flush=True)

if __name__=='__main__':
 with threadpool_limits(1):main()
