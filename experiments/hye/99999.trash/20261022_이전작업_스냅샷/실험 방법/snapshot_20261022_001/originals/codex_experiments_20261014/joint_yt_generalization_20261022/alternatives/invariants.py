"""Coordinate-only robust geometry and invariant feature development screen."""
from screen import *
from engine import extract

def homography(points,d,w):
 src=(points-points.mean(0))/100;dst=src+d/100;u,v=src.T;xx,yy=dst.T;n=len(points)
 A=np.zeros((n,2,8));A[:,0,:3]=np.c_[u,v,np.ones(n)];A[:,1,3:6]=np.c_[u,v,np.ones(n)];A[:,0,6:]=-np.c_[xx*u,xx*v];A[:,1,6:]=-np.c_[yy*u,yy*v]
 sw=np.repeat(np.sqrt(np.maximum(w,.001)),2);c=np.linalg.lstsq(A.reshape(-1,8)*sw[:,None],dst.ravel()*sw,rcond=None)[0]
 mat=np.r_[c,1].reshape(3,3);iv=np.linalg.inv(mat);iv/=iv[2,2]
 return c-np.array([1,0,0,0,1,0,0,0]),iv.ravel()[:8]-np.array([1,0,0,0,1,0,0,0])

def main():
 data=ROOT/'joint_center_630_20261022/results';m=pd.read_csv(data/'manifest.csv');a=np.load(data/'measurements.npz');points=a['points'];D=a['D'];Q=a['quality'];regions=a['regions'];bank={}
 for agg in ['uniform','photo','stability','contrast','huber','soft_l1','cauchy']:
  for feat in ['global1','global2','global3','local1','local2','homography']:
   key=agg+'_'+feat;bank[key]=np.stack([extract(points,d,q,agg,feat) for d,q in zip(D,Q)]);print('features',key,flush=True)
 for weighting in ['uniform','photo','stability']:
  f=[];inv=[]
  for d,q in zip(D,Q):
   w=np.ones(len(points)) if weighting=='uniform' else 1/(1+(q[:,0]/10)**2) if weighting=='photo' else 1/(1+(q[:,1]/.2)**2)
   result=[homography(points[regions==r],d[regions==r],w[regions==r]) for r in range(3)]
   f.append(np.concatenate([z[0] for z in result]));inv.append(np.concatenate([z[1] for z in result]))
  bank['local_homo_'+weighting]=np.array(f);bank['local_homo_inverse_'+weighting]=np.array(inv)
 # Multiplication by fixed-scale observed image translation, no learned labels.
 raw=D.reshape(len(D),-1);shift=D.mean(1)[:,0:1]/100
 bank['raw_xshift']=np.c_[raw,raw*shift]
 bank['raw_xshift2']=np.c_[raw,raw*shift,raw*shift*shift]
 maps=FeatureMaps(points,regions)
 for name in ['P12','QR18','QR36']:
  xx=maps.transform(D,name);bank[name+'_xshift']=np.c_[xx,xx*shift]
 bank['raw']=raw
 np.savez_compressed(H/'invariant_features.npz',**bank)
 splits=json.loads((data/'expanded_features/splits.json').read_text());fold=np.full(len(m),-1)
 for f,s in enumerate(splits):fold[m.relative.isin(s['test'])]=f
 new=m.relative.str.startswith('20261020').to_numpy();y=m[['Y','T']].to_numpy();cfg=[]
 for feat in bank:
  cfg += [dict(feature=feat,family='gls',alpha=v) for v in [.01,.1,.5,1.]]
  cfg += [dict(feature=feat,family='ridge',alpha=v) for v in [.0001,.01,1.]]
 (H/'invariant_protocol.json').write_text(json.dumps(dict(configs=cfg,features='positive-weight630fixedIDs; per-image geometry fitting never useslabels',scope='development grouped4fold',source=str(data)),indent=2))
 records=[];allpred=[]
 for regime in ['combined','new_only']:
  for k,c in enumerate(cfg):
   X=bank[c['feature']];pred=np.full(y.shape,np.nan)
   for f in range(4):
    tr=(fold!=f)&(new if regime=='new_only' else np.ones(len(m),bool));te=(fold==f)&new
    pred[te]=make(c).fit(X[tr],y[tr]).predict(X[te])
   scores=metrics(y[new],pred[new]);records.append(dict(regime=regime,candidate=k,**c,**scores))
   pp=m.loc[new,['index','file','relative','Y','T','scope','condition']].copy();pp['regime']=regime;pp['candidate']=k;pp['fold']=fold[new];pp['predY']=pred[new,0];pp['predT']=pred[new,1];allpred.append(pp)
   if k%25==0:
    pd.DataFrame(records).to_csv(H/'invariant_summary.csv',index=False);print('screen',regime,k,scores,flush=True)
  df=pd.DataFrame(records);df.to_csv(H/'invariant_summary.csv',index=False);pd.concat(allpred,ignore_index=True).to_csv(H/'invariant_predictions.csv',index=False)
  print('BEST',regime,df[df.regime==regime].sort_values(['passed','Y_max'],ascending=[False,True]).head(15).to_string(index=False),flush=True)

if __name__=='__main__':
 with threadpool_limits(1):main()
