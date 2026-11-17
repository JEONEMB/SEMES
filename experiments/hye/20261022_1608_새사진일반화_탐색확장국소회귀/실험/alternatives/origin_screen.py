from screen import *
import re
sys.path.insert(0,str(H.parent))
from data import load,features
from origin_gls import OriginGLS

def main():
 m,a=load();bank=features(a);y=m[['Y','T']].to_numpy();fold=m.fold.to_numpy();new=m.dataset.to_numpy()==20261020
 src=pd.read_csv(ROOT/'joint_center_630_20261022/results/manifest.csv');groups=[]
 for row in m.itertuples():
  if row.dataset==20261014:g='old:'+str(src.iloc[row.expanded_cache_index]['group'])
  elif row.scope=='center':g='new:center'
  else:g='new:capture_hypothesis_'+re.search(r'\((\d+)\)',row.file).group(1)
  groups.append(g)
 groups=np.array(groups)
 cfg=[dict(feature=f,shrink=s,mode=mode,k=k) for f in ['raw1260','raw_shift'] for s in [.01,.1,.3] for mode,k in [('capture',3),('kmeans',1),('kmeans',3),('kmeans',6)]]
 (H/'origin_protocol.json').write_text(json.dumps(dict(configs=cfg,scope='development screen, outer folds already used forscreening',training_capture_groups='historical6groups andnewrepeat1/2/3 are hypothesis, usedonlyfitting',prediction='select candidate exclusively by image Mahalanobis residual; predict accepts Xonly',test_centers='allnewcenter3 in outerfold0'),indent=2))
 rec=[];preds=[];audit=[]
 for k,c in enumerate(cfg):
  X=bank[c['feature']];p=np.full(y.shape,np.nan);centers=np.zeros(len(m),int);gap=np.zeros(len(m));alt=np.full(y.shape,np.nan)
  for f in range(4):
   tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);model=OriginGLS(c['shrink'],c['mode'],c['k']).fit(X[tr],y[tr],groups[tr]);p[te]=model.predict(X[te]);centers[te]=model.last['chosen'];gap[te]=model.last['gap'];alt[te]=model.last['alternative']
   audit.append(dict(candidate=k,fold=f,training_indices=tr.tolist(),test_indices=te.tolist(),candidate_names=model.origin_names))
  for subset,mask in [('new60',new),('old281',~new),('all341',np.ones(len(m),bool))]:rec.append(dict(candidate=k,**c,subset=subset,**metrics(y[mask],p[mask])))
  pp=m[['index','file','Y','T','scope','condition','dataset']].copy();pp['candidate']=k;pp['fold']=fold;pp['predY']=p[:,0];pp['predT']=p[:,1];pp['candidate_chosen']=centers;pp['cost_gap']=gap;pp['alternateY']=alt[:,0];pp['alternateT']=alt[:,1];preds.append(pp)
  print(k,c,metrics(y[new],p[new]),flush=True)
 df=pd.DataFrame(rec);df.to_csv(H/'origin_summary.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(H/'origin_predictions.csv',index=False);(H/'origin_audit.json').write_text(json.dumps(audit,indent=2));print('BEST',df[df.subset=='new60'].sort_values(['passed','Y_max'],ascending=[False,True]).head(12).to_string(index=False),flush=True)

if __name__=='__main__':
 with threadpool_limits(1):main()
