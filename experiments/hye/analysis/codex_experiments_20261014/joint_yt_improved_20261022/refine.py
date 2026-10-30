"""Keep original P12: test whether greedy geometry selection discarded useful cues."""
import json,joblib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from engine import *
from experiment import Study,metrics

def main():
    s=Study();folds=json.loads((H/'splits.json').read_text());out=[];synthetic=[];base=dict(measurement='uniform',feature='global2',policy='none')
    for fold in range(5):
        if fold==4:tr=s.ids;te=None
        else:tr=np.array([s.lookup[n] for n in folds[fold]['train']]);te=np.array([s.lookup[n] for n in folds[fold]['test']])
        c=s.select([dict(base,model=m) for m in model_configs()],tr,fold,'fixed_C')
        c=s.select([dict(c,policy=policy) for policy in POLICIES],tr,fold,'fixed_D')
        mdl=s.fit(c,tr);bundle=dict(config=c,model=mdl,ids=s.obs['ids'],points=s.obs['points'])
        joblib.dump(bundle,H/('fixedfeature_model.joblib' if te is None else f'fixedfeature_fold{fold}.joblib'))
        if te is None:continue
        X=s.features(c);p=mdl.predict(X[te]);y=s.y[te]
        out.append(pd.DataFrame(dict(file=s.names[te],axis=s.m.axis.iloc[te].to_numpy(),fold=fold,true_Y=y[:,0],true_T=y[:,1],pred_Y=p[:,0],pred_T=p[:,1],both_pass09=np.all(abs(p-y)<=.09,axis=1))))
        yi=te[s.y[te,0]!=0];ti=te[s.y[te,1]!=0];ii=np.repeat(yi,len(ti));jj=np.tile(ti,len(yi))
        origin=np.linalg.lstsq(np.c_[np.ones(len(tr)),s.y[tr],s.y[tr]**2],X[tr],rcond=None)[0][0]
        pp=mdl.predict(X[ii]+X[jj]-origin);tt=s.y[ii]+s.y[jj]
        synthetic.append(pd.DataFrame(dict(fold=fold,true_Y=tt[:,0],true_T=tt[:,1],pred_Y=pp[:,0],pred_T=pp[:,1])))
        print('fixed feature outer',fold,metrics(p,y),flush=True)
    p=pd.concat(out,ignore_index=True);sy=pd.concat(synthetic,ignore_index=True)
    p.to_csv(H/'fixedfeature_oof.csv',index=False);sy.to_csv(H/'fixedfeature_synthetic.csv',index=False)
    report=dict(real_oof=metrics(p[['pred_Y','pred_T']].to_numpy(),p[['true_Y','true_T']].to_numpy()),synthetic_mixed=metrics(sy[['pred_Y','pred_T']].to_numpy(),sy[['true_Y','true_T']].to_numpy()),selection=s.selections)
    (H/'fixedfeature_summary.json').write_text(json.dumps(report,indent=2));(H/'fixedfeature_search.json').write_text(json.dumps(s.search,indent=2))
    path=H/'report.html';text=path.read_text(encoding='utf-8');text+='<h2>원래 P12 특징을 유지한 개선</h2><p>A/B단계의 GLS 기준 greedy 선택이 비선형 회귀에 유용한 특징을 버릴 가능성을 점검했다. 측정과 global2는 원래대로 고정하고 C 회귀28개 설정과 D 안정화8개를 학습 내부에서 선택했다. 외부 정답으로 후보를 선택하지 않았다. 추가 개발 탐색이므로 새로운 독립 검증과 구분한다.</p><p>OOF: '+str(report['real_oof'])+'</p><p>합성 혼합: '+str(report['synthetic_mixed'])+'</p><p>fixedfeature_model.joblib에 별도 저장.</p>'
    path.write_text(text,encoding='utf-8');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
