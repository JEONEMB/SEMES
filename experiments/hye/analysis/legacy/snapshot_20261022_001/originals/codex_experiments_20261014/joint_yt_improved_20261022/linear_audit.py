"""Additive linear track and controlled model-family comparison on uniform P12.
Selections use already saved inner scores, never best outer test performance.
"""
import json,joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from engine import *
from experiment import Study,metrics,rank_score

def eligible(c):
    m=c['model'];f=m['family']
    return f in ['ridge','bayes','ard','pcr','pls'] or (f=='forward' and not m['quad'] and m['loss']=='linear')

def main():
    study=Study();splits=json.loads((H/'splits.json').read_text());search=json.loads((H/'search.json').read_text());out=[];synthetic=[];selected=[];controlled=[]
    for fold,split in enumerate(splits):
        tr=np.array([study.lookup[n] for n in split['train']]);te=np.array([study.lookup[n] for n in split['test']])
        candidates=[r for r in search if r['fold']==fold and r['stage']=='C_models' and eligible(r['config'])]
        chosen=min(candidates,key=lambda r:(*r['inner_score'],r['candidate']));c=chosen['config'];selected.append(dict(fold=fold,config=c,inner_score=chosen['inner_score']))
        mdl=study.fit(c,tr);X=study.features(c);p=mdl.predict(X[te]);y=study.y[te]
        out.append(pd.DataFrame(dict(file=study.names[te],fold=fold,axis=study.m.axis.iloc[te].to_numpy(),true_Y=y[:,0],true_T=y[:,1],pred_Y=p[:,0],pred_T=p[:,1],both_pass09=np.all(abs(p-y)<=.09,axis=1))))
        joblib.dump(dict(config=c,model=mdl,ids=study.obs['ids'],points=study.obs['points']),H/f'linear_fold{fold}.joblib')
        yi=te[study.y[te,0]!=0];ti=te[study.y[te,1]!=0];ii=np.repeat(yi,len(ti));jj=np.tile(ti,len(yi))
        origin=np.linalg.lstsq(np.c_[np.ones(len(tr)),study.y[tr],study.y[tr]**2],X[tr],rcond=None)[0][0]
        pred=mdl.predict(X[ii]+X[jj]-origin);truth=study.y[ii]+study.y[jj]
        synthetic.append(pd.DataFrame(dict(fold=fold,Y_source=study.names[ii],T_source=study.names[jj],true_Y=truth[:,0],true_T=truth[:,1],pred_Y=pred[:,0],pred_T=pred[:,1])))
        # One-factor comparison: every family gets the same original unweighted P12.
        base=dict(measurement='uniform',feature='global2',policy='none')
        inner=list(StratifiedKFold(3,shuffle=True,random_state=900+fold).split(tr,study.m.session.iloc[tr]))
        models=model_configs();families=sorted(set(m['family'] for m in models))
        for family in families:
            ranks=[]
            for k,mc in enumerate(models):
                if mc['family']!=family:continue
                cfg=dict(base,model=mc);ip=[];it=[]
                for a,b in inner:
                    mm=study.fit(cfg,tr[a]);ip.append(mm.predict(study.features(cfg)[tr[b]]));it.append(study.y[tr[b]])
                ranks.append((*rank_score(np.vstack(ip),np.vstack(it)),k))
            cc=dict(base,model=models[min(ranks)[-1]]);pp=study.fit(cc,tr).predict(study.features(cc)[te])
            controlled.append(pd.DataFrame(dict(file=study.names[te],fold=fold,family=family,true_Y=study.y[te,0],true_T=study.y[te,1],pred_Y=pp[:,0],pred_T=pp[:,1])))
        print('linear and controlled',fold,flush=True)
    out=pd.concat(out,ignore_index=True);out.to_csv(H/'linear_oof.csv',index=False)
    synthetic=pd.concat(synthetic,ignore_index=True);synthetic.to_csv(H/'linear_synthetic_mixed.csv',index=False)
    cc=pd.concat(controlled,ignore_index=True);cc.to_csv(H/'controlled_model_predictions.csv',index=False)
    stats=[]
    for family,g in cc.groupby('family'):stats.append(dict(family=family,**metrics(g[['pred_Y','pred_T']].to_numpy(),g[['true_Y','true_T']].to_numpy())))
    pd.DataFrame(stats).to_csv(H/'controlled_model_summary.csv',index=False)
    report=dict(selected=selected,real_oof=metrics(out[['pred_Y','pred_T']].to_numpy(),out[['true_Y','true_T']].to_numpy()),synthetic_mixed=metrics(synthetic[['pred_Y','pred_T']].to_numpy(),synthetic[['true_Y','true_T']].to_numpy()))
    # Final additive linear model: configure geometry from all-data training selection,
    # then choose its linear estimator inside all280 training3-fold.
    final=json.loads((H/'final_selection.json').read_text())['config'];candidates=[dict(final,model=m,policy='none') for m in model_configs() if eligible(dict(model=m))]
    final=study.select(candidates,study.ids,-1,'linear_final');mdl=study.fit(final,study.ids)
    joblib.dump(dict(config=final,model=mdl,ids=study.obs['ids'],points=study.obs['points']),H/'linear_model.joblib')
    report['final_config']=final;report['interpretation']='Affine mapping preserves superposition around learned origin, but its pure-axis OOF and synthetic superposition are not real mixed-pose validation.'
    (H/'linear_audit.json').write_text(json.dumps(report,indent=2))
    path=H/'report.html';text=path.read_text(encoding='utf-8');text+='<h2>가산 선형 모델을 별도로 비교</h2><p>비선형 모델이 단일 축 학습 분포에 의존할 가능성을 점검하기 위해 C단계의 선형 후보만 내부 점수로 선택했다. 별도 linear_model.joblib로 저장했다. 실제 OOF: '+str(report['real_oof'])+'</p><p>선형 합성 혼합(실제 혼합 사진 아님): '+str(report['synthetic_mixed'])+'</p><h2>모델만 바꾼 통제 실험</h2><p>관측 uniform/global2(P12)를 고정하고 모델군별 설정만 학습 내부에서 선택했다. 네 단계 greedy 선택의 효과와 구분한다.</p>'+pd.DataFrame(stats).to_html(index=False,float_format=lambda v:f'{v:.6f}')
    path.write_text(text,encoding='utf-8');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
