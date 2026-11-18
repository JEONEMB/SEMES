"""Separate joint Y/T experiment. Frozen boundary630 features; nested 4-fold OOF."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import StratifiedKFold

H=Path(__file__).resolve().parent; B=H.parent
sys.path.insert(0,str(B))
import zero_experiment as Z

def basis(y,quad):
    return np.c_[y,y*y] if quad else y

def fit_joint(X,y,tr,quad,shrink):
    A=basis(y[tr],quad); co=np.linalg.lstsq(A,X[tr],rcond=None)[0]
    r=X[tr]-A@co; S=r.T@r/(len(tr)-len(co))
    std=np.sqrt(np.maximum(np.diag(S),1e-14)); C=S/std[:,None]/std[None,:]
    C=(1-shrink)*C+shrink*np.eye(len(C))
    W=np.linalg.cholesky(np.linalg.inv(C)).T/std[None,:]
    return dict(co=co,W=W,quad=quad,shrink=shrink)

def predict_joint(X,model):
    co,W,quad=model['co'],model['W'],model['quad']
    A=co@W.T; z=X@W.T
    initial=np.linalg.lstsq(A[:2].T,z.T,rcond=None)[0].T
    if not quad:return initial
    out=[]
    for zz,start in zip(z,initial):
        def fun(v):return basis(v[None],True)[0]@A-zz
        def jac(v):return np.stack([A[0]+2*v[0]*A[2],A[1]+2*v[1]*A[3]],axis=1)
        # Fixed multiple starts, without true axis/sign/domain constraints.
        starts=[start,np.zeros(2),np.array([start[0],0]),np.array([0,start[1]])]
        results=[least_squares(fun,s,jac=jac,max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10) for s in starts]
        best=min(results,key=lambda r:np.sum(r.fun*r.fun))
        if not best.success:raise RuntimeError('joint optimization failed')
        out.append(best.x)
    return np.array(out)

def score(pred,true):
    e=abs(pred-true)
    return (int(np.any(e>.09,axis=1).sum()),float(np.max(e)),float(np.mean(e)))

def rows(m,y,ids,pred,method,protocol,fold):
    g=m.iloc[ids][['file','axis','value']].copy()
    g['true_Y']=y[ids,0];g['true_T']=y[ids,1]
    g['pred_Y']=pred[:,0];g['pred_T']=pred[:,1]
    g['eY']=abs(pred[:,0]-y[ids,0]);g['eT']=abs(pred[:,1]-y[ids,1])
    g['both_pass09']=(g.eY<=.09)&(g.eT<=.09)
    active=np.where(g.axis.to_numpy()=='Y',pred[:,0],pred[:,1])
    g['active_sign_correct']=np.sign(active)==np.sign(g.value)
    g['inactive_abs']=np.where(g.axis.to_numpy()=='Y',abs(pred[:,1]),abs(pred[:,0]))
    # Diagnostic only: no dominant-axis conversion of joint outputs.
    g['method']=method;g['protocol']=protocol;g['fold']=fold
    return g

def main():
    source=B/'center_patch_v2_20261021'
    protected=[source/'v1_edge630_features.npy',source/'v1_edge630_model.json',H.parents[1]/'motion_web_20261022/server.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    names=np.load(source/'tracks.npz')['names'];m=Z.manifest(names)
    X=np.load(source/'v1_edge630_features.npy');ids=np.flatnonzero(m.axis!='C')
    y=np.c_[np.where(m.axis=='Y',m.value,0),np.where(m.axis=='T',m.value,0)]
    saved=json.loads((B/'multiref_4fold_20261021/splits.json').read_text())
    (H/'splits.json').write_text(json.dumps(saved,indent=2))
    lookup={n:i for i,n in enumerate(names)}
    configs=[(quad,shrink) for quad in [False,True] for shrink in [.001,.05,.2,1.]]
    results=[];selection=[];conditions=[]
    for fold,s in enumerate(saved):
        tr=np.array([lookup[n] for n in s['train']]);te=np.array([lookup[n] for n in s['test']])
        assert len(tr)==210 and len(te)==70 and not set(tr)&set(te)
        inner=list(StratifiedKFold(3,shuffle=True,random_state=100+fold).split(tr,m.session.iloc[tr]))
        ranks=[]
        for k,(quad,shrink) in enumerate(configs):
            ip=[];it=[]
            for a,b in inner:
                mdl=fit_joint(X,y,tr[a],quad,shrink);ip.append(predict_joint(X[tr[b]],mdl));it.append(y[tr[b]])
            ranks.append((*score(np.vstack(ip),np.vstack(it)),k))
        chosen=min(ranks)[-1];selection.append(dict(fold=fold,config=chosen,ranking=sorted(ranks)))
        for k,(quad,shrink) in enumerate(configs):
            mdl=fit_joint(X,y,tr,quad,shrink);p=predict_joint(X[te],mdl)
            tag=f'joint_{"quadratic" if quad else "linear"}_s{shrink}'
            results.append(rows(m,y,te,p,tag,'oof4',fold))
            if k==chosen:results.append(rows(m,y,te,p,'joint_nested_selected','oof4',fold))
            linear=(mdl['co']@mdl['W'].T)[:2].T
            cosine=float(linear[:,0]@linear[:,1]/np.prod(np.linalg.norm(linear,axis=0)))
            conditions.append(dict(fold=fold,method=tag,whitened_cosine=cosine,normalized_condition=float(np.linalg.cond(linear/np.linalg.norm(linear,axis=0)))))
            np.savez(H/f'fold{fold}_{tag}.npz',**mdl)
        old=Z.predict(m,X,te,Z.fit(m,X,tr,m.value.to_numpy(),'center',True),'center')
        p=np.c_[np.where(old.pred_axis=='Y',old.pred,0),np.where(old.pred_axis=='T',old.pred,0)]
        r=rows(m,y,te,p,'existing_Y_or_T','oof4',fold)
        r['axis_correct']=old.pred_axis.to_numpy()==old.axis.to_numpy()
        r['axis_sign_correct']=(old.pred_axis.to_numpy()==old.axis.to_numpy())&(np.sign(old.pred.to_numpy())==np.sign(old.value.to_numpy()))
        results.append(r)
        print('Completed fold',fold,'selected',configs[chosen],flush=True)
    # Deployment selection uses internal CV on all available training photos.
    ranks=[]
    for k,(quad,shrink) in enumerate(configs):
        ip=[];it=[]
        for s in saved:
            tr=np.array([lookup[n] for n in s['train']]);te=np.array([lookup[n] for n in s['test']])
            ip.append(predict_joint(X[te],fit_joint(X,y,tr,quad,shrink)));it.append(y[te])
        ranks.append((*score(np.vstack(ip),np.vstack(it)),k))
    chosen=min(ranks)[-1];quad,shrink=configs[chosen]
    mdl=fit_joint(X,y,ids,quad,shrink);np.savez(H/'joint_model.npz',**mdl)
    results.append(rows(m,y,ids,predict_joint(X[ids],mdl),'joint_final_model','resub280',-1))
    old=Z.predict(m,X,ids,Z.fit(m,X,ids,m.value.to_numpy(),'center',True),'center')
    oldp=np.c_[np.where(old.pred_axis=='Y',old.pred,0),np.where(old.pred_axis=='T',old.pred,0)]
    results.append(rows(m,y,ids,oldp,'existing_Y_or_T','resub280',-1))
    p=pd.concat(results,ignore_index=True)
    p.to_csv(H/'predictions.csv',index=False,encoding='utf-8-sig')
    p[~p.both_pass09].to_csv(H/'failures.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for (method,protocol),g in p.groupby(['method','protocol']):
        groups={'all':g,'Y':g[g.axis=='Y'],'T':g[g.axis=='T'],'small_Y_le0.30':g[(g.axis=='Y')&(abs(g.value)<=.30)],'small_T_le0.05':g[(g.axis=='T')&(abs(g.value)<=.05)]}
        for subset,gg in groups.items():
            summary.append(dict(method=method,protocol=protocol,subset=subset,n=len(gg),passed=int(gg.both_pass09.sum()),pass_rate=gg.both_pass09.mean(),Y_MAE=gg.eY.mean(),Y_max=gg.eY.max(),T_MAE=gg.eT.mean(),T_max=gg.eT.max(),active_sign_accuracy=gg.active_sign_correct.mean(),inactive_MAE=gg.inactive_abs.mean(),inactive_max=gg.inactive_abs.max(),axis_accuracy=gg.axis_correct.mean() if 'axis_correct' in gg else None,axis_sign_accuracy=gg.axis_sign_correct.mean() if 'axis_sign_correct' in gg else None))
    summary=pd.DataFrame(summary);summary.to_csv(H/'summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(conditions).to_csv(H/'identifiability.csv',index=False)
    (H/'selection.json').write_text(json.dumps(dict(configs=configs,outer=selection,final_config=chosen,final_ranking=sorted(ranks)),indent=2))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    oof=p[(p.protocol=='oof4')&(p.method=='joint_nested_selected')]
    assert len(oof)==280 and oof.file.nunique()==280 and np.isfinite(oof[['pred_Y','pred_T']]).all().all()
    assert np.array_equal(predict_joint(np.zeros((1,12)),mdl),np.zeros((1,2)))
    audit=dict(preserved_sha256=hashes,feature_source=str(source/'v1_edge630_features.npy'),feature_sha256=hashes[str(source/'v1_edge630_features.npy')],measurement='fixed630 blurred boundary, all points including uncertain; existing cached features',rules='../../ALGORITHM_RULES.md',images=280,final_config=configs[chosen],model='X=sY*Y+sT*T+cY*Y^2+cT*T^2 (quadratic only); no interaction, no offset, no clipping, no hard zero axis',selection='3-fold inner CV within each outer210, lexicographic vector failures/max error/MAE; final full-data configuration selection has separate resub label',limitations=['Commanded labels, not independently measured physical poses','No mixed Y+T photographs: additivity and zero Y*T interaction unvalidated','Same acquisition data, not independent capture','Exact center gives exact zero; no repeated neutral captures to measure false alarms'])
    (H/'audit.json').write_text(json.dumps(audit,indent=2))
    important=summary[(summary.protocol=='oof4')&summary.method.isin(['existing_Y_or_T','joint_nested_selected'])]
    report='<meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1400px;margin:40px auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}a{color:#1264a3}</style><h1>Y·T 동시 추정 — 기존 방식 비교</h1><p><a href="../../ALGORITHM_RULES.md">공통 알고리즘 규칙</a></p><p>고정 630개 구조 모서리 경계 측정과 P12 특징을 양쪽 모델에 동일하게 사용. 기존 알고리즘·웹·결과 보존. 고정4-fold에서 학습210/시험70, 총280장 OOF. 동시 모델 설정은 학습210장 내부3-fold에서 선택.</p><p>주 지표: |Y오차|≤0.09 AND |T오차|≤0.09. Y 사진 정답=(명령Y,0), T 사진 정답=(0,명령T). 축 선택 성공을 통과조건으로 추가하지 않으며 기존 축·부호 정확도는 별도 표시. 활성 축 부호는 동시 모델에서도 평가. 비활성 축 누출은 표의 inactive 열. small 구간은 진단용이며 모델 선택에 별도 가중하지 않음. T 물리 단위는 미확인.</p>'+important.to_html(index=False,float_format=lambda v:f'{v:.6f}')+'<h2>모델과 검증 한계</h2><p>동시 모델은 두 값을 자유롭게 맞추는 GLS 선형 또는 가산 이차 모델이다. 한 축을 선택하거나 0으로 강제하지 않는다. Y×T 항은 현재 자료에서 항상0이라 식별할 수 없어 사용하지 않는다. 실제 혼합 자세의 정확도는 미검증이며, 현재 결과를 혼합 성능으로 주장할 수 없다. 정답은 파일명 명령값이며 독립 실측 자세가 아니다.</p><h2>전체 설정과 재대입 (주 성적과 구분)</h2>'+summary.to_html(index=False,float_format=lambda v:f'{v:.6f}')+'<p><a href="predictions.csv">사진별 예측</a> · <a href="failures.csv">실패 목록</a> · <a href="selection.json">내부 선택</a> · <a href="identifiability.csv">이동 모양 유사도 진단</a> · <a href="audit.json">출처·보존 검증</a></p>'
    (H/'report.html').write_text(report,encoding='utf-8')
    print(important.to_string(index=False),flush=True)

if __name__=='__main__':main()
