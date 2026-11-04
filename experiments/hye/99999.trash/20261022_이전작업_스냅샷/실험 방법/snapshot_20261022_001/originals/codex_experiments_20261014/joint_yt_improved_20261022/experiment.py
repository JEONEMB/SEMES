"""Four staged searches, all selection inside each outer training partition."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,hashlib,warnings,html
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from engine import *
warnings.filterwarnings('ignore',category=ConvergenceWarning)
sys.path.insert(0,str(B));import zero_experiment as Z

BASE_MODEL=dict(family='forward',cov='shrink',quad=False,loss='linear')

def rank_score(pred,true):
    if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
    e=abs(pred-true)
    return [int(np.any(e>.09,axis=1).sum()),float(e.max()),float(e.mean())]

def metrics(p,y):
    e=abs(p-y);active=abs(y)>0;inactive=~active
    return dict(n=len(y),passed=int(np.all(e<=.09,axis=1).sum()),pass_rate=float(np.all(e<=.09,axis=1).mean()),Y_MAE=float(e[:,0].mean()),Y_max=float(e[:,0].max()),T_MAE=float(e[:,1].mean()),T_max=float(e[:,1].max()),active_sign_accuracy=float((np.sign(p[active])==np.sign(y[active])).mean()),inactive_MAE=float(abs(p[inactive]).mean()) if inactive.any() else None,inactive_max=float(abs(p[inactive]).max()) if inactive.any() else None)

class Study:
    def __init__(self):
        self.obs=load_observations();self.names=self.obs['names'];self.m=Z.manifest(self.names)
        self.y=np.c_[np.where(self.m.axis=='Y',self.m.value,0),np.where(self.m.axis=='T',self.m.value,0)]
        self.ids=np.flatnonzero(self.m.axis!='C');self.lookup={n:i for i,n in enumerate(self.names)}
        self.cache={};self.records=[];self.selections=[];self.search=[];self.saved=[]
    def features(self,c):
        key=(c['measurement'],c['feature'])
        if key not in self.cache:
            path=H/'features'/('_'.join(key)+'.npy');path.parent.mkdir(exist_ok=True)
            if path.exists():X=np.load(path)
            else:
                X=np.stack([extract(self.obs['points'],d,q,*key) for d,q in zip(self.obs['D'],self.obs['q'])]);np.save(path,X)
            assert np.isfinite(X).all();self.cache[key]=X
            print('features',key,X.shape,flush=True)
        return self.cache[key]
    def fit(self,c,tr):return Stabilized(c['model'],c['policy']).fit(self.features(c)[tr],self.y[tr])
    def rows(self,p,te,name,fold,stage,config,protocol='oof4'):
        out=pd.DataFrame(dict(file=self.names[te],axis=self.m.axis.iloc[te].to_numpy(),value=self.m.value.iloc[te].to_numpy(),true_Y=self.y[te,0],true_T=self.y[te,1],pred_Y=p[:,0],pred_T=p[:,1]))
        e=abs(p-self.y[te]);out['eY']=e[:,0];out['eT']=e[:,1];out['both_pass09']=np.all(e<=.09,axis=1)
        out['method']=name;out['stage']=stage;out['fold']=fold;out['protocol']=protocol;out['config']=json.dumps(config,sort_keys=True)
        self.records.append(out)
    def select(self,candidates,tr,fold,stage,te=None):
        inner=list(StratifiedKFold(3,shuffle=True,random_state=900+max(fold,0)).split(tr,self.m.session.iloc[tr]))
        ranks=[]
        for k,c in enumerate(candidates):
            pred=[];truth=[];start=time.perf_counter()
            for aa,bb in inner:
                model=self.fit(c,tr[aa]);pred.append(model.predict(self.features(c)[tr[bb]]));truth.append(self.y[tr[bb]])
            score=rank_score(np.vstack(pred),np.vstack(truth));ranks.append((*score,k))
            self.search.append(dict(fold=fold,stage=stage,candidate=k,config=c,inner_score=score,seconds=time.perf_counter()-start))
            if te is not None:
                model=self.fit(c,tr);p=model.predict(self.features(c)[te]);self.rows(p,te,f'{stage}_candidate{k:02d}',fold,stage,c)
        chosen=min(ranks)[-1];best=candidates[chosen]
        self.selections.append(dict(fold=fold,stage=stage,chosen=chosen,config=best,inner_score=list(min(ranks)[:3])))
        print('selection',fold,stage,chosen,best,'inner',min(ranks)[:3],flush=True)
        return best
    def pipeline(self,tr,fold,te=None):
        c=dict(measurement='uniform',feature='global2',model=BASE_MODEL,policy='none')
        if te is not None:self.rows(self.fit(c,tr).predict(self.features(c)[te]),te,'stage0_common_origin',fold,'0',c)
        candidates=[dict(c,measurement=mm) for mm in MEASUREMENTS]
        c=self.select(candidates,tr,fold,'A_measurement',te)
        if te is not None:self.rows(self.fit(c,tr).predict(self.features(c)[te]),te,'stageA_selected',fold,'A',c)
        c=self.select([dict(c,feature=f) for f in FEATURES],tr,fold,'B_features',te)
        if te is not None:self.rows(self.fit(c,tr).predict(self.features(c)[te]),te,'stageB_selected',fold,'B',c)
        c=self.select([dict(c,model=mm) for mm in model_configs()],tr,fold,'C_models',te)
        if te is not None:self.rows(self.fit(c,tr).predict(self.features(c)[te]),te,'stageC_selected',fold,'C',c)
        c=self.select([dict(c,policy=pp) for pp in POLICIES],tr,fold,'D_stabilization',te)
        model=self.fit(c,tr)
        if te is not None:
            p=model.predict(self.features(c)[te]);self.rows(p,te,'improved_nested',fold,'D',c)
            self.saved.append((fold,c,model,te,tr))
            joblib.dump(dict(config=c,model=model,ids=self.obs['ids'],points=self.obs['points']),H/f'fold{fold}_model.joblib')
            self.diagnostics(fold,tr,te,c,p)
        return c,model
    def diagnostics(self,fold,tr,te,c,p):
        X=self.features(dict(measurement='uniform',feature='global2'))
        y=self.y;A=np.c_[np.ones(len(tr)),y[tr]];co=np.linalg.lstsq(A,X[tr],rcond=None)[0];r=X[tr]-A@co
        std=np.maximum(np.std(r,axis=0),1e-8);C=(r/std).T@(r/std)/len(r);C=.999*C+.001*np.eye(len(std))
        vals,U=np.linalg.eigh(C);W=((U/np.sqrt(np.maximum(vals,1e-8)))@U.T)/std[:,None]
        directions=co[1:]@W;cos=float(directions[0]@directions[1]/np.prod(np.linalg.norm(directions,axis=1)))
        fisher=directions@directions.T;cov=np.linalg.inv(fisher)
        ev,vec=np.linalg.eigh(cov)
        # Residual includes model error and label error; this is not a physical sensor CRLB.
        diagnostic=dict(fold=fold,cosine=cos,angle_degrees=float(np.degrees(np.arccos(np.clip(cos,-1,1)))),normalized_condition=float(np.linalg.cond(directions.T/np.linalg.norm(directions,axis=1))),residual_based_Y_std=float(np.sqrt(cov[0,0])),residual_based_T_std=float(np.sqrt(cov[1,1])),weak_direction=vec[:,-1].tolist(),error_correlation=float(np.corrcoef((p-y[te]).T)[0,1]))
        (H/f'diagnostic_fold{fold}.json').write_text(json.dumps(diagnostic,indent=2))

def protected_hashes():
    roots=[B/'joint_yt_20261022',B.parent/'motion_web_20261022',B/'center_patch_v2_20261021']
    files=[]
    for root in roots:
        files.extend(p for p in root.iterdir() if p.is_file() and p.suffix in ['.py','.json','.npz','.npy','.csv','.html'])
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def main():
    start=time.perf_counter();protected=protected_hashes();s=Study()
    resume='--resume' in sys.argv
    splits=json.loads((B/'joint_yt_20261022/splits.json').read_text());(H/'splits.json').write_text(json.dumps(splits,indent=2))
    registry=dict(measurements=MEASUREMENTS,features=FEATURES,models=model_configs(),policies=POLICIES,selection='greedy staged inner3-fold; failures,maxerror,MAE; same preset method menu in all outer folds',outer='saved 4-fold; development comparison on reused historical splits')
    (H/'registered_methods.json').write_text(json.dumps(registry,indent=2))
    for f,split in enumerate(splits):
        tr=np.array([s.lookup[n] for n in split['train']]);te=np.array([s.lookup[n] for n in split['test']])
        if resume:
            bundle=joblib.load(H/f'fold{f}_model.joblib');s.saved.append((f,bundle['config'],bundle['model'],te,tr));continue
        s.pipeline(tr,f,te)
        pd.concat(s.records,ignore_index=True).to_csv(H/'progress_predictions.csv',index=False)
        (H/'progress_selection.json').write_text(json.dumps(s.selections,indent=2))
        print('finished outer',f,'elapsed',round(time.perf_counter()-start),flush=True)
    if resume:
        s.records=[pd.read_csv(H/'progress_predictions.csv')]
        s.selections=json.loads((H/'selection.json').read_text());s.search=json.loads((H/'search.json').read_text())
    # Prior method scores stay byte-for-byte preserved; import their recorded OOF predictions.
    old=pd.read_csv(B/'joint_yt_20261022/predictions.csv')
    old=old[(old.protocol=='oof4')&old.method.isin(['existing_Y_or_T','joint_nested_selected'])].copy();old['stage']='prior';old['config']='previous experiment'
    p=pd.concat(s.records+[old],ignore_index=True)
    p.to_csv(H/'predictions.csv',index=False,encoding='utf-8-sig');p[~p.both_pass09].to_csv(H/'failures.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for method,g in p.groupby('method'):
        subsets={'all':g,'Y':g[g.axis=='Y'],'T':g[g.axis=='T'],'smallY':g[(g.axis=='Y')&(abs(g.value)<=.30)],'smallT':g[(g.axis=='T')&(abs(g.value)<=.05)]}
        for subset,gg in subsets.items():
            summary.append(dict(method=method,subset=subset,**metrics(gg[['pred_Y','pred_T']].to_numpy(),gg[['true_Y','true_T']].to_numpy())))
    summary=pd.DataFrame(summary);summary.to_csv(H/'summary.csv',index=False,encoding='utf-8-sig')
    (H/'selection.json').write_text(json.dumps(s.selections,indent=2));(H/'search.json').write_text(json.dumps(s.search,indent=2))
    # The all-data final model is selected separately using training-only3-fold.
    if resume:
        bundle=joblib.load(H/'model.joblib');final_config,model=bundle['config'],bundle['model']
        s.selections+=json.loads((H/'final_selection.json').read_text())['selections']
    else:final_config,model=s.pipeline(s.ids,-1)
    joblib.dump(dict(config=final_config,model=model,ids=s.obs['ids'],points=s.obs['points']),H/'model.joblib')
    finalp=model.predict(s.features(final_config)[s.ids]);resub=metrics(finalp,s.y[s.ids])
    pd.DataFrame(dict(file=s.names[s.ids],Y=finalp[:,0],T=finalp[:,1])).to_csv(H/'resub_predictions.csv',index=False)
    (H/'final_selection.json').write_text(json.dumps(dict(config=final_config,selections=[a for a in s.selections if a['fold']==-1],resub=resub),indent=2))
    synthetic=[];noise=[]
    for f,c,mdl,te,tr in s.saved:
        X=s.features(c);rng=np.random.default_rng(200+f);yi=te[s.y[te,0]!=0];ti=te[s.y[te,1]!=0]
        i=np.repeat(yi,len(ti));j=np.tile(ti,len(yi))
        origin=np.linalg.lstsq(np.c_[np.ones(len(tr)),s.y[tr],s.y[tr]**2],X[tr],rcond=None)[0][0]
        pp=mdl.predict(X[i]+X[j]-origin);truth=s.y[i]+s.y[j]
        synthetic.append(pd.DataFrame(dict(fold=f,Y_source=s.names[i],T_source=s.names[j],true_Y=truth[:,0],true_T=truth[:,1],pred_Y=pp[:,0],pred_T=pp[:,1])))
        A=np.c_[np.ones(len(tr)),s.y[tr],s.y[tr]**2];r=X[tr]-A@np.linalg.lstsq(A,X[tr],rcond=None)[0]
        for scale in [.25,.5,1.]:
            rr=rng.choice(len(r),len(te));perturbed=X[te]+scale*r[rr]
            noise.append(dict(fold=f,residual_noise_scale=scale,**metrics(mdl.predict(perturbed),s.y[te])))
    synth=pd.concat(synthetic,ignore_index=True);synth.to_csv(H/'synthetic_mixed_predictions.csv',index=False)
    synthstats=metrics(synth[['pred_Y','pred_T']].to_numpy(),synth[['true_Y','true_T']].to_numpy())
    (H/'synthetic_mixed_summary.json').write_text(json.dumps(synthstats,indent=2));pd.DataFrame(noise).to_csv(H/'feature_noise_stress.csv',index=False)
    # Paired descriptive bootstrap. Historical images are not independent new sessions.
    primary=p[p.method=='improved_nested'].set_index('file').sort_index();prior=p[p.method=='joint_nested_selected'].set_index('file').loc[primary.index]
    gain=primary.both_pass09.astype(int).to_numpy()-prior.both_pass09.astype(int).to_numpy();rng=np.random.default_rng(77)
    boot=np.mean(gain[rng.integers(len(gain),size=(2000,len(gain)))],axis=1)
    assert len(primary)==280 and primary.index.nunique()==280 and np.isfinite(primary[['pred_Y','pred_T']]).all().all()
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==v for path,v in protected.items())
    audit=dict(protected_sha256=protected,preserved_files=len(protected),observation_ids=630,diagnostic_valid_fraction=float(s.obs['valid'][s.ids].mean()),minimum_photo_valid_fraction=float(s.obs['valid'][s.ids].mean(1).min()),total_seconds=time.perf_counter()-start,final_config=final_config,paired_pass_gain=float(gain.mean()),descriptive_bootstrap_gain_interval=np.quantile(boot,[.025,.975]).tolist(),synthetic_mixed=synthstats,notes=['No hard zero-axis conversion or axis classification at inference','Nonlinear regressors learn from axis-only support; this can implicitly favor single-axis answers','No new QR, photometric, blur or actual mixed capture generalization validation','Feature residual noise is model/label error plus image noise, not independent sensor noise','Repeated use of historical folds is development evaluation; not blind independent validation'])
    (H/'audit.json').write_text(json.dumps(audit,indent=2));write_report(summary,p,audit)
    print('PRIMARY',summary[(summary.subset=='all')&summary.method.isin(['existing_Y_or_T','joint_nested_selected','stage0_common_origin','stageA_selected','stageB_selected','stageC_selected','improved_nested'])].to_string(index=False),flush=True)

def write_report(summary,p,audit):
    keys=['existing_Y_or_T','joint_nested_selected','stage0_common_origin','stageA_selected','stageB_selected','stageC_selected','improved_nested']
    main=summary[(summary.subset=='all')&summary.method.isin(keys)].set_index('method').loc[keys].reset_index()
    research=json.loads((H/'research.json').read_text())
    refs=''.join('<li><a href="'+html.escape(r['url'])+'">'+html.escape(r['title'])+'</a> — '+html.escape(r['applied'])+'<br>'+html.escape(r['scope'])+'</li>' for r in research)
    diagnostics=[json.loads((H/f'diagnostic_fold{f}.json').read_text()) for f in range(4)]
    plan=[('A 관측 집계','밝기 오차 가중 / 창 안정성 가중 / 대비 종합 가중 / Huber / soft-L1 / Cauchy','픽셀 좌표는 기존 경계 측정 그대로. 모든 점의 가중치≥0.001, 점 교체·제거 없음.'),('B 특징 표현','전체 affine / 전체 cubic / QR별 affine / QR별 quadratic / 12분면 평균 / homography','A에서 고른 집계로 표현을 바꿈. global2 기준 포함 7개.'),('C 회귀·역추정','GLS 공분산 5종, 이차 및 강건 역추정, Ridge / Bayesian / ARD / PCR / PLS / KernelRidge / SVR / ExtraTrees','공통 절편·스케일·차원축소·공분산은 학습 분할에서만 추정. 두 값을 연속 출력.'),('D 안정화','bootstrap 평균 / bootstrap 중앙값 / subagging / 잡음 증강 / convex mixup / 가산 Y+T / 축 균형 재표본화','C의 선택 모델을 유지하고 추가. 합성은 학습 데이터로만 생성. none 대조군 포함 8개.')]
    body='<meta charset="utf-8"><title>Y·T 동시 추정 개선 실험</title><style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;line-height:1.65;color:#172b44}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #d8e0e8;text-align:right}th:first-child,td:first-child{text-align:left}h1,h2{color:#164979}a{color:#1264a3}.note{padding:16px;background:#fff2d9;border-radius:8px}img{width:100%;max-width:1100px}</style><h1>Y·T 동시 추정: 원인 분석과 단계별 개선</h1><p><a href="../../ALGORITHM_RULES.md">공통 규칙</a> · 기존 알고리즘·첫 동시 모델·웹 보존 SHA-256 '+str(audit['preserved_files'])+'개 파일 확인.</p><p class="note">실제 혼합 사진이 없는 개발용 비교다. 같은 4-fold를 반복 사용했고 새 독립 촬영 성적이 아니다. 비선형 모델은 단일 축 학습 분포를 익혀 성적이 높아질 수 있으므로, 단일 축 성능 개선만으로 혼합 추정 능력을 주장하지 않는다.</p><h2>같은 280장 OOF 비교</h2><p>두 축 절대오차 모두≤0.09가 통과. 정답은 명령값. 각 외부 fold 학습210/시험70, 네 단계의 방법 선택은 학습210장 내부3-fold. greedy 단계 선택이며 전 조합 최적화가 아니다. stage0는 공통 학습 절편을 추가한 선형 GLS 기준으로 첫 원점통과 모델과 구분한다.</p>'+main.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'<h2>단계마다 추가된 것</h2>'+pd.DataFrame(plan,columns=['단계','추가 방법','적용 범위']).to_html(index=False)+'<h2>왜 떨어졌나</h2><p>학습 잔차로 백색화한 Y/T 방향의 각도와 조건수를 계산했다. 방향이 비슷하면 작은 잔차를 두 값으로 나눌 때 서로 상쇄되는 Y/T 오차가 커진다. 아래 표의 residual_based 표준편차는 모델·명령값 오차를 포함하므로 센서의 물리적 한계나 순수 촬영 잡음으로 해석할 수 없다.</p>'+pd.DataFrame(diagnostics).to_html(index=False,float_format=lambda x:f'{x:.5f}')+'<h2>원인과 개선 그림</h2><img src="analysis.png"><h2>최종 모델의 혼합·잡음 보조 진단</h2><p>합성 혼합은 각 외부 시험 fold의 Y 특징+T 특징−학습 공통 원점으로 구성했다. 실제 혼합 영상이 아니며 가산 가정이 틀리면 이 진단도 틀린다. '+html.escape(json.dumps(audit['synthetic_mixed']))+'</p><p>학습 잔차를 시험 특징에 추가한 feature_noise_stress.csv도 저장했다. 픽셀 밝기·QR 내용 변경·흐림의 실제 검증은 아니다.</p><h2>전체 후보와 세부 구간</h2><p>A/B/C/D_candidate 번호는 fold별 이전 단계 선택에 이어 적용한 조건부 후보다. 후보 자체 OOF 최고치를 최종 성능으로 고르지 않았다. 주 성적은 improved_nested 행이다.</p>'+summary.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'<h2>찾아보고 실제 적용한 논문·공식 자료</h2><ol>'+refs+'</ol><h2>결과 파일</h2><p><a href="predictions.csv">사진별 예측</a> · <a href="failures.csv">실패</a> · <a href="selection.json">외부 fold 내부 선택</a> · <a href="registered_methods.json">방법 목록</a> · <a href="final_selection.json">최종 모델과 재대입</a> · <a href="synthetic_mixed_predictions.csv">합성 혼합 보조 진단</a> · <a href="audit.json">검증·보존 기록</a></p>'
    (H/'report.html').write_text(body,encoding='utf-8')
    make_figure(p,main)

def make_figure(p,main):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    labels=['Existing OR','First joint','Common origin','A weighting','B geometry','C regression','D stabilization']
    axs[0,0].bar(labels,main.pass_rate.to_numpy()*100,color=['#888','#c65a46','#789','#5e98ad','#5e98ad','#297b9b','#12507c']);axs[0,0].set_ylim(0,105);axs[0,0].set_ylabel('Both-axis pass (%)');axs[0,0].tick_params(axis='x',rotation=25)
    for method,color in [('joint_nested_selected','#c65a46'),('improved_nested','#12507c')]:
        g=p[p.method==method];axs[0,1].scatter(g.pred_Y-g.true_Y,g.pred_T-g.true_T,s=12,alpha=.5,label=method,color=color)
    axs[0,1].set_xlabel('Y error');axs[0,1].set_ylabel('T error');axs[0,1].legend(fontsize=8)
    for col,axis in enumerate(['Y','T']):
        g=p[(p.method=='improved_nested')&(p.axis==axis)]
        axs[1,col].scatter(g['true_'+axis],g['pred_'+axis],s=15,alpha=.6,label='active estimate')
        other='T' if axis=='Y' else 'Y';axs[1,col].scatter(g['true_'+axis],g['pred_'+other],s=12,alpha=.5,label='inactive leakage')
        xx=np.array([g['true_'+axis].min(),g['true_'+axis].max()]);axs[1,col].plot(xx,xx,'k--',lw=1);axs[1,col].axhline(0,color='gray',lw=1)
        axs[1,col].set_xlabel('Command '+axis);axs[1,col].set_ylabel('Predicted coordinate');axs[1,col].legend()
    for ax in axs.ravel():ax.grid(alpha=.2)
    fig.suptitle('Fixed 630 boundary observations | nested staged selection | historical OOF, no real mixed poses')
    fig.tight_layout();fig.savefig(H/'analysis.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
