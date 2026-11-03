"""Audit native CSV results and render the development comparison."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
OLD=P.parent/'20261102_173432_t_six_methods'

def metrics(pred,target):
    e=(np.asarray(pred)-np.asarray(target))/.01
    assert np.isfinite(e).all()
    return {'n':len(e),'MAE_tick':np.abs(e).mean(),'RMSE_tick':np.sqrt(np.mean(e*e)),
            'max_tick':np.abs(e).max(),'pass_1tick':int((np.abs(e)<=1+1e-12).sum()),'bias_tick':e.mean()}

def main():
    excluded=pd.read_csv(P/'exclusions.csv');ex=set(excluded.path)
    original=pd.read_csv(OLD/'observations.csv');rawfold={}
    def fold(y,t):
        h=2166136261
        for c in f'{round(y*100)}:{round(t*100)}'.encode():h=((h^c)*16777619)&0xffffffff
        return h%5
    for r in original.itertuples():rawfold[r.path]=fold(r.Y,r.T)
    expected=set(original.loc[(original.cohort!='reference')&~original.path.isin(ex),'path'])
    refs=set(original.loc[original.cohort=='reference','path'])
    records=[];cohorts=[];checks=[];data={}
    names=['nominal_exclude0','nominal_exclude8','nominal_exclude5','standard_crossfit_exclude8','nominal_group_exclude8']
    methods=['raw','global_oof','session_oof','session_shrink30','reference_zero']
    for name in names:
        base=P/'results'/name;d=pd.read_csv(base/'predictions.csv');data[name]=d
        assert not d.path.duplicated().any()
        if 'group' not in name:assert all(int(r.fold)==rawfold[r.path] for r in d.itertuples())
        wanted=set(original.loc[original.cohort!='reference','path']) if 'exclude0' in name else expected if 'exclude8' in name else set(original.loc[(original.cohort!='reference')&~original.path.isin(set(excluded.loc[excluded.sensitivity_exclude==1,'path'])),'path'])
        assert set(d.path)==wanted
        assert d[methods].apply(np.isfinite).all().all()
        audit=pd.read_csv(base/'isolation.csv');assert audit.overlap.eq(0).all()
        inner=pd.read_csv(base/'inner_predictions.csv');cal=pd.read_csv(base/'calibration.csv')
        for outer,part in inner.groupby('outer'):
            held=set(d.loc[d.fold==outer,'path']);assert not held&set(part.path)
            banned=ex if 'exclude8' in name else set(excluded.loc[excluded.sensitivity_exclude==1,'path']) if 'exclude5' in name else set()
            assert not set(part.path)&banned
            if 'group' not in name:assert all(rawfold[path]!=outer for path in part.path)
            used=part[part.used_for_calibration==1].copy();used['delta']=used.target-used.prediction
            global_shift=used.delta.mean();c=cal[cal.fold==outer]
            assert np.allclose(c.global_shift,global_shift,atol=1e-12)
            for row in c.itertuples():
                q=used[used.group==row.group];assert row.n==len(q)
                assert abs(row.session_shift-q.delta.mean())<1e-12
                assert abs(row.shrink30_shift-(q.delta.sum()+30*global_shift)/(len(q)+30))<1e-12
            for row in d[d.fold==outer].itertuples():
                match=c[c.group==row.group]
                assert abs(row.global_oof-row.raw-global_shift)<1e-12
                for column,key in [('session_oof','session_shift'),('session_shrink30','shrink30_shift')]:
                    shift=match.iloc[0][key] if len(match) else global_shift
                    assert abs(getattr(row,column)-row.raw-shift)<1e-12
        if 'group' in name:assert d.known_group.eq(0).all()
        for target in ['T_original','T_target']:
            q=d if target=='T_original' else d[d.target_available==1]
            for method in methods:
                records.append({'experiment':name,'method':method,'target':target,**metrics(q[method],q[target])})
        for cohort,q in d.groupby('cohort'):
            for method in ['raw','session_shrink30']:
                cohorts.append({'experiment':name,'cohort':cohort,'method':method,**metrics(q[method],q.T_original)})
        failure=d.assign(error_ticks=(d.raw-d.T_original)/.01)
        failure=failure[failure.error_ticks.abs()>1].sort_values('error_ticks',key=abs,ascending=False)
        failure.to_csv(base/'failures_nominal_raw.csv',index=False)
        checks.append({'case':name,'n':len(d),'unique':True,'calibration_audit':True,'outer_leakage':0})
    # Audit every origin's training membership independently of its exported coordinates.
    origin_checks=[]
    for file in sorted((P/'origins').glob('*.json')):
        info=json.loads(file.read_text(encoding='utf-8'));train=set(info['train_paths']);assert not train&ex;assert refs<=train
        if file.stem!='full':
            parts=file.stem.split('_');heldfolds=[int(v.replace('outer','').replace('inner','')) for v in parts]
            assert all(path in refs or rawfold[path] not in heldfolds for path in train)
        origin_checks.append({'case':file.stem,'n':len(train),'bound_hits':info['raw_bound_hits']})
    assert len(origin_checks)==26
    pd.DataFrame(origin_checks).to_csv(P/'results/origin_isolation.csv',index=False)
    # Compare both models against exactly the same fold-specific standard coordinates.
    nominal=data['nominal_exclude8'];standard=data['standard_crossfit_exclude8']
    cross=nominal[['path','raw']].rename(columns={'raw':'nominal_prediction'}).merge(standard,on='path',validate='one_to_one')
    cross=cross[cross.target_available==1]
    for method in ['nominal_prediction','raw']:
        records.append({'experiment':'same_standard_target','method':method,'target':'T_target',**metrics(cross[method],cross.T_target)})
    old=pd.read_csv(OLD/'results/predictions_method7.csv')
    control=data['nominal_exclude0'].merge(old[['path','T_pred']],on='path',validate='one_to_one')
    reproduced_difference=float((control.raw-control.T_pred).abs().max())
    assert reproduced_difference<1e-10
    for name,q in [('historical_all',old),('historical_filtered8',old[~old.path.isin(ex)])]:
        records.append({'experiment':name,'method':'frozen_OOF_diagnostic','target':'T_original',**metrics(q.T_pred,q.T_true)})
    # Oracle is explicitly descriptive and never used to change a prediction/model.
    q=nominal.copy();e=q.raw-q.T_original;oracle=q.raw-e.groupby(q.group).transform('mean')
    records.append({'experiment':'oracle_NOT_predictive','method':'test_session_mean_removed','target':'T_original',**metrics(oracle,q.T_original)})
    summary=pd.DataFrame(records);summary.to_csv(P/'results/summary.csv',index=False)
    pd.DataFrame(cohorts).to_csv(P/'results/cohorts.csv',index=False)
    saved=original.merge(excluded,on='path',validate='one_to_one');saved.to_csv(P/'results/excluded_samples.csv',index=False)
    verification={'historical571_prediction_max_difference_deg':reproduced_difference,'native_cases':checks,'origin_fits':26,'origin_train_test_and_exclusion_checks':True,
                  'calibration_recomputed_from_training_inner_OOF':True,'all_prediction_values_finite':True,
                  'standard_target_unavailable':standard.loc[standard.target_available==0,'path'].tolist(),
                  'origin_bound_hit_range':[min(v['bound_hits'] for v in origin_checks),max(v['bound_hits'] for v in origin_checks)]}
    (P/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    fig,ax=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    main=summary[(summary.experiment=='nominal_exclude8')&(summary.target=='T_original')].set_index('method').loc[methods]
    labels=['보정 없음','공통 절편','세션 평균','세션 축소 보정','기준4장 원점']
    x=np.arange(len(labels))
    for key,offset,color,label in [('MAE_tick',-.25,'#0072B2','MAE'),('RMSE_tick',0,'#009E73','RMSE'),('max_tick',.25,'#D55E00','최대 절대오차')]:
        bars=ax[0].bar(x+offset,main[key],.24,color=color,label=label)
        ax[0].bar_label(bars,fmt='%.2f',fontsize=8)
    ax[0].axhline(1,color='#777',ls='--',lw=1);ax[0].set_xticks(x,labels,rotation=15);ax[0].set_ylabel('tick (1 tick = 0.01°)');ax[0].legend(fontsize=9);ax[0].set_title('8장 제외 후 563장: 원래 명목 라벨 기준')
    q=nominal.copy();q['raw_error']=(q.raw-q.T_original)/.01;q['corrected_error']=(q.session_shrink30-q.T_original)/.01
    for method,color,label in [('raw_error','#888888','보정 없음'),('corrected_error','#0072B2','세션 축소 보정')]:
        g=q.groupby('group')[method].mean();ax[1].plot(g.index,g.values,'o-',label=label,color=color)
    ax[1].axhline(0,color='black',lw=.7);ax[1].set_xlabel('수정시각/스캔으로 나눈 구간 ID');ax[1].set_ylabel('평균 부호 오차 (tick)');ax[1].set_title('평균 바이어스는 줄어도 최악 오차는 남음');ax[1].legend()
    fig.suptitle('동일 자세를 함께 보류한 개발 교차검증 — 시험 라벨로 보정하지 않음');fig.savefig(P/'comparison.png',dpi=170);plt.close(fig)
    def tab(frame):
        lines=['| 방법 | n | MAE tick | RMSE tick | 최대 tick | 1tick 이내 |','|---|---:|---:|---:|---:|---:|']
        for r in frame.itertuples():lines.append(f'| {r.method} | {r.n} | {r.MAE_tick:.4f} | {r.RMSE_tick:.4f} | {r.max_tick:.4f} | {r.pass_1tick}/{r.n} |')
        return '\n'.join(lines)
    text='# T 라벨 감사 후 재평가 결과\n\n'
    text+='주 평가: 이름 충돌1+Actual 충돌3+반복 위치불일치4 제외. 기준4장은 별도 보정용. 총563장. 원본은 보존했다.\n\n'
    text+='## 같은 명목 라벨 기준\n\n'+tab(main.reset_index())+'\n\n'
    text+='세션 보정은 학습 내부 OOF 잔차에서 계산했다. 순수 세션 평균은 MAE가 작아져도 최대오차를 악화시켰다. 축소 보정은 평균 지표를 줄이지만 최대1tick은 달성하지 못했다. 공통 원점을 기준4장 평균0으로 강제하는 방식은 명목 라벨과 불일치를 키웠다.\n\n'
    text+='## .57 세 장을 복원한 민감도: 566장\n\n'+tab(summary[(summary.experiment=='nominal_exclude5')&(summary.target=='T_original')])+'\n\n'
    text+='## 표준 원점 v3 가정\n\n원점도 제외8장을 빼고 26회 재적합했다. 원점 보정이 식별되지 않는3장은 표준 좌표 지표에서 제외하고, 명목 라벨 지표에서는 포함했다.\n\n'
    text+=tab(summary[summary.experiment=='same_standard_target'])+'\n\n'
    text+='위 표는 같은 통계적 목표 좌표에 두 모델을 비교한다. 원점 모델이 만든 좌표에 잘 맞는다는 뜻이며 물리적 정확도의 증거가 아니다. 표준 모델의 원래 라벨 기준 점수는 다음과 같다.\n\n'
    text+=tab(summary[(summary.experiment=='standard_crossfit_exclude8')&(summary.target=='T_original')&(summary.method=='raw')])+'\n\n'
    text+='## 촬영 구간 통째 보류\n\n'+tab(summary[(summary.experiment=='nominal_group_exclude8')&(summary.target=='T_original')&(summary.method.isin(['raw','global_oof','session_oof','session_shrink30']))])+'\n\n'
    text+='처음 보는 구간의 바이어스는 학습에서 알 수 없어 공통 절편으로 돌아간다. 구간별 자세 범위도 달라 이 결과는 촬영 세션 영향만 분리한 실험이 아니다.\n\n'
    text+='## 과거 예측 고정 대조군\n\n'+tab(summary[summary.experiment.isin(['historical_all','historical_filtered8'])])+'\n\n'
    text+='두 행은 각각 전체571장과 제외 후563장이다. 이미 학습된 예측을 필터링했으므로 제외 후 재학습 성능으로 사용하지 않는다. 삭제와 재선택이 성능을 반드시 높이지는 않는다.\n\n'
    text+='제외하지 않은571장을 새 도구로 재실행한 대조군은 과거 저장 예측과 최대차이0으로 일치했다. 따라서 새 평가 도구의 구현 차이로 생긴 성능 변화는 아니다. 제외 후 재학습·후보 및 결합 재선택의 변화가 포함된다.\n\n'
    text+='## 판단\n\n1. 명목 라벨 재현이 목표면 제외 후 원래 라벨과의 직접 오차가 주 지표로 가장 명확하다.\n2. 알려진 구간을 별도 보정하고 운영 시 구간ID를 제공할 수 있다면 학습/calibration 잔차를 쓴 축소 보정을 보조 지표로 쓸 수 있다. 순수 사진 추론과 조건이 다르다.\n3. 표준 좌표는 공통 원점 규약에 대한 조건부 분석으로 유지한다. 보정량이 제약 경계에 닿아 있고 원점/바이어스를 독립 계측하지 못했으므로 새 정답으로 확정하지 않는다.\n4. 시험 정답으로 구간 평균을 빼거나 원점 좌표를 고르면 예측 성능으로 타당하지 않다. summary.csv의 oracle 행은 진단 전용이다.\n5. 모든 성적은 이미 탐색한 개발 자료다. 1tick 목표를 달성한 방식은 없다.\n\n'
    text+='상세 예측·실패 목록·구간별 calibration·학습 내부 OOF·분할 격리·표준 좌표 식별 여부를 results/와 origins/에 보존했다. verification.json은 보정량과 학습/시험 격리를 독립 재계산한 결과다.\n'
    (P/'RESULTS.md').write_text(text,encoding='utf-8')
    hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P.rglob('*')) if f.is_file() and f.suffix in ['.cpp','.py','.csv','.json','.md'] and f.name!='SHA256.json'}
    (P/'SHA256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[['experiment','method','target','n','MAE_tick','RMSE_tick','max_tick','pass_1tick']].to_string(index=False))
    print(json.dumps(verification,ensure_ascii=False))
if __name__=='__main__':main()
