"""Scientific evaluation artifact: error-first, separate calibration protocols."""
from pathlib import Path
import json,html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import H,metric

def table(df):return df.to_html(index=False,float_format=lambda x:f'{x:.5f}',escape=True,na_rep='측정 실패')

def main():
    p=pd.read_csv(H/'predictions.csv');summary=pd.read_csv(H/'summary.csv')
    q=p[(p.method=='selected_joint')&(p.protocol=='nested_group_oof')&(p.dataset==20261020)].sort_values('index')
    old=pd.read_csv(H.parent/'evaluate_archives_20261022/predictions.csv');old=old[(old.model=='improved_nonlinear')&(old.archive==20261020)].sort_values('index')
    stats=metric(q[['Y','T']],q[['predY','predT']]);before=metric(old[['true_Y','true_T']],old[['pred_Y','pred_T']])
    failures=q[~q.both_within09].sort_values('eY',ascending=False)
    q.to_csv(H/'new60_oof_corrections.csv',index=False,encoding='utf-8-sig');failures.to_csv(H/'new60_remaining_failures.csv',index=False,encoding='utf-8-sig')
    stages=[dict(단계='기존 모델 그대로 · 기존 날짜만 학습',**before)]
    recovered=pd.read_csv(H/'measurements/frozen_model_recovered12.csv')
    expanded=old.copy().set_index('index')
    for _,row in recovered.iterrows():expanded.loc[row['index'],['pred_Y','pred_T']]=[row.pred_Y,row.pred_T]
    stages.append(dict(단계='QR 검색 범위만 확대 · 기존 모델 그대로',**metric(expanded[['true_Y','true_T']],expanded[['pred_Y','pred_T']])))
    for name,label in [('raw_global','630점 전체 좌표 + GLS'),('interaction_global','좌표 × 수평 이동량 추가'),('interaction_local','영상 이동량에 따른 국소 가중 추가'),('selected_joint','내부 선택 포함 · 주 결과')]:
        protocol='nested_group_oof' if name=='selected_joint' else 'fixed_config_group_oof'
        sub=summary[(summary.method==name)&(summary.protocol==protocol)&(summary.dataset.astype(str)=='20261020')].iloc[0]
        stages.append(dict(단계=label,**{k:sub[k] for k in ['n','failures','passed','Y_MAE','Y_max','T_MAE','T_max']}))
    stages=pd.DataFrame(stages)[['단계','Y_MAE','Y_max','T_MAE','T_max','passed','n','failures']]
    stages.to_csv(H/'stage_comparison.csv',index=False,encoding='utf-8-sig')
    labels={'Y_MAE':'Y 평균오차','Y_max':'Y 최대오차','T_MAE':'T 평균오차','T_max':'T 최대오차','passed':'두 축≤0.09','n':'전체','failures':'측정 실패'}
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    fig,axs=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for ax,key in zip(axs[:2],['Y','T']):
        actual=q[key].to_numpy();estimated=q['pred'+key].to_numpy();lo=min(actual.min(),estimated.min())-.06;hi=max(actual.max(),estimated.max())+.06
        xx=np.array([lo,hi]);ax.fill_between(xx,xx-.09,xx+.09,color='#dbeee2',label='±0.09 허용범위');ax.plot(xx,xx,color='#55636c',lw=1)
        ax.scatter(actual,estimated,c=np.where(q.both_within09,'#087e8b','#dc593e'),s=35,alpha=.85,edgecolors='white',linewidth=.5)
        ax.set(xlabel=f'파일명 명령 {key}',ylabel=f'학습에서 뺀 사진의 추정 {key}',title=f'{key}: 평균 {stats[key+"_MAE"]:.4f}, 최대 {stats[key+"_max"]:.4f}')
        ax.grid(alpha=.15);ax.legend(fontsize=9)
    axs[2].plot(np.sort(old.eY.dropna()),label='기존 모델: 추정된 48장',color='#a3acb4',lw=2)
    axs[2].plot(np.sort(q.eY),label='개선: 60장 그룹 검증',color='#087e8b',lw=2)
    axs[2].axhline(.09,color='#dc593e',ls='--',label='허용오차 0.09');axs[2].set(xlabel='각 결과를 오차 순으로 정렬',ylabel='Y 절대오차',title='Y 오차 감소 · 일부 실패 잔존');axs[2].legend(fontsize=9);axs[2].grid(alpha=.15)
    fig.savefig(H/'error_comparison.png',dpi=180);plt.close(fig)
    conditions=q.groupby(['Y','T'],as_index=False).agg(사진수=('file','size'),통과=('both_within09','sum'),Y평균오차=('eY','mean'),Y최대오차=('eY','max'),T평균오차=('eT','mean'),T최대오차=('eT','max'))
    conditions.to_csv(H/'new60_conditions.csv',index=False,encoding='utf-8-sig')
    failview=failures[['file','Y','T','predY','predT','correctionY','correctionT','eY','eT']].rename(columns={'file':'사진','Y':'명령 Y','T':'명령 T','predY':'추정 Y','predT':'추정 T','correctionY':'반대 보정 Y','correctionT':'반대 보정 T','eY':'Y 오차','eT':'T 오차'})
    fullview=q[['file','Y','T','predY','predT','correctionY','correctionT','eY','eT','both_within09','fold']].rename(columns={'file':'사진','Y':'명령 Y','T':'명령 T','predY':'추정 Y','predT':'추정 T','correctionY':'반대 보정 Y','correctionT':'반대 보정 T','eY':'Y 오차','eT':'T 오차','both_within09':'동시 통과'})
    config=json.loads((H/'final_config.json').read_text());oldstats=summary[(summary.method=='selected_joint')&(summary.protocol=='nested_group_oof')&(summary.dataset.astype(str)=='20261014')].iloc[0]
    resub=summary[(summary.method=='selected_joint')&(summary.protocol=='resub341')&(summary.dataset.astype(str)=='20261020')].iloc[0]
    document=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>새 사진 Y·T 동시 추정 개선</title>
    <style>body{{font:16px/1.7 system-ui,"Malgun Gothic";background:#f3f6f8;color:#203240;margin:0}}main{{max-width:1250px;margin:auto;padding:32px 24px 70px}}h1{{line-height:1.35}}h2{{margin-top:40px;color:#154f66}}.card{{background:white;padding:24px;border:1px solid #d8e2e8;border-radius:12px;margin:16px 0}}.alert{{background:#fff3df;border-left:6px solid #ce7b2e;padding:20px}}.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.kpi{{background:white;padding:18px;border-radius:10px}}.kpi strong{{display:block;font-size:27px;color:#087e8b}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}th,td{{border:1px solid #d8e2e8;padding:8px;text-align:right}}th{{background:#e9f1f5}}td:first-child{{text-align:left}}.scroll{{overflow:auto}}img{{width:100%}}code,pre{{background:#e6edf1;padding:3px 6px}}pre{{overflow:auto;padding:16px}}a{{color:#116b92}}.small{{font-size:14px;color:#4b606e}}@media(max-width:700px){{.kpis{{grid-template-columns:1fr}}}}</style></head><body><main>
    <p class="small">2026-10-22 · <a href="../../ALGORITHM_RULES.md">공통 알고리즘 규칙</a> · 기존 알고리즘/모델/보고서 보존</p>
    <h1>새 사진에서도 Y와 T를 함께 추정하도록 개선</h1>
    <div class="alert"><b>성능은 개선됐지만, 모든 사진이 두 축 0.09 이내라는 목표는 아직 미달이다.</b><br>주 검증은 60장 중 {stats['passed']}장 통과. 나머지 {60-stats['passed']}장은 Y 오차가 0.09를 넘었다. 최종 모델을 저장했지만 기존 기본 알고리즘은 교체하지 않았다.</div>
    <div class="kpis"><div class="kpi">Y 평균 / 최대오차<strong>{stats['Y_MAE']:.4f} / {stats['Y_max']:.4f}</strong>기존 0.2234 / 1.0000</div><div class="kpi">T 평균 / 최대오차<strong>{stats['T_MAE']:.4f} / {stats['T_max']:.4f}</strong>기존 0.0189 / 0.0751</div><div class="kpi">두 축 허용오차 동시 통과<strong>{stats['passed']}/60 · {stats['passed']/60:.1%}</strong>측정 실패 12장 → 0장</div></div>
    <p><b>비교 조건:</b> 기존 모델은 20261014만 학습한 뒤 20261020에 적용했다. 개선 모델은 20261020의 <b>학습용 조건도 교정에 추가</b>했다. 각 시험 사진과 같은 명령의 반복 3장은 전부 학습에서 제외했다. 이 차이를 감춘 동일 학습량의 모델 비교가 아니다.</p>
    <h2>왜 Y가 틀렸는가</h2><div class="card"><ol>
    <li><b>단일축 학습 트리가 혼합 자세를 표현하지 못했다.</b> 학습은 Y 단독 80장, T 단독 200장, 혼합 0장이다. T=1 혼합 사진 12장에서는 80개 트리가 모두 Y=0을 출력했다. 평균으로 출력을 만드는 기존 트리의 학습 정답 범위를 계산하면 새 60장 중 18장은 구조상 두 축 0.09 동시 통과가 불가능했다. 이는 이미지의 근본 한계가 아니라 해당 모델의 한계다.</li>
    <li><b>큰 이동 12장은 검색 범위 밖이었다.</b> 영상 수평 이동 160.8~174.5px를 ±130px로 찾으려 했다. ±260px로 넓히고 같은 630점 경계 맞춤을 적용하니 모두 복구됐다. 복구 후에도 기존 트리는 T=2를 약0.99로 출력했다.</li>
    <li><b>Y와 T가 만드는 영상 이동이 매우 비슷하다.</b> 전체 좌표의 선형 반응 방향 유사도는 0.999997이다. T 영향과 구별되는 Y 정보가 작아 12개 계수로 요약하거나 반복 측정 잡음이 섞이면 Y 오차가 커진다. 공분산을 고려한 전체점 회귀로 구별 정보를 더 이용했다.</li>
    <li><b>새 center도 기존 원점과 차이가 있다.</b> 기존 모델은 새 center를 Y=0.095~0.111로 추정했다. 그러나 원점만 빼는 진단에서도 Y 최대오차1.0은 남았다. 원점 하나의 문제로 설명되지 않는다.</li>
    </ol><p class="small">영상 정보가 부족하다는 수학적 불가능성은 주장하지 않는다. 예를 들어 Y 정답 차이가0.5인 두 사진의 전체 좌표 차이 RMS가0.03382px로 매우 작지만, 더 좋은 측정·표현이 이를 구분할 가능성은 남아 있다.</p></div>
    <h2>단계별로 추가한 것</h2><div class="scroll">{table(stages.rename(columns=labels))}</div>
    <p class="small">전체점 GLS부터는 혼합 사진의 학습용 조건을 추가했다. 중간 3개 회귀 행은 고정 후보의 개발용 그룹 검증이다. 마지막 행은 매 외부 시험 fold를 빼고 내부3-fold에서 모델을 선택한 주 결과다. 후보를 개발하는 데 이 자료를 이미 보았으므로, 주 결과도 완전히 새로운 날짜의 독립 검증은 아니다. 기존 모델 평균·최대는 추정된48장 기준이고 실패12장은 통과 분모60에 포함했다. 검색 확대만 한 행은60장 전체 오차다.</p>
    <div class="card"><p>최종 경로: <b>사진 → 동일630점 측정 → 전체1260 좌표 → 좌표와 영상 수평 이동량의 곱 추가 → 정규화한 GLS 공동 추정 → (−Y,−T) 보정값</b>.</p><p>국소 가중 후보는 현재 사진의 영상 이동량과 가까운 학습 사진에 더 큰 가중치를 준다. 모든 학습 사진에 양의 가중치를 유지하며, 시험 정답·파일명·반복 번호를 사용하지 않는다. 점을 제외하거나 Y/T 중 하나를 강제로0으로 만들지 않는다.</p><p>Ridge, PLS, 다항식, RBF, SVR, 호모그래피, 가중 공간 집계, 광도 특징, 선형/비선형 순방향 모델도 별도 비교했다. 개선되지 않은 후보의 결과도 보존했다.</p></div>
    <h2>학습에서 뺀 사진의 오차</h2><img src="error_comparison.png" alt="Y와 T의 명령값 대비 추정값 및 Y오차 비교"><p class="small">청록 점은 두 축 동시 통과, 주황 점은 동시 실패다. 이번 주 검증의 주황8장은 모두 Y 실패이며 T는 허용범위 안이다.</p>
    <h2>아직 실패하는 {len(failures)}장</h2><div class="scroll">{table(failview)}</div>
    <p>반대 보정값은 추정값에 마이너스를 붙인 값이다. 실제 로봇을 구동하지 않았으며, 파일명 명령을 기준으로 계산한 오차다. 보정 후 실제 센터 복귀를 실측한 결과가 아니다.</p>
    <h2>검증 조건과 기존 자료의 성능</h2><div class="card"><p>기존280장의4-fold는 보존했다. 새 혼합사진은 같은(Y,T)조건의 반복3장을 같은fold에, 새center3장도 같은fold에 넣었다. 외부4-fold마다 학습 데이터만으로 내부3-fold 선택·원점·공분산을 추정했다. 주 선택 기준은 내부 새 날짜 사진의 허용오차 실패 수, 최대오차, 평균오차 순이다.</p>
    <p><b>새 자료에 맞춘 개선이 기존 자료 성능까지 유지하지는 못했다.</b> 기존281장(기준center 포함)의 새 모델 교차검증은 {int(oldstats.passed)}/281, Y 최대{oldstats.Y_max:.4f}, T 최대{oldstats.T_max:.4f}다. 따라서 기존 알고리즘을 대체하지 않고 별도 실험 모델로 보존한다.</p>
    <p>전체341장을 학습한 최종 모델의 새60장 재대입은 {int(resub.passed)}/60, Y 최대{resub.Y_max:.5f}다. 학습한 사진을 다시 맞힌 값이며 새 사진 성능으로 제시하지 않는다. 최종 설정: <code>{html.escape(str(config))}</code>.</p></div>
    <h2>사진별 추정과 보정값</h2><div class="scroll">{table(fullview)}</div>
    <h2>재현과 저장 파일</h2><p><a href="new60_oof_corrections.csv">60장 사진별 교차검증·반대 보정값</a> · <a href="new60_remaining_failures.csv">남은 실패</a> · <a href="new60_conditions.csv">명령 조건별 결과</a> · <a href="summary.csv">전체 요약</a> · <a href="model.joblib">최종 실험 모델</a> · <a href="predict.py">실제 사진 추론 코드</a> · <a href="protocol.json">평가 기준</a> · <a href="selection.json">내부 선택 기록</a> · <a href="verification.json">독립 검증</a> · <a href="smoke_test.json">실제 이미지 추론 확인</a></p>
    <pre>python predict.py "현재사진.bmp" --output prediction.json\npython evaluate.py\npython report.py</pre>
    <p class="small">추론 코드의 입력은 이미지 바이트와 보존된 고정center다. 출력은 Y/T와 반대 보정값이며 로봇 명령을 실행하지 않는다. 원본 사진 해시·기존 모델 보존·630개ID·측정 방법과 실패 복구는 <a href="measurements/audit.json">측정 출처</a>와 <a href="diagnostics/diagnosis.json">원인 진단</a>에 기록했다.</p>
    <h2>적용 근거와 참고자료</h2><ul>
    <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html">DecisionTreeRegressor 공식 문서</a>: 제곱오차 트리의 leaf 평균. 단일축 정답의 볼록평균 출력 한계 분석에 사용.</li>
    <li><a href="https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf">Chaumette &amp; Hutchinson, Visual Servo Control I (2006)</a>: 영상 특징 변화와 움직임의 관계. 두 축 순방향 반응을 함께 풀고 반대 보정값을 계산하는 근거. 실제 제어 안정성을 이번 실험에서 검증한 것은 아님.</li>
    <li><a href="https://scikit-learn.org/stable/modules/covariance.html">공분산 추정과 shrinkage</a>: 상관된630점 잡음으로 인한 역행렬 불안정을 정규화. 실제 shrink 비율은 학습 내부에서 선택.</li>
    <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">다항식 상호작용 특징</a>: 좌표와 영상 이동량의 곱을 추가하여 위치에 따른 응답 변화를 표현.</li>
    <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html">StratifiedGroupKFold</a>: 반복 사진이 학습·시험 양쪽에 들어가는 누출을 방지.</li>
    </ul></main></body></html>'''
    (H/'report.html').write_text(document,encoding='utf-8')
    (H/'result.json').write_text(json.dumps(dict(before=before,nested_new60=stats,remaining_Y_failures=len(failures),goal_all_within09=False,final_config=config,calibration_used_new_date_training_conditions=True),indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
