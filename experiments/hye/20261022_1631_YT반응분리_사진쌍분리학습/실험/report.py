"""Report actual mixed-pose training and what held-out axis separation achieves."""
from pathlib import Path
import json,html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;PREV=H.parent/'joint_yt_generalization_20261022'

def table(x):return x.to_html(index=False,float_format=lambda v:f'{v:.5f}',escape=True)
def best(x):return x.sort_values(['passed','Y_max'],ascending=[False,True]).iloc[0]

def main():
    p=pd.read_csv(H/'predictions.csv');s=pd.read_csv(H/'summary.csv');research=json.loads((H/'research/research.json').read_text(encoding='utf-8'));coverage=json.loads((H/'research/coverage.json').read_text(encoding='utf-8'))
    new=p[(p.method=='pass_first')&(p.protocol=='nested_group_oof')&(p.dataset==20261020)].sort_values('index')
    train=p[(p.method=='pass_first')&(p.protocol=='resub341')&(p.dataset==20261020)].sort_values('index')
    prev=pd.read_csv(PREV/'predictions.csv');prev=prev[(prev.method=='selected_joint')&(prev.protocol=='nested_group_oof')&(prev.dataset==20261020)].sort_values('index')
    new.to_csv(H/'new60_group_oof_corrections.csv',index=False,encoding='utf-8-sig');train.to_csv(H/'new60_training_reapplication.csv',index=False,encoding='utf-8-sig');new[~new.passed].to_csv(H/'new60_failures.csv',index=False,encoding='utf-8-sig')
    geometric=pd.read_csv(H/'geometric/summary.csv').query("subset=='new'")
    conditional=pd.read_csv(H/'conditional/summary.csv').query("scope=='new60'")
    neural=pd.read_csv(H/'neural_summary.csv').query("subset=='new60'")
    paired=pd.read_csv(H/'paired_summary.csv').query("subset=='new60'")
    methods=[]
    for name,row in [('실제 조건 평균으로 반응 학습',best(geometric[geometric.fit_mode=='new_condition_means'])),('한 축 고정 사진쌍의 차이 학습',best(geometric[geometric.fit_mode=='axis_pairs'])),('사진쌍·반복 일관성 제약 회귀',best(paired)),('QR 중심·모양 모멘트 분리',best(geometric[geometric.feature=='moments'])),('추정한 T 제거 후 Y 학습',best(conditional)),('두 출력 신경망 + 사진쌍 제약',best(neural[neural.pair_weight>0]))]:
        methods.append(dict(방법=name,**{key:row[key] for key in ['Y_MAE','Y_max','T_MAE','T_max','passed']}))
    methods=pd.DataFrame(methods);methods.to_csv(H/'method_comparison.csv',index=False,encoding='utf-8-sig')
    labels={'Y_MAE':'Y 평균오차','Y_max':'Y 최대오차','T_MAE':'T 평균오차','T_max':'T 최대오차','passed':'두 축≤0.09 장수'}
    comparisons=[]
    for name,z,passcol in [('직전 모델 · 그룹 교차검증',prev,'both_within09'),('이번 분리학습 · 그룹 교차검증',new,'passed'),('이번 최종 모델 · 학습 사진 재대입',train,'passed')]:
        comparisons.append(dict(평가=name,Y_MAE=z.eY.mean(),Y_max=z.eY.max(),T_MAE=z.eT.mean(),T_max=z.eT.max(),passed=int(z[passcol].sum())))
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(H/'honest_comparison.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    ys=coverage['Y_levels'];ts=coverage['T_levels'];grid=np.zeros((len(ys),len(ts)),int)
    for _,row in new[new.scope=='mixed'].groupby(['Y','T']).size().reset_index(name='n').iterrows():grid[ys.index(row.Y),ts.index(row['T'])]=int(row['n'])
    axs[0].imshow(grid,cmap='Blues',vmin=0,vmax=4,aspect='auto');axs[0].set_xticks(range(len(ts)),[str(t) for t in ts]);axs[0].set_yticks(range(len(ys)),[str(y) for y in ys]);axs[0].set(xlabel='T 명령',ylabel='Y 명령',title='실제 혼합 사진: 19조건 × 3장')
    for i in range(len(ys)):
        for j in range(len(ts)):axs[0].text(j,i,str(grid[i,j]) if grid[i,j] else '없음',ha='center',va='center',color='white' if grid[i,j] else '#66747f')
    for ax,key in zip(axs[1:],['Y','T']):
        lo=min(new[key].min(),new['pred'+key].min())-.07;hi=max(new[key].max(),new['pred'+key].max())+.07;x=np.array([lo,hi]);ax.fill_between(x,x-.09,x+.09,color='#e4efe9',label='±0.09');ax.plot(x,x,color='#687681',lw=1)
        ax.scatter(train[key],train['pred'+key],s=24,label='학습 사진 재대입',color='#177e89',alpha=.7)
        ax.scatter(new[key],new['pred'+key],s=38,label='그 조건을 뺀 교차검증',marker='x',color='#d95f42')
        ax.set(xlabel=f'명령 {key}',ylabel=f'추정 {key}',title=f'{key}: 학습 성적과 새 조건 성적');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(H/'training_vs_heldout.png',dpi=180);plt.close(fig)
    sources=''.join(f'<li><a href="{html.escape(r["url"])}">{html.escape(r["title"])}</a> — {html.escape(r["supports"])}</li>' for r in research['sources'])
    rows=new[['file','Y','T','predY','predT','correctionY','correctionT','eY','eT','passed']].rename(columns={'file':'사진','Y':'명령 Y','T':'명령 T','predY':'추정 Y','predT':'추정 T','correctionY':'반대 보정 Y','correctionT':'반대 보정 T','eY':'Y 오차','eT':'T 오차','passed':'동시 통과'})
    failures=rows.loc[~new.passed]
    old=s[(s.method=='pass_first')&(s.protocol=='nested_group_oof')&(s.dataset.astype(str)=='20261014')].iloc[0]
    # Concrete capture proposal only; no robot command interface is called.
    captures=[]
    for y in [-1.,-.5,0.,.5,1.]:
        for t in [-1.,-.1,0.,.1,1.]:captures.append(dict(proposed_Y=y,proposed_T=t,repeats=5,status='proposal_not_executed',T_physical_unit='unconfirmed'))
    pd.DataFrame(captures).to_csv(H/'proposed_capture_conditions.csv',index=False,encoding='utf-8-sig')
    doc=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>혼합 자세 학습과 Y·T 분리 검증</title><style>body{{margin:0;background:#f3f6f7;color:#213441;font:16px/1.75 system-ui,"Malgun Gothic"}}main{{max-width:1250px;padding:32px 24px 70px;margin:auto}}h1{{line-height:1.35}}h2{{color:#14586c;margin-top:36px}}.card{{padding:22px;background:white;border:1px solid #d8e4e8;border-radius:12px;margin:16px 0}}.alert{{padding:22px;background:#fff0d9;border-left:6px solid #cb7b36}}table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}th,td{{border:1px solid #d9e2e7;padding:8px;text-align:right}}th{{background:#e8f0f4}}td:first-child{{text-align:left}}.scroll{{overflow:auto}}img{{width:100%}}a{{color:#096d93}}.small{{font-size:14px;color:#4e6674}}code,pre{{background:#e6edf0;padding:3px 6px}}pre{{padding:16px;overflow:auto}}</style></head><body><main>
    <p class="small">2026-10-22 · <a href="../../ALGORITHM_RULES.md">공통 규칙</a> · 이전 알고리즘과 결과 보존</p>
    <h1>실제 혼합 자세를 학습하고<br>Y와 T의 영상 반응을 분리해 검증</h1>
    <div class="alert"><b>“혼합 학습이 없다”는 설명은 초기 모델에 해당한다.</b> 직전52/60 모델에는 이미 혼합 사진 학습이 들어 있었다. 이번에는 실제 한 축 고정 사진쌍, 반복 잡음, 기하 특징, 조건부 회귀, 두 출력 신경망까지 조사·학습했다.<br><b>학습 사진 재대입60/60과, 그 조건을 빼고 평가한50/60을 구분해야 한다. 새 조건 전부0.09 목표는 아직 미달이다.</b></div>
    <h2>무엇을 학습했는가</h2><div class="card"><p>실제 혼합 데이터는 <b>19가지(Y,T) 조합 × 3장 = 57장</b>, 새center3장이다. Y4수준 × T4수준의 완전한 격자가 포함되어 있어 현재 자료로 두 축을 분리하는 학습을 할 수 있다.</p><ul>
    <li><b>T가 같고 Y만 다른 조건쌍24개</b>: Y가 만드는 영상 변화량을 학습.</li>
    <li><b>Y가 같고 T만 다른 조건쌍37개</b>: T가 만드는 영상 변화량을 학습.</li>
    <li><b>같은(Y,T)를 반복한 사진</b>: 움직임 차이와 반복 측정 변동을 구별.</li>
    </ul><p>쌍의 양 끝, 조건 평균, 공분산, 신경망과 PCA는 모두 해당 학습fold 안의 사진으로만 만들었다. 시험 사진의 실제Y/T를 조건부 입력으로 주지 않았다. 현재 사진의 파일명·반복번호도 추론에 사용하지 않는다.</p></div>
    <img src="training_vs_heldout.png" alt="혼합 조건 격자와 학습 재대입·학습에서 뺀 조건의 YT추정 비교">
    <h2>실제로 적용한 분리 방법</h2><div class="scroll">{table(methods.rename(columns=labels))}</div>
    <p class="small">표는 각 계열의 개발용 고정후보 비교다. 동일조건3장을 같은시험fold로 묶었다. 후보를 보고 가장 좋은 결과만 최종 성적으로 선택하지 않고, 아래에서는 내부3-fold 선택까지 포함했다. 한 축 고정 쌍의 총수24/37은 전체자료 설명이며 실제 각fold학습에 쓴 쌍은 그보다 적다.</p>
    <h2>학습 성적과 실제 검증 성적</h2><div class="scroll">{table(comparisons.rename(columns=labels))}</div>
    <p><b>이번 모델은 최대Y오차를0.3282에서0.2574로 낮췄지만, 동시 통과는52장에서50장으로 줄었다.</b> 전체 성능이 더 좋다고 결론내리지 않았고 기존 모델을 교체하지 않았다. 학습에 쓴60장을 다시 넣으면 Y최대0.07955·T최대0.01374로 모두통과하지만, 이는 새 사진 성능이 아니다.</p>
    <p>주 검증은 이전과 같은 외부4-fold에 내부3-fold 선택을 사용한다. 새center3장도 함께 시험에서 제외했다. 모델 계열 개발에 이미 이 데이터를 사용했으므로 완전히 손대지 않은 미래 촬영 검증은 아니다. 기존281장에서는 이번 모델이 {int(old.passed)}/281통과, Y최대{old.Y_max:.4f}, T최대{old.T_max:.4f}였다. 최대오차를 우선 최소화하는 별도 선택 실험도 저장했지만 새60장44통과로 더 나빠 주 결과로 바꾸지 않았다.</p>
    <h2>왜 단순히 두 출력을 나누면 안 되는가</h2><div class="card"><p>현재 자료에서 수평 영상 반응은 Y당 약14.24px, T당 약80.15px다. T를 먼저 추정하고 그 영향을 빼는 구조에서는 <b>T오차가Y에 약5.626배</b>로 전달될 수 있다. T오차0.02만으로 Y오차가약0.113이 되는 셈이다. 따라서 T자체가0.09이내라는 사실만으로 Y도맞는 것은 아니다.</p>
    <p>공통 수평 이동을 제거한 모양만 사용하면 Y/T의 혼선뿐 아니라 필요한 이동량 정보도 잃었다. 같은 선형모형을 직교화해 다시 쓰는 방법은 기존GLS와 수학적으로 같아 새로운 정보가 생기지 않는다. 반복 사진만으로 잡음 공분산을 구하는 방법도 모델의 자세별 오차를 충분히 설명하지 못했다.</p>
    <p>학습한 반응을 Y→(1,0), T→(0,1)로 만드는 제약은 수치적으로 만족했지만, 실제 시험 사진의 오차까지0이 되지는 않았다. 이 학습식의 성질을 새 사진 분리 성공으로 제시하지 않는다.</p></div>
    <h2>현재 데이터로 확인할 수 있는 범위</h2><div class="card"><p>혼합 사진은 전부양수Y/T이며, 새날짜의Y단독·T단독사진은 없다. 이것은 부호·원점·날짜 변화 검증의 제한이다. 현재양수영역오차가 전부 이누락때문이라고 단정하지 않는다. 실제명령의상관도0.111로 작아 강한정답상관이원인이라는 설명도사용하지 않았다.</p>
    <p>보정된3D자세 모델도 조사했지만 저장소에 카메라내부행렬·왜곡계수, QR실제3D배치, 로봇좌표변환, T단위·회전축이 확인되지 않았다. 임의값을 넣어 회전/병진 분리라고 주장하지 않았다. <a href="geometric/camera_geometry_audit.json">근거 파일과 조사 결과</a></p>
    <p>다음 촬영은 같은날짜·같은원점에서 Y/T단독축을 포함한 부호있는5×5격자, 조건당5회반복을 제안한다. <a href="proposed_capture_conditions.csv">25조건·125장 제안</a>은 기존관측범위안의명령예시이며 실행하지 않았다. 블록마다 순서를무작위화하고center와실제엔코더자세·접근방향을함께기록해야 영상잡음과기계반복오차를구분할 수 있다. 최종검증은 학습이끝난뒤 다른촬영일블록으로진행한다.</p></div>
    <h2>이번 그룹 검증에서 남은10장</h2><div class="scroll">{table(failures)}</div>
    <h2>60장 전체 추정·반대 보정값</h2><div class="scroll">{table(rows)}</div>
    <p class="small">정답은 파일명명령이며 실제자세계측값은 아니다. 반대보정값은 추정값의음수다. 로봇동작이나센터복귀를실측하지 않았다.</p>
    <h2>코드·학습 모델·평가 기록</h2><p><a href="predict.py">실제사진추론</a> · <a href="pass_first_model.joblib">이번최종실험모델</a> · <a href="new60_group_oof_corrections.csv">그룹검증60장</a> · <a href="new60_training_reapplication.csv">학습재대입60장</a> · <a href="selection.json">내부선택</a> · <a href="protocol.json">평가기준</a> · <a href="verification.json">독립검증</a> · <a href="smoke_test.json">실제영상실행확인</a></p>
    <pre>python predict.py "현재사진.bmp" --output prediction.json\npython evaluate.py\npython report.py</pre>
    <h2>논문·공식자료와 적용 근거</h2><p>조사한방법별수식·구현·적용한계는 <a href="research/research.json">연구정리</a>에보존했다. 사진쌍제약신경망은그룹감독아이디어를응용했으며원논문의VAE재현으로표현하지않았다.</p><ul>{sources}</ul>
    </main></body></html>'''
    (H/'report.html').write_text(doc,encoding='utf-8')
    (H/'result.json').write_text(json.dumps(dict(previous_nested_pass=52,current_nested_pass=50,current_nested_Y_max=float(new.eY.max()),training_reapplication_pass=int(train.passed.sum()),training_reapplication_Y_max=float(train.eY.max()),generalization_goal_met=False,existing_model_replaced=False),indent=2))
    print(comparisons.to_string(index=False))

if __name__=='__main__':main()
