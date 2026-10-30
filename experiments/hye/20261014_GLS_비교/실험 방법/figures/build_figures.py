"""Scientific figures from the completed seven-method GLS comparison.

Read-only inputs: metrics.csv, predictions.csv, experiment_plan.json, checks.json.
Writes figures and their exact aggregate values only in this figures directory.
"""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
METHODS=['baseline','ols_euclidean','ols_wls','ols_gls_lw','spatial_wls','spatial_gls_lw','spatial_gls_lw_metric_lw']
LABELS=['기존 기준\n공간 OLS · 거리 기존','유클리드 거리\n공간 OLS · 거리 I','대각 가중 거리\n공간 OLS · 거리 대각','거리 GLS-LW\n공간 OLS · 거리 LW','공간 WLS\n공간 대각 · 거리 기존','공간 GLS-LW\n공간 LW · 거리 기존','공간 + 거리 GLS-LW\n공간 LW · 거리 LW']
SHORT=['기존 기준','유클리드','거리 대각','거리 LW','공간 WLS','공간 LW','공간+거리 LW']
COLORS=['#2477AA','#94A3B5','#94A3B5','#139780','#D1A03F','#8B70AC','#8B70AC']
CAPTION='같은 개발 사진 280장을 20번 나눠 평가한 결과입니다. 5,600건은 반복 예측 횟수이며 독립 사진 5,600장이 아닙니다.'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def style(ax,xlabel):
    ax.set_xlabel(xlabel,fontsize=12,labelpad=10)
    ax.grid(axis='x',color='#DCE3EA',alpha=.7,zorder=0)
    ax.set_axisbelow(True)
    for key in ['top','right','left']:ax.spines[key].set_visible(False)
    ax.spines['bottom'].set_color('#BCC6D1')
    ax.tick_params(axis='y',length=0,labelsize=11)
    ax.tick_params(axis='x',labelsize=10)

def save(fig,stem):
    fig.savefig(HERE/(stem+'.png'),dpi=150,facecolor='white')
    fig.savefig(HERE/(stem+'.svg'),facecolor='white')
    plt.close(fig)

def failure_figure(rows):
    fig=plt.figure(figsize=(16,10),facecolor='white')
    fig.text(.04,.953,'GLS를 추가하면 실패가 줄어드는가?',fontsize=24,fontweight='bold',color='#162E45')
    fig.text(.04,.908,'QR 205점 동일 · 7가지 방법 · 5-fold OOF를 20개 seed로 반복 · 각 라운드 280장',fontsize=13,color='#405468')
    y=np.arange(len(rows));left=fig.add_axes([.235,.30,.355,.525]);right=fig.add_axes([.675,.30,.28,.525])
    failures=[r['repeated_total_failures'] for r in rows];rounds=[r['perfect_rounds_of20'] for r in rows]
    left.barh(y,failures,color=COLORS,height=.57,zorder=3)
    left.set_yticks(y,LABELS);left.invert_yaxis();left.set_xlim(0,90);left.set_xticks([0,20,40,60,80])
    left.set_title('전체 실패 총횟수',loc='left',fontsize=16,fontweight='bold',pad=20)
    style(left,'실패 횟수 [5,600회 반복 예측 중]')
    for yi,value in zip(y,failures):left.text(value+1.6,yi,str(value),va='center',fontsize=13,fontweight='bold',color='#253D53')
    right.barh(y,rounds,color=COLORS,height=.57,zorder=3)
    right.set_yticks(y,['']*len(rows));right.invert_yaxis();right.set_xlim(0,20);right.set_xticks([0,5,10,15,20])
    right.set_title('280장 모두 합격한 라운드',loc='left',fontsize=16,fontweight='bold',pad=20)
    style(right,'전부 합격한 라운드 [20회 중]')
    for yi,value in zip(y,rounds):right.text(value+.45,yi,f'{value} / 20',va='center',fontsize=12,fontweight='bold',color='#253D53')
    fig.text(.235,.225,'실패 정의: 축 또는 부호 오류, 또는 Y/T 중 하나라도 절대오차 0.09 초과',fontsize=11,color='#405468')
    fig.text(.04,.18,'전체 280장 재대입 실패 횟수',fontsize=11,fontweight='bold',color='#253D53')
    tax=fig.add_axes([.04,.105,.915,.065]);tax.axis('off')
    table=tax.table(cellText=[[f"{r['full280_failures']} / 280" for r in rows]],colLabels=SHORT,cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.55)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#DFE6EC');cell.set_linewidth(.7)
        if r==0:cell.set_facecolor('#ECF1F6');cell.set_text_props(weight='bold',color='#253D53')
        else:cell.set_facecolor('#FAFBFC')
    fig.text(.04,.063,'위 재대입은 학습에 쓴 280장에 다시 예측한 성적이며, 반복 OOF 평가와 별도입니다.',fontsize=10.5,color='#56697A')
    fig.text(.04,.026,CAPTION,fontsize=11,color='#43586A')
    save(fig,'01_repeated_failures_and_perfect_rounds')

def error_figure(rows):
    fig=plt.figure(figsize=(16,10),facecolor='white')
    fig.text(.04,.953,'오분류까지 포함하면 최대 오차는 얼마나 남는가?',fontsize=24,fontweight='bold',color='#162E45')
    fig.text(.04,.908,'20회 반복 OOF에서 관측한 최댓값 · 예측 결과를 (Y, T) 벡터로 환산 · 모든 축은 0부터 표시',fontsize=13,color='#405468')
    y=np.arange(len(rows));axes=[fig.add_axes([.235,.245,.34,.58]),fig.add_axes([.66,.245,.295,.58])]
    for ax,key,unit,title,xmax in [(axes[0],'Y','mm','Y 최대 절대오차',.35),(axes[1],'T','명령 단위','T 최대 절대오차',.105)]:
        vals=[r[f'repeated_max_error_{key}'] for r in rows];full=[r[f'full280_max_error_{key}'] for r in rows]
        ax.barh(y,vals,color=COLORS,height=.57,zorder=3)
        ax.scatter(full,y,marker='D',s=52,facecolors='white',edgecolors='#1B354B',linewidths=1.3,zorder=5,label='전체 280장 재대입 최댓값')
        ax.axvline(.09,color='#C85544',ls='--',lw=1.4,zorder=4,label='허용오차 0.09')
        ax.set_xlim(0,xmax);ax.set_yticks(y,LABELS if key=='Y' else ['']*len(rows));ax.invert_yaxis()
        ax.set_title(title,loc='left',fontsize=16,fontweight='bold',pad=20);style(ax,f'최대 절대오차 [{unit}]')
        if key=='Y':ax.set_xticks([0,.05,.10,.15,.20,.25,.30,.35])
        else:ax.set_xticks([0,.025,.05,.075,.10])
        for yi,value in zip(y,vals):ax.text(value+xmax*.018,yi,f'{value:.5f}',va='center',fontsize=11,fontweight='bold',color='#253D53')
    axes[1].text(.09,.99,'허용 0.09',transform=axes[1].get_xaxis_transform(),color='#C85544',rotation=90,ha='right',va='top',fontsize=10)
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.232,.146),ncol=2,frameon=False,fontsize=11)
    fig.text(.235,.118,'막대: 반복 OOF 최댓값     ◇: 전체 280장 재대입 최댓값',fontsize=11,color='#405468')
    fig.text(.04,.078,'실제로 움직이지 않은 축에 잘못 출력된 값도 오차입니다. T 오차가 0.09 이내여도 축을 틀리면 전체 판정은 실패합니다.',fontsize=11,color='#43586A')
    fig.text(.04,.045,'Y 단위: mm.  T는 데이터의 회전 명령 단위를 사용하며, 도(degree)로 독립 검증한 단위가 아닙니다.',fontsize=10.5,color='#56697A')
    fig.text(.04,.015,CAPTION,fontsize=11,color='#43586A')
    save(fig,'02_max_vector_errors_including_misclassification')

def main():
    font=Path('C:/Windows/Fonts/malgun.ttf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'svg.fonttype':'none'})
    checks=json.loads((SOURCE/'checks.json').read_text(encoding='utf8'))
    assert checks['original_files_unchanged'] and checks['all_protocols_each_have_280_unique']
    assert checks['independent_test'] is False
    inputs=[SOURCE/n for n in ['metrics.csv','predictions.csv','experiment_plan.json','checks.json']]
    before={p.name:sha(p) for p in inputs}
    metrics=list(csv.DictReader((SOURCE/'metrics.csv').open(encoding='utf8')))
    predictions=list(csv.DictReader((SOURCE/'predictions.csv').open(encoding='utf8')))
    summary=[]
    for m,label in zip(METHODS,LABELS):
        rr=[r for r in metrics if r['method']==m and r['protocol'].startswith('repeated5_')]
        pred=[r for r in predictions if r['method']==m and r['protocol'].startswith('repeated5_')]
        full=next(r for r in metrics if r['method']==m and r['protocol']=='full280')
        assert len(rr)==20 and len(pred)==5600 and all(int(r['n'])==280 for r in rr)
        failures=sum(int(r['failed']) for r in rr)
        assert failures==sum(r['pass09'].lower()=='false' for r in pred)
        row=dict(method=m,label_ko=label.replace('\n',' / '),repeated_decisions=5600,unique_images=280,rounds=20,repeated_total_failures=failures,perfect_rounds_of20=sum(int(r['failed'])==0 for r in rr),min_failures_per_round=min(int(r['failed']) for r in rr),max_failures_per_round=max(int(r['failed']) for r in rr),repeated_max_error_Y=max(float(r['vector_Y_max']) for r in rr),repeated_max_error_T=max(float(r['vector_T_max']) for r in rr),full280_failures=int(full['failed']),full280_max_error_Y=float(full['vector_Y_max']),full280_max_error_T=float(full['vector_T_max']))
        assert abs(row['repeated_max_error_Y']-max(float(p['error_Y']) for p in pred))<1e-12
        assert abs(row['repeated_max_error_T']-max(float(p['error_T']) for p in pred))<1e-12
        summary.append(row)
    (HERE/'figure_data.json').write_text(json.dumps({'rows':summary,'scope_note':CAPTION,'source_checks':checks,'input_sha256':before},ensure_ascii=False,indent=2),encoding='utf8')
    with (HERE/'figure_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    failure_figure(summary);error_figure(summary)
    assert before=={p.name:sha(p) for p in inputs}
    print(json.dumps({'figures':2,'png_size':[2400,1500],'svg':2,'input_files_unchanged':True,'aggregate_failure_counts':[r['repeated_total_failures'] for r in summary],'perfect_rounds':[r['perfect_rounds_of20'] for r in summary]}))

if __name__=='__main__':main()
