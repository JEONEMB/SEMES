"""Scientific comparison figures and a self-contained local results report."""
from pathlib import Path
import html,json,platform,sys,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import experiment as E

HERE=E.HERE
LABEL={'baseline':'기존 GN','pixel_noise_gn':'픽셀 잡음 가중 GN'}
COLOR={'baseline':'#2477AA','pixel_noise_gn':'#DE772B'}
PROTOCOLS=['full280','interleaved','stratified5','blocked_within_session5','repeated20']
PNAMES={'full280':'280장 재대입','interleaved':'짝홀 OOF','stratified5':'층화 5분할 OOF','blocked_within_session5':'시행 내 연속구간 OOF','repeated20':'반복 5분할 20회 합계','session_out_missing_origin':'시행 제외 진단'}

def fmt(v,n=6):return f'{v:.{n}f}'
def esc(v):return html.escape(str(v))
def save(fig,name):
 fig.savefig(HERE/'figures'/f'{name}.png',dpi=160,facecolor='white')
 fig.savefig(HERE/'figures'/f'{name}.svg',facecolor='white')
 plt.close(fig)

def setup():
 font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
 plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
 (HERE/'figures').mkdir(exist_ok=True)

def clean(ax):
 for s in ['top','right']:ax.spines[s].set_visible(False)
 ax.grid(axis='x',alpha=.22);ax.set_axisbelow(True)

def summarize(pred,metrics):
 rows=[]
 for protocol in PROTOCOLS:
  sub=pred[pred.protocol.str.startswith('repeated5_seed')] if protocol=='repeated20' else pred[pred.protocol==protocol]
  for method in E.METHODS:
   g=sub[sub.method==method]
   r=dict(method=method,protocol=protocol,rejected=int(g.rejected.sum()),**E.C.score(g))
   r['Y_axis_correct']=int(((g.axis=='Y')&g.correct_axis).sum())
   r['T_axis_correct']=int(((g.axis=='T')&g.correct_axis).sum())
   r['Y_axis_sign_correct']=int(((g.axis=='Y')&g.correct).sum())
   r['T_axis_sign_correct']=int(((g.axis=='T')&g.correct).sum())
   rows.append(r)
 return pd.DataFrame(rows)

def accuracy_figure(summary):
 fig,axs=plt.subplots(2,2,figsize=(16,10.5));fig.subplots_adjust(left=.19,right=.95,top=.84,bottom=.12,wspace=.30,hspace=.38)
 fig.suptitle('같은 사진·같은 분할에서 축과 부호 정확도 비교',x=.035,ha='left',y=.965,fontsize=23,fontweight='bold',color='#17354C')
 fig.text(.035,.916,'기존 205점·GN17·흐림 σ0.8 유지 / 앞단의 픽셀 가중치만 변경 / 점과 숫자로 정확도 표시',fontsize=12,color='#42586C')
 for ax,(axis,sign) in zip(axs.ravel(),[('Y',False),('T',False),('Y',True),('T',True)]):
  column=axis+('_axis_sign_accuracy' if sign else '_axis_accuracy')
  vals=summary[column].to_numpy()*100;lower=max(0,min(vals)-max(.15,(100-min(vals))*.28))
  span=100-lower;upper=100+span*.50
  for method,offset,marker in [('baseline',-.14,'o'),('pixel_noise_gn',.14,'D')]:
   d=summary[summary.method==method].set_index('protocol').loc[PROTOCOLS]
   x=d[column].to_numpy()*100;y=np.arange(5)+offset
   ax.scatter(x,y,color=COLOR[method],marker=marker,s=45,zorder=3,label=LABEL[method])
   for xi,yi,(_,r) in zip(x,y,d.iterrows()):
    key=axis+('_axis_sign_correct' if sign else '_axis_correct')
    ax.text(xi+span*.025,yi,f'{xi:.4f}%  ({int(r[key])}/{int(r[axis+"_n"])})',va='center',fontsize=9.5,color=COLOR[method])
  ax.set_yticks(range(5),[PNAMES[p].replace('반복 5분할 20회 합계','반복 20회 합계') for p in PROTOCOLS] if axis=='Y' else ['']*5)
  ax.set_ylim(4.5,-.5);ax.set_xlim(lower,upper);ax.set_xticks(np.linspace(lower,100,4));ax.set_xticklabels([f'{v:.2f}' for v in np.linspace(lower,100,4)])
  ax.set_xlabel('정확도 [%] · 차이를 보기 위한 확대 축')
  ax.set_title(axis+(' 축·부호 모두 정답' if sign else ' 축 정답'),loc='left',fontweight='bold',fontsize=15);clean(ax)
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='lower left',bbox_to_anchor=(.185,.045),frameon=False)
 fig.text(.035,.02,'각 일반 평가: Y 80장 / T 200장. 반복 합계: Y 1,600회 / T 4,000회. 같은 280장의 반복 예측이며 독립 새 사진 시험이 아닙니다.',fontsize=10.5,color='#42586C')
 save(fig,'01_axis_and_sign_accuracy')

def error_figure(summary):
 fig,axs=plt.subplots(1,2,figsize=(16,8.6));fig.subplots_adjust(left=.19,right=.96,top=.80,bottom=.22,wspace=.27)
 fig.suptitle('오분류까지 포함한 Y/T 최대 절대오차',x=.035,ha='left',y=.96,fontsize=23,fontweight='bold',color='#17354C')
 fig.text(.035,.9,'예측을 (Y, T) 벡터로 환산 / 실제로 움직이지 않은 축에 잘못 출력한 값도 오차에 포함',fontsize=12,color='#42586C')
 for ax,axis in zip(axs,['Y','T']):
  col='vector_'+axis+'_max';maximum=max(.09,summary[col].max());limit=maximum*1.43
  for method,offset in [('baseline',-.18),('pixel_noise_gn',.18)]:
   d=summary[summary.method==method].set_index('protocol').loc[PROTOCOLS]
   x=d[col].to_numpy();y=np.arange(5)+offset
   ax.barh(y,x,height=.31,color=COLOR[method],label=LABEL[method],zorder=3)
   for xi,yi in zip(x,y):ax.text(xi+maximum*.025,yi,f'{xi:.6f}',va='center',fontsize=11,color=COLOR[method])
  ax.axvline(.09,color='#B9473F',ls='--',lw=1.5,label='허용 0.09')
  ax.set_xlim(0,limit);ax.set_yticks(range(5),[PNAMES[p] for p in PROTOCOLS] if axis=='Y' else ['']*5)
  ax.set_ylim(4.6,-.6);ax.set_title(axis+' 최대 절대오차',loc='left',fontweight='bold',fontsize=16)
  ax.set_xlabel('오차 [mm]' if axis=='Y' else '오차 [T 명령 단위]');clean(ax)
 handles,labels=axs[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.185,.115),ncol=3,frameon=False)
 fig.text(.035,.072,'전체 합격은 축·부호 정답과 두 축의 오차 ≤0.09를 모두 요구합니다. T 물리 단위는 확인되지 않아 원래 명령 단위를 사용합니다.',fontsize=10.5,color='#42586C')
 fig.text(.035,.032,'280장 재대입은 학습에 사용한 사진의 성적입니다. OOF도 같은 촬영 자료 내부 평가이며, 새 시행의 성능을 보장하지 않습니다.',fontsize=10.5,color='#42586C')
 save(fig,'02_max_errors_all_predictions')

def repeated_figure(metrics):
 fig,ax=plt.subplots(figsize=(16,8.8));fig.subplots_adjust(left=.08,right=.96,top=.78,bottom=.22)
 fig.suptitle('같은 20개 무작위 분할에서 실패가 얼마나 바뀌었나',x=.035,ha='left',y=.96,fontsize=23,fontweight='bold',color='#17354C')
 stats={}
 for method,offset in [('baseline',-.18),('pixel_noise_gn',.18)]:
  d=metrics[(metrics.method==method)&metrics.protocol.str.startswith('repeated5_seed')].copy()
  d['seed']=d.protocol.str.replace('repeated5_seed','',regex=False).astype(int);d=d.sort_values('seed')
  ax.bar(d.seed+offset,d.failed,width=.34,color=COLOR[method],label=LABEL[method])
  for x,y in zip(d.seed+offset,d.failed):ax.text(x,y+.05,str(y),ha='center',va='bottom',fontsize=10,color=COLOR[method])
  stats[method]={'failed':int(d.failed.sum()),'perfect':int((d.failed==0).sum()),'worst':int(d.failed.max())}
 ax.set_xticks(range(20),[str(7000+i) for i in range(20)],rotation=40,ha='right')
 ax.set_xlabel('기존과 동일한 분할 seed');ax.set_ylabel('실패 횟수 [라운드당 280회 예측]')
 ax.set_ylim(0,max(s['worst'] for s in stats.values())+1);ax.yaxis.set_major_locator(MaxNLocator(integer=True));clean(ax)
 ax.grid(axis='y',alpha=.22);ax.grid(axis='x',visible=False)
 fig.text(.08,.865,'실패: 축·부호 오류 또는 Y/T 중 하나라도 오차 0.09 초과 또는 입력 거절',fontsize=12,color='#42586C')
 for x,method in [(.08,'baseline'),(.52,'pixel_noise_gn')]:
  s=stats[method];fig.text(x,.13,f'{LABEL[method]}: 총 {s["failed"]}/5,600회 실패 · 전부 합격 {s["perfect"]}/20회',fontsize=13,color=COLOR[method],fontweight='bold')
 fig.legend(loc='lower left',bbox_to_anchor=(.075,.065),ncol=2,frameon=False)
 fig.text(.035,.025,'각 라운드에서 280장 모두 한 번씩 학습 제외 예측합니다. 20회 합계는 같은 사진을 반복 예측한 수이며 독립 표본 수가 아닙니다.',fontsize=10.5,color='#42586C')
 save(fig,'03_repeated_cv_failures')
 return stats

def noise_figure():
 curves=[]
 for p in (HERE/'folds').glob('*/noise_model.json'):
  if not p.parent.name.startswith('full280'):curves.append(json.loads(p.read_text(encoding='utf8'))['sd'])
 full=json.loads((HERE/'folds/full280__all/noise_model.json').read_text(encoding='utf8'));x=np.array(full['centers']);a=np.array(curves)
 fig,ax=plt.subplots(figsize=(12,6.4));fig.subplots_adjust(left=.10,right=.96,top=.78,bottom=.22)
 fig.suptitle('가중치에 사용한 밝기별 잡음 추정',x=.04,ha='left',y=.95,fontsize=22,fontweight='bold',color='#17354C')
 ax.fill_between(x,a.min(0),a.max(0),alpha=.20,color='#DE772B',label='교차검증 학습 분할별 추정 범위')
 ax.plot(x,full['sd'],color='#DE772B',marker='o',label='전체 280장 보정 모델의 추정값')
 ax.set_xlabel('QR 픽셀 밝기 [0–255]');ax.set_ylabel('한 영상의 잡음 표준편차 추정 [밝기 단위]');ax.set_ylim(0,max(a.max(),max(full['sd']))*1.2);ax.set_xlim(0,255)
 ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=10)
 fig.text(.04,.12,'QR 세 영역의 학습 Y 사진 8쌍에서 밝기·국소 이동을 보정한 잔차로 추정했습니다. 음영은 신뢰구간이 아닙니다.',fontsize=10.5,color='#42586C')
 fig.text(.04,.065,'정지 반복 촬영에서 분리한 센서 잡음이 아니며, 무상관·동일 분산 가정을 포함한 경험적 가중치 후보입니다.',fontsize=10.5,color='#42586C')
 save(fig,'04_train_only_noise_curve')

def table(summary):
 rows=[]
 for _,r in summary.iterrows():
  cells=[PNAMES[r.protocol],LABEL[r.method]]
  for a in ['Y','T']:
   cells += [f'{r[a+"_axis_accuracy"]*100:.4f}% ({int(r[a+"_axis_correct"])}/{int(r[a+"_n"])})',
             f'{r[a+"_axis_sign_accuracy"]*100:.4f}% ({int(r[a+"_axis_sign_correct"])}/{int(r[a+"_n"])})']
  cells += [fmt(r.vector_Y_max),fmt(r.vector_T_max),f'{int(r.failed)}/{int(r.n)}',str(int(r.rejected))]
  rows.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in cells)+'</tr>')
 headers=['평가','방법','Y 축','Y 축·부호','T 축','T 축·부호','Y 최대오차 mm','T 최대오차','전체 실패','거절']
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'

def main():
 setup();pred=pd.read_csv(HERE/'predictions.csv');metrics=pd.read_csv(HERE/'metrics.csv')
 summary=summarize(pred,metrics);summary.to_csv(HERE/'comparison_summary.csv',index=False,encoding='utf-8-sig')
 accuracy_figure(summary);error_figure(summary);stats=repeated_figure(metrics);noise_figure()
 pair=[]
 for p in PROTOCOLS:
  g=pred[pred.protocol.str.startswith('repeated5_seed')] if p=='repeated20' else pred[pred.protocol==p]
  b=g[g.method=='baseline'];w=g[g.method=='pixel_noise_gn']
  d=b.merge(w,on=['protocol','file'],suffixes=('_old','_new'));assert len(d)==len(b)
  pair.append(dict(protocol=p,n=len(d),fixed=int((~d.pass09_old&d.pass09_new).sum()),regressed=int((d.pass09_old&~d.pass09_new).sum()),
   both_failed=int((~d.pass09_old&~d.pass09_new).sum()),both_passed=int((d.pass09_old&d.pass09_new).sum())))
 pd.DataFrame(pair).to_csv(HERE/'paired_comparison.csv',index=False,encoding='utf-8-sig')
 repeated=pred[pred.protocol.str.startswith('repeated5_seed')]
 failure_rows=[]
 for file,g in repeated.groupby('file'):
  b=g[g.method=='baseline'];w=g[g.method=='pixel_noise_gn']
  if (~b.pass09).any() or (~w.pass09).any():
   failure_rows.append(dict(file=file,axis=g.axis.iloc[0],value=float(g.value.iloc[0]),old_failed=int((~b.pass09).sum()),new_failed=int((~w.pass09).sum()),
    old_Y_max=float(b.error_Y.max()),new_Y_max=float(w.error_Y.max()),old_T_max=float(b.error_T.max()),new_T_max=float(w.error_T.max())))
 failure_table=pd.DataFrame(failure_rows).sort_values(['new_failed','old_failed'],ascending=False)
 failure_table.to_csv(HERE/'failed_image_frequency.csv',index=False,encoding='utf-8-sig')
 checks=json.loads((HERE/'checks.json').read_text());audit=json.loads((HERE/'audit_checks.json').read_text());full=summary[summary.protocol=='full280'].set_index('method')
 old=stats['baseline'];new=stats['pixel_noise_gn']
 direction='감소' if new['failed']<old['failed'] else ('증가' if new['failed']>old['failed'] else '동일')
 conclusion=f'반복 교차검증 실패는 {old["failed"]}회에서 {new["failed"]}회로 {direction}했습니다. 전체 합격 라운드는 {old["perfect"]}/20회에서 {new["perfect"]}/20회입니다.'
 E.dump(HERE/'report_data.json',dict(repeated_stats=stats,paired=pair,conclusion=conclusion,summary=summary.replace({np.nan:None}).to_dict(orient='records')))
 pieces=['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>픽셀 잡음 가중 GN 비교</title><style>
 body{margin:0;background:#f5f7fa;color:#17354c;font:16px/1.8 "Malgun Gothic",sans-serif}main{max-width:1280px;margin:auto;padding:32px 28px 70px}h1{font-size:31px;line-height:1.4}h2{margin-top:48px;font-size:24px}h3{margin-top:32px}p{max-width:1120px}a{color:#087a94}nav a{margin-right:18px}figure{margin:30px 0}img{width:100%;height:auto;display:block;background:white}figcaption{font-size:14px;color:#4b6174}.scroll{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:10px 6px;border-bottom:1px solid #d7e0e8;text-align:left;vertical-align:top;white-space:nowrap}th{background:#19394f;color:white}tr:nth-child(even){background:#edf3f7}code{font-family:Consolas,monospace;background:#e6edf3;padding:2px 4px}li{margin:9px 0}.formula{padding:16px;background:#e6edf3;overflow-wrap:anywhere}details{margin:24px 0}summary{font-weight:bold;cursor:pointer} @media(max-width:600px){main{padding:18px 14px 40px}h1{font-size:25px}body{font-size:15px}nav a{display:inline-block}h2{font-size:21px}}
 </style><main><h1>픽셀 잡음 가중 GN을 적용하면 성능이 좋아지는가?</h1><p>2026-10-15 · QR 세 개 · 센터와 현재 사진 두 장 · 기존 205점과 판정 구조 유지</p>''']
 pieces.append('<p><strong>'+esc(conclusion)+'</strong></p>')
 pieces.append('<p>재대입에서 Y 최대오차는 줄었지만, 학습에서 제외한 사진에서는 그만큼의 감소가 이어지지 않았습니다. 이번 설정을 기존 방법보다 일관되게 우수하다고 판단할 근거는 부족합니다.</p>')
 pieces.append('<p>두 방법의 전체 280장 재대입 축·부호 정답은 각각 '+str(int(full.loc['baseline','correct']))+'/280장, '+str(int(full.loc['pixel_noise_gn','correct']))+'/280장입니다. 재대입과 교차검증은 구분합니다. 두 평가 모두 기존 촬영 자료에서 얻은 성적이며, 새 시행의 독립 시험은 아닙니다.</p>')
 pieces.append('<nav><a href="#results">수치 비교</a><a href="#figures">시각화</a><a href="#method">적용 방법</a><a href="#failures">실패 사진</a><a href="#audit">검증 및 기록</a></nav>')
 pieces.append('<h2 id="results">같은 데이터와 같은 분할의 결과</h2><p>최대오차는 모든 예측을 (Y,T) 벡터로 계산했습니다. 축을 틀린 사진과 실제로 움직이지 않은 축의 출력도 포함합니다. 전체 실패는 축·부호 오류, 두 축 중 오차 0.09 초과, 품질 거절 중 하나라도 있는 경우입니다.</p>')
 pieces.append(table(summary))
 pieces.append('<p>일반 평가별 사진 수는 Y 80장/T 200장입니다. 반복 합계는 같은 사진을 20번 예측한 Y 1,600회/T 4,000회이며, 독립 사진 5,600장이 아닙니다. Y 단위는 mm이며 T는 원래 명령 단위입니다. 최소 명령 간격 Y 0.05/T 0.01과 허용오차 0.09를 혼동하지 않습니다.</p>')
 pieces.append('<h2 id="figures">성능 시각화</h2>')
 for stem,caption in [('01_axis_and_sign_accuracy','축 정답률과 축·부호 정답률. 차이를 볼 수 있도록 정확도 축을 확대했으며 분자와 분모를 함께 표시했습니다.'),('02_max_errors_all_predictions','전체 입력에 대한 Y/T 최대오차. 붉은 점선은 허용오차 0.09입니다.'),('03_repeated_cv_failures','같은 seed를 사용한 20회 반복 OOF의 실패 횟수. 각 라운드에서 280장을 모두 학습 제외 예측합니다.'),('04_train_only_noise_curve','각 학습 분할에서만 추정한 잡음 곡선. 정지 반복 촬영으로 분리한 센서 잡음과는 구분합니다.')]:
  pieces.append(f'<figure><img src="figures/{stem}.png" alt="{esc(caption)}"><figcaption>{esc(caption)} <a href="figures/{stem}.svg">SVG</a></figcaption></figure>')
 pieces.append('''<h2 id="method">실제로 적용한 방법</h2><ol>
 <li><strong>기존 측정·모델 고정:</strong> QR 205점, QR별 독립 영상 범위, LK15 초기화, GN17, 흐림 σ0.8, cubic 보간, 밝기 배율·오프셋, P12 공간식, 6개 절편 후보, 축별 2차 운동식과 기존 잔차 공분산 방식을 유지했습니다. 기존 소스를 수정하지 않고 새 후보만 구현했습니다.</li>
 <li><strong>학습 전용 잡음 추정:</strong> 각 분할의 학습 Y 사진 중 정수 이동에 가까운 8쌍을 선택했습니다. 세 QR 영역에서 전역 밝기 배율·오프셋과 9×9 국소 이동 성분을 제거하고, 밝기 16단계 bin의 잔차 분산/2로 한 영상의 분산을 근사했습니다. 표본이 100개 미만인 bin은 보간하고 표준편차 최솟값은 0.5로 미리 고정했습니다.</li>
 <li><strong>양쪽 영상과 흐림 고려:</strong> 센터의 텍스처와 해당 현재 사진의 기존 GN 배율·오프셋으로 픽셀별 분산을 예측했습니다. σ0.8 Gaussian kernel의 제곱으로 분산을 전파하고, 현재 영상의 분산과 배율 제곱을 곱한 센터 분산을 더했습니다. 픽셀 간 상관과 보간 후 완전한 공분산은 생략한 대각 근사입니다.</li>
 <li><strong>가중 GN:</strong> 그 역분산을 가중치로 사용해 이동 두 값·밝기 배율·오프셋 네 변수를 같은 가중치로 맞췄습니다. 가중치는 가중 GN을 시작하기 전에 고정합니다. GN 초기 좌표와 반복 횟수·종료 조건은 기존과 같습니다.</li>
 <li><strong>판정 모델 재학습:</strong> 각 분할의 잡음 모델로 학습 사진과 시험 사진을 모두 측정한 뒤, 해당 학습 사진만으로 기존 구조의 운동 곡선과 공분산을 다시 맞췄습니다. 시험 사진의 명령값·시행 ID는 추론에 입력하지 않습니다.</li>
 </ol><div class="formula">잔차 r = 현재 영상(W(p)) − a·센터 영상(p) − b<br>픽셀 가중치 w ≈ 1 / (현재 영상 분산 + a²·센터 영상 분산)<br>GN 계산: (Jᵀ W J) Δ = Jᵀ W 잔차</div>
 <p>이 잡음 모델은 이동 사진 쌍의 잔차에서 얻은 경험적 후보입니다. 두 영상의 잡음이 독립이고 같은 분산이라는 보정 가정, 밝기 모델, 대각 근사를 포함합니다. 잡음의 물리 원인이 확정되었거나 실제 위치 불확도가 측정되었다는 뜻은 아닙니다. 점 좌표가 기존과 달라진 양은 실제 정답에 대한 좌표 정확도와 구분합니다.</p>
 <p>실험 설정은 결과를 보기 전에 저장했습니다. 가중치·창 크기·흐림·판정기를 결과에 따라 재선택하지 않았습니다. v7b 2단계 판정기나 H8 직접 정합은 이번 비교에 추가하지 않았습니다.</p>''')
 pieces.append('<h2 id="failures">같은 사진에서 개선·악화된 예측</h2>')
 pairdf=pd.DataFrame(pair).copy();pairdf.protocol=pairdf.protocol.map(PNAMES)
 pairdf=pairdf.rename(columns={'protocol':'평가','n':'예측 수','fixed':'기존 실패→새 합격','regressed':'기존 합격→새 실패','both_failed':'둘 다 실패','both_passed':'둘 다 합격'})
 pieces.append('<div class="scroll">'+pairdf.to_html(index=False,border=0)+'</div>')
 pieces.append('<h3>반복 OOF에서 한 번이라도 실패한 원본 사진</h3><p>아래 횟수는 사진별 20회 중 실패 횟수입니다. 전체 실패 목록에는 예측 축·부호와 Y/T 오차를 모두 저장했습니다.</p>')
 pieces.append('<div class="scroll">'+failure_table.rename(columns={'file':'사진','axis':'축','value':'명령값','old_failed':'기존 실패/20','new_failed':'가중 실패/20','old_Y_max':'기존 Y 최대','new_Y_max':'가중 Y 최대','old_T_max':'기존 T 최대','new_T_max':'가중 T 최대'}).to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}')+'</div>')
 diagnostic=metrics[metrics.protocol=='session_out_missing_origin'].copy()
 pieces.append('<details><summary>시행 전체 제외 진단: 기존 원점 후보가 사라지는 별도 조건</summary><p>이 분할은 학습에서 특정 시행·부호 후보가 사라집니다. 특히 T 부호 분기가 없는 경우까지 포함하므로, 여섯 원점을 모두 보정한 조건의 일반화 성적과 직접 비교하지 않습니다. 두 방법에 동일한 기존 분할을 적용한 진단값을 기록합니다.</p>')
 pieces.append('<div class="scroll">'+diagnostic[['method','axis_accuracy','axis_sign_accuracy','vector_Y_max','vector_T_max','failed','rejected']].to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}')+'</div></details>')
 pieces.append('<h2 id="audit">검증과 기존 결과 보존</h2><ul>')
 facts=[f'기존 측정·모델·소스·결과 {len(json.loads((HERE/"baseline_snapshot/hashes.json").read_text()))}개를 별도 복사하고 SHA256으로 보존 여부를 확인했습니다.',
  '원본 281개 영상의 SHA256도 전후 동일합니다.',
  '전처리에서 기존 GN을 원본 281장 모두 다시 실행해 저장 좌표와 차이 0을 확인했습니다.',
  f'픽셀 잡음값을 상수로 주면 기존 GN 좌표와 최대 차이 {audit["constant_variance_check"]["max_abs_coordinate_difference_px"]:.3g}px로 일치합니다.',
  f'기존 기준 예측은 모든 분할에서 다시 계산했으며, 저장된 결과와 최대 차이는 {checks["baseline_max_prediction_difference"]:.3g}입니다.',
  f'{audit["all_prediction_rows_independently_scored"]:,}개 예측 행의 정확도·축별 최대오차·합격 여부를 별도 코드로 다시 계산했습니다.',
  f'119개 잡음 모델의 학습 사진 목록과 {audit["training_only_noise_pair_records_checked"]:,}개 QR 쌍 기록을 검사했습니다. 시험 영상 접근을 차단한 상태에서도 잡음 추정 결과가 같았습니다.',
  '익명 파일명으로 복사한 원본 네 장을 저장 모델로 실행해 캐시 좌표·예측과 일치하는지 확인했습니다.',
  '동일 개발 데이터 내부 비교입니다. 새 기준점·카메라 조건·혼합 Y/T·독립 실제 위치 정답에 대한 검증은 수행하지 않았습니다.']
 pieces.extend('<li>'+esc(v)+'</li>' for v in facts)
 pieces.append('''</ul><p><a href="comparison_summary.csv">핵심 수치 CSV</a> · <a href="metrics.csv">모든 평가 수치</a> · <a href="predictions.csv">전체 예측</a> · <a href="failures.csv">실패 예측</a> · <a href="paired_comparison.csv">개선·악화 대응 비교</a> · <a href="fold_diagnostics.csv">분할별 진단</a> · <a href="experiment_plan.json">사전 실험 설정</a> · <a href="audit_checks.json">검증 기록</a> · <a href="weighted_model_full280.json">저장된 후보 모델</a></p>
 <p>재현 코드: <a href="experiment.py">실험</a> · <a href="predict.py">두 영상 추론</a> · <a href="verify.py">검증</a> · <a href="report.py">시각화</a> · <a href="run.ps1">실행 스크립트</a></p>
 <p>잡음 추정의 출발점은 <a href="../../analysis_v8/r11_noise_model.py">v8 잡음 분석 코드</a>입니다. 이전 비교는 <a href="../gls_comparison_20261014/GLS_비교결과.html">좌표·특징 공분산 GLS 실험</a>에 보존되어 있습니다.</p></main></html>''')
 (HERE/'픽셀잡음가중_GN_비교결과.html').write_text(''.join(pieces),encoding='utf8')
 E.dump(HERE/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,opencv=E.cv2.__version__))
 print(conclusion)
 print(summary[['method','protocol','Y_axis_sign_accuracy','T_axis_sign_accuracy','vector_Y_max','vector_T_max','failed']].to_string(index=False))

if __name__=='__main__':main()
