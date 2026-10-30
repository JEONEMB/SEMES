"""Target candidate costs, displacement fields and four-way comparison figures."""
from pathlib import Path
import json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import experiment as N

HERE=N.HERE
LABEL={'gn_six':'기존 GN + 6후보','pixel_six':'가중 GN + 6후보','gn_v7b':'기존 GN + v7b 2단계','pixel_v7b':'가중 GN + v7b 2단계'}
COLORS={'gn_six':'#2477AA','pixel_six':'#DE772B','gn_v7b':'#149783','pixel_v7b':'#8764A5'}
AXISCOLOR={'Y':'#2477AA','T':'#DE772B'}
PROT=['full280','interleaved','stratified5','blocked_within_session5','repeated20']
PN={'full280':'280장 재대입','interleaved':'짝홀 OOF','stratified5':'층화 5분할 OOF','blocked_within_session5':'시행 내 연속구간 OOF','repeated20':'반복 5분할 20회 합계'}
def esc(s):return html.escape(str(s))
def save(fig,name):
 fig.savefig(HERE/'figures'/f'{name}.png',dpi=155,facecolor='white');fig.savefig(HERE/'figures'/f'{name}.svg',facecolor='white');plt.close(fig)
def setup():
 (HERE/'figures').mkdir(exist_ok=True)
 font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
def style(ax):
 for s in ['top','right']:ax.spines[s].set_visible(False)
 ax.set_axisbelow(True);ax.grid(alpha=.20)
def summary(pred):
 rows=[]
 for p in PROT:
  sub=pred[pred.protocol.str.startswith('repeated5_seed')] if p=='repeated20' else pred[pred.protocol==p]
  for method in N.METHODS:
   g=sub[sub.method==method];r=dict(method=method,protocol=p,**N.E.C.score(g))
   for a in ['Y','T']:
    r[a+'_correct']=int(((g.axis==a)&g.correct).sum())
   rows.append(r)
 return pd.DataFrame(rows)
def select_cases(target):
 rows=target[(target.protocol.str.startswith('repeated5_seed'))&target.method.isin(['gn_six','pixel_six'])]
 p=rows.pivot(index=['protocol','fold'],columns='method',values='pred_axis').reset_index()
 p['seed']=p.protocol.str.replace('repeated5_seed','',regex=False).astype(int);p=p.sort_values('seed')
 cases=[dict(id='full280__all',label='280장 재대입',protocol='full280',fold='all')]
 for name,mask in [('both_wrong',(p.gn_six=='Y')&(p.pixel_six=='Y')),('regression',(p.gn_six=='T')&(p.pixel_six=='Y')),('both_correct',(p.gn_six=='T')&(p.pixel_six=='T'))]:
  if mask.any():
   r=p[mask].iloc[0];cases.append(dict(id=r.protocol+'__'+str(r.fold),label=f'seed {7000+int(r.seed)} · '+{'both_wrong':'두 6후보 방법 모두 오류','regression':'가중 GN에서 새로 오류','both_correct':'두 6후보 방법 모두 정답'}[name],protocol=r.protocol,fold=str(r.fold),kind=name))
 return cases
def load_case(case,method):return np.load(HERE/'folds'/case['id']/(method+'_target.npz'))
def best(c,axis):
 names=c['candidate_names'].astype(str);ids=[i for i,n in enumerate(names) if n.startswith(axis) and c['allowed'][i]]
 return min(ids,key=lambda i:c['costs'][i])
def case_numbers(case,z,m,Xbase):
 folder=HERE/'folds'/case['id'];complete=json.loads((folder/'complete.json').read_text());data=np.load(HERE/'measurements'/complete['measurement_cache'])
 index={n:i for i,n in enumerate(m.file)};tr=np.array([index[n] for n in complete['training_files']]);target=index[N.PLAN['target']]
 models={'gn':N.Z.fit(m,Xbase,tr,m.value.to_numpy(),'inferred_session',True),'pixel':N.Z.fit(m,data['X'],tr,m.value.to_numpy(),'inferred_session',True)}
 crossed=[]
 for measurement,XX in [('gn',Xbase),('pixel',data['X'])]:
  for fit,model in models.items():
   rows=N.six_candidates(XX[target],model);chosen,byaxis=N.choose_candidates(rows)
   cy,ct=byaxis['Y']['cost'],byaxis['T']['cost']
   crossed.append(dict(case=case['id'],measurement=measurement,model_fit=fit,cost_Y=cy,cost_T=ct,log_cost_ratio=float(np.log(cy/ct)),pred_axis=chosen['axis'],pred_value=chosen['value']))
 diag=[]
 for method in N.METHODS:
  c=load_case(case,method);iy,it=best(c,'Y'),best(c,'T');B=c['B'];f=c['X'];W=c['W'];actual=c['raw_displacement'];smooth=B@f.reshape(6,2)
  fy,ft=c['candidate_features'][iy],c['candidate_features'][it]
  yres=W@(f-fy);tres=W@(f-ft)
  assert np.allclose([yres@yres,tres@tres],c['costs'][[iy,it]],rtol=1e-7,atol=1e-5)
  perqr=[]
  for k in range(3):
   sel=c['regions']==k
   perqr.append(dict(QR=k+1,mean_dx=float(actual[sel,0].mean()),mean_dy=float(actual[sel,1].mean()),
    Y_P12_residual_rms_px=float(np.sqrt(np.mean((B[sel]@(f-fy).reshape(6,2))**2))),
    T_P12_residual_rms_px=float(np.sqrt(np.mean((B[sel]@(f-ft).reshape(6,2))**2)))))
  diag.append(dict(case=case['id'],label=case['label'],method=method,mean_dx=float(actual[:,0].mean()),mean_dy=float(actual[:,1].mean()),
   weak_P12_rms_px=float(np.sqrt(np.mean((smooth-smooth.mean(0))**2))),
   discarded_point_residual_rms_px=float(np.sqrt(np.mean((actual-smooth)**2))),
   candidate_field_separation_rms_px=float(np.sqrt(np.mean((B@(fy-ft).reshape(6,2))**2))),
   best_Y_branch=str(c['candidate_names'][iy]),best_T_branch=str(c['candidate_names'][it]),
   Y_value=float(c['candidate_values'][iy]),T_value=float(c['candidate_values'][it]),
   cost_Y=float(c['costs'][iy]),cost_T=float(c['costs'][it]),
   Y_P12_residual_rms_px=float(np.sqrt(np.mean((B@(f-fy).reshape(6,2))**2))),
   T_P12_residual_rms_px=float(np.sqrt(np.mean((B@(f-ft).reshape(6,2))**2))),
   whitened_Y_components_squared=(yres*yres).tolist(),whitened_T_components_squared=(tres*tres).tolist(),perQR=perqr))
 return diag,crossed

def target_ratio_plot(target):
 fig,ax=plt.subplots(figsize=(16,7.5));fig.subplots_adjust(left=.08,right=.97,top=.8,bottom=.23)
 fig.suptitle('같은 T−0.05 사진의 후보 비용 순위는 분할마다 달라진다',x=.035,ha='left',y=.96,fontsize=23,fontweight='bold',color='#17354c')
 for method in N.METHODS:
  g=target[(target.method==method)&target.protocol.str.startswith('repeated5_seed')].copy();g['seed']=g.protocol.str.replace('repeated5_seed','',regex=False).astype(int);g=g.sort_values('seed')
  assert len(g)==20
  ax.plot(7000+g.seed,g.log_cost_ratio,'o-',color=COLORS[method],label=LABEL[method],lw=1.7,ms=4.5)
 ax.axhline(0,color='#b3453c',lw=1.3,ls='--');ax.set_xticks(range(7000,7020));ax.tick_params(axis='x',rotation=40)
 ax.set_xlabel('기존과 동일한 분할 seed');ax.set_ylabel('log(Y 최소 비용 / T 최소 비용)');style(ax)
 fig.text(.08,.86,'0 위: T 선택(정답) / 0 아래: Y 선택(오류). 각 방법의 같은 비용 공간 안에서 두 후보를 비교합니다.',fontsize=12,color='#42586c')
 fig.legend(ncol=2,loc='lower left',bbox_to_anchor=(.075,.08),frameon=False)
 fig.text(.035,.025,'현재 사진은 매 라운드 한 번씩 학습에서 제외됩니다. 비용 비율은 정답 확률이 아니며 절대 비용의 크기는 방법·분할 간 직접 비교하지 않습니다.',fontsize=10.5,color='#42586c')
 save(fig,'01_T005_cost_ratio_by_split')

def candidate_plot(case):
 fig,axs=plt.subplots(2,2,figsize=(16,11));fig.subplots_adjust(left=.12,right=.95,top=.81,bottom=.14,wspace=.34,hspace=.48)
 fig.suptitle('T−0.05의 후보별 비용과 이동량',x=.035,ha='left',y=.965,fontsize=23,fontweight='bold',color='#17354c')
 fig.text(.035,.91,case['label'],fontsize=13,color='#42586c')
 for ax,method in zip(axs.ravel(),N.METHODS):
  c=load_case(case,method);names=c['candidate_names'].astype(str);costs=c['costs'];allowed=c['allowed'];values=c['candidate_values']
  good=np.flatnonzero(allowed);winner=good[np.argmin(costs[good])];maximum=float(np.nanmax(costs[good]))
  for i,name in enumerate(names):
   if allowed[i]:
    ax.barh(i,costs[i],height=.53,color=AXISCOLOR[name[0]],edgecolor='#172f44' if i==winner else 'none',linewidth=2)
    ax.text(costs[i]+maximum*.025,i,f'{costs[i]:.3f} · v={values[i]:+.5f}',va='center',fontsize=10)
   else:ax.text(maximum*.025,i,'부호 조건 불일치로 제외',va='center',color='#8996a1',fontsize=10)
  ax.set_yticks(range(len(names)),names);ax.set_ylim(len(names)-.5,-.5);ax.set_xlim(0,maximum*1.75)
  ax.set_title(LABEL[method]+' · 선택 '+str(names[winner]),loc='left',fontsize=14,fontweight='bold')
  ax.set_xlabel('해당 모델의 공분산 정규화 비용');style(ax)
 fig.text(.035,.07,'6후보는 부호 조건을 통과한 후보를 비교합니다. v7b는 1단계 축 판정 비용과 그 단계의 적합값을 표시합니다.',fontsize=11,color='#42586c')
 fig.text(.035,.032,'v7b의 최종 이동량은 별도의 2단계 곡선으로 계산합니다. 네 패널의 절대 비용을 서로 직접 비교해서는 안 됩니다.',fontsize=11,color='#42586c')
 save(fig,'02_T005_candidates')

def movement_plot(case):
 c=load_case(case,'pixel_six');iy,it=best(c,'Y'),best(c,'T');B=c['B'];f=c['X'];raw=c['raw_displacement'];smooth=B@f.reshape(6,2)
 residualY=B@(f-c['candidate_features'][iy]).reshape(6,2);residualT=B@(f-c['candidate_features'][it]).reshape(6,2)
 ref=N.E.read(N.E.ROOT/'center.bmp');x0,x1,y0,y1=1520,2280,940,1190;image=ref[y0-N.E.CROP[1]:y1-N.E.CROP[1],x0-N.E.CROP[0]:x1-N.E.CROP[0]]
 fig,axs=plt.subplots(2,2,figsize=(18,10));fig.subplots_adjust(left=.045,right=.98,top=.81,bottom=.13,wspace=.12,hspace=.35)
 fig.suptitle('205개 특징점 이동과 Y/T 후보가 남기는 미세 잔차',x=.035,ha='left',y=.96,fontsize=23,fontweight='bold',color='#17354c')
 fig.text(.035,.902,case['label']+' / 가중 GN 측정 / 배경은 센터 영상',fontsize=12,color='#42586c')
 fields=[(raw,8,'실제 측정 이동 ×8'),(smooth-smooth.mean(0),400,'주 평행이동을 제거한 P12 변형 ×400'),
  (residualY,1000,'Y 후보의 P12 잔차 ×1000'),(residualT,1000,'T 후보의 P12 잔차 ×1000')]
 for ax,(d,scale,title) in zip(axs.ravel(),fields):
  ax.imshow(image,cmap='gray',vmin=0,vmax=255,extent=[x0,x1,y1,y0],alpha=.60)
  p=c['points'];ax.quiver(p[:,0],p[:,1],d[:,0]*scale,d[:,1]*scale,color='#bd2e3b',angles='xy',scale_units='xy',scale=1,width=.0023,headwidth=3,headlength=3.5)
  ax.scatter(p[:,0],p[:,1],s=3,color='#143f5c');ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
  ax.set_title(title+f' · 좌표 RMS {np.sqrt(np.mean(d*d)):.5f}px',loc='left',fontweight='bold',fontsize=12)
  ax.set_xlabel('센터 x [원본 px]');ax.set_ylabel('센터 y [원본 px]')
 fig.text(.035,.065,f'평균 이동: Δx={raw[:,0].mean():.6f}px, Δy={raw[:,1].mean():.6f}px. 잔차 두 패널은 같은 배율입니다.',fontsize=11,color='#42586c')
 fig.text(.035,.025,'잔차 화살표는 판정에 사용한 12계수 공간 변형의 차이입니다. 실제 위치 정답에 대한 추적 오차가 아니며, 판정 비용은 이 잔차의 단순 픽셀 제곱합과 다릅니다.',fontsize=10.5,color='#42586c')
 save(fig,'03_T005_feature_motion')

def cost_components(case):
 fig,axs=plt.subplots(1,2,figsize=(16,7.5));fig.subplots_adjust(left=.07,right=.97,top=.79,bottom=.24,wspace=.18)
 fig.suptitle('공분산으로 정규화한 어떤 성분이 Y 선택을 만드는가?',x=.035,ha='left',y=.96,fontsize=22,fontweight='bold',color='#17354c')
 for ax,method in zip(axs,['gn_six','pixel_six']):
  c=load_case(case,method);iy,it=best(c,'Y'),best(c,'T');ys=c['W']@(c['X']-c['candidate_features'][iy]);ts=c['W']@(c['X']-c['candidate_features'][it])
  x=np.arange(1,13);ax.bar(x-.18,ys*ys,width=.35,color=AXISCOLOR['Y'],label=f'Y 합계 {ys@ys:.3f}');ax.bar(x+.18,ts*ts,width=.35,color=AXISCOLOR['T'],label=f'T 합계 {ts@ts:.3f}')
  ax.set_xticks(x);ax.set_xlabel('백색화된 성분 번호 [1–12]');ax.set_ylabel('후보 비용 기여량');ax.set_title(LABEL[method],loc='left',fontweight='bold');ax.legend(frameon=False);style(ax)
 fig.text(.035,.86,case['label'],fontsize=12,color='#42586c')
 fig.text(.035,.10,'각 색의 12개 막대를 더하면 해당 후보 비용이 됩니다. 성분은 공분산이 반영된 좌표이며, 특정 QR나 특정 점 하나를 뜻하지 않습니다.',fontsize=10.5,color='#42586c')
 fig.text(.035,.05,'따라서 화면에서 보이는 잔차 화살표 크기만으로 후보 순위를 판단할 수 없습니다. 두 방법의 백색화 좌표도 서로 다릅니다.',fontsize=10.5,color='#42586c')
 save(fig,'04_T005_cost_components')

def crossed_plot(case,rows):
 sub=[r for r in rows if r['case']==case['id']];a=np.array([[next(r['log_cost_ratio'] for r in sub if r['measurement']==m and r['model_fit']==f) for f in ['gn','pixel']] for m in ['gn','pixel']])
 fig,ax=plt.subplots(figsize=(9.5,6.8));fig.subplots_adjust(left=.22,right=.88,top=.74,bottom=.23)
 vmax=max(np.abs(a).max(),.01);im=ax.imshow(a,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{a[i,j]:+.5f}\n'+('T 선택 · 정답' if a[i,j]>0 else 'Y 선택 · 오류'),ha='center',va='center',fontsize=13,color='white' if abs(a[i,j])>.6*vmax else '#17354c')
 ax.set_xticks([0,1],['기존 측정으로\n학습한 모델','가중 측정으로\n학습한 모델']);ax.set_yticks([0,1],['기존 측정값\n(현재 사진)','가중 측정값\n(현재 사진)']);fig.colorbar(im,ax=ax,shrink=.75,label='log(Y/T 비용) · 0 위 T / 0 아래 Y')
 fig.suptitle('측정값 변화와 학습 모델 변화를 교차해 확인',x=.04,ha='left',y=.95,fontsize=20,fontweight='bold',color='#17354c')
 fig.text(.04,.84,case['label'],fontsize=12,color='#42586c');fig.text(.04,.10,'6후보 판정기의 진단용 교차 적용입니다. 새 배포 방법의 성적이 아니며, 두 변경의 영향을 분리해 보는 데 사용합니다.',fontsize=10.5,color='#42586c')
 save(fig,'05_T005_measurement_model_crossing')

def performance_plot(s,metrics):
 fig,axs=plt.subplots(2,2,figsize=(16,10));fig.subplots_adjust(left=.24,right=.96,top=.82,bottom=.16,hspace=.38,wspace=.25)
 fig.suptitle('같은 280장·같은 분할에서 v7b 2단계 판정 비교',x=.035,ha='left',y=.96,fontsize=23,fontweight='bold',color='#17354c')
 fig.text(.035,.90,'반복 5분할 20회 합계 / QR205·원래 명령값 / 기존 6후보와 v7b 구조를 각각 재학습',fontsize=12,color='#42586c')
 data=s[s.protocol=='repeated20'].set_index('method').loc[N.METHODS];y=np.arange(4)
 for ax,a in zip(axs[0],['Y','T']):
  vals=data[a+'_axis_sign_accuracy'].to_numpy()*100
  ax.barh(y,vals,color=[COLORS[m] for m in N.METHODS],height=.58);ax.set_yticks(y,[LABEL[m] for m in N.METHODS] if a=='Y' else ['']*4);ax.invert_yaxis();ax.set_xlim(0,118)
  for yi,v,(_,r) in zip(y,vals,data.iterrows()):ax.text(v+1,yi,f'{v:.4f}%\n{int(r[a+"_correct"])}/{int(r[a+"_n"])}',va='center',fontsize=10)
  ax.set_title(a+' 축·부호 정확도',loc='left',fontsize=15,fontweight='bold');ax.set_xlabel('정확도 [%]');ax.set_xticks([0,25,50,75,100]);style(ax)
 for ax,a in zip(axs[1],['Y','T']):
  vals=data['vector_'+a+'_max'].to_numpy();maximum=max(vals.max(),.09)
  ax.barh(y,vals,color=[COLORS[m] for m in N.METHODS],height=.58);ax.set_yticks(y,[LABEL[m] for m in N.METHODS] if a=='Y' else ['']*4);ax.invert_yaxis();ax.set_xlim(0,maximum*1.45)
  ax.axvline(.09,color='#b3453c',ls='--')
  for yi,v in zip(y,vals):ax.text(v+maximum*.03,yi,f'{v:.6f}',va='center',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',pad=.7))
  ax.set_title(a+' 최대오차 · 오분류 포함',loc='left',fontsize=15,fontweight='bold');ax.set_xlabel('오차 [mm]' if a=='Y' else '오차 [T 명령 단위]');style(ax)
 fig.text(.035,.065,'v7b 구조: 0 근처 선형·센터 고정으로 축 판정, 전체 범위의 자유 절편 2차식으로 이동량 추정. 두 단계에 원래의 방해 성분을 포함합니다.',fontsize=10.5,color='#42586c')
 fig.text(.035,.025,'5,600회는 같은 280장의 반복 예측이며 독립 새 촬영 수가 아닙니다. 실제 v7b 원본의 QR232·영상 보정 라벨 성적과 동일한 실험이 아닙니다.',fontsize=10.5,color='#42586c')
 save(fig,'06_four_method_cv_comparison')

def html_table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda v:f'{v:.6f}')+'</div>'
def main():
 setup();pred=pd.read_csv(HERE/'predictions.csv');metrics=pd.read_csv(HERE/'metrics.csv');target=pd.read_csv(HERE/'T005_predictions_and_costs.csv',dtype={'fold':str})
 s=summary(pred);s.to_csv(HERE/'comparison_summary.csv',index=False,encoding='utf-8-sig');cases=select_cases(target)
 z,m,Xbase=N.E.load_data();numbers=[];crossed=[]
 for case in cases:
  n,c=case_numbers(case,z,m,Xbase);numbers+=n;crossed+=c
 N.dump(HERE/'case_analysis.json',dict(cases=cases,measurements_and_costs=numbers,crossed_measurement_and_model=crossed))
 pd.DataFrame(crossed).to_csv(HERE/'crossed_measurement_model.csv',index=False,encoding='utf-8-sig')
 selected=next(c for c in cases if c.get('kind')=='both_wrong');regression=next((c for c in cases if c.get('kind')=='regression'),selected)
 target_ratio_plot(target);candidate_plot(selected);movement_plot(selected);cost_components(selected);crossed_plot(regression,crossed);performance_plot(s,metrics)
 repeated=metrics[metrics.protocol.str.startswith('repeated5_seed')]
 overview=[]
 for method in N.METHODS:
  g=repeated[repeated.method==method];t=target[(target.method==method)&target.protocol.str.startswith('repeated5_seed')]
  overview.append(dict(method=method,repeat_failed=int(g.failed.sum()),perfect_rounds=int((g.failed==0).sum()),target_wrong=int((t.pred_axis!='T').sum())))
 N.dump(HERE/'overview.json',overview)
 rows=[]
 for _,r in s.iterrows():
  rows.append({'평가':PN[r.protocol],'방법':LABEL[r.method],'Y 축':f'{r.Y_axis_accuracy*100:.4f}%',
   'Y 축·부호':f'{r.Y_axis_sign_accuracy*100:.4f}% ({int(r.Y_correct)}/{int(r.Y_n)})',
   'T 축':f'{r.T_axis_accuracy*100:.4f}%','T 축·부호':f'{r.T_axis_sign_accuracy*100:.4f}% ({int(r.T_correct)}/{int(r.T_n)})',
   'Y 최대오차 mm':r.vector_Y_max,'T 최대오차':r.vector_T_max,'전체 실패':f'{int(r.failed)}/{int(r.n)}'})
 display=pd.DataFrame(rows)
 primary=next(r for r in numbers if r['case']==selected['id'] and r['method']=='pixel_six')
 parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T−0.05 오분류와 v7b 비교</title><style>body{margin:0;background:#f6f8fa;color:#17354c;font:16px/1.85 "Malgun Gothic",sans-serif}main{max-width:1280px;margin:auto;padding:30px 26px 70px}h1{font-size:30px;line-height:1.5}h2{margin-top:44px;font-size:24px}h3{margin-top:28px}a{color:#087b90}nav a{display:inline-block;margin-right:18px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:10px 7px;white-space:nowrap;text-align:left;border-bottom:1px solid #d9e2e9}th{background:#1b3b50;color:white}tr:nth-child(even){background:#edf3f7}figure{margin:28px 0}img{width:100%;height:auto;display:block;background:white}figcaption{font-size:14px;color:#4b6174}li{margin:10px 0}code{font-family:Consolas,monospace}@media(max-width:650px){main{padding:18px 14px 40px}h1{font-size:24px}h2{font-size:21px}}</style><main><h1>T−0.05는 왜 Y로 틀리는가? v7b 2단계 판정도 같은 조건에서 비교</h1><p>2026-10-15 · 기존 결과 보존 · QR205 · 원래 명령값 · 동일 119개 학습 분할</p><nav><a href="#cause">오분류 분석</a><a href="#comparison">v7b 비교</a><a href="#cases">경우별 비용</a><a href="#audit">검증·기록</a></nav>''']
 parts.append('<h2 id="cause">1. 유효점 205개에서도 바뀌는 Y/T 후보 비용 순위</h2>')
 parts.append(f'<p>대표 실패 분할({esc(selected["label"])})에서 T−0.05의 평균 점 이동은 Δx={primary["mean_dx"]:.6f}px, Δy={primary["mean_dy"]:.6f}px입니다. 큰 공통 이동을 제거한 P12 변형의 RMS는 {primary["weak_P12_rms_px"]:.6f}px, 가장 가까운 Y/T 후보가 예측한 점 이동 사이의 RMS 차이는 {primary["candidate_field_separation_rms_px"]:.6f}px입니다.</p>')
 parts.append(f'<p>가중 GN의 205점은 모두 추적 유효 조건을 통과했습니다. 하지만 해당 분할에서 Y 후보 {esc(primary["best_Y_branch"])}의 비용 {primary["cost_Y"]:.6f}가 T 후보 {esc(primary["best_T_branch"])}의 비용 {primary["cost_T"]:.6f}보다 작아, Y={primary["Y_value"]:.6f}로 선택됩니다. 이것은 알고리즘이 어떤 계산으로 오분류했는지 보여주며, 카메라 잡음·표면 반사·기구 변동 중 물리 원인을 확정한 결과는 아닙니다.</p>')
 parts.append('<h3>QR별 이동과 후보 잔차</h3>')
 parts.append(html_table(pd.DataFrame(primary['perQR']).rename(columns={'mean_dx':'평균 Δx px','mean_dy':'평균 Δy px','Y_P12_residual_rms_px':'Y 후보 좌표 RMS px','T_P12_residual_rms_px':'T 후보 좌표 RMS px'})))
 parts.append('<p>이 사례의 왼쪽·가운데 QR은 픽셀 잔차 RMS 기준으로 T 후보에 조금 더 가깝고, 오른쪽 QR은 Y 후보에 더 가깝습니다. 다만 최종 비용은 QR별 RMS의 단순 합이 아니라 12계수 전체 공분산 거리이므로, 특정 QR 하나가 물리 원인이라고 단정하지 않습니다.</p>')
 for name,caption in [('01_T005_cost_ratio_by_split','현재 사진은 동일하지만 학습 자료에 따라 곡선·절편·공분산이 달라지며 Y/T 비용의 순위가 바뀝니다.'),('03_T005_feature_motion','큰 공통 이동과 작은 판 변형을 나눠 표시했습니다. 확대 배율과 잔차 정의를 확인하십시오.'),('02_T005_candidates','각 후보의 최적 이동량과 비용. 부호 불일치 후보는 원래 추론과 같이 제외합니다.'),('04_T005_cost_components','단순한 픽셀 잔차 RMS보다 공분산으로 정규화한 전체 잔차가 최종 비용을 결정합니다.'),('05_T005_measurement_model_crossing','기존·가중 측정값과 두 학습 모델을 교차 적용한 진단. 앞단 측정 변화와 뒷단 보정 변화의 영향을 분리합니다.')]:
  parts.append(f'<figure><img src="figures/{name}.png" alt="{esc(caption)}"><figcaption>{esc(caption)} <a href="figures/{name}.svg">SVG</a></figcaption></figure>')
 crossed_case=[r for r in crossed if r['case']==regression['id']]
 cby={(r['measurement'],r['model_fit']):r for r in crossed_case}
 if cby['gn','gn']['pred_axis']=='T' and cby['pixel','gn']['pred_axis']=='Y' and cby['gn','pixel']['pred_axis']=='T':
  parts.append('<p><strong>'+esc(regression['label'])+'에서는 기존 모델을 그대로 둔 채 현재 사진의 측정값만 가중 GN 결과로 바꿔도 T에서 Y로 판정이 바뀝니다.</strong> 반대로 기존 측정값에 가중 측정으로 학습한 모델만 적용하면 T를 유지합니다. 이 분할의 새 오류에는 측정값 변화가 판정 경계를 넘긴 영향이 확인됩니다. 이는 알고리즘 변경의 진단이며, 실제 물리 위치나 잡음 원인의 인과관계를 확정한 것은 아닙니다.</p>')
 parts.append('<h2 id="comparison">2. v7b 2단계 판정 비교</h2><p>판정기는 0 근처 선형·센터 고정 모델로 축을 고르고, 전체 학습 범위의 자유 절편 2차식으로 이동량을 계산합니다. 원본 v7b의 세로 이동·화면 내 회전 방해 성분과 단계별 공분산, 8회 최적화를 포함했습니다. 영상 보정 라벨을 사용하지 않고 각 외부 학습 분할 안에서 원래 명령값으로 맞췄습니다.</p>')
 parts.append(html_table(pd.DataFrame([dict(방법=LABEL[r['method']],반복실패=f'{r["repeat_failed"]}/5600',오류0라운드=f'{r["perfect_rounds"]}/20',T005오분류=f'{r["target_wrong"]}/20') for r in overview])))
 parts.append('<p><strong>이번 공통 입력 조건에서는 v7b 2단계 판정의 반복 실패가 154회·138회로, 기존 6후보의 13회·12회보다 많았습니다. T−0.05도 두 v7b 조합 모두 20회 중 20회 Y로 오분류했습니다.</strong> v7b의 원래 QR232·영상 보정 라벨 성적과 이 실험은 입력·정답 조건이 다르며, 이 결과를 모든 조건의 방법 순위로 일반화하지 않습니다.</p>')
 parts.append('<figure><img src="figures/06_four_method_cv_comparison.png" alt="네 방법의 반복 교차검증 정확도와 최대오차"><figcaption>전체 280장 재대입·짝홀·층화·연속구간·반복 교차검증의 상세 값은 아래 표에 모두 제시합니다.</figcaption></figure>')
 parts.append(html_table(display).replace('<table ','<table id="main-results" ',1))
 diagnostic=metrics[metrics.protocol=='session_out_missing_origin'][['method','axis_accuracy','axis_sign_accuracy','vector_Y_max','vector_T_max','failed']].copy()
 diagnostic['method']=diagnostic['method'].map(LABEL)
 parts.append('<details><summary>보정 후보 일부가 사라지는 시행 전체 제외 진단</summary><p>기존과 같은 분할도 실행했습니다. 6후보 방법에서는 일부 원점·부호 분기가 학습에 없으므로, 여섯 분기를 보정한 상태의 새 시행 검증과 구분해야 합니다.</p>'+html_table(diagnostic)+'</details>')
 parts.append('<p>오분류도 포함한 (Y,T) 벡터 최대오차입니다. 실패는 축·부호 오류 또는 어느 축이든 오차 0.09 초과입니다. 같은 280장을 반복 평가한 결과로 독립적인 새 촬영의 성능은 아닙니다. T는 원래 명령 단위를 유지합니다.</p>')
 parts.append('<h2 id="cases">3. 재대입·실패·새 오류·정답 분할의 비용을 함께 보기</h2>')
 for case in cases:
  sub=target[(target.protocol==case['protocol'])&(target.fold==case['fold'])].copy();sub.method=sub.method.map(LABEL)
  parts.append('<h3>'+esc(case['label'])+'</h3>')
  parts.append(html_table(sub[['method','pred_axis','pred','branch','cost_Y','cost_T','log_cost_ratio']].rename(columns={'method':'방법','pred_axis':'예측 축','pred':'최종 이동량','branch':'선택 후보','cost_Y':'Y 비용','cost_T':'T 비용','log_cost_ratio':'log(Y/T 비용)'})))
 parts.append('<p>표의 최종 이동량과 v7b 1단계 후보 그림의 이동량은 다른 계산 단계입니다. 후보의 절대 비용을 서로 다른 판정기나 서로 다른 학습 분할 간에 정답 확률처럼 비교하지 않습니다.</p>')
 parts.append('<h2 id="audit">4. 검증과 기존 결과 보존</h2><ul><li>직전 두 방법의 모든 예측 축·이동량을 동일 분할에서 재현했습니다.</li><li>원본 v7b의 Model 클래스만 분리 실행해 이식한 판정·이동량 계산과 수치 비교했습니다. 원본 모듈의 보정 라벨 계산은 실행하지 않았습니다.</li><li>각 시험 사진의 특징·명령값을 바꿔도 학습 모델이 달라지지 않는지 검사했습니다.</li><li>기존 결과와 소스의 해시를 전후 비교했습니다. 원본 영상·기존 모델·기존 실험 결과를 덮어쓰지 않았습니다.</li><li>이 실험은 구조를 고정한 비교입니다. 결과를 보고 v7b의 학습 범위·문턱·방해 성분을 재선택하지 않았습니다.</li></ul>')
 parts.append('''<p><a href="comparison_summary.csv">핵심 수치</a> · <a href="metrics.csv">모든 평가</a> · <a href="predictions.csv">전체 예측</a> · <a href="T005_candidate_costs.csv">모든 후보 비용</a> · <a href="T005_predictions_and_costs.csv">분할별 T−0.05 비용</a> · <a href="case_analysis.json">이동장·잔차 분석 수치</a> · <a href="crossed_measurement_model.csv">측정·모델 교차 진단</a> · <a href="checks.json">검증 기록</a> · <a href="audit_checks.json">독립 재채점·보존 검사</a> · <a href="source_fidelity_checks.json">원본 v7b 수치 대조</a> · <a href="experiment_plan.json">고정한 비교 설정</a></p><p><a href="../colleague_brief_20261015/동료설명_QR_YT_알고리즘.html">동료 설명용 알고리즘 정리</a> · <a href="../pixel_weighted_gn_20261015/픽셀잡음가중_GN_비교결과.html">직전 완료 실험</a></p></main></html>''')
 (HERE/'T005_원인분석_v7b_비교.html').write_text(''.join(parts),encoding='utf8')
 print(pd.DataFrame(overview).to_string(index=False));print(s[['method','protocol','Y_axis_sign_accuracy','T_axis_sign_accuracy','vector_Y_max','vector_T_max','failed']].to_string(index=False))
 print('CASE',json.dumps(primary,ensure_ascii=False))
 print('CROSSED',json.dumps([r for r in crossed if r['case']==regression['id']],ensure_ascii=False))

if __name__=='__main__':main()
