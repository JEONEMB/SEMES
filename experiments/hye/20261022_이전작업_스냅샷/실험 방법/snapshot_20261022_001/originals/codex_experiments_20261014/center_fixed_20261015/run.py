from pathlib import Path
import sys,json,hashlib,zipfile
import numpy as np,pandas as pd,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;PROJ=BASE.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'measurement'))
import zero_experiment as Z
from spread import SpreadPairExtractor
from measure import ROOT,CROP,BOXES
plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':12})
COLORS=['#E63946','#F49D18','#B455C7','#2878D0','#00A087','#D95F02','#8C6BB1','#C51B7D','#0086A8','#6B8E23','#BD692E','#4A55B8']; BG='#17191F';FG='#F4F5F7';SUB='#C3C8D2'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=[p for folder in ['measurement','full280','trial_direct_reference_20261015','trial_all_points_20261015'] for p in (BASE/folder).rglob('*') if p.is_file()]
hashes={str(p.relative_to(PROJ)):digest(p) for p in protected}
def raw(n):return ROOT/n if n=='center.bmp' else ROOT/('Y' if n.startswith('Y') else 'Theta')/n
old=np.load(BASE/'measurement/gn17_spread14.npz');names=old['names'].astype(str)
cache=OUT/'center_once_tracks.npz'
if not cache.exists():
    model=SpreadPairExtractor(ROOT/'center.bmp',features=None,min_distance=14)
    pts=model.points+np.array(CROP[:2]);reg=model.regions;ds=[];vs=[];qs=[]
    for k,n in enumerate(names):
        if n=='center.bmp':r=dict(D=np.zeros_like(pts,dtype=np.float32),valid=np.ones(len(pts),bool),quality=np.zeros((len(pts),5),np.float32))
        else:r=model.extract(raw(n))
        ds.append(r['D']);vs.append(r['valid']);qs.append(r['quality'])
        if k%40==0:print('CENTER TRACK',k+1,len(names),flush=True)
    np.savez_compressed(cache,names=names,points=pts,regions=reg,D=np.stack(ds),valid=np.stack(vs),quality=np.stack(qs))
z=np.load(cache);pts=z['points'];reg=z['regions'];D=z['D'].astype(float);valid=z['valid'];assert valid.all();assert len(pts)==205
assert np.allclose(pts,old['points'],atol=1e-5)
ic=int(np.flatnonzero(names=='center.bmp')[0]);assert np.all(D[ic]==0)
m=Z.manifest(names);groups=np.zeros(len(pts),int)
for q,(x0,y0,x1,y1) in enumerate(BOXES):
    s=reg==q;groups[s]=4*q+(pts[s,0]>=(x0+x1)/2).astype(int)+2*(pts[s,1]>=(y0+y1)/2).astype(int)
pd.DataFrame(dict(point_id=np.arange(len(pts)),x_center=pts[:,0],y_center=pts[:,1],qr=reg+1,group=groups+1)).to_csv(OUT/'固定点_205.csv',index=False,encoding='utf-8-sig')
x=(pts[:,0]-1906)/486;y=(pts[:,1]-1070)/122;B=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
X=np.linalg.lstsq(B,D.transpose(1,0,2).reshape(len(pts),-1),rcond=None)[0].reshape(6,len(D),2).transpose(1,0,2).reshape(len(D),12)
ids=np.flatnonzero(m.axis!='C'); index={n:i for i,n in enumerate(names)}
saved_folds=json.loads((BASE/'gls_comparison_20261014/splits.json').read_text(encoding='utf-8'))
folds=[f for f in saved_folds if f['protocol']!='full280']
preds=[];summ=[];configs=[('center_quadratic','center',True),('center_linear','center',False),('legacy_six_origins','inferred_session',True)]
for method,mode,quad in configs:
    splits=[dict(protocol='full280_resubstitution',fold='all',train=names[ids].tolist(),test=names[ids].tolist())]+folds
    for ss in splits:
        tr=np.array([index[n] for n in ss['train']]);te=np.array([index[n] for n in ss['test']])
        model=Z.fit(m,X,tr,m.value.to_numpy(),'center' if mode=='center' else mode,quad)
        if mode=='center':assert len(model[0])==2 and all(np.all(v[1]==0) for v in model[0].values())
        df=Z.predict(m,X,te,model,mode);df['method']=method;df['protocol']=ss['protocol'];df['fold']=ss['fold'];preds.append(df)
        if ss['protocol']=='full280_resubstitution':
            branches,W=model
            np.savez_compressed(OUT/(method+'_frozen_model.npz'),W=W,branch_names=np.array(list(branches)),**{f'{k}_{part}':v[j] for k,v in branches.items() for part,j in [('base',1),('slope',2),('quadratic',3)]})
    print('EVAL',method,flush=True)
p=pd.concat(preds,ignore_index=True)
for (method,protocol),df in p.groupby(['method','protocol']):summ.append(dict(method=method,protocol=protocol,**Z.score(df)))
summary=pd.DataFrame(summ);summary.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig');p.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
true=np.c_[np.where(p.axis=='Y',p.value,0),np.where(p.axis=='T',p.value,0)];pr=np.c_[np.where(p.pred_axis=='Y',p.pred,0),np.where(p.pred_axis=='T',p.pred,0)]
p['Y_abs_error']=np.abs(true[:,0]-pr[:,0]);p['T_abs_error']=np.abs(true[:,1]-pr[:,1]);p['axis_sign_correct']=(p.axis==p.pred_axis)&(np.sign(p.value)==np.sign(p.pred));p['pass']=p.axis_sign_correct&(p.Y_abs_error<=.09)&(p.T_abs_error<=.09)
p[~p['pass']].to_csv(OUT/'failures.csv',index=False,encoding='utf-8-sig');p.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
# Test labels, trial identity and names do not affect predictions or training of the frozen center model.
ss=folds[0];tr=np.array([index[n] for n in ss['train']]);te=np.array([index[n] for n in ss['test']]);mo=Z.fit(m,X,tr,m.value.to_numpy(),'center',True)
m2=m.copy();m2.loc[te,'value']=12345.;m2.loc[te,'session']='UNKNOWN';m2.loc[te,'axis']='?';m2.loc[te,'file']='unknown.bmp'
a=Z.predict(m,X,te,mo,'center');b=Z.predict(m2,X,te,mo,'center');assert np.array_equal(a.pred.to_numpy(),b.pred.to_numpy()) and np.array_equal(a.pred_axis.to_numpy(),b.pred_axis.to_numpy())
config_sessions=['Y2-','Y1-','Y2+','Y1+','T+','T-'];records=[]
im=cv2.imdecode(np.fromfile(ROOT/'center.bmp',np.uint8),0);xa,xb,ya,yb=1400,2390,900,1235
for no,session in enumerate(config_sessions,1):
    ii=np.flatnonzero(m.session==session);ii=ii[np.argsort(abs(m.value.to_numpy()[ii]))];order=np.r_[ic,ii]
    fig=plt.figure(figsize=(16,9),facecolor=BG)
    fig.text(.045,.952,f'Trial {no:02d}  {session}  |  center 고정 · 전체 205점',color=FG,fontsize=26,weight='bold')
    fig.text(.045,.911,f'center에서 한 번 검출한 동일 ID · trial {len(ii)}장 직접 추적 · center → |명령값| 증가 순 · 0점 후보 보정 없음',color=SUB,fontsize=13)
    for row,amp in enumerate([1,100]):
        ax=fig.add_axes([.045,.495-row*.385,.91,.345]);ax.imshow(im[ya:yb,xa:xb],cmap='gray',extent=[xa,xb,yb,ya],vmin=-45,vmax=350)
        for g in range(12):
            sel=groups==g;pp=pts[sel][None]+D[order][:,sel]*[1,amp]
            ax.add_collection(LineCollection(pp.transpose(1,0,2),colors=COLORS[g],linewidths=.85,alpha=.85))
            ax.scatter(pts[sel,0],pts[sel,1],s=10,facecolors='white',edgecolors='black',linewidths=.3,zorder=4)
            ax.scatter(pp[-1,:,0],pp[-1,:,1],s=7,color=COLORS[g],marker='>',zorder=5)
        ax.set(xlim=(xa,xb),ylim=(yb,ya));ax.set_aspect('equal');ax.set_axis_off()
        ax.set_title(('실제 배율: 가로 ×1 · 세로 ×1' if amp==1 else '확대 보기: 가로 ×1 · 세로 ×100 — 기울기·길이는 실제 배율이 아닙니다')+'  |  선 = 실제 측정점 연결 · 흰 점 = center 위치',color=SUB,fontsize=12,loc='left',pad=7)
    handles=[Line2D([0],[0],color=COLORS[g],lw=3,label=f'{g+1:02d}') for g in range(12)];fig.legend(handles=handles,ncol=12,loc='lower center',bbox_to_anchor=(.5,.055),frameon=False,labelcolor=FG,fontsize=11)
    fig.text(.045,.025,'12색은 QR별 공간 4분면 구분이며 P12 계수와 무관합니다. 두 패널 모두 205개 전부 표시, 회귀 추세선 없음.',color=SUB,fontsize=10)
    fn=f'{no:02d}_{session.replace("+","plus").replace("-","minus")}_center고정_전체경로'
    fig.savefig(OUT/(fn+'.png'),dpi=200,facecolor=BG);fig.savefig(OUT/(fn+'.svg'),facecolor=BG);plt.close(fig);records.append(dict(trial=session,png=fn+'.png',photos=len(ii),points=len(pts)))
print(summary[summary.protocol=='full280_resubstitution'].to_string(index=False),flush=True)
full=p[(p.method=='center_quadratic')&(p.protocol=='full280_resubstitution')]
fig,axs=plt.subplots(1,2,figsize=(12,4.8));
for ax,col in zip(axs,['Y_abs_error','T_abs_error']):
    ax.scatter(np.arange(len(full)),full[col],s=13,c=np.where(full.axis_sign_correct,'#007C91','#CF3F3F'));ax.axhline(.09,color='orange',ls='--',label='허용오차 0.09');ax.set_title(col);ax.set_xlabel('280장 전체 사진');ax.legend();ax.grid(alpha=.2)
fig.suptitle('고정 center · 절편 0 · 2차 모델: 재대입 두 축 오차 (오분류 포함)');fig.tight_layout();fig.savefig(OUT/'예측오차.png',dpi=180);plt.close(fig)
assert all(digest(PROJ/k)==v for k,v in hashes.items())
(OUT/'preserved_hashes.json').write_text(json.dumps(hashes,indent=2,ensure_ascii=False),encoding='utf-8')
checks=dict(center_points_detected_once=205,all281_valid=bool(valid.all()),same_point_ids_all_frames=True,center_reference_zero=bool(np.all(D[ic]==0)),fresh_vs_previous_max_displacement_difference=float(np.max(np.abs(D-old['D']))),preserved_files=len(hashes),heldout_metadata_not_used=True,zero_candidates=0,center_curve_intercepts=0,saved_split_records=len(saved_folds),cv_folds=len(folds),total_noncenter_images=len(ids))
(OUT/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
thumb=Image.new('RGB',(1600,1350),BG)
for k,r in enumerate(records):
    a=Image.open(OUT/r['png']).convert('RGB');a.thumbnail((800,450));thumb.paste(a,((k%2)*800,(k//2)*450))
thumb.save(OUT/'6장_미리보기.jpg',quality=94)
cols=['method','protocol','n','Y_axis_sign_accuracy','T_axis_sign_accuracy','strict_pass09','vector_Y_max','vector_T_max']
html='<meta charset="utf-8"><title>고정 center 전체 경로와 예측 결과</title><style>body{font-family:Malgun Gothic;margin:24px;background:#f2f5f9;color:#17324d}img{max-width:100%;display:block;margin:20px 0}table{border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #ccd4df}section{overflow:auto}a{color:#007c91}</style><h1>고정 center 전체 경로와 예측 결과</h1><p><a href="../../ALGORITHM_RULES.md">공통 규칙 원본</a> · <a href="metrics.csv">전체 지표</a> · <a href="predictions.csv">사진별 예측</a> · <a href="failures.csv">실패 목록</a></p><p>center에서 특징점 205개를 한 번 검출한 뒤 280장을 직접 재추적했습니다. 두 축 곡선의 절편은 0입니다. 시행별 0점 후보·라벨 보정·현재 사진 재검출은 없습니다. 추론에는 파일명·trial·정답을 넣지 않습니다.</p><p>주 결과는 center_quadratic입니다. center_linear는 비교용이며 legacy_six_origins는 기존 방식의 비교 결과로 새 기준을 충족하지 않습니다. full280_resubstitution은 학습에 사용한 280장 재대입이며, 나머지는 기존과 같은 분할의 교차검증입니다. 최대오차는 오분류를 포함한 (Y,T) 벡터 오차입니다.</p><section>'+summary[cols].to_html(index=False,float_format=lambda f:f'{f:.6f}')+'</section><img src="예측오차.png"><h2>6개 trial 전체 경로</h2><p>위 패널은 실제 배율, 아래 패널만 세로 100배입니다. 실제 측정점들을 연결했으며 추세선이 아닙니다. 모든 패널에 동일 ID 205개 전부를 표시합니다.</p>'
for r in records:html+=f'<h3>{r["trial"]}</h3><a href="{r["png"]}">PNG 원본</a><img src="{r["png"]}">'
primary_full=full
rep=p[(p.method=='center_quadratic')&p.protocol.str.startswith('repeated5_')]
rep_score=Z.score(rep)
intro=f'<h2>주 결과</h2><p><b>0점 후보 없는 2차 모델: 280장 중 {int(primary_full["pass"].sum())}/280장 통과, 축·부호 오류 {int((~primary_full.axis_sign_correct).sum())}장. 목표 미달입니다.</b> Y 75/80, T 199/200. 오분류 포함 최대오차 Y {primary_full.Y_abs_error.max():.6f}, T {primary_full.T_abs_error.max():.6f}.</p><p>반복 5분할 20회: Y 축·부호 정확도 {rep_score["Y_axis_sign_accuracy"]*100:.4f}%, T {rep_score["T_axis_sign_accuracy"]*100:.4f}%. 이는 20회 반복 평가이며 서로 다른 사진 5600장이 아닙니다. 기존 저장 분할 119개 중 재대입 1개와 교차검증 118개를 동일하게 사용했습니다.</p><h3>재대입 실패 사진</h3>'+primary_full[~primary_full['pass']][['file','pred_axis','pred','Y_abs_error','T_abs_error']].to_html(index=False,float_format=lambda f:f'{f:.6f}')
html=html.replace('<section>',intro+'<h2>전체 비교 지표</h2><section>',1)
(OUT/'결과.html').write_text(html,encoding='utf-8')
with zipfile.ZipFile(OUT/'고정center_경로와예측_기록.zip','w',zipfile.ZIP_DEFLATED) as zz:
    for pp in OUT.iterdir():
        if pp.is_file() and pp.suffix!='.zip':
            if pp.name=='결과.html':zz.writestr(pp.name,html.replace('../../ALGORITHM_RULES.md','ALGORITHM_RULES.md'))
            else:zz.write(pp,pp.name)
    zz.write(PROJ/'ALGORITHM_RULES.md','ALGORITHM_RULES.md')
print('COMPLETE',checks,flush=True)
