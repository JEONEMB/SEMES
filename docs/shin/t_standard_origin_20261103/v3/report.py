from pathlib import Path
import os,json,zipfile,shutil
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape,A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
from html import escape

P=Path(__file__).resolve().parent;ROOT=P.parents[2]
AUDIT=ROOT/'analysis/frame_link_audit_20261103'
EXP=ROOT/'qrwithbar-lanczos-src/experiments/hye/20261102_173432_t_six_methods'
RAW=Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})

def metric(e):
    e=np.asarray(e,float);e=e[np.isfinite(e)]
    return {'n':len(e),'MAE_ticks':float(np.mean(np.abs(e))),'RMSE_ticks':float(np.sqrt(np.mean(e*e))),
        'max_ticks':float(np.max(np.abs(e))),'pass_1tick':int(np.sum(np.abs(e)<=1)),'pass_percent':float(np.mean(np.abs(e)<=1)*100)}

def main():
    manifest=pd.read_csv(P/'manifest.csv').set_index('path');nom=manifest[manifest.cohort!='reference']
    variants=['original','corrected','fixed_v3','crossfit','prior_half','prior_double','bound_double']
    runs={};records=[];matrix=[]
    for v in variants:
        for family,directory in [('QR+line','results'),('scene extension','scene_results')]:
            q=pd.read_csv(P/v/directory/'predictions.csv').set_index('path');r=manifest.loc[q.index]
            q=q.join(r[['Y','T_original','T','T_standard','session','group','date','Actual_log_conflict']])
            q['abs_ticks']=q.error_ticks.abs();q['variant']=v;q['family']=family
            q.to_csv(P/v/directory/'samples_with_coordinates.csv')
            runs[(v,family)]=q;ok=q.target_available==1
            records.append({'variant':v,'family':family,'target':'own fitted coordinate' if v=='crossfit' else v,**metric(q.loc[ok,'error_ticks']),
                'attempted':len(q),'unidentified_targets':int((~ok).sum()),'inference_failures':int((q.status!=0).sum())})
            for target,col in [('original nominal','T_original'),('corrected nominal','T'),('fixed v3','T_standard')]:
                matrix.append({'variant':v,'family':family,'target':target,**metric((q.T_pred-q[col])*100)})
    legacy=pd.read_csv(EXP/'results/predictions_method7.csv').set_index('path')
    legacy_std=metric((legacy.T_pred-manifest.loc[legacy.index,'T_standard'])*100)
    summary=pd.DataFrame(records);summary.to_csv(P/'metrics.csv',index=False)
    targetmatrix=pd.DataFrame(matrix);targetmatrix.to_csv(P/'three_target_metrics.csv',index=False)
    selected=runs[('fixed_v3','QR+line')]
    before=runs[('original','QR+line')]
    session=[]
    for (v,family),q in runs.items():
        if v not in ['original','corrected','fixed_v3','crossfit']:continue
        for group,g in q.groupby('group'):
            available=g.target_available==1
            if available.any():session.append({'variant':v,'family':family,'group':group,**metric(g.loc[available,'error_ticks'])})
    pd.DataFrame(session).to_csv(P/'by_scan_group.csv',index=False)
    failures=selected[selected.abs_ticks>1].sort_values('abs_ticks',ascending=False);failures.to_csv(P/'failures_fixed_v3.csv')
    # A comparison on the SAME assumed coordinate, alongside nominal-label diagnostics.
    b=metric((before.T_pred-before.T_standard)*100);a=metric(selected.error_ticks)
    comparison={'legacy_method7_on_fixed_v3':legacy_std,'controlled_nominal_model_on_fixed_v3':b,'retrained_v3_on_fixed_v3':a,
        'same_target_RMSE_reduction_percent':(1-a['RMSE_ticks']/b['RMSE_ticks'])*100,
        'same_target_max_reduction_percent':(1-a['max_ticks']/b['max_ticks'])*100,
        'conclusion':'v3 convention can be learned substantially better after retraining; all-image 1tick target remains unmet; physical truth uncertified.'}
    controls=pd.read_csv(P/'global_calibration_predictions.csv')
    for col in ['shift_only','affine']:comparison[col]=metric((controls[col]-controls.T_target)*100)
    (P/'result_summary.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')

    fig,axs=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    for ax,col,title in zip(axs,['MAE_ticks','RMSE_ticks','max_ticks'],['MAE','RMSE','최대 절대오차']):
        vals=[legacy_std[col],b[col],a[col]];ax.bar(range(3),vals,color=['#aab5bc','#688a9e','#24897e'])
        ax.set(xticks=range(3),xticklabels=['기존 method7\n미변환 출력','통제 대조군\n명목 좌표 학습','v3 좌표로\n재학습'],ylabel='tick',title=title+' · 모두 동일 v3 목표 기준')
        for i,z in enumerate(vals):ax.text(i,z+.04,f'{z:.3f}',ha='center')
    fig.savefig(P/'01_same_target.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,4.5),layout='constrained')
    for name,c in [('original','#9daab3'),('fixed_v3','#277e94')]:
        q=runs[(name,'QR+line')];e=(q.T_pred-q.T_standard)*100
        axs[0].scatter(q['T'],e,s=13,alpha=.7,label='명목 학습' if name=='original' else 'v3 학습')
    axs[0].set(xlabel='정정 명목 T (°)',ylabel='가정 v3 좌표에 대한 부호 오차 (tick)',title='동일 좌표에서 비교');axs[0].legend()
    for group,g in selected.groupby('group'):axs[1].scatter(g.Y,g.error_ticks,s=15,label=str(group))
    axs[1].set(xlabel='명목 Y (mm)',ylabel='v3 재학습 오차 (tick)',title='Y 근처 원점 스캔에도 큰 오차가 남음')
    for ax in axs:
        ax.axhline(1,color='#bd7443',ls='--');ax.axhline(-1,color='#bd7443',ls='--');ax.grid(alpha=.15)
    fig.savefig(P/'02_failure_distribution.png',dpi=150);plt.close(fig)
    # Show exactly the 630 measured QR corners, original line samples, and A/B patch roles.
    f=pd.read_csv(ROOT/'analysis/full_scene_origin_20261102/feature_definitions.csv').set_index('feature_id')
    spec=json.loads((AUDIT/'standard_origin_v3.json').read_text())
    ids=[int(s['feature_id']) for s in spec['origin_blocks'] if s['feature_id'] is not None and s['importance']>0]
    parts=pd.read_csv(ROOT/'analysis/full_scene_origin_20261102/feature_partitions.csv')
    b_ids=parts[parts.partition=='B_right'].feature_id.tolist()
    photo=cv2.imdecode(np.fromfile(str(RAW/'center.bmp'),np.uint8),cv2.IMREAD_GRAYSCALE)
    inventory=pd.read_csv(ROOT/'analysis/session_eda_20261102/image_inventory.csv')
    idx=int(inventory[inventory.path=='center.bmp'].iloc[0]['index'])
    fs=cv2.FileStorage(str(ROOT/f'analysis/session_eda_20261102/remeasured/cache/frame{idx}.yml.gz'),cv2.FILE_STORAGE_READ)
    pts=fs.getNode('points').mat();edges=fs.getNode('edges').mat();fs.release()
    fig,ax=plt.subplots(figsize=(16,9),layout='constrained');ax.imshow(photo,cmap='gray',vmin=0,vmax=255)
    ax.scatter(pts[:,0],pts[:,1],s=2,c='#39a9ff',label='기존 추정기: QR 실제 모서리 630점')
    ax.scatter(edges[:,0],edges[:,1],s=10,c='#ffda45',label='기존 추정기: 상·하 선 법선 측정 표본')
    for j,fid in enumerate(ids):
        x,y=f.loc[fid,['reference_x','reference_y']]
        ax.add_patch(Rectangle((x-20,y-20),40,40,fill=False,color='#ff56be',lw=1.3));ax.text(x+23,y-10,f'A{j+1}',color='#ff8ed1',fontsize=8)
    ax.scatter([],[],marker='s',facecolors='none',edgecolors='#ff56be',label=f'확장 추정기: 엄격 품질 검사 A 패치 {len(ids)}개')
    ax.set(xlim=(0,3840),ylim=(2160,0),title='실제 사용 영역 · 네모는 패치 범위이며 중심을 모서리라고 가정하지 않음')
    ax.legend(loc='lower left',fontsize=11,facecolor='#18232d',labelcolor='white');ax.axis('off')
    fig.savefig(P/'03_used_features.png',dpi=180);plt.close(fig)
    # Worst examples retain every test image, including soft-weighted training outliers.
    worst=failures.head(3)
    fig,axs=plt.subplots(3,2,figsize=(15,8.4),layout='constrained',gridspec_kw={'width_ratios':[2,1]})
    for i,(path,row) in enumerate(worst.iterrows()):
        image=cv2.imdecode(np.fromfile(str(RAW/path),np.uint8),cv2.IMREAD_GRAYSCALE)
        axs[i,0].imshow(image,cmap='gray');axs[i,0].set_xlim(1300,2500);axs[i,0].set_ylim(1200,900);axs[i,0].axis('off')
        axs[i,0].set_title(path,fontsize=10,loc='left')
        text=f"스캔 후보 G{int(row.group)} / fold {int(row.fold)}\n원래 T {row.T_original:.5f}° / 정정 T {row['T']:.5f}°\nv3 목표 {row.T_standard:.6f}°\nOOF 예측 {row.T_pred:.6f}°\n오차 {row.error_ticks:+.3f} tick"
        axs[i,1].text(0,.95,text,transform=axs[i,1].transAxes,va='top',fontsize=12,linespacing=1.5);axs[i,1].axis('off')
    fig.savefig(P/'04_failure_photos.png',dpi=160);plt.close(fig)
    sensitivity=pd.read_csv(P/'origin_sensitivity.csv')
    fig,axs=plt.subplots(1,2,figsize=(14,4.5),layout='constrained')
    labels=['v3 기준','사전척도 0.5배','사전척도 2배','경계 2배'];names=['fixed_v3','prior_half','prior_double','bound_double']
    for family,offset,color in [('QR+line',-.16,'#277e94'),('scene extension',.16,'#d7a248')]:
        vals=[float(summary[(summary.variant==v)&(summary.family==family)].iloc[0].max_ticks) for v in names]
        axs[0].bar(np.arange(4)+offset,vals,width=.31,color=color,label=family)
    axs[0].set(xticks=range(4),xticklabels=labels,ylabel='각 가정 좌표에서의 최대 오차 (tick)',title='가정 변경 후 각각 재학습 · 목표 좌표도 달라짐');axs[0].legend()
    axs[1].bar(range(3),sensitivity.target_delta_RMS_ticks,color='#9c80b3');axs[1].set(xticks=range(3),xticklabels=labels[1:],ylabel='v3 기준 대비 목표 T 변화 RMS (tick)',title='추정기가 아니라 목표 좌표 자체의 민감도')
    fig.savefig(P/'05_sensitivity.png',dpi=150);plt.close(fig)

    pdfmetrics.registerFont(TTFont('Malgun','C:/Windows/Fonts/malgun.ttf'))
    pdfmetrics.registerFont(TTFont('MalgunBold','C:/Windows/Fonts/malgunbd.ttf'))
    body=ParagraphStyle('body',fontName='Malgun',fontSize=10,leading=15,spaceAfter=6)
    title=ParagraphStyle('title',parent=body,fontName='MalgunBold',fontSize=20,leading=26,spaceAfter=12,textColor=colors.HexColor('#143c50'))
    small=ParagraphStyle('small',parent=body,fontSize=8.5,leading=12)
    story=[]
    def para(text,style=body):story.append(Paragraph(text,style))
    def heading(text):para(text,title)
    def table(rows,widths=None):
        cells=[[Paragraph(escape(str(v)),small) for v in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2edf1')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#638492')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d5dfe4')),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,5*mm))
    def image(name,w=255,h=None):
        from PIL import Image as PIL
        iw,ih=PIL.open(P/name).size
        story.append(Image(str(P/name),width=w*mm,height=(h if h else w*ih/iw)*mm))
    def page():story.append(PageBreak())
    def metricrow(label,m):return [label,str(m['n']),f"{m['MAE_ticks']:.3f}",f"{m['RMSE_ticks']:.3f}",f"{m['max_ticks']:.3f}",f"{m['pass_1tick']}/{m['n']} ({m['pass_percent']:.1f}%)"]
    heading('표준 원점 v3 반영 및 T 추정 평가')
    para('2026-11-03 · 1 tick = 0.01° · 사진만으로 추론 · 개발 자료 575장 보존')
    para('<b>가정한 v3 좌표를 학습하도록 모델을 수정했고, 같은 v3 목표에서 오차가 줄었습니다. 모든 사진 1 tick 이내라는 목표는 달성하지 못했습니다.</b>')
    table([['동일 v3 목표에 대한 평가','n','MAE','RMSE','최대 |오차|','1 tick 이내'],
        metricrow('기존 method7 출력 그대로',legacy_std),metricrow('통제 대조군: 명목 좌표 학습',b),metricrow('전역 원점 상수만 재보정',comparison['shift_only']),metricrow('전역 절편·기울기 재보정',comparison['affine']),metricrow('v3 좌표로 재학습',a)], [90*mm,15*mm,28*mm,28*mm,30*mm,63*mm])
    para(f"통제 대조군과 같은 목표로 비교할 때 RMSE {b['RMSE_ticks']:.3f} → {a['RMSE_ticks']:.3f} tick ({comparison['same_target_RMSE_reduction_percent']:.1f}% 감소), 최대오차 {b['max_ticks']:.3f} → {a['max_ticks']:.3f} tick입니다. 기존 method7은 이전 분할의 저장 OOF이며, 통제 대조군은 이번 분할로 다시 계산했습니다.")
    para('이 점수는 통계적 표준 좌표를 정답으로 가정했을 때의 적합 성능입니다. 원래 명목 라벨에서 얻었던 최대 2.640 tick과 직접 비교해 개선율을 계산하지 않습니다. 물리적 T 정확도를 인증하는 외부 참값은 없습니다.')
    image('01_same_target.png',235)
    page();heading('무엇을 수정했는가')
    para('QR·외부 수평선의 기존 104개 측정값은 그대로 재사용했습니다. 원점 생성에서 구한 스캔별 Y/T 보정량을 학습 좌표에 반영하고, 공분산의 반복 그룹은 정정 명목 자세로 유지했습니다. 반복 편차 네 장은 학습에서만 부드러운 가중치를 주며 시험 점수에서는 동일하게 한 장으로 계산했습니다.')
    table([['항목','이번 처리'],['기존 모델 계열','회귀·잡음 보정·기하·RBF·최악 그룹 회귀의 5개 구성원을 유지. 이전 배포 모델의 하이퍼파라미터를 고정해 좌표 변경 효과를 통제.'],['결합 비율','외부 학습 집합 안의 내부 보류 예측으로 선택. 최대오차 + 0.1 × RMSE를 최소화. 외부 시험 정답은 사용하지 않음.'],['v3 원점','네 기준 사진의 평균 잠재 좌표 = 0이라는 규약. 움직임 스캔과 원점 복귀 노출을 분리한 17개 그룹.'],['라벨 감사','.33 파일 한 장만 파생 T=.32로 정정. .57 세 장은 포함하며 Actual 로그 .56과의 충돌을 유지.'],['추론 입력','사진 한 장. 정답 Y/T, 세션 ID, 시각, 파일명에서 해석한 라벨을 추론 입력에 넣지 않음.'],['추가 개선 시도','원점 생성 A 자연 패치도 입력하는 9개 ridge/RBF 후보의 결합을 추가. 품질 불합격 관측은 학습 평균으로 대체하고 특징 선택·정규화·중심은 학습에서만 적합.']], [38*mm,216*mm])
    para('사진의 원시 특징을 하나의 공통 상수만큼 평행 이동하는 것은 학습 평균으로 정규화하는 모델의 성능을 본질적으로 개선하지 않습니다. 출력에 전역 상수만 더하거나 전역 절편·기울기를 다시 맞추는 대조 실험도 외부 학습 집합에서만 보정했습니다. 이번 수정의 핵심은 스캔마다 다른 잠재 좌표와 반복 관측의 학습 가중치를 반영한 재학습입니다.')
    para('추가 특징 모델은 v3 결과를 본 뒤 확장한 탐색입니다. 이 자료는 이미 모델·원점 선택에 여러 번 사용된 개발 자료이며 신규 장비/세션 일반화 검증으로 부르지 않습니다.')
    page();heading('평가 분할과 원점 재적합 검증')
    table([['구분','학습·평가 원칙','무엇을 뜻하는가'],['고정 v3 조건부 CV','전체 자료로 만들어진 v3 좌표를 고정. 정정 명목 (Y,T) 자세를 FNV-1a 5-fold로 통째 보류. 모델 변환·학습·결합 선택은 각 학습 집합에서 수행.','주어진 표준 좌표를 얼마나 잘 학습하는지. 원점 생성까지 독립적인 검증은 아님.'],['원점까지 재적합','외부 fold마다 원점을 다시 적합하고, 내부 결합 검증에서도 외부·내부 두 fold를 모두 제외한 원점으로 다시 적합. 총 25개 원점 적합. 네 원점 앵커는 보정에 유지.','시험 영상/라벨을 그 fold의 원점 적합에서 제외. 기존 개발 선택과 네 앵커를 조건으로 한 재검증.'],['식별되지 않은 표준 목표','G11의 Y=.1, T=.01 반복 3장은 같은 자세 묶음이 전부 보류되면 그 그룹의 학습 관측이 없음.','임의의 0 보정량을 정답으로 점수화하지 않음. 571장 모두 예측하되 표준 좌표 오차는 식별 가능한 568장에만 계산.'],['분모','575장 = 일반 자료 571장 + 기준 사진 4장. 이전 전방 원점 감사의 555장과 분할·분모가 다름.','추론 실패 0과 1 tick 정확도 실패는 별개. 이번 평가에서 시험 사진을 삭제하지 않음.']], [40*mm,110*mm,104*mm])
    qrref=summary[(summary.variant=='crossfit')&(summary.family=='QR+line')].iloc[0].to_dict()
    scref=summary[(summary.variant=='crossfit')&(summary.family=='scene extension')].iloc[0].to_dict()
    table([['원점도 재적합한 표준 좌표 일관성','n','MAE','RMSE','최대 |오차|','1 tick 이내'],metricrow('QR+선 결합',qrref),metricrow('A 자연 특징 확장',scref)], [90*mm,15*mm,28*mm,28*mm,30*mm,63*mm])
    para('보류 좌표 자체도 학습에서 추정한 바이어스를 명목 라벨에 더한 값입니다. 따라서 원점까지 재적합해도 독립적인 물리적 T 정답은 생기지 않습니다. 관측된 전방 영상 잔차의 감소를 T 정확도 향상으로 바꾸어 주장하지 않습니다.')
    page();heading('원래 라벨·정정 라벨·v3 목표를 구분')
    labels={'original':'원래 명목 학습','corrected':'정정 명목 학습','fixed_v3':'v3 학습','crossfit':'원점도 재적합'}
    for target in ['original nominal','corrected nominal','fixed v3']:
        text={'original nominal':'원래 파일명 유래 T에 대한 점수','corrected nominal':'감사 후 정정 명목 T에 대한 점수','fixed v3':'고정 v3 표준 T에 대한 점수'}[target]
        para('<b>'+text+'</b>')
        rows=[['모델','n','MAE','RMSE','최대 |오차|','1 tick 이내']]
        for v in ['original','corrected','fixed_v3']:
            m=targetmatrix[(targetmatrix.variant==v)&(targetmatrix.family=='QR+line')&(targetmatrix.target==target)].iloc[0].to_dict();rows.append(metricrow(labels[v],m))
        table(rows,[90*mm,15*mm,28*mm,28*mm,30*mm,63*mm])
    para('각 표 안에서는 목표가 동일합니다. 표 사이의 점수를 섞어 개선율을 계산하면 안 됩니다. v3 모델이 명목 라벨에서 크게 벗어나는 것은 의도한 좌표 변경을 포함합니다. 세 평가 모두 같은 571장 전체에 대한 결과입니다.')
    page();heading('실제 사진에서 사용한 특징')
    image('03_used_features.png',250,140.6)
    para('파랑: 기존 QR 630개 코너(212/210/208). 노랑: 기존 상·하 선의 법선 관측. 분홍 A: 확장 모델에 추가한 나사·지그 등의 패치 22개. 긴 직선상의 임의 점을 2차원 정확한 점으로 취급하지 않습니다. v3 원점 생성은 하부 선 품질 미달로 제외했지만, 기존 추정기 대조를 위해 기존 선 특징 처리는 보존했습니다.',small)
    page();heading('남은 큰 오차와 실패 사진')
    image('04_failure_photos.png',250,140)
    para(f"v3 고정 목표에서 571장 중 {len(failures)}장이 1 tick을 초과했습니다. 그림은 시험 fold에서 보류된 사진이며 전체 데이터로 학습한 배포 모델의 재대입 예측과 다릅니다. 특히 Y 단독 자료의 원점 부근에서도 큰 오차가 남았습니다. 사진 몇 장만으로 원인을 카메라 이동·기구 오차·검출 잡음 중 하나로 확정할 수는 없습니다.",small)
    page();heading('특징 확장과 잔차 분포')
    sf=summary[(summary.variant=='fixed_v3')&(summary.family=='scene extension')].iloc[0].to_dict()
    table([['고정 v3 목표','n','MAE','RMSE','최대 |오차|','1 tick 이내'],metricrow('QR+선 결합',a),metricrow('A 자연 특징 확장',sf)], [90*mm,15*mm,28*mm,28*mm,30*mm,63*mm])
    para('자연 특징을 실제 추론 입력에 추가해도 이번 결합 방식에서는 성능이 좋아지지 않았습니다. 단순히 원점 생성 특징을 더 넣는 것만으로 해결되지 않음을 확인했습니다. 기존 구성원 중 RBF 단독은 고정 v3 목표 최대 4.334 tick으로 결합보다 작았지만, 외부 점수를 본 사후 관찰이므로 이를 새 독립 성적으로 채택하지 않습니다.')
    image('02_failure_distribution.png',242)
    para('전수 오차와 그룹별 집계는 samples_with_coordinates.csv, by_scan_group.csv, failures_fixed_v3.csv에 저장했습니다. G 번호는 촬영 로그로 확정된 물리 세션이 아니라 시간·방향·원점 노출에 따른 분석 그룹입니다.',small)
    page();heading('원점 경계와 사전척도 민감도')
    image('05_sensitivity.png',245)
    rows=[['원점 가정','Y 사전척도','T 사전척도','경계','경계 도달','표준 T 변화 RMS / 최대 (tick)']]
    for r in sensitivity.itertuples():rows.append([r.variant,f'{r.sigma_Y:.3f} mm',f'{r.sigma_T:.3f}°',r.bound,r.bound_hits,f'{r.target_delta_RMS_ticks:.3f} / {r.target_delta_max_ticks:.3f}'])
    table(rows,[45*mm,35*mm,35*mm,22*mm,27*mm,90*mm])
    para('v3 기준 자체의 원시 보정 변수 경계 도달은 12개입니다. 사전척도를 절반·두 배로 하거나 원시 경계를 3에서 6으로 넓힌 뒤 원점과 모델을 각각 다시 적합했습니다. 이 결과는 좌표 가정과 학습 난이도가 함께 바뀐 것이며 서로 다른 물리 정답의 정확도 비교가 아닙니다. 가정한 T 좌표 자체가 1 tick 이상 달라지는 민감도가 남습니다.')
    page();heading('수치·원본 영상 실행 검증')
    checks=json.loads((P/'verification.json').read_text())
    table([['검사','결과'],['원본 수와 분모','575장, 원점 앵커 4장, 일반 평가 571장. 14개 학습 실행의 경로·중복·fold·오차 산식 검사 통과.'],['원점 적합 격리','외부/내부 25개 원점 적합에서 보류 non-reference row가 학습 row 목록에 없음을 검사. 모든 원점 앵커는 보정에 포함.'],['정정 자세 묶음','감사 후 같은 (Y,T) 반복은 한 fold에만 존재. .32 파일 오기 한 장도 올바른 반복 묶음에 포함.'],['저장·재로딩','C++ 각 모델의 전체 575장 예측 차이 1e-10° 미만. 확장 모델 JSON 재로딩 최대 차이 4.44e-16°.'],['원본 사진 직접 추론','T=.50, Y=.10/T=0, Y=1/T=2의 세 원본에서 두 추정기 실행. QR 104특징과 A 패치 위치·유효성 모두 캐시와 일치, 최종 배포 예측과 일치.'],['원본 보존','시험 원본 3장 해시 전후 일치. v3 원점 JSON·좌표 CSV·원본 관측 CSV 해시 일치. sources/ 및 원본 영상은 편집하지 않음.'],['v2 기록 보존','기존 v2 571장 결과와 최대 5.156 tick 기록을 그대로 유지. 이 값은 v2 가정 목표이며 v3와 같은 정답 기준이 아님.']], [44*mm,210*mm])
    para('이 검사는 구현 일관성·분할 격리·사진 입력 실행을 확인합니다. 기존 데이터의 적응적 탐색, 원점 규약·앵커의 불확실성, 미촬영 영역 및 새로운 세션의 물리적 T 정확도까지 검증하지는 않습니다.')
    page();heading('산출물과 재현')
    para('<b>기존 권장 모델은 교체하지 않았습니다.</b> 새 v3 모델과 특징 확장 모델은 가정 좌표를 출력하는 개발 후보로 따로 저장했습니다. 출력에는 standard_origin_v3_assumed와 physical_accuracy_certified=false를 명시합니다.')
    table([['파일/폴더','내용'],['deployment/qr_model.json + runtime/v3_estimate.exe','QR·선 결합의 사진 입력 실행 모델. 표준 좌표 출력.'],['deployment/scene_model.json + scene_estimator.py','A 자연 패치까지 사용하는 확장 추정기. Python NumPy/SciPy/Pandas/OpenCV 필요.'],['deployment/qr_assets, reference_scene.png, runtime','고정 QR 기준·자연 패치 기준 사진·네이티브 측정 DLL. 명목 정답 CSV나 영상 파일명 라벨 없이 추론.'],['metrics.csv / three_target_metrics.csv','자체 목표 및 원래/정정/v3 세 목표의 MAE·RMSE·최대오차·1 tick 비율.'],['crossfit/outer*/origin_fit.json','각 원점의 학습 row ID·보정량·경계·수렴 로그. 가정 표준 좌표가 식별되지 않는 시험 3장 기록.'],['prepare_v3.py / train.cpp / scene_estimator.py','원점 25회 재적합, C++ 기존 계열 재학습, 자연 특징 확장 학습·추론 소스.'],['verification.json / reproduce.ps1','검증 결과와 전체 재현 순서. 전체 소스·결과·배포 자산은 ZIP에 포함.']], [85*mm,169*mm])
    para('사진만 입력하는 C++ 실행 예: deployment/runtime/v3_estimate.exe IMAGE.bmp deployment/qr_assets deployment/qr_model.json',small)
    para('확장 모델 실행 예: python deployment/scene_estimator.py --image IMAGE.bmp --model deployment/scene_model.json',small)
    para('표준 원점을 맞다고 가정하면 재학습으로 그 좌표의 예측 오차는 줄었습니다. 하지만 최대 1 tick 성능은 확인하지 못했으며, 물리 참값 미확정과 원점 가정 민감도를 함께 남겼습니다.')
    output=ROOT/'output/pdf/T_axis_standard_origin_v3_model_evaluation_20261103.pdf';output.parent.mkdir(exist_ok=True,parents=True)
    def footer(c,doc):
        c.setFont('Malgun',8);c.setFillColor(colors.HexColor('#657b86'));c.drawString(14*mm,9*mm,'T-axis · assumed standard origin v3 · development evaluation');c.drawRightString(283*mm,9*mm,str(doc.page))
    doc=SimpleDocTemplate(str(output),pagesize=landscape(A4),rightMargin=14*mm,leftMargin=14*mm,topMargin=12*mm,bottomMargin=16*mm)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(str(output));assert len(reader.pages)==10,len(reader.pages)
    text='\n'.join(p.extract_text() for p in reader.pages)
    assert '4.792' in text and '568' in text and '표준' in text
    (P/'pdf_validation.json').write_text(json.dumps({'pages':len(reader.pages),'text_checks':'PASS','render_inspection':'pending'},indent=2))
    write_readme(P,comparison)
    print('PDF',output,'pages',len(reader.pages));print(summary[['variant','family','n','RMSE_ticks','max_ticks','pass_1tick']].to_string(index=False))

def write_readme(p,comparison):
    (p/'README.md').write_text('''# v3 표준 좌표 기반 T 추정 모델 평가

575장 유지, 기준 4장 제외 일반 예측 571장. 가정 v3 좌표로 재학습했고 원점까지 외부/내부 fold마다 재적합한 개발 검증도 수행했다. 원래/정정 명목 라벨과 표준 좌표에 대한 점수는 `three_target_metrics.csv`에서 분리한다.

동일 고정 v3 목표에서 통제 명목 모델 RMSE 2.713 / 최대 5.309 tick → v3 재학습 RMSE 1.004 / 최대 4.792 tick, 451/571 통과. 원점까지 재적합하면 표준 좌표가 식별되는 568장 RMSE 1.005 / 최대 4.802 tick, 450/568 통과. 미식별 3장도 추론은 수행했으며 명목 라벨 기준 평가는 전체 571장이다. 나사·지그 A 패치 확장은 개선되지 않았다.

## 평가 조건

- 기존 구성원 배포 하이퍼파라미터를 고정하고 학습 좌표·가중치를 변경했다. 이번 결합 가중치는 학습 내부에서 선택했다. 이전의 모든 하이퍼파라미터 검색을 반복한 실험과 구분한다.
- 정정 명목 자세를 한 fold에 묶어 .32 오기 이미지도 반복과 함께 보류한다. 예전 저장 OOF와 한 장의 fold가 달라질 수 있다.
- 전체자료 v3 좌표를 고정한 결과는 조건부 적합 평가다. 원점 재적합도 네 앵커·기존 감사/모델 선택을 조건으로 한 개발 검증이다. 물리적 정답이나 신규 세션의 1 tick 성능을 인증하지 않는다.
- `crossfit`는 25개 적합(외부5 + 내부20). 시험 구간이 학습에서 사라져 바이어스를 식별할 수 없는 경우 `target_available=0`으로 기록한다.
- 기존 v2 실험, 원본 영상, 이전 모델, sources/를 보존했다. 개선 모델이 기존 권장 모델을 대체하지 않는다.

## 실행

프로젝트 루트에서 `analysis/t_standard_origin_20261103/v3/reproduce.ps1`로 원점·빌드·학습·평가·PDF를 재현한다. 이미 만들어진 배포 폴더만으로 사진 추론:

```powershell
deployment/runtime/v3_estimate.exe IMAGE.bmp deployment/qr_assets deployment/qr_model.json
python deployment/scene_estimator.py --image IMAGE.bmp --model deployment/scene_model.json
```

C++ 실행기는 Windows x64/MSVC 런타임을 사용한다. 확장 모델 Python은 numpy, scipy, pandas, opencv-python이 필요하다. 동봉 DLL은 MSVC v141/OpenCV3.4.15로 빌드됐다. 추론에는 명목 라벨·세션 ID·시각·개발 CSV가 필요 없다. 기준 사진은 동봉한 고정 reference_scene.png와 qr_assets를 쓴다.

`verification.json`: 14개 실행의 571장 예측 일관성, 25개 원점 학습/시험 격리, 저장·재로딩, 원본 사진 3장 직접 추론·측정 캐시 동일성, 입력 해시 보존 검증 결과.
''',encoding='utf-8')
    shutil.copy2(p/'README.md',p/'deployment/README.md')

if __name__=='__main__':main()
