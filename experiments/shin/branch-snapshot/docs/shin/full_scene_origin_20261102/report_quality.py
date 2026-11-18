from model_scene import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape,A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
from html import escape
import zipfile
import cv2

plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
d,features,array=load();j=json.loads((P/'standard_origin_v2.json').read_text())
summary=pd.read_csv(P/'strict_cv_summary.csv').set_index('config')
pred=pd.read_csv(P/'strict_cv_predictions.csv');post=pd.read_csv(P/'heldout_reference_checks.csv')
blocked=pd.read_csv(P/'contiguous_cv_predictions.csv')
quality=pd.read_csv(P/'directional_feature_quality.csv')
active=pd.read_csv(P/'pilot_feature_quality.csv');eligible=d[~d.suspect].copy()
A=[b['feature_id'] for b in j['origin_blocks'] if b['feature_id'] is not None]
bs,_=blocks(eligible,features,array,'B_right',True);B=[b['fid'] for b in bs if b['fid'] is not None]
ref_overlap=[(a,b) for a in A for b in B if abs(features.iloc[a].reference_x-features.iloc[b].reference_x)<=40 and abs(features.iloc[a].reference_y-features.iloc[b].reference_y)<=40]
v=array['v'];delta=v[:,A,None,:2]-v[:,None,B,:2]
overlap=(np.abs(delta[:,:,:,0])<=40)&(np.abs(delta[:,:,:,1])<=40)&array['valid'][:,A,None]&array['valid'][:,None,B]
assert not ref_overlap and overlap.sum()==0

refpath=Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set\center.bmp')
ref=cv2.imdecode(np.fromfile(str(refpath),np.uint8),cv2.IMREAD_GRAYSCALE)
fig,ax=plt.subplots(figsize=(16,9),layout='constrained');ax.imshow(ref,cmap='gray',vmin=0,vmax=255)
others=features[~features.feature_id.isin(A+B)]
ax.scatter(others.reference_x,others.reference_y,s=9,c='#aaaaaa',alpha=.7,label='최종 구성/검증에 채택하지 않은 후보')
for ids,color,label in [(A,'#f4b128','원점 구성: 왼쪽 특징 A'),(B,'#17bc83','보류 검증: 오른쪽 특징 B')]:
    q=features.iloc[ids];ax.scatter(q.reference_x,q.reference_y,s=48,facecolors='none',edgecolors=color,linewidths=1.5,label=label)
ax.axvline(2000,color='#69cddd',ls='--',lw=.8)
ax.set(title='후보 표시와 실제 채택을 구분: 구성 A와 검증 B는 패치도 겹치지 않음',xlim=(0,3840),ylim=(2160,0));ax.legend(loc='lower left',fontsize=9)
fig.savefig(P/'02_accepted_and_validation_map.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
q=quality[quality.feature_id.isin(active.loc[active.active,'feature_id'])]
axs[0].scatter(q.median_condition,q.strict_valid_fraction,s=25);axs[0].axvline(50,color='#c55',ls='--');axs[0].set(xscale='log',xlabel='위치 정보 행렬의 조건수 (작을수록 양방향 안정)',ylabel='엄격한 정합 유효 비율',title='정·역방향 일치에 방향별 불확실성 검사를 추가')
axs[0].set_xticks([1,10,50,100,1000],['1','10','50','100','1000']);axs[0].minorticks_off()
for fid in [0,1,6,12]:
    z=quality[quality.feature_id==fid].iloc[0]
    axs[1].bar(str(fid),z.strict_valid_fraction)
axs[1].set(ylim=(0,1.05),xlabel='수동 나사·구멍 후보 번호',ylabel='엄격한 정합 유효 비율',title='나사·구멍이라는 이유만으로 채택하지 않음')
fig.savefig(P/'03_directional_quality.png',dpi=160);plt.close(fig)

names=['qr_line_bias','scene_no_bias','scene_bias','scene_bias_gain'];labels=['QR·상부선\n일정 바이어스','외부 특징 추가\n바이어스 없음','외부 특징 추가\n일정 바이어스','외부 특징 추가\n바이어스+배율']
fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
for ax,col,title in [(axs[0],'QR_RMSE_px','QR 보류 오차'),(axs[1],'right_RMSE_px','원점 생성에 쓰지 않은 오른쪽 특징의 보류 오차')]:
    ax.bar(np.arange(4),summary.loc[names,col],color=['#8999a6','#a9bfcb','#277e94','#68a1a5']);ax.set(xticks=range(4),xticklabels=labels,ylabel='좌표 성분 RMSE (px)',title=title)
fig.savefig(P/'04_independent_validation.png',dpi=160);plt.close(fig)

cv95=pred[pred.config=='scene_bias'].right_feature_component_RMS_px.quantile(.95)
fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for name,color in [('qr_line_bias','#8999a6'),('scene_bias','#277e94'),('scene_bias_gain','#68a1a5')]:
    q=blocked[blocked.config==name].groupby('fold').right_feature_component_RMS_px.median();axs[0].plot(q.index,q.values,'o-',label=name)
axs[0].set(xlabel='연속 자세 블록 보류 fold',ylabel='오른쪽 특징 좌표 성분 RMS 중앙값 (px)',title='인접 자세를 묶어서 보류한 검사');axs[0].legend(fontsize=8)
axs[1].bar(['center1','center2','center3'],post.right_component_RMS_px,color=['#5c99ac','#5c99ac','#d35e48'])
axs[1].axhline(cv95,ls='--',color='#777',label=f'일반 자세 보류 오차 95백분위 ({cv95:.3f}px)')
axs[1].set(ylabel='보류 오른쪽 특징 좌표 성분 RMS (px)',title='기준 사진을 빼고 원점을 다시 만든 후 예측');axs[1].legend(fontsize=8)
fig.savefig(P/'05_post_origin_checks.png',dpi=160);plt.close(fig)

# Pairwise uncertainty of the comparison, resampling complete pose groups, not individual points.
wide=pred.pivot(index=['pose','path'],columns='config',values='right_feature_component_RMS_px').reset_index()
groups=wide.groupby('pose')[['qr_line_bias','scene_bias']].mean();rng=np.random.default_rng(507)
gains=[]
for _ in range(2000):
    g=groups.iloc[rng.integers(0,len(groups),len(groups))]
    gains.append(float((g.qr_line_bias-g.scene_bias).mean()))
gain=100*(1-summary.loc['scene_bias','right_RMSE_px']/summary.loc['qr_line_bias','right_RMSE_px'])
qr_change=100*(summary.loc['scene_bias','QR_RMSE_px']/summary.loc['qr_line_bias','QR_RMSE_px']-1)
ci=np.quantile(gains,[.025,.975])
status={'candidate_features':len(features),'pilot_active_features':int(active.active.sum()),'features_strict_valid_at_least_90pct':int((quality.strict_valid_fraction>=.9).sum()),
    'origin_A_features':len(A),'validation_B_features':len(B),'A_B_reference_patch_overlaps':len(ref_overlap),'A_B_target_valid_patch_overlaps':int(overlap.sum()),
    'CV_images':int(summary.loc['scene_bias','n_test']),'right_RMSE_reduction_vs_quality_baseline_percent':gain,
    'QR_RMSE_increase_vs_quality_baseline_percent':qr_change,
    'paired_pose_bootstrap_mean_RMS_difference_interval_px':ci.tolist(),
    'zero_reference_holdout_right_RMS_px':post.right_component_RMS_px.tolist(),'ordinary_pose_right_RMS_95pct_px':cv95,
    'conclusion':'Additional features improve independent-field consistency, but zero-reference holdout fails for center3. Common origin remains unvalidated; v1 improvement must not be interpreted as physical-origin accuracy.'}
save('quality_validation_summary.json',status)
j['status']='validation_not_sufficient_for_confirmed_common_origin'
j['post_validation_status']='center3 held-out zero-reference residual exceeds ordinary-pose empirical 95th percentile'
j['quality_validation_summary']=status;save('standard_origin_v2.json',j)

pdfmetrics.registerFont(TTFont('Malgun',r'C:\Windows\Fonts\malgun.ttf'));pdfmetrics.registerFont(TTFont('MalgunBold',r'C:\Windows\Fonts\malgunbd.ttf'))
pdfmetrics.registerFontFamily('Malgun',normal='Malgun',bold='MalgunBold',italic='Malgun',boldItalic='MalgunBold')
body=ParagraphStyle('body',fontName='Malgun',fontSize=11,leading=18,spaceAfter=10,wordWrap='CJK',textColor=colors.HexColor('#273b4d'))
head=ParagraphStyle('head',parent=body,fontName='MalgunBold',fontSize=20,leading=28,spaceAfter=15)
small=ParagraphStyle('small',parent=body,fontSize=9,leading=14)
cell=ParagraphStyle('cell',parent=body,fontSize=9,leading=13,spaceAfter=0)
story=[]
def p(t,style=body):story.append(Paragraph(t,style))
def page(title):
    if story:story.append(PageBreak())
    p(title,head)
def tab(rows,widths):
    data=[[Paragraph(escape(str(x)),cell) for x in row] for row in rows]
    t=Table(data,colWidths=[x*mm for x in widths],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2edf4')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#d7e1ea'))]))
    story.append(t);story.append(Spacer(1,3*mm))
def fig(name,title,caption):
    page(title);w,h=PILImage.open(P/name).size;scale=min(261*mm/w,148*mm/h)
    story.append(Image(str(P/name),width=w*scale,height=h*scale));story.append(Spacer(1,2*mm));p(caption,small)

page('특징점 신뢰도를 전 단계에 반영한 공통 원점 재검증')
p('<b>사용자의 지적을 반영해 특징 측정, 일정 바이어스 가설 검증, 원점 생성 및 생성 후 검증을 모두 수정했습니다.</b> 직선 위 점을 정밀한 2차원 위치로 취급하지 않았고, 원점 생성에 쓴 특징과 검증 특징도 분리했습니다.')
p(f'원본 575장에 대해 QR 밖 나사·구멍·지그·판 외곽·배경의 154개 후보를 조사했습니다. 초기 정합 검사 후 75개를 전수 추적했고, 방향별 불확실성까지 포함한 엄격한 검사를 전체 사진의 90% 이상에서 통과한 후보는 {status["features_strict_valid_at_least_90pct"]}개였습니다.')
p(f'<b>결과는 단순한 전면 개선이 아닙니다.</b> 외부 특징을 추가한 일정 바이어스 모델은 생성에 쓰지 않은 오른쪽 특징의 보류 RMSE를 약 {gain:.1f}% 줄였지만, QR 자체의 보류 RMSE는 약 {qr_change:.1f}% 커졌습니다.')
p('<b>핵심 사후 검증:</b> 원점 기준 사진을 한 장씩 빼고 다시 원점을 생성한 뒤 예측했을 때 center3에서 오른쪽 보류 특징의 RMS가 0.668px였습니다. 일반 자세 보류 오차의 95백분위 0.342px보다 큽니다. 따라서 현재 공통 원점은 확정된 결과로 채택할 수 없습니다.')
p('v2 후보 모델과 좌표는 보존하되 검증 미충족 상태를 명시했습니다. 이전 v1의 35.7% 개선은 영상 특징 적합 지표의 개선이었으며, 물리적 원점 또는 T 참값 정확도의 검증으로 해석하면 안 됩니다.')
p('2026-11-02 · 전수 원본은 읽기만 했으며 파일명·라벨·기존 모델은 변경하지 않았습니다.',small)

page('1. 꼭짓점이 아닌 후보는 어떤 기준으로 걸렀는가?')
p('점 표시는 기하학적 꼭짓점과 영상 패치의 기준 위치를 구분해야 합니다. 패치 중심이 꼭짓점일 필요는 없지만, 가로·세로 이동이 모두 구분될 정도의 무늬가 있어야 합니다. 긴 직선 하나만 있는 패치는 선에 평행한 위치가 결정되지 않으므로 2차원 점으로 쓰지 않습니다.')
p('최종 좌표는 밝기 gain·offset을 포함한 Lanczos-3 패치 정합으로 추출했습니다. 해석적 영상 미분으로 위치 정보 행렬을 계산하고, gain·offset에 해당하는 자유도를 Schur complement로 제거한 뒤 두 방향 위치의 조건부 공분산을 얻었습니다.')
p('개별 측정의 채택 기준은 NCC&gt;0.8, 정·역방향 오차&lt;0.1px, 정규화 밝기 RMSE&lt;0.2, 최적화 이동 경계 비접촉, 정보 행렬 조건수&lt;50, 위치 불확실성의 긴 축 표준편차&lt;0.1px입니다. 평평한 패치와 직선만 있는 패치를 거르는 수치 시험도 통과했습니다.')
p('이 불확실성은 패치·잡음 모델을 가정한 값입니다. 실제 광학·반사·형태 변화까지 포함한 절대오차 보증은 아닙니다. 모델 가중치에는 같은 자세 반복의 실측 산포를 추가했고, 같은 부위에 점이 많다는 이유로 과도한 신뢰를 주지 않도록 부위별 총 가중치를 제한했습니다.')
p('원형 나사·구멍 후보도 자동으로 믿지 않았습니다. 예를 들어 후보 0의 엄격한 유효 비율은 99.3%지만, 후보 1은 36.3%, 후보 12는 22.4%입니다. 반사·가림·대응 실패가 있으면 생김새가 분명해도 탈락합니다.')
fig('03_directional_quality.png','정합 품질과 방향별 불확실성을 함께 검사','조건부 표준편차는 실제 위치 오차의 보증이 아닙니다. 해당 점의 관측 가능성·반복 일관성을 판단하는 기준으로 사용했습니다. 직선만 있는 후보를 정밀한 2차원 점으로 승격하지 않았습니다.')
fig('02_accepted_and_validation_map.png','실제 채택한 점과 검증용 점을 구분','회색은 최초 후보 중 최종 원점 구성·검증에 채택하지 않은 점입니다. 노란색 A는 원점 생성, 초록색 B는 독립 특징 검증에 사용했습니다. 기준 영상과 유효 측정 영상 모두에서 A·B 패치가 겹친 경우는 0건입니다.')

page('2. 가설 검증에서도 같은 측정 신뢰도를 적용했다')
p('초기 14장 파일럿은 특징 후보의 측정 품질을 평가하는 보정 자료로만 사용했습니다. 이 사진들과 같은 자세의 반복은 모두 학습에 남기고 시험에서 제외했습니다. 라벨 이상 4장은 이전과 같이 적합·평가에서 제외했습니다. 보정 자료 20장을 제외한 551장을 보류 평가했습니다.')
p('각 fold마다 학습 영상에서만 특징별 유효 비율·공분산을 계산했습니다. 자연 특징은 학습 내 유효 비율 85% 이상이고 유효 원점 기준이 2장 이상일 때 사용했으며, 각 사진에서는 위의 엄격한 개별 정합 검사를 다시 적용했습니다. 따라서 90% 이상 통과 후보 수 37개와 최종 85% 훈련 유효율 기준의 특징 수는 다릅니다.')
p('기존 수평선도 별도로 확인했습니다. 위치·기울기를 함께 설명할 만큼 유효 표본 수와 x 범위가 확보된 상부 선만 사용했고, 이 기준을 만족하지 못한 하부 선은 이번 모델에서 제외했습니다. 하부 선 자체가 영원히 쓸 수 없다는 뜻은 아니며, 현재 계측 결과로 두 계수를 신뢰하기 어렵다는 뜻입니다.')
p('가설은 일정 스캔 바이어스 모델과, 긴 T 스캔에 작은 배율 차이까지 허용한 모델을 비교했습니다. 같은 자세의 반복은 항상 함께 보류했고, 추가로 연속한 자세 구간 전체를 보류했습니다. 선형 스캔의 높은 R²만으로 누적 오차가 없다고 결론내리지 않았습니다.')
p('오른쪽 검증 특징 B는 원점·구간 바이어스 최적화에 넣지 않았습니다. 학습 자세에서 별도의 B 반응식을 적합한 뒤, 보류 사진에서 B 위치를 예측했습니다. 영상 정합 실패는 유효 관측 수로 기록했고 이번 551장 모두에서 최소 3개 이상의 B 특징을 검증할 수 있었습니다.')

page('3. 보류 영상·보류 특징에서 얻은 결과')
rows=[['방법','QR RMSE(px)','오른쪽 보류 RMSE(px)','오른쪽 중앙 RMS(px)']]
display=['QR+상부선·일정 바이어스','외부 특징·바이어스 없음','외부 특징·일정 바이어스','외부 특징·바이어스+배율']
for name,label in zip(names,display):
    r=summary.loc[name];rows.append([label,f'{r.QR_RMSE_px:.4f}',f'{r.right_RMSE_px:.4f}',f'{r.right_median_px:.4f}'])
tab(rows,[93,52,58,59])
p(f'외부 특징을 쓰면 QR에만 가장 잘 맞추던 모델보다 오른쪽 검증 특징의 RMSE는 {gain:.1f}% 감소합니다. 반면 QR RMSE는 {qr_change:.1f}% 증가합니다. 따라서 모든 관측이 같은 비율로 좋아졌다는 결론은 부적절하며, 관측군 사이에 남는 불일치를 보여 줍니다.')
p('같은 외부 특징 모델끼리 비교하면 일정 바이어스를 허용하는 것이 바이어스 없는 모델보다 좋았습니다. 배율 항을 추가했을 때 오른쪽 중앙 오차는 조금 줄지만 전체 RMSE는 오히려 약간 커졌습니다. 시험한 허용폭 안에서는 복잡한 배율 모델이 일관되게 우월하지 않았습니다. 일정 바이어스는 유용한 근사지만 참으로 입증된 전제는 아닙니다.')
p('표의 오차는 영상 좌표 성분의 전방 예측 오차입니다. T를 역추정한 참값 오차나 새 세션에서의 제어 정확도와는 다릅니다. 이번 자료는 개발 자료이므로 독립적인 새 촬영 검증이라는 주장도 하지 않습니다.',small)
fig('04_independent_validation.png','추가 특징은 독립 부위의 일관성을 높이지만 QR 오차와 상충','모든 모델에서 같은 품질 기준·보정 사진·보류 자세를 사용했습니다. 원점 생성에 쓴 왼쪽 자연 특징과 검증 오른쪽 자연 특징은 측정 패치도 공유하지 않습니다.')

page('4. 공통 원점 생성 후 검증에서 무엇이 남았는가?')
p('공통 원점 v2 후보는 유효한 QR 관측, 상부 선, 왼쪽 자연 특징 22개로 생성했습니다. 오른쪽 자연 특징 17개는 원점 최적화에 넣지 않고 검사에 남겼습니다. 원점 기준 사진 4장의 추정 좌표 평균을 0으로 하는 좌표 규약과 날짜 영상 등록 규약은 명시적으로 유지했습니다.')
rows=[['제외한 원점 사진','QR RMS(px)','오른쪽 보류 RMS(px)','오른쪽 특징 수']]
for r in post.itertuples():rows.append([Path(r.path).name,f'{r.QR_component_RMS_px:.4f}',f'{r.right_component_RMS_px:.4f}',r.right_features])
tab(rows,[79,55,71,57])
p('각 검사는 해당 기준 사진을 원점 생성·영상 등록·바이어스 추정에서 제외하고 다시 적합했습니다. center1·2는 기준 사진만 있던 구간이므로, 사진을 빼면 그 구간 바이어스는 추정할 수 없어 0으로 복귀합니다. center3은 같은 구간의 다른 자세가 학습에 남아 있어 일정 바이어스 가정의 전이도 검사합니다.')
p(f'center3의 오른쪽 잔차 {post.iloc[2].right_component_RMS_px:.3f}px는 일반 자세 보류 잔차의 경험적 95백분위 {cv95:.3f}px보다 컸습니다. 이 비교가 물리적 원점의 통계 검정은 아니지만, 현재 규정으로 만든 원점이 모든 원점 사진에 안정적으로 전이된다고 말할 수 없음을 보여 줍니다.')
p('10/4 기준 사진은 영상 좌표계 정의에 사용되므로 독립 보류 검증을 했다고 주장하지 않았습니다. 기준을 정한 사진 자체에 잘 맞는 것을 원점 정확도의 증거로 삼지 않았습니다.')
fig('05_post_origin_checks.png','연속 구간 검증과 기준 사진 사후 검증','오른쪽 점선은 별도 일반 자세 보류 평가의 경험적 95백분위입니다. center3은 이 범위를 넘었습니다. 성능 향상이 있는 부분과 원점 확정에 실패한 부분을 분리해 보고합니다.')

page('5. 결과의 채택 상태와 이전 해석의 수정')
p('<b>채택할 결과:</b> 방향별 측정 불확실성과 관측 품질을 전 단계에 반영하는 절차, 추가 특징이 독립된 부위의 설명에 도움이 된다는 개발 근거, 일정 바이어스가 유용한 1차 근사라는 결과입니다.')
p('<b>확정하지 않을 결과:</b> v1 또는 v2가 단 하나의 물리적 참원점을 찾았다는 주장, 구간별 Y/T 보정량이 라벨의 실제 오류라는 주장, 영상 특징 개선율이 T 정확도 개선율이라는 해석입니다.')
p('v2는 QR 630점의 가상 기준과 왼쪽 자연 특징의 원점 좌표, 사진별 잠정 변환을 저장했으나, JSON 상태를 검증 미충족으로 표시했습니다. 최적화 변수도 경계에 닿는 경우가 남습니다. 따라서 기존 파일명을 이 값으로 수정하거나 확정된 표준으로 대체하지 않았습니다.')
p('다음 개선에서는 불일치를 더 많은 자유도로 억지로 흡수하기보다, 방향별 스캔 차이·영상 반응 모델·가림과 반사·명령/Actual 로그의 의미를 분리해 검토해야 합니다. 새로운 독립 공통 자세와 실제 위치 기록이 생기면 원점의 물리적 정확도를 추가 검증할 수 있습니다.')
p('원본 575장은 변경하지 않았습니다. natural_lanczos.cpp와 quality_scene.py는 측정·불확실성 검사, model_scene.py는 학습/검증 분리와 v2 구성, report_quality.py는 이 PDF와 결과 묶음을 재생성합니다. 재현용 CSV·JSON·코드를 ZIP에 포함했습니다.',small)

def footer(c,doc):
    c.saveState();c.setStrokeColor(colors.HexColor('#d3dee8'));c.line(15*mm,11*mm,282*mm,11*mm)
    c.setFont('Malgun',8);c.setFillColor(colors.HexColor('#63758a'));c.drawString(15*mm,6*mm,'특징 신뢰도와 공통 원점 재검증 | 2026-11-02');c.drawRightString(282*mm,6*mm,str(doc.page));c.restoreState()
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
pdf=OUT/'T_axis_origin_quality_revalidation_v2.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=landscape(A4),leftMargin=15*mm,rightMargin=15*mm,topMargin=12*mm,bottomMargin=17*mm,title='특징 신뢰도를 반영한 공통 원점 재검증 v2',author='Codex')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(pdf);text='\n'.join(p.extract_text() for p in reader.pages)
assert all(s in text for s in ['551','154','0.668','검증','37'])
save('pdf_check.json',{'pages':len(reader.pages),'text_characters':len(text),'path':str(pdf)})
with zipfile.ZipFile(P/'quality_origin_v2_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in P.iterdir():
        if f.is_file() and f.suffix in ['.csv','.json','.py','.cpp','.png','.md','.txt']:z.write(f,f.name)
    z.write(pdf,pdf.name)
print('PDF',pdf,'PAGES',len(reader.pages));print('STATUS',status['conclusion'])
