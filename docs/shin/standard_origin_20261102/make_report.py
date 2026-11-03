from fit_origin import *
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape,A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
from pypdf import PdfReader
from html import escape
import zipfile

full,L=data();d=full[~full.suspect].copy();model=json.loads((P/'standard_origin.json').read_text())
m=Model(d,L,model['degree'],True,model['sigma_session_label_Y_mm'],model['sigma_session_label_T_deg'])
B=np.asarray(model['scan_group_bias_YT']);u=((B[m.free]-B[m.fixed])/[m.sy,m.st]).ravel()
m.u=u;m.objective(u,pd.read_csv(P/'fit_residuals.csv').robust_weight.to_numpy(),True)
cv=pd.read_csv(P/'cv_summary.csv');sens=pd.read_csv(P/'prior_and_shape_sensitivity.csv')
points=pd.read_csv(P/'standard_origin_630points.csv')[['x_standard_px','y_standard_px']].to_numpy()
figures(m,d,cv,sens,points)
model['status']='provisional_statistical_reference'
model['session_biases_status']='not certified ground truth; regularizer-sensitive and bound-limited'
model['frame_registration']='Oct-15 common image translation is fixed from nominal-zero reference-image means; this is a coordinate-frame convention, not causal camera-motion identification.'
model['reference_constraint']='Additional four-reference-equivalent penalty on forward-model intercept, around frame-registered reference-feature mean.'
save('standard_origin.json',model)

validation=json.loads((P/'validation.json').read_text())
tr=pd.read_csv(P/'within_session_tracks.csv')
selected=cv[cv.config=='bias_deg3_t010'].iloc[0];base=cv[cv.config=='no_bias_deg3'].iloc[0]
gain=100*(1-selected.median_normalized_RMS/base.median_normalized_RMS)
qr_gain=100*(1-selected.QR_RMSE_px/base.QR_RMSE_px)
interval=pd.read_csv(P/'conditional_origin_intervals.csv')
centers=pd.read_csv(P/'standard_origin_qr_centers.csv')
starts=pd.read_csv(P/'multistart_stability.csv')
summary={'selected_model':selected.to_dict(),'same_degree_baseline':base.to_dict(),
         'normalized_forward_median_reduction_percent':gain,'QR_forward_RMSE_reduction_percent':qr_gain,
         'origin_f0_prior_shape_range':[sens.origin_f0.min(),sens.origin_f0.max()],
         'multistart_origin_f0_range':[starts.origin_f0.min(),starts.origin_f0.max()],
         'status':'virtual standard reference constructed; individual Y/T bias decomposition remains model-dependent',
         'not_a_T_accuracy_measurement':True}
save('result_summary.json',summary)

pdfmetrics.registerFont(TTFont('Malgun',r'C:\Windows\Fonts\malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold',r'C:\Windows\Fonts\malgunbd.ttf'))
pdfmetrics.registerFontFamily('Malgun',normal='Malgun',bold='MalgunBold',italic='Malgun',boldItalic='MalgunBold')
body=ParagraphStyle('body',fontName='Malgun',fontSize=11,leading=18,spaceAfter=10,wordWrap='CJK',textColor=colors.HexColor('#26394d'))
head=ParagraphStyle('head',parent=body,fontName='MalgunBold',fontSize=21,leading=29,spaceAfter=15)
small=ParagraphStyle('small',parent=body,fontSize=9,leading=14,spaceAfter=7)
cell=ParagraphStyle('cell',parent=body,fontSize=9,leading=13,spaceAfter=0)
story=[]
def p(text,style=body):story.append(Paragraph(text,style))
def page(title):
    if story:story.append(PageBreak())
    p(title,head)
def tab(rows,widths):
    rows=[[Paragraph(escape(str(v)),cell) for v in row] for row in rows]
    t=Table(rows,colWidths=[v*mm for v in widths],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2edf5')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#d9e2eb'))]))
    story.append(t);story.append(Spacer(1,3*mm))
def fig(name,title,caption):
    page(title);im=P/name;w,h=PILImage.open(im).size;s=min(261*mm/w,148*mm/h)
    story.append(Image(str(im),width=w*s,height=h*s));story.append(Spacer(1,2*mm));p(caption,small)

page('연속 스캔의 일정 바이어스를 이용한 단일 표준 원점 v1')
p('<b>요청한 가설을 모델로 구현해, 전체 자료가 공유하는 가상 표준 원점을 구성했습니다.</b> 원본 575장의 Lanczos 재측정 결과를 이용했으며, 라벨 이상 4장은 적합에서 제외하고 그대로 보존했습니다.')
p('표준 원점은 하나의 이미지 파일을 임의로 고르는 대신, 기준 사진과 연속 스캔을 함께 설명하는 공통 영상 반응의 (Y,T)=(0,0)으로 정의했습니다. 결과는 QR 630점, 외부 선 120표본, 세션별 좌표 변환 및 수치 모델로 저장했습니다.')
p(f'<b>가설의 유용성:</b> 같은 자세 반복을 함께 보류한 5-fold 개발 검증에서 영상 특징의 정규화 RMS 중앙값은 {base.median_normalized_RMS:.3f} → {selected.median_normalized_RMS:.3f}, 약 {gain:.1f}% 감소했습니다. QR 특징의 보류 RMSE도 {base.QR_RMSE_px:.3f} → {selected.QR_RMSE_px:.3f}px로 감소했습니다.')
p('<b>발견한 추가 구조:</b> 긴 T 촬영 구간은 양의 T와 음의 T 두 스캔을 포함합니다. 각각은 매우 좁은 직선 띠지만, 둘을 하나의 일정 바이어스로 묶는 것은 부적절했습니다. 두 스캔을 별도 그룹으로 분리했습니다.')
p('<b>채택 범위:</b> 이 결과는 재현 가능한 통계적 표준 기준입니다. 실제 장비의 물리적 참원점이나 각 사진의 참좌표를 확정한 결과는 아닙니다. 특히 Y와 T에 나눠 배분한 구간별 바이어스는 가정과 경계 제한에 민감합니다.')
p('분석일 2026-11-02 · Y: mm · T: degree · T 1 tick=0.01° · 원본 및 기존 모델 불변',small)

page('1. 사용자의 관찰을 데이터에서 확인한 결과')
p('미세 틱 스캔에서 QR 수평 변위와 T 라벨을 직선으로 적합했습니다. 직선 주변의 폭이 작다는 관찰은 확인됩니다. 다만 이것이 누적 오차가 없다는 증거는 아닙니다. 일정한 틱 배율 오차나 시간에 비례한 드리프트도 순차 스캔에서는 직선의 기울기로 흡수될 수 있습니다.')
rows=[['스캔','사진 수','기울기(px/°)','절편(px)','직선 R²','잔차 SD(px)']]
for sid,branch,name in [(3,'positive','S03 양의 T'),(3,'negative','S03 음의 T'),(8,'all','S08 미세 T')]:
    q=tr[(tr.session==sid)&(tr.branch==branch)&(tr.feature=='f0')].iloc[0]
    rows.append([name,int(q.n),f'{q.slope:.4f}',f'{q.intercept:.4f}',f'{q.linear_R2:.8f}',f'{q.residual_SD_px:.4f}'])
tab(rows,[47,23,42,42,44,45])
p('S03의 양·음 스캔을 전체 범위에서 각각 직선 적합하면 절편 차이는 0.639px이며, 평균 수평 반응으로 환산한 차이는 약 0.79 tick입니다. 원점 가까운 |T|≤0.1°만 적합하면 약 0.52 tick이므로, 이 숫자 자체도 적합 범위에 의존합니다. 원점 차이·방향 의존성·비선형 반응의 분리가 필요합니다.')
p('시간 간격으로 나눈 기존 14개 후보 중 S03만 두 연속 스캔으로 분리해 총 15개 보정 그룹을 사용했습니다. 그룹 3은 양의 T, 그룹 15는 음의 T이며 나머지 그룹 번호는 기존 세션 후보와 같습니다. 시간 구간은 실제 재원점 이벤트를 확인한 기록이 아닙니다.')
fig('01_track_hypothesis.png','미세 틱 스캔: 직선 띠와 잔차','왼쪽 S03은 양·음 스캔을 함께 표시해 직선 잔차의 분기를 보여줍니다. 오른쪽 S08은 좁은 띠를 이루지만 작은 구조적 잔차가 남습니다. 큰 R²만으로 바이어스만 존재한다고 확정하지 않았습니다.')

page('2. 하나의 원점을 구성한 방식과 식별 기준')
p('<b>좌표 모델:</b> 각 사진의 표준 좌표 = 명목 라벨 + 그 연속 스캔의 일정한 (Y,T) 바이어스. 스캔 내부의 틱 간격은 유지하며, 사진별 자유 오프셋은 주지 않았습니다. 따라서 라벨을 사진마다 임의로 바꿔 오차를 없애는 모델이 아닙니다.')
p('<b>영상 모델:</b> QR 평균 변위 6개와 상·하 선 위치·기울기 4개를 공통 2차 또는 3차 전방 반응으로 설명합니다. 반복 사진에서 얻은 공분산을 25% 대각 축소해 특징을 백색화하고, 큰 영상 잔차는 IRLS로 완화했습니다. 선과 QR를 하나의 물리 평면으로 가정하지 않았습니다.')
p(f'<b>영상 좌표계 등록:</b> 기준 사진만으로 10/15 영상의 공통 이동을 (dx,dy)=({m.D[0]:.4f}, {m.D[1]:.4f})px로 계산해 10/4 영상 좌표계에 등록했습니다. 이 이동이 카메라 때문인지 기계 원점 때문인지는 판별하지 않고, 표준 영상 좌표계의 규약으로 고정했습니다.')
p('<b>원점의 정의:</b> 원점 기준 사진 4장의 추정 실제 좌표 평균을 (0,0)으로 두어 전체 좌표 이동의 자유도를 고정했습니다. 추가로 공통 반응의 절편이 등록한 기준 사진들의 특징 평균에서 과도하게 벗어나지 않도록 기준 사진 4장 상당의 제약을 적용했습니다. 완전히 자유로운 원점 추정은 아닙니다.')
p('<b>채택 규정:</b> 개발 검증에서 선택한 3차 공통 반응, 스캔 바이어스 사전 표준편차 Y=0.05mm·T=0.01°를 v1으로 채택했습니다. 게이지를 제거한 최적화 변수에는 ±3의 경계를 적용했습니다. 이는 라벨 오차의 참분포가 아니라 과도한 보정을 막기 위한 규정입니다.')
p('추정 과정에서는 선형 반응 계수를 조건부 최소제곱으로 제거하고, 스캔 바이어스를 해석적 기울기의 L-BFGS-B로 최적화했습니다. 기울기 유한차분 검증의 상대오차는 '+f'{validation["analytic_gradient_relative_error"]:.2e}'+'였습니다.',small)

page('3. 채택한 표준 원점의 수치 표현')
p('아래 값은 표준 좌표 (Y,T)=(0,0)에서 예측되는 QR 평균 변위입니다. 기준 자산 reference.png에서 추출한 같은 QR 점들에 대한 차이이며, 이 값 자체가 T 또는 Y 오차는 아닙니다. 수치는 재현을 위해 보존하고, 해석에서는 불확실성보다 많은 유효 숫자를 주장하지 않습니다.')
rows=[['관측','표준 dx(px)','표준 dy(px)','dx 조건부 95% 구간','dy 조건부 95% 구간']]
for k,name in enumerate(['좌측 QR','중앙 QR','우측 QR']):
    x=interval.iloc[k*2];y=interval.iloc[k*2+1]
    rows.append([name,f'{x.estimate:.4f}',f'{y.estimate:.4f}',f'{x.conditional_lo:.4f} ~ {x.conditional_hi:.4f}',f'{y.conditional_lo:.4f} ~ {y.conditional_hi:.4f}'])
tab(rows,[40,38,38,63,63])
p('QR 630개 점 각각의 표준 x/y 좌표는 standard_origin_630points.csv, 외부 선 120표본은 standard_origin_120edges.csv에 저장했습니다. 사진별 변환표 coordinates_in_standard_frame.csv에는 원래 라벨, 추정 스캔 바이어스, 표준 좌표, 적합 제외 여부가 함께 있습니다.')
p('<b>조건부 구간의 의미:</b> 기준 사진·영상 좌표 등록·모델 차수·사전 허용폭을 고정한 상태에서, 같은 자세 반복을 함께 묶어 60회 bootstrap한 근사 구간입니다. 물리적 참원점의 신뢰구간이 아니며, 모델 가정의 오차는 포함하지 않습니다.')
fig('05_virtual_reference.png','구성한 단일 가상 표준 원점: QR 630점','서로 다른 원본 사진을 픽셀 평균해 흐려진 이미지를 만들지 않았습니다. 하나의 공통 표준 위치에 해당하는 기하 좌표를 저장했습니다. 새 서브픽셀 보간 없이, 전수 Lanczos-3 재측정 좌표에서 추정했습니다.')
fig('02_standard_origin_tracks.png','공통 원점에 연결한 스캔 궤적','모델이 추정한 Y 영향과 날짜 영상 이동을 제거한 표현입니다. 빨간 별은 공통 (Y,T)=(0,0)의 특징값입니다. 이 그림은 적합된 모델의 표현이며, 일반화 검증은 다음의 보류 자세 결과로 확인해야 합니다.')

page('4. 보류 자세에서 가설이 실제로 도움이 되었는가?')
p('기준 원점 사진 4장은 보정 앵커로 항상 유지하고, 나머지 567장을 같은 (Y,T) 명령 좌표의 반복 묶음으로 5-fold 분리했습니다. 시험 사진의 영상·라벨을 이용해 스캔 바이어스를 다시 맞추지 않았습니다. 각 fold의 측정 공분산도 학습 반복 사진만으로 추정했습니다. 시험은 이미 보정한 촬영 묶음 안에서 보지 않은 자세를 설명하는 능력을 평가합니다.')
rows=[['동일 3차 반응 비교','원점 바이어스 없음','스캔 일정 바이어스 허용','변화']]
rows.extend([
 ['정규화 RMS 중앙값',f'{base.median_normalized_RMS:.3f}',f'{selected.median_normalized_RMS:.3f}',f'{gain:.1f}% 감소'],
 ['QR 전방 RMSE(px)',f'{base.QR_RMSE_px:.4f}',f'{selected.QR_RMSE_px:.4f}',f'{qr_gain:.1f}% 감소'],
 ['외부 선 전방 RMSE(px)',f'{base.line_RMSE_px:.4f}',f'{selected.line_RMSE_px:.4f}',f'{100*(1-selected.line_RMSE_px/base.line_RMSE_px):.1f}% 감소'],
 ['정규화 RMS 95백분위',f'{base.p95_normalized_RMS:.3f}',f'{selected.p95_normalized_RMS:.3f}','일부 어려운 표본은 남음']])
tab(rows,[65,59,65,60])
p('<b>해석:</b> 일정한 스캔 바이어스를 허용하는 것이 영상 자료의 공통 설명에 도움이 된다는 개발 근거가 있습니다. 그러나 이 점수는 사진에서 T를 역추정한 참값 오차가 아닙니다. 원점을 선택한 뒤 바꾼 라벨과 예측을 비교해 정확도가 높아졌다고 계산하지 않았습니다.')
p('모델 차수와 사전 폭은 같은 개발 CV에서 선택했으므로 선택 편향이 남습니다. 완전히 새로운 촬영 세션의 성능이나 모든 사진의 최대 T 오차 1 tick 이내를 입증한 결과는 아닙니다.',small)
fig('04_forward_validation.png','일정 바이어스를 허용한 모델의 전방 검증','숫자가 작을수록 보류 사진의 QR·외부 선 특징을 더 잘 설명합니다. 비교 모델들은 동일한 원점 기준 사진과 날짜 영상 좌표 등록을 사용했습니다. T 추정 정확도와 구분해야 합니다.')

page('5. 원점 위치와 구간별 Y/T 보정량의 안정성은 다르다')
p(f'같은 조건에서 초기값을 바꿔 5회 다시 적합한 좌측 QR 원점 dx 범위는 {starts.origin_f0.min():.5f} ~ {starts.origin_f0.max():.5f}px였습니다. 다만 이것은 같은 모델 안의 국소 안정성 검사이며 전역 최적해의 증명은 아닙니다.')
p(f'모델을 2·3차로 바꾸고 T 바이어스 사전 폭을 0.25~4 tick으로 바꾸면, 좌측 QR 원점 dx는 {sens.origin_f0.min():.3f} ~ {sens.origin_f0.max():.3f}px로 달라졌습니다. 같은 조건의 bootstrap 구간보다 넓으므로, 원점의 최종 불확실성에서는 모델 가정의 영향도 포함해야 합니다.')
p('더 큰 문제는 구간별 Y/T 바이어스 분해입니다. 예를 들어 S03 양의 T에 배정한 T 보정은 작은 사전 폭에서는 0 tick 부근, 큰 사전 폭에서는 10 tick 이상으로 달라집니다. 동시에 Y 보정도 달라져 비슷한 영상을 설명합니다. 원점 영상 좌표는 상대적으로 좁은 범위라도 물리 좌표 보정량은 유일하지 않습니다.')
p(f'채택 모델의 게이지 제거 변수 {len(m.u)}개 중 {validation["raw_parameter_bound_hits"]}개가 경계에 닿았습니다. 따라서 session_biases.csv의 수치는 확정된 라벨 수정량이 아니라, 채택 규정 안에서 얻은 잠정값입니다. 원본 파일명을 이 값으로 일괄 변경하지 않았습니다.')
p('영상 등록 이동에 ±0.08px를 가정해 흔들어 보는 별도 민감도도 계산했습니다. 기준 사진을 고정한 bootstrap만으로 원점의 정밀도를 과신하지 않도록, 해당 결과는 anchor_frame_sensitivity.csv에 구분해 저장했습니다.')
fig('06_origin_stability.png','표본 변동과 모델 의존성을 구분한 원점 안정성','왼쪽은 기준 사진과 모델 가정을 고정한 60회 bootstrap, 오른쪽은 모델 차수·사전 허용폭을 바꾼 결과입니다. 좁은 표본 구간을 물리적 원점의 확정 오차로 해석하면 안 됩니다.')
fig('03_bias_sensitivity.png','각 스캔의 보정량은 아직 모델 의존적이다','그룹 3은 S03 양의 T, 그룹 15는 S03 음의 T입니다. 추정 보정량이 차수와 사전 폭에 따라 크게 바뀌는 구간은 라벨의 실제 오류가 그 값이라고 확정할 수 없습니다.')

page('6. 지금부터 사용할 수 있는 결과와 남은 한계')
p('<b>현재 사용할 수 있는 것:</b> standard_origin.json에 규정을 명시한 표준 원점 v1, QR 630점·선 120표본의 가상 기준, 모든 사진을 같은 통계적 좌표계로 표시하는 잠정 변환표가 있습니다. 기준 규정을 고정하면 같은 데이터에서 재현할 수 있습니다.')
p('<b>현재 확정할 수 없는 것:</b> 단 하나의 물리적 참원점이 현재 자료로 유일하게 복원되었다는 주장, 구간별 Y/T 보정량을 실제 장비 오차로 해석하는 것, 수정 좌표를 참값으로 사용해 T 정확도를 평가하는 것입니다.')
p('<b>이번 가설의 결론:</b> 연속 스캔 내부의 일정한 바이어스라는 가정은 자료 설명과 보류 자세 예측에 유용했습니다. 다만 직선 띠만으로 틱 배율 오차·방향 의존성·느린 드리프트가 없다고 증명되지는 않습니다. 기존 시간 구간보다 실제 연속 스캔을 구분하는 것이 중요했습니다.')
p('최적화가 원점 차이 대신 카메라·치구 변화나 부족한 반응 모델까지 흡수하지 않도록, 날짜 영상 등록을 기준 사진에서 고정하고, 구간 바이어스에 사전 규정과 경계 제한을 적용했습니다. 자유도를 무제한 늘려 잘 맞는 해를 최종 원점으로 채택하지 않았습니다.')
p('추후 새로운 공통 자세 앵커 또는 독립 실제 위치 로그가 추가되면, 같은 표준 기준을 유지하면서 Y/T 바이어스의 불확실성을 줄일 수 있습니다. 이번 v1은 그 후속 검증을 위한 재현 가능한 기준입니다.')
p('원본: 기존 분석의 해시 검증·575장 Lanczos 전수 재측정 자료. 새로 이미지 보간이나 원본 수정은 하지 않았습니다. 분석 재현: fit_origin.py → validate_origin.py → make_report.py. PDF와 함께 JSON·CSV·코드를 묶은 ZIP을 제공합니다.',small)

def footer(c,doc):
    c.saveState();c.setStrokeColor(colors.HexColor('#d2dce6'));c.line(15*mm,11*mm,282*mm,11*mm)
    c.setFont('Malgun',8);c.setFillColor(colors.HexColor('#63758a'));c.drawString(15*mm,6*mm,'연속 스캔 기반 표준 원점 v1 | 2026-11-02')
    c.drawRightString(282*mm,6*mm,str(doc.page));c.restoreState()
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
target=OUT/'T_axis_standard_origin_v1.pdf'
doc=SimpleDocTemplate(str(target),pagesize=landscape(A4),leftMargin=15*mm,rightMargin=15*mm,topMargin=12*mm,bottomMargin=17*mm,
    title='연속 스캔의 일정 바이어스로 구성한 표준 원점 v1',author='Codex')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(target);text='\n'.join(page.extract_text() for page in r.pages)
assert all(v in text for v in ['630','567','바이어스','통계적','물리적'])
save('pdf_validation.json',{'pages':len(r.pages),'text_characters':len(text),'pdf':str(target)})
with zipfile.ZipFile(P/'standard_origin_v1_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in P.iterdir():
        if f.is_file() and f.suffix in ['.json','.csv','.py','.png','.md','.ps1']:z.write(f,f.name)
    z.write(target,target.name)
print('PDF',target,'PAGES',len(r.pages));print('FORWARD_GAIN',gain)
