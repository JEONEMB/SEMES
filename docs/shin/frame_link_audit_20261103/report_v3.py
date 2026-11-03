from forensics import *
import cv2,zipfile
from PIL import Image as PILImage,ImageDraw,ImageFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape,A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
from html import escape

data,features,arrays=load();result=json.loads((P/'result_summary_v3.json').read_text());j=json.loads((P/'standard_origin_v3.json').read_text())
summary=pd.read_csv(P/'refinement_summary.csv');post=pd.read_csv(P/'refinement_zero_holdouts.csv');selected=result['selected']
coords=pd.read_csv(P/'coordinates_standard_v3.csv');intervals=pd.read_csv(P/'conditional_origin_intervals_v3.csv')
outliers=coords[coords.training_image_weight<.999].merge(pd.read_csv(P/'final_repeat_diagnostics.csv')[['path','common_norm','region_disagreement']],on='path')
outliers.to_csv(P/'outlier_actions.csv',index=False)

# A transparent photo map of the final origin-construction and held-out landmarks.
photo=PILImage.open(Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set\center.bmp')).convert('RGB')
draw=ImageDraw.Draw(photo);font=ImageFont.truetype(r'C:\Windows\Fonts\malgunbd.ttf',22)
A=[b['feature_id'] for b in j['origin_blocks'] if b['feature_id'] is not None and b['importance']>0]
featuremap=pd.read_csv(V2/'annotated_feature_ID_map.csv');B=featuremap.loc[featuremap.role=='heldout_validation_B','feature_id'].to_list()
for ids,color,prefix in [(A,'#ed45b4','A'),(B,'#00c98b','B')]:
    for fid in ids:
        f=features.iloc[fid];x,y=f.reference_x,f.reference_y
        draw.rectangle((x-20,y-20,x+20,y+20),outline=color,width=4)
        box=draw.textbbox((x-22,y-49),f'{prefix}{fid:02}',font=font);draw.rectangle((box[0]-3,box[1]-2,box[2]+3,box[3]+2),fill='#10202c')
        draw.text((x-22,y-49),f'{prefix}{fid:02}',font=font,fill=color)
frame=575-4
fs=cv2.FileStorage(str(ROOT/'analysis/session_eda_20261102/remeasured/cache'/f'frame{frame}.yml.gz'),cv2.FILE_STORAGE_READ)
pts=fs.getNode('points').mat();edges=fs.getNode('edges').mat();fs.release()
for q in range(3):
    xy=pts[pts[:,2]==q,:2]
    draw.rectangle((*xy.min(0)-8,*xy.max(0)+8),outline='#278fff',width=4)
    for x,y in xy:draw.ellipse((x-1,y-1,x+1,y+1),fill='#278fff')
for x,y,*_ in edges[:60][edges[:60,5]>0]:draw.rectangle((x-3,y-7,x+3,y+7),outline='#ffbf20',width=2)
banner=PILImage.new('RGB',(3840,2310),'#10202c');banner.paste(photo,(0,150));dr=ImageDraw.Draw(banner)
dr.text((35,20),'표준 원점 v3 구성에 반영한 관측 영역',font=ImageFont.truetype(r'C:\Windows\Fonts\malgunbd.ttf',42),fill='white')
dr.text((35,84),f'분홍 A {len(A)}개: 원점 구성 / 초록 B {len(B)}개: 보류 검증 / 파랑: QR 630점 / 노랑: 상부 선',font=ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf',28),fill='#d2e0eb')
banner.save(P/'07_standard_origin_v3_features.png')

pdfmetrics.registerFont(TTFont('Malgun',r'C:\Windows\Fonts\malgun.ttf'));pdfmetrics.registerFont(TTFont('MalgunBold',r'C:\Windows\Fonts\malgunbd.ttf'))
pdfmetrics.registerFontFamily('Malgun',normal='Malgun',bold='MalgunBold',italic='Malgun',boldItalic='MalgunBold')
body=ParagraphStyle('body',fontName='Malgun',fontSize=11,leading=18,spaceAfter=10,wordWrap='CJK',textColor=colors.HexColor('#27394d'))
head=ParagraphStyle('head',parent=body,fontName='MalgunBold',fontSize=20,leading=28,spaceAfter=14)
small=ParagraphStyle('small',parent=body,fontSize=9,leading=14)
cell=ParagraphStyle('cell',parent=body,fontSize=9,leading=13,spaceAfter=0)
story=[]
def p(s,style=body):story.append(Paragraph(s,style))
def page(title):
    if story:story.append(PageBreak())
    p(title,head)
def tab(rows,widths):
    t=Table([[Paragraph(escape(str(x)),cell) for x in row] for row in rows],colWidths=[w*mm for w in widths],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1edf4')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#dbe3eb'))]))
    story.append(t);story.append(Spacer(1,3*mm))
def fig(name,title,caption):
    page(title);w,h=PILImage.open(P/name).size;scale=min(262*mm/w,148*mm/h)
    story.append(Image(str(P/name),width=w*scale,height=h*scale));story.append(Spacer(1,2*mm));p(caption,small)

page('공통 좌표 연결의 단절 진단과 표준 원점 v3')
p('<b>단일 사진의 나쁜 특징점만이 문제가 아니었습니다.</b> 원점 복귀 사진을 인접 스캔의 일정 바이어스와 묶은 구간 규칙, 배경 반응식의 부족, 파일명·로그 불일치, 반복 촬영 위치 변동을 구분했습니다. 확인한 처리를 표준 원점 v3에 반영했습니다.')
p('<b>데이터 처리:</b> 원본 575장은 모두 보존·사용했습니다. .32 폴더의 .33 파일명 1건은 파생 표의 T를 .32로 정정했습니다. .57 사진 3장은 영상 연속성이 라벨을 지지해 적합 자료로 복원하고 Actual 로그 불일치는 별도 기록했습니다. 반복 편차가 큰 4장은 삭제하지 않고 가중치를 낮췄습니다.')
p(f'<b>핵심 구간 수정:</b> center3를 보류한 원점 검증의 QR 잔차는 {result["center3_QR_before_after"][0]:.3f} → {result["center3_QR_before_after"][1]:.3f}px, 원점 생성에 쓰지 않은 오른쪽 특징 잔차는 {result["center3_right_before_after"][0]:.3f} → {result["center3_right_before_after"][1]:.3f}px로 감소했습니다.')
z=result['changes']['pose'];b=result['changes']['contiguous']
p(f'<b>일반 자세 보류:</b> QR RMSE {z["QR_RMSE_before"]:.3f} → {z["QR_RMSE_after"]:.3f}px, 오른쪽 특징 {z["right_RMSE_before"]:.3f} → {z["right_RMSE_after"]:.3f}px. 연속 자세 블록 보류에서도 QR {b["QR_RMSE_before"]:.3f} → {b["QR_RMSE_after"]:.3f}px, 오른쪽 {b["right_RMSE_before"]:.3f} → {b["right_RMSE_after"]:.3f}px로 개선됐습니다.')
p('<b>채택 상태:</b> v3는 현재 자료에서 재검증한 통계적 표준 원점입니다. 원점 영상·부위 간 일관성은 개선됐지만, 물리적 참원점이나 최대 T 오차 1 tick 이내를 입증한 결과는 아닙니다. 이미 탐색한 자료에서 얻은 개발 결과입니다.')
p('2026-11-03 · Y: mm · T: degree · 1 tick=0.01° · 새 이미지 보간 없이 기존 Lanczos-3 정밀 계측 좌표를 사용',small)

page('1. 무엇을 아웃라이어로 보았고 무엇을 보지 않았는가?')
tab([['구분','관측 근거','반영 방식'],
 ['파일명 라벨 오류 1건','.32 폴더/대응 TXT와 QR·좌우 궤적이 모두 .32를 지지','파생 표 T=.32; 원본 이름 유지'],
 ['Actual 로그 불일치 3장','.57 사진은 주변 .56/.58 사이에 정상적으로 위치','사진 복원, .57 유지; 로그 충돌 기록'],
 ['반복 위치 편차 후보 4장','QR·여러 외부 부위가 함께 변위; 지역 간 차이는 작음','정합 실패로 삭제하지 않고 soft weighting'],
 ['구간 연결 오류 후보','46초 뒤 원점 복귀를 120초 공백 기준으로 같은 구간에 묶음','모든 원점 기준 노출을 스캔과 분리'],
 ['부위별 체계적 모델 잔차','양의 Y 스캔에서 본체와 배경의 잔차가 일관되게 다름','Y 고차 반응 항의 보류 성능 비교']], [51,118,93])
p('큰 모델 잔차만으로 사진을 제거하지 않았습니다. 반복 촬영의 서로 다른 부위가 같은 방향으로 움직이면 실제 촬영 위치 변동 또는 공통 영상 변동일 수 있습니다. 카메라·기구·진동 중 원인을 특정할 독립 로그는 없습니다.')

page('2. 라벨 오류와 로그 오류를 영상 궤적으로 구분')
p('전역 원점을 먼저 가정하지 않고, Y=.05로 고정된 인접 미세 T 촬영에서 문제 사진을 제외한 국소 반응을 적합했습니다. QR, 왼쪽 자연 특징 A, 오른쪽 B를 각각 사용해 국소 순서 좌표를 비교했습니다. 여기서 구한 T는 주변 명목 라벨을 기준으로 한 상대 좌표이며 독립적인 물리적 참값은 아닙니다.')
p('.32 폴더의 .33 파일명 사진은 QR=0.32043°, A=0.32027°, B=0.32043°에 해당했습니다. 파일명 .33보다 폴더 .32와 일치합니다. 원본은 변경하지 않고 파생 표에 정정 근거를 기록했습니다.')
p('.57 사진 3장의 국소 좌표는 대체로 .569~.571°였습니다. Actual 로그로 환산한 .56° 위치와는 맞지 않으며, .57 영상이 주변 영상 사이의 한 틱 진행을 끊는 모습은 없습니다. 따라서 이 3장을 영상 아웃라이어로 제외했던 보수적 처리를 철회했습니다. Actual 값의 기록 의미·동기화 오류 여부는 미확정입니다.')
fig('02_label_evidence.png','라벨 .32/.33과 Actual .56/.57의 다른 성격','왼쪽 화살표는 파생 라벨 .33→.32 정정입니다. 오른쪽 빨간 X는 Actual 로그를 곧바로 참값으로 사용했을 때의 위치입니다. 해당 위치보다 .57 라벨이 영상 궤적과 일관됩니다.')

page('3. 좌표 연결이 크게 깨진 두 곳')
p('<b>첫째, S11의 원점 복귀:</b> Y=.1,T=.01의 반복 3장 뒤에 center3가 촬영됐습니다. 마지막 반복과 center3의 수정 시각 간격은 46초여서 기존 120초 공백 규칙은 같은 세션 후보로 묶었습니다. 그러나 명령 위치는 (0,0)으로 복귀했습니다. 같은 일정 바이어스로 강제하는 것은 별개의 가정입니다.')
p('원점 사진은 정상 영상일 수 있는데, 인접 스캔의 바이어스가 이 사진에 잘못 전이돼 큰 잔차를 만들 수 있었습니다. 모든 원점 기준 노출을 움직임 스캔과 분리하는 일반 규칙을 적용하자 center3의 보류 오차가 크게 줄었습니다. 실제 장비의 재원점·백래시 이벤트가 있었다는 물리적 증명은 아닙니다.')
p('<b>둘째, 양의 Y 단독 스캔의 배경:</b> S01·S04에서 QR·본체 나사 잔차는 약 .05~.08px지만 좌우 배경은 약 .38~.53px였습니다. 한 장의 무작위 실패보다 여러 자세에서 반복되는 부위별 차이였습니다. 모든 특징에 같은 3차 반응식 구조를 쓰는 가정의 부족을 점검했습니다.')
p('Y의 4·5차 항을 소수 추가하자 보류한 배경 설명이 개선됐습니다. 이는 영상 반응식 부족을 지지하지만, 그 원인이 렌즈 왜곡인지 깊이·투영 관계인지 기구 움직임인지는 현재 결과만으로 결정하지 않았습니다.')
fig('01_connection_break_heatmap.png','기존 모델의 구간·관측군별 잔차','이 진단표는 이전 v2의 551장 보류 자료입니다. QR·본체가 함께 어긋나는 S11과, 본체보다 배경이 크게 어긋나는 양의 Y 스캔을 구분했습니다. 뒤의 전략 비교는 모두 감사 후 같은 575장과 555장 보류를 사용합니다.')
fig('03_zero_reset_timeline.png','시간 공백만으로는 원점 복귀를 구분하지 못했다','시각은 파일 수정 시각의 대리값입니다. 실제 촬영 로그라고 단정하지 않았습니다. 반영한 규칙은 특정 center3만 예외 처리하는 것이 아니라 모든 원점 기준 노출을 인접 움직임 스캔과 분리하는 것입니다.')

page('4. 반복 촬영의 큰 편차는 어떻게 처리했는가?')
p('같은 감사 후 (Y,T) 좌표의 반복 사진에서 QR 평균 이동과 왼쪽 외부 부위별 중앙 이동을 계산했습니다. 지역마다 점 수가 달라도 한 부위가 과도한 표를 갖지 않도록 부위 중앙값을 사용했습니다. 각 사진을 나머지 반복과 비교한 공통 변위가 중앙값+4×강건 표준편차를 넘으면 후보로 표시했습니다.')
tab([['사진','공통 반복 편차(px)','지역 간 차이(px)','학습 가중치']]+[[Path(r.path).name,f'{r.common_norm:.3f}',f'{r.region_disagreement:.3f}',f'{r.training_image_weight:.3f}']for r in outliers.itertuples()], [139,42,42,39])
p('최대 후보는 Y=.1,T=.1의 두 번째 사진으로 공통 변위 약 .580px에 비해 지역 간 차이는 .014px였습니다. 여러 곳이 함께 움직였으므로 특정 패치의 오정합이라고 볼 근거가 약합니다. 삭제 대신 가중치를 낮췄으며, 보류 평가에서는 불리한 사진도 그대로 점수에 포함했습니다.')
p('각 fold의 후보 기준과 가중치는 그 fold의 학습 반복에서만 다시 계산했습니다. 검증용 오른쪽 B의 잔차를 보고 학습 사진을 제거하거나 가중치를 맞추지 않았습니다. 이 조정의 전체 성능 영향은 작았고, 주된 개선은 구간 규칙과 반응식 수정에서 나왔습니다.')
fig('04_repeat_deviation_photo.png','실제 위치 변동 후보: 부위들이 함께 움직인다','화살표는 가시성을 위해 실제 변위의 200배로 표시했습니다. 배경을 고정물로 가정하지 않았습니다. 이 그림은 원본 사진의 불량 판정이 아니라 동일 명령 반복 내 공통 영상 변동의 증거입니다.')

page('5. 어떤 수정을 채택했는가?')
p('동일한 감사 후 575장에 대해 여섯 전략을 비교했습니다. 특징 선택용 고정 보정 자료와 같은 자세 반복 20장을 학습에 남겼고, 나머지 555장을 동일 자세 묶음 및 연속 자세 블록으로 각각 보류했습니다. 특징 품질 기준, 부위별 공분산, 반복 편차 가중치는 학습 자료에서만 추정했습니다.')
p('수정 단계는 기존 시간 구간, 원점 노출 분리, 반복 편차 강건 가중, 본체 위주 구성, 별도 배경 Y5 반응, 전체 특징 Y5 반응입니다. 배경을 제외하면 QR은 잘 맞지만 배경 설명은 악화되는 상충도 확인했습니다.')
p('<b>채택 규칙:</b> 두 보류 방식에서 QR RMSE를 기준 대비 5% 넘게 악화시키지 않는 후보 중, 원점 생성에 사용하지 않은 오른쪽 특징의 연속 블록 RMSE가 가장 작은 후보를 선택했습니다. 선택 결과는 원점 분리+반복 강건 가중+전체 Y4/Y5 반응 보완입니다. T 방향은 기존 3차 범위를 유지했습니다.')
tab([['평가','반영 전 QR','v3 QR','반영 전 오른쪽','v3 오른쪽']]+[[label,f'{result["changes"][scheme]["QR_RMSE_before"]:.4f}',f'{result["changes"][scheme]["QR_RMSE_after"]:.4f}',f'{result["changes"][scheme]["right_RMSE_before"]:.4f}',f'{result["changes"][scheme]["right_RMSE_after"]:.4f}']for scheme,label in [('pose','동일 자세 묶음'),('contiguous','연속 자세 블록')]], [62,49,49,51,51])
p('단위는 영상 좌표 성분의 전방 RMSE(px)입니다. 이전 v2 표와 이미지 수가 다르므로 직접 비교하지 않고, 같은 감사 후 데이터의 반영 전후만 비교했습니다. 이 선택은 이미 탐색한 개발 자료에 대한 것이며 새 촬영 독립 검증은 아닙니다.',small)
fig('05_refinement_validation.png','단계별 비교: 큰 잔차를 제거하지 않고 원인을 나눠 반영','QR만 좋게 만드는 후보 대신 QR와 원점 생성 미사용 오른쪽 특징을 함께 확인했습니다. 모든 시험 사진을 포함한 결과이며, 특정 실패 사진을 시험 점수에서 빼지 않았습니다.')

page('6. 원점 생성 후 보류 기준 사진 재검증')
p('각 원점 기준 사진을 한 장씩 제외하고 날짜 영상 등록·원점·스캔 바이어스를 다시 적합했습니다. 제외한 사진의 관측값을 사용해 해당 구간 오프셋을 맞추지 않았습니다. 10/4 기준은 영상 좌표계를 정의하므로 별도의 독립 보류 성공을 주장하지 않습니다.')
before=post[post.config=='audited_time'].set_index('path');after=post[post.config==selected].set_index('path')
rows=[['기준 사진','반영 전 QR','v3 QR','반영 전 오른쪽','v3 오른쪽']]
for path,r in after.iterrows():
    b=before.loc[path];rows.append([Path(path).name,f'{b.QR_RMS_px:.4f}',f'{r.QR_RMS_px:.4f}',f'{b.right_RMS_px:.4f}',f'{r.right_RMS_px:.4f}'])
tab(rows,[62,49,49,51,51])
p('특히 center3는 더 이상 Y=.1,T=.01 스캔의 바이어스를 물려받지 않습니다. QR 잔차 개선이 큰 반면, 오른쪽의 잔차는 여전히 약 .2px이며 다른 원점 사진에서도 0이 되지 않습니다. 새 원점에서 모든 현상이 완전히 설명됐다는 뜻은 아닙니다.')
p(f'채택 모델의 일반 자세 보류 오른쪽 RMS 95백분위는 {result["new_right_pose_95pct"]:.3f}px입니다. 표의 경험적 크기를 비교할 수 있으나, 기준 사진 3장만으로 물리적 원점의 통계적 정확도나 모든 미래 세션 성능을 보증할 수 없습니다.')
fig('06_zero_holdout_before_after.png','이상 사진 제거 없이 원점 기준의 전이 오차를 줄였다','같은 기준 사진을 학습에서 제외한 상태의 반영 전후입니다. 각 사진·관측군이 모두 같은 비율로 개선되지는 않습니다. 이는 현재 자료에서의 개발 재검증입니다.')

page('검증이 타당한 이유: 무엇을 숨기고 무엇을 예측했는가?')
p('학습에 사용한 영상에 다시 맞춰 본 잔차만으로는 원점을 검증할 수 없습니다. 이번 비교는 다음의 정보 차단을 적용한 전방 예측 검사입니다. 즉 학습 영상에서 좌표-특징 관계를 만든 뒤, 보류 영상에서 실제 계측한 특징을 얼마나 설명하는지 확인합니다.')
tab([['검증 장치','차단한 정보와 이유','확인하는 내용'],
 ['같은 자세 반복을 통째로 보류','같은 (Y,T) 반복이 학습·시험 양쪽에 들어가면 사실상 같은 위치를 미리 본 효과가 생김','반복 사진 복제가 아닌 보류 자세 설명력'],
 ['연속 자세 블록 보류','한 구간의 이웃 자세를 묶어 제외. 고정 보정용 20장은 학습에 유지','촘촘한 무작위 분할에만 의존하지 않는지'],
 ['원점 생성용 A와 검사 B 분리','서로 겹치지 않는 오른쪽 B 패치는 원점 적합 목적함수에 미사용. B 반응식은 학습 영상에서만 보정','원점 생성에 쓰지 않은 부위로의 전이'],
 ['원점 사진 한 장씩 제외','제외 사진 없이 날짜 등록·원점·바이어스 재적합. 제외 사진 관측으로 오프셋 보정하지 않음','다른 원점 기준에서 제외 원점으로의 전이'],
 ['불리한 시험 사진도 포함','각 방식 555장 동일 분모. 학습 반복에서만 가중치 계산. 큰 시험 잔차로 사진을 삭제하지 않음','좋은 사례 선별에 의한 개선 방지']], [49,135,78])
p('특징 선택용 고정 보정 자료와 같은 자세 반복은 모두 학습에 남겼습니다. 따라서 이 평가는 현재 촬영 범위와 보정 앵커가 있는 조건의 전이 검사입니다. 보정 앵커가 전혀 없는 새 범위 또는 새 세션 성능을 뜻하지 않습니다.',small)

page('검증이 말할 수 있는 범위와 추가 독립 검증')
p('<b>반영 전후 비교가 의미 있는 이유:</b> 같은 라벨 감사, 같은 575장, 같은 보류 분할과 555장 평가 분모에서 규칙만 바꿨습니다. 따라서 .57 사진 복원으로 분모가 바뀐 과거 v2 표와 달리, 이번 표의 차이는 비교 조건 변화로 설명되지 않습니다. QR과 생성 미사용 B가 두 분할 방식에서 함께 개선됐으며, center2의 QR은 소폭 악화됐다는 결과도 그대로 공개했습니다.')
p('<b>순환 검증을 완전히 없앴다고 말하지 않는 이유:</b> B 반응식은 학습 영상의 추정 표준 좌표를 기준으로 보정했습니다. 따라서 B는 원점 목적함수 밖의 관측이지만 독립적인 위치 참값은 아닙니다. 파일명 정정의 국소 점검에도 A/B를 참고했고, 전체 자료의 진단과 보류 결과를 보고 구간 규칙·후보 모델을 선택했습니다. 해당 결과는 적응적으로 탐색한 개발 재검증이며, 최종 모델의 편향 없는 외부 성능 추정치는 아닙니다.')
p('<b>이 검증으로 지지되는 주장:</b> 기존의 시간 공백 기반 연결과 반응식보다 v3가 현재 자료의 여러 부위와 보류 원점 노출을 더 일관되게 설명합니다. 수치 기울기 검사는 계산 구현의 일관성을 확인하고, 조건부 bootstrap은 고정한 규약 아래의 민감도를 보여 줍니다. 둘 다 물리적 참원점의 증명은 아닙니다.')
p('<b>물리적 정확도까지 확인하려면:</b> v3의 원점 정의·파라미터·특징·평가 규칙을 먼저 고정한 뒤, 별도 촬영 세션에서 원점 복귀와 공통 자세를 다시 촬영해야 합니다. 명령/Actual의 동기화와 독립 위치 기준이 있는 상태에서 영상으로 추정한 T를 비교해야 1 tick=0.01° 정확도 주장을 검사할 수 있습니다. 현 단계 모델 평가에서는 원래 명령 라벨 기준과 가정한 표준 좌표 기준을 각각 보고해야 합니다.')

page('7. 표준 원점 v3의 정의와 저장 결과')
p('표준 원점은 원점 기준 사진 4장의 추정 Y/T 평균을 (0,0)으로 고정한 가상 기준입니다. 10/15 영상은 기준 사진의 QR 공통 이동으로 10/4 좌표계에 등록합니다. 이 등록 이동의 물리적 원인을 카메라·기계 원점 중 하나로 특정하지 않았습니다.')
rows=[['관측','표준 dx(px)','표준 dy(px)','dx 조건부 구간','dy 조건부 구간']]
for k,name in enumerate(['좌측 QR','중앙 QR','우측 QR']):
    x=intervals.iloc[k*2];y=intervals.iloc[k*2+1];rows.append([name,f'{x.estimate:.4f}',f'{y.estimate:.4f}',f'{x.conditional_lo:.4f} ~ {x.conditional_hi:.4f}',f'{y.conditional_lo:.4f} ~ {y.conditional_hi:.4f}'])
tab(rows,[42,39,39,71,71])
p('표준 QR 630점은 standard_origin_v3_QR630.csv, 자연 특징 원점은 standard_origin_v3_natural.csv, 전체 575장의 원래 라벨·정정 라벨·표준 좌표·학습 가중치는 coordinates_standard_v3.csv에 저장했습니다. 원본 이미지는 삭제·이동·이름 변경하지 않았습니다.')
p('표의 구간은 기준 사진·좌표 규약·모델 선택을 고정하고 같은 자세 묶음을 40회 resampling한 조건부 bootstrap 구간입니다. 모델 선택과 기준 사진 자체의 불확실성을 포함하지 않으므로 물리적 원점 신뢰구간으로 해석하면 안 됩니다. 스캔 Y/T 바이어스 분해는 여전히 정규화와 범위 제한의 영향을 받습니다.')
p(f'<b>경계 민감도:</b> 최종 적합의 원시 구간 보정 변수 {j["raw_bound_hits"]}개가 설정한 ±3 탐색 경계에 도달했습니다. 원시 변수는 Y 사전척도 .05mm, T 사전척도 .01°로 환산한 뒤 원점 평균 제약을 적용합니다. 따라서 최종 구간 보정량을 물리적으로 식별한 참값으로 취급하면 안 됩니다. 모델 반영 시 이 좌표는 가정한 목표이며, 경계·사전척도 민감도와 원래 라벨 기준 성능을 함께 확인해야 합니다.',small)
fig('07_standard_origin_v3_features.png','표준 원점 v3에 실제 반영한 관측 영역','분홍은 원점 생성, 초록은 오른쪽 보류 검증, 파랑은 QR 모서리, 노랑은 상부 선 표본입니다. 사진은 사용 영역을 보여 주는 원본 center.bmp이며 가상 원점 영상 자체는 아닙니다.')

page('8. 이번 자료가 지지하는 결론과 남은 한계')
p('<b>지지하는 결론:</b> center3의 큰 잔차는 사진 불량의 증거로 취급하면 안 됐습니다. 원점 복귀를 시간 공백만으로 스캔에 묶은 규칙을 바꾸자 잔차가 크게 줄었습니다. 양의 Y 스캔의 부위별 체계적 잔차에는 더 유연한 Y 반응식이 도움이 됐습니다. 일부 반복 사진에는 공통 위치 변동이 있었지만 그 영향은 주원인보다 작았습니다.')
p('<b>표준 원점에 반영한 것:</b> 파생 라벨 1건 정정, 영상 연속성이 지지하는 .57 자료 복원, 원점 노출과 움직임 스캔의 바이어스 분리, 반복 공통 편차 soft weighting, Y4/Y5 반응 보완입니다. v3는 이 규약과 근거를 JSON·CSV로 재현할 수 있습니다.')
p('<b>아직 입증하지 못한 것:</b> 물리적 참원점의 유일 복원, 구간별 Y/T 보정량의 실제 장비 오차 여부, T 최대 1 tick 정확도, 새 촬영 세션의 성능입니다. 같은 자료를 보고 규칙을 선택했으므로 적응적 개발 편향이 남습니다. 검증의 개선과 물리적 원점의 확정은 구분해야 합니다.')
p('추후에는 원점 복귀·이동 방향·명령 및 Actual 기록의 시각을 이미지와 연결하면 이번에 드러난 모호함을 줄일 수 있습니다. 새로운 공통 자세 앵커와 실제 위치 기록은 v3를 유지하면서 독립적으로 검증하는 데 사용해야 합니다.')
p('재현 순서: forensics.py → refine_origin.py → export_v3.py → report_v3.py. 원본은 read-only로 사용했고 기존 Lanczos 계측 좌표를 재사용했습니다. QR 630점의 원점 좌표 평균과 내보낸 QR 특징값의 일치, 기울기 수치 검증, 평가 표의 중복 여부도 확인했습니다.',small)

def footer(c,doc):
    c.saveState();c.setStrokeColor(colors.HexColor('#d4dfe8'));c.line(15*mm,11*mm,282*mm,11*mm);c.setFont('Malgun',8);c.setFillColor(colors.HexColor('#667a8c'))
    c.drawString(15*mm,6*mm,'공통 좌표 연결 진단 · 표준 원점 v3 | 2026-11-03');c.drawRightString(282*mm,6*mm,str(doc.page));c.restoreState()
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True);target=OUT/'T_axis_common_origin_v3_audit_20261103.pdf'
doc=SimpleDocTemplate(str(target),pagesize=landscape(A4),leftMargin=15*mm,rightMargin=15*mm,topMargin=12*mm,bottomMargin=17*mm,
    title='공통 좌표 연결 단절과 아웃라이어 감사 및 표준 원점 v3',author='Codex')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(target);text='\n'.join(p.extract_text() for p in r.pages)
assert all(s in text for s in ['575','555','center3','아웃라이어','조건부'])
save('pdf_validation.json',{'pages':len(r.pages),'text_characters':len(text),'path':str(target)})
print('PDF',target,'PAGES',len(r.pages))
