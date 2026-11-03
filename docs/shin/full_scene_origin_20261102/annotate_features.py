from pathlib import Path
import json,sys,re,collections,math
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont,ImageOps
import cv2
from reportlab.platypus import SimpleDocTemplate,Paragraph,Image as RLImage,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import landscape,A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(P));from model_scene import load,blocks
d,features,array=load();quality=pd.read_csv(P/'directional_feature_quality.csv').set_index('feature_id')
j=json.loads((P/'standard_origin_v2.json').read_text())
A=[q['feature_id'] for q in j['origin_blocks'] if q['feature_id'] is not None]
specs,_=blocks(d[~d.suspect],features,array,'B_right',True);B=[s['fid'] for s in specs if s['fid'] is not None]
raw=Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set\center.bmp')
original=Image.open(raw).convert('RGB')
fontpath=r'C:\Windows\Fonts\malgun.ttf';boldpath=r'C:\Windows\Fonts\malgunbd.ttf'
def font(size,bold=False):return ImageFont.truetype(boldpath if bold else fontpath,size)
BLUE='#308bff';GOLD='#ffc107';PINK='#ec47b8';GREEN='#00cf91';WHITE='#ffffff';DARK='#102131'
idx=int(d.loc[d.path=='center.bmp','index'].iloc[0]);fs=cv2.FileStorage(str(ROOT/'analysis/session_eda_20261102/remeasured/cache'/f'frame{idx}.yml.gz'),cv2.FILE_STORAGE_READ)
qr=fs.getNode('points').mat();edges=fs.getNode('edges').mat();fs.release()

photo=original.copy();draw=ImageDraw.Draw(photo);occupied=[]
def label(draw,xy,text,color,size=21):
    origin=xy;chosen=None
    for dx,dy in [(0,0),(0,-30),(45,0),(-45,0),(0,35),(60,-35),(-65,-35),(60,35),(-65,35)]:
        pos=(xy[0]+dx,xy[1]+dy);box=draw.textbbox(pos,text,font=font(size,True));box=(box[0]-4,box[1]-3,box[2]+4,box[3]+3)
        if not any(box[0]<b[2] and box[2]>b[0] and box[1]<b[3] and box[3]>b[1] for b in occupied):chosen=(pos,box);break
    if chosen is None:
        pos=xy;box=draw.textbbox(pos,text,font=font(size,True));chosen=(pos,(box[0]-4,box[1]-3,box[2]+4,box[3]+3))
    pos,box=chosen
    if pos!=origin:draw.line((origin[0]+12,origin[1]+45,pos[0]+12,pos[1]+25),fill=color,width=1)
    draw.rectangle(box,fill=DARK);draw.text(pos,text,font=font(size,True),fill=color);occupied.append(box)
for k in range(3):
    a=qr[qr[:,2]==k,:2];left,top=a.min(0)-8;right,bottom=a.max(0)+8
    draw.rectangle((left,top,right,bottom),outline=BLUE,width=4)
    label(draw,(left,top-38),f'QR{k}  실제 모서리 {len(a)}점',BLUE,20)
    for x,y in a:draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill=BLUE)
good=edges[:60,5]>0
for x,y,*_ in edges[:60][good]:draw.rectangle((x-3,y-7,x+3,y+7),outline=GOLD,width=2)
label(draw,(260,655),'L0: 상부 수평선의 유효 7×15 표본',GOLD,24)
for ids,color,role in [(A,PINK,'A'),(B,GREEN,'B')]:
    for fid in ids:
        f=features.iloc[fid];x,y=f.reference_x,f.reference_y
        draw.rectangle((x-20,y-20,x+20,y+20),outline=color,width=4)
        draw.line((x-4,y,x+4,y),fill=color,width=2);draw.line((x,y-4,x,y+4),fill=color,width=2)
        label(draw,(x-22,y-52),f'{role}{fid:02}',color,22)

def add_header(image,title,sub):
    out=Image.new('RGB',(image.width,image.height+175),DARK);out.paste(image,(0,175));dr=ImageDraw.Draw(out)
    dr.text((40,20),title,font=font(44,True),fill=WHITE)
    dr.text((40,87),sub,font=font(27),fill='#d4e3ee')
    return out
overview=add_header(photo,'실제로 사용한 특징: 원점 생성과 검증을 구분','파랑 QR 모서리 630점  |  노랑 상부선 표본  |  분홍 A: 원점 생성 22개  |  초록 B: 생성에 쓰지 않은 검증 17개')
overview.save(P/'features_on_original_photo.png')
center=photo.crop((1340,270,2580,1560));center=ImageOps.scale(center,1.6,Image.Resampling.LANCZOS)
add_header(center,'지그·QR·나사 확대','네모는 실제 정합 영역입니다. A/B 중심은 패치 기준 위치이며, 기하학적 꼭짓점이라는 뜻은 아닙니다.').save(P/'features_central_zoom.png')

bg=Image.new('RGB',(2200,1450),DARK);dr=ImageDraw.Draw(bg)
dr.text((40,20),'배경 나사·구멍 확대: 왼쪽은 원점 생성, 오른쪽은 보류 검증',font=font(34,True),fill=WHITE)
for col,(box,title) in enumerate([((820,1250,1180,1815),'왼쪽 패널·하부 구조: A'),((2630,1250,3040,1815),'오른쪽 패널·하부 구조: B')]):
    crop=photo.crop(box);crop=ImageOps.contain(crop,(1000,1250),Image.Resampling.LANCZOS)
    x=50+col*1100;bg.paste(crop,(x,150));dr.text((x,95),title,font=font(29,True),fill=PINK if col==0 else GREEN)
bg.save(P/'features_background_zoom.png')

def contact(ids,role,color,outname):
    cols=6;rows=math.ceil(len(ids)/cols);tw,th=320,330
    sheet=Image.new('RGB',(cols*tw,rows*th+190),DARK);dr=ImageDraw.Draw(sheet)
    title='A: 공통 원점 생성에 사용한 자연 특징' if role=='A' else 'B: 원점 생성에서 보류한 검증 특징'
    dr.text((25,20),title,font=font(36,True),fill=color)
    dr.text((25,78),'색 네모 = 실제 41×41 정합 패치 / 주변 사진은 위치 확인용 확대',font=font(25),fill=WHITE)
    dr.text((25,120),'한 점의 실제 위치오차를 보증하는 표시는 아닙니다. 각 영상에서 품질 검사 후 사용했습니다.',font=font(21),fill='#cfdae5')
    for i,fid in enumerate(ids):
        f=features.iloc[fid];x,y=int(f.reference_x),int(f.reference_y)
        crop=original.crop((x-65,y-65,x+66,y+66)).resize((250,250),Image.Resampling.LANCZOS)
        cd=ImageDraw.Draw(crop);scale=250/131
        cd.rectangle((45*scale,45*scale,85*scale,85*scale),outline=color,width=4)
        tx=(i%cols)*tw+30;ty=(i//cols)*th+185
        sheet.paste(crop,(tx,ty));dr.text((tx,ty+254),f'{role}{fid:02}  ({x}, {y})',font=font(22,True),fill=color)
        rate=quality.loc[fid,'strict_valid_fraction']*100
        dr.text((tx,ty+285),f'개별 품질 검사 유효: {rate:.1f}%',font=font(20),fill=WHITE)
    sheet.save(P/outname)
contact(A,'A',PINK,'features_A_all_patches.png');contact(B,'B',GREEN,'features_B_all_patches.png')

sheet=Image.new('RGB',(2100,1000),DARK);sd=ImageDraw.Draw(sheet)
counts=[int(np.sum(qr[:,2]==k)) for k in range(3)]
sd.text((30,20),f'QR 실제 사용점 확대: {counts[0]} + {counts[1]} + {counts[2]} = 630개 모서리',font=font(38,True),fill=BLUE)
sd.text((30,85),'파란 점 = 추출 좌표 / 주황 네모 = 9×9 정합 패치 예시 1개',font=font(27),fill=WHITE)
for k in range(3):
    a=qr[qr[:,2]==k,:2];left,top=np.floor(a.min(0)-14).astype(int);right,bottom=np.ceil(a.max(0)+14).astype(int)
    crop=original.crop((left,top,right,bottom));dr=ImageDraw.Draw(crop)
    for x,y in a:dr.ellipse((x-left-1,y-top-1,x-left+1,y-top+1),fill=BLUE)
    x,y=a[0];dr.rectangle((x-left-4,y-top-4,x-left+4,y-top+4),outline=GOLD,width=1)
    crop=ImageOps.contain(crop,(640,740),Image.Resampling.LANCZOS);sheet.paste(crop,(k*700+20,180))
    sd.text((k*700+30,135),f'QR{k}: {len(a)}점',font=font(30,True),fill=BLUE)
sheet.save(P/'features_QR_actual_corners.png')

# A mapping table connects every displayed ID to a real pixel coordinate and its role.
mapping=[]
for ids,role in [(A,'origin_construction_A'),(B,'heldout_validation_B')]:
    for fid in ids:
        f=features.iloc[fid];mapping.append({'display_id':f'{"A" if role.startswith("origin") else "B"}{fid:02}',
            'feature_id':fid,'role':role,'reference_x':f.reference_x,'reference_y':f.reference_y,'patch_width':41,'patch_height':41,
            'strict_valid_fraction':quality.loc[fid,'strict_valid_fraction']})
pd.DataFrame(mapping).to_csv(P/'annotated_feature_ID_map.csv',index=False)

pdfmetrics.registerFont(TTFont('Malgun',fontpath));pdfmetrics.registerFont(TTFont('MalgunBold',boldpath))
pdfmetrics.registerFontFamily('Malgun',normal='Malgun',bold='MalgunBold',italic='Malgun',boldItalic='MalgunBold')
head=ParagraphStyle('head',fontName='MalgunBold',fontSize=18,leading=25,spaceAfter=8)
cap=ParagraphStyle('cap',fontName='Malgun',fontSize=9,leading=14,wordWrap='CJK')
story=[]
pages=[('features_on_original_photo.png','사진 위 실제 사용 영역','파랑은 QR 모서리, 노랑은 상부 수평선 표본, 분홍 A는 원점 생성, 초록 B는 생성에 쓰지 않은 검증 특징입니다. 아래쪽 수평선과 회색 탈락 후보는 사용 사진에 표시하지 않았습니다.'),
('features_central_zoom.png','지그·QR·체결부 확대','A와 B의 네모는 41×41 범위이며, 그 안에서 2px 간격의 21×21 밝기 표본을 정합했습니다. 중심은 패치 기준 위치이며 꼭짓점이라는 뜻은 아닙니다. 각 영상에서 방향별 불확실성과 정·역방향 검사를 통과한 관측만 사용했습니다.'),
('features_background_zoom.png','배경 나사·구멍 확대','배경이라는 이유만으로 고정물로 가정하지 않았습니다. 현재 영상에서는 배경 구멍도 큰 움직임을 보이므로, 각 특징의 영상 반응을 따로 적합했습니다.'),
('features_A_all_patches.png','원점 생성에 쓴 A 특징 전체','패치 바깥의 주변 영상은 위치 확인을 위한 확대입니다. 유효 비율은 575장에 개별 품질 검사를 적용한 비율입니다. 잘 보이는 구멍이라도 반사·가림·대응 불안정 때문에 항상 유효한 것은 아닙니다.'),
('features_B_all_patches.png','검증용으로 보류한 B 특징 전체','B 특징은 공통 원점·스캔 바이어스 최적화에 넣지 않았습니다. 학습 자세에서 별도 반응식을 적합한 뒤 보류 사진에서 검사했습니다. A와 B의 측정 패치는 겹치지 않습니다.'),
('features_QR_actual_corners.png','QR 630점의 실제 위치','QR 좌표는 원본 center.bmp에서 추출한 위치입니다. 생성 템플릿의 630개 점은 모두 흑백 4분면의 1개 또는 3개가 다른 모서리 패턴이며, 직선만 있는 패턴은 없습니다. 주황 네모는 실제 9×9 정합 패치 크기 예시입니다.')]
for i,(name,title,caption) in enumerate(pages):
    if i:story.append(PageBreak())
    story.append(Paragraph(title,head));w,h=Image.open(P/name).size;scale=min(264*mm/w,157*mm/h)
    story.append(RLImage(str(P/name),width=w*scale,height=h*scale));story.append(Spacer(1,2*mm));story.append(Paragraph(caption,cap))
def foot(c,doc):c.setFont('Malgun',8);c.drawString(14*mm,6*mm,'사용 특징 표시 | 원본: center.bmp | 2026-11-02');c.drawRightString(283*mm,6*mm,str(doc.page))
OUT=ROOT/'output/pdf';out=OUT/'T_axis_used_features_photo_atlas.pdf'
doc=SimpleDocTemplate(str(out),pagesize=landscape(A4),leftMargin=14*mm,rightMargin=14*mm,topMargin=10*mm,bottomMargin=15*mm,title='실제 사용 특징의 사진 표시',author='Codex')
doc.build(story,onFirstPage=foot,onLaterPages=foot)
assert len(PdfReader(out).pages)==6
print('PDF',out,'PAGES',6,'A',len(A),'B',len(B))
