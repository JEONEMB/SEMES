from pathlib import Path
from html import escape
import re
import pandas as pd
from html.parser import HTMLParser
class Node:
    def __init__(self,name,attrs=()):self.name=name;self.attrs=dict(attrs);self.children=[]
    def get(self,key,default=None):
        val=self.attrs.get(key,default)
        return val.split() if key=='class' and isinstance(val,str) else val
    def __getitem__(self,key):return self.attrs[key]
    def get_text(self,sep=' ',strip=True):
        values=[x.get_text(sep,strip) if isinstance(x,Node) else x for x in self.children]
        return sep.join(x.strip() if strip else x for x in values if x.strip())
    def find_all(self,name,recursive=True):
        out=[]
        for child in self.children:
            if isinstance(child,Node):
                if child.name==name:out.append(child)
                if recursive:out.extend(child.find_all(name))
        return out
    def find(self,name):return next(iter(self.find_all(name)),None)
    def select(self,selector):
        if selector=='main > section':return self.find('main').find_all('section',recursive=False)
        return self.find_all(selector)
    def select_one(self,selector):
        if selector=='header .lead':return next(n for n in self.find('header').find_all('p') if 'lead' in n.get('class',[]))
        if selector=='.callout':return next(n for n in self.find_all('section') if 'callout' in n.get('class',[]))
        return self.find(selector)
class Parser(HTMLParser):
    def __init__(self):super().__init__();self.root=Node('root');self.stack=[self.root]
    def handle_starttag(self,tag,attrs):
        n=Node(tag,attrs);self.stack[-1].children.append(n)
        if tag not in ['meta','img','br','hr','input','link']:self.stack.append(n)
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].name==tag:self.stack=self.stack[:i];break
    def handle_data(self,data):self.stack[-1].children.append(data)
def BeautifulSoup(data,unused):
    p=Parser();p.feed(data);return p.root
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
from pypdf import PdfReader

P=Path(__file__).resolve().parent
W=P.parents[1]
OUT=W/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
TMP=W/'tmp/pdfs';TMP.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Malgun',r'C:\Windows\Fonts\malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold',r'C:\Windows\Fonts\malgunbd.ttf'))
pdfmetrics.registerFontFamily('Malgun',normal='Malgun',bold='MalgunBold',italic='Malgun',boldItalic='MalgunBold')
body=ParagraphStyle('body',fontName='Malgun',fontSize=10.5,leading=17,spaceAfter=9,wordWrap='CJK',textColor=colors.HexColor('#243247'))
h1=ParagraphStyle('h1',parent=body,fontName='MalgunBold',fontSize=25,leading=35,spaceAfter=18)
h2=ParagraphStyle('h2',parent=body,fontName='MalgunBold',fontSize=17,leading=25,spaceAfter=14)
cap=ParagraphStyle('caption',parent=body,fontSize=9,leading=14,textColor=colors.HexColor('#52647a'))
cell=ParagraphStyle('cell',parent=body,fontSize=8,leading=12,spaceAfter=0)
story=[]
soup=BeautifulSoup((P/'report.html').read_text(encoding='utf-8'),'html.parser')
def txt(node):
    s=escape(node.get_text(' ',strip=True))
    return s.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
def para(node,style=body):return Paragraph(txt(node),style)
story.append(Paragraph('T축 촬영 세션·원점 EDA',h1))
story.append(Paragraph('원본 575장 전수 재측정 | 기준 영상 4장 | 기존 OOF 평가 571장',body))
story.append(para(soup.select_one('header .lead')))
story.append(Spacer(1,5*mm))
for li in soup.select_one('.callout').select('li'):
    story.append(Paragraph('• '+txt(li),body))
story.append(Spacer(1,5*mm))
story.append(Paragraph('분석 기준일: 2026-11-02 · Y: mm · T: degree · 1 tick = 0.01°',cap))
story.append(Paragraph('이 PDF에는 웹 보고서의 분석 내용, 세션 통계, 그래프 6개를 포함했습니다. 원본 이미지와 라벨은 변경하지 않았습니다.',body))

def figure(node):
    name=node.find('img')['src'];im=P/name
    story.append(PageBreak())
    story.append(Paragraph({'01':'시간 메타데이터와 세션 후보','02':'세션별 오차와 편향','03':'원점 차이의 근거와 식별 한계','04':'반복 촬영 변동과 선 품질','05':'파일명·폴더 라벨 불일치','06':'각 촬영 세션 후보의 오차 경향'}[name[:2]],h2))
    iw,ih=PILImage.open(im).size
    scale=min(258*mm/iw,148*mm/ih)
    story.append(Image(str(im),width=iw*scale,height=ih*scale))
    story.append(Spacer(1,2*mm));story.append(para(node.find('figcaption'),cap))

for sec in soup.select('main > section'):
    if 'callout' in sec.get('class',[]):continue
    if sec.get('id')=='files':continue
    story.append(PageBreak())
    heading=sec.find('h2');story.append(para(heading,h2))
    for node in sec.children:
        if not getattr(node,'name',None):continue
        if node.name=='p':story.append(para(node))
        elif node.name=='ul':
            for li in node.find_all('li',recursive=False):story.append(Paragraph('• '+txt(li),body))
    for node in sec.find_all('figure',recursive=False):figure(node)
    if sec.get('id')=='time':
        story.append(PageBreak());story.append(Paragraph('촬영 세션 후보별 통계',h2))
        df=pd.read_csv(P/'session_summary.csv')
        data=[['후보','기록 일시(KST)','이미지/평가','Y 범위(mm)','T 범위(°)','편향(tick)','MAE(tick)','최대(tick)','1tick 통과']]
        for r in df.itertuples():
            f=lambda v:'-' if pd.isna(v) else f'{v:.3f}'
            data.append([f'S{r.session:02}',r.start_KST[:19]+' ~ '+r.end_KST[11:19],f'{r.images}/{r.evaluated}',f'{r.Y_min:.2f} ~ {r.Y_max:.2f}',f'{r.T_min:.2f} ~ {r.T_max:.2f}',f(r.bias_ticks),f(r.MAE_ticks),f(r.max_abs_ticks),f'{r.pass_1tick}/{r.evaluated}' if r.evaluated else '-'])
        data=[[Paragraph(escape(str(x)),cell) for x in row] for row in data]
        table=Table(data,colWidths=[15*mm,60*mm,24*mm,31*mm,31*mm,23*mm,23*mm,23*mm,23*mm],repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeaf3')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#dce3eb'))]))
        story.append(table);story.append(Spacer(1,3*mm));story.append(Paragraph('14개는 수정 시각의 120초 초과 공백으로 나눈 후보이며, 실제 원점 재설정 횟수가 아닙니다. 기준 전용 S06·S07에는 평가 점수가 없습니다.',cap))
    if sec.get('id')=='origin':
        story.append(PageBreak());story.append(Paragraph('원점 기준 영상의 픽셀 차이',h2))
        d=pd.read_csv(P/'reference_pixel_displacements.csv')
        data=[['영상','평균 dx(px)','평균 dy(px)','이동 크기(px)','이동 이외 RMS(px)']]
        for row in d.itertuples():data.append([Path(row.path).name]+[f'{v:.4f}' for v in row[2:]])
        table=Table(data,colWidths=[64*mm,47*mm,47*mm,47*mm,47*mm])
        table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Malgun'),('FONTSIZE',(0,0),(-1,-1),10),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeaf3')),('TOPPADDING',(0,0),(-1,-1),13),('BOTTOMPADDING',(0,0),(-1,-1),13),('LINEBELOW',(0,0),(-1,-1),.5,colors.HexColor('#dce3eb'))]))
        story.append(table);story.append(Spacer(1,7*mm));story.append(Paragraph('10/4 center.bmp 대비 동일 QR 630점의 평균 이동입니다. 픽셀 차이는 확인되지만, 이를 T축 원점 이동량으로 안정적으로 환산할 수 없습니다.',body))

def footer(c,doc):
    c.saveState();c.setStrokeColor(colors.HexColor('#ccd7e2'));c.line(15*mm,11*mm,282*mm,11*mm)
    c.setFont('Malgun',8);c.setFillColor(colors.HexColor('#63758a'))
    c.drawString(15*mm,6*mm,'촬영 세션·원점 EDA | 2026-11-02')
    c.drawRightString(282*mm,6*mm,str(doc.page));c.restoreState()
target=OUT/'T_axis_session_origin_EDA.pdf'
doc=SimpleDocTemplate(str(target),pagesize=landscape(A4),rightMargin=15*mm,leftMargin=15*mm,topMargin=12*mm,bottomMargin=17*mm,title='T축 촬영 세션·원점 EDA',author='Codex')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(target)
text='\n'.join(p.extract_text() for p in r.pages)
assert len(r.pages)>=10
assert all(key in text for key in ['575','Lanczos','0.729','0.709','0.57'])
print(str(target));print('PAGES',len(r.pages),'TEXT',len(text))
(TMP/'page_text.txt').write_text(text,encoding='utf-8')
