# '기초 정보' 다음에 '분석에 필요한 정보' 2장(기준 위치, 이동 관계)을 추가한다
# 스타일은 기초 정보 슬라이드와 같음: 머리말(1-1 ...), 제목, 왼쪽 그림, 오른쪽 '항목 / 값 / 가는 선' 표
#   사용: python add_analysis_slides.py 입력.pptx 출력.pptx
import copy, sys, pathlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.oxml.ns import qn

HERE = pathlib.Path(__file__).parent
F, OUT = HERE / sys.argv[1], HERE / sys.argv[2]
SUB, LINE, FAINT = RGBColor.from_string('4B5563'), RGBColor.from_string('E5E7EB'), RGBColor.from_string('9CA3AF')
FONT, E = 'Noto Sans KR', 914400
KX, KW, VX, VW, LX, LW = 7.21, 1.3, 8.51, 4.03, 7.21, 5.33   # 기초 정보 표와 같은 위치
TOP, BOTTOM, LH = 2.03, 6.43, 0.2
FX, FW = 0.8, 5.78                                          # 왼쪽 그림 영역 (기초 정보의 QR 묶음과 같은 폭)

SLIDES = [
    dict(layout='CONTENT', title='분석에 필요한 정보: 기준 위치', img='ref_center.png', size=(1960, 770),
         cap='center 사진의 QR 3개. 파란 네모 = QR 영역, 점 = 중심 (좌표 단위 px)',
         rows=[('QR 중심 좌표', ['L (1642, 1068)', 'C (1900, 1068)', 'R (2159, 1071)']),
               ('QR 크기', ['약 160 × 172px (폭 × 높이)']),
               ('QR 간격', ['258px (L와 C), 259px (C와 R)']),
               ('QR 줄 기울기', ['R이 L보다 3px 낮음 (약 0.3°)']),
               ('사용 영역', ['QR 3개', '판 나사 2개는 비교 실험에서만']),
               ('판까지 거리', ['약 320mm (추정)', 'Θ가 도 단위라고 가정'])],
         notes='분석의 기준이 되는 center 사진의 QR 위치입니다. QR 3개의 중심 좌표, 크기, 간격을 쟀고, 오른쪽 QR이 왼쪽보다 3픽셀 낮아 카메라가 약 0.3도 기울어 있습니다. 분석에는 QR 3개만 쓰고, 판 나사는 비교 실험에서만 씁니다.'),
    dict(layout='CONTENT_GRAY', title='분석에 필요한 정보: 이동 관계', img='linear.png', size=(1960, 1330),
         cap='명령값과 QR 가로 이동 (QR 3개 평균). 오른쪽은 직선에서 벗어난 양을 확대',
         rows=[('Y 1mm', ['QR 약 14.2px 이동']),
               ('Θ 1', ['QR 약 80px 이동']),
               ('한 단계 이동', ['Y 0.05 ≈ 0.7px,  Y 0.1 ≈ 1.4px', 'Θ 0.01 ≈ 0.8px']),
               ('Y, Θ 겹침', ['Θ 0.178 ≈ Y 1mm', '평균 이동이 같아서 구분 안 됨']),
               ('구분 단서', ['약 0.05px (전체 이동의 0.4%)']),
               ('어려운 구간', ['|Y| 0.5mm 이하', '|Θ| 0.09 이하'])],
         notes='명령값과 QR 이동의 관계입니다. Y 1mm에 약 14.2픽셀, Θ 1에 약 80픽셀로 거의 직선입니다. 한 단계 이동은 1픽셀 안팎이고, Θ 0.178과 Y 1mm는 평균 이동이 같아 구분이 안 됩니다. 둘을 가르는 단서는 약 0.05픽셀이라 0 근처가 가장 어렵습니다.'),
]

prs = Presentation(F)
base = next(sl for sl in prs.slides if any(sh.has_text_frame and sh.text_frame.text == '기초 정보' for sh in sl.shapes))
base_idx = list(prs.slides).index(base)
layouts = {l.name: l for l in base.slide_layout.slide_master.slide_layouts}
keep = [sh for sh in base.shapes if (sh.has_text_frame and sh.text_frame.text in ('기초 정보',)) or (sh.has_text_frame and sh.text_frame.text.startswith('1-1')) or (sh.top / E < 0.55 and sh.height == 0) or (sh.is_placeholder and 'Slide Number' in sh.name)]

def text(s, x, y, w, h, lines, size, color=SUB):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

new = []
for d in SLIDES:
    s = prs.slides.add_slide(layouts[d['layout']])
    for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)   # 레이아웃이 넣은 빈 자리표시 지움
    for sh in keep: s.shapes._spTree.append(copy.deepcopy(sh._element))
    t = next(sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text == '기초 정보')
    t.text_frame.paragraphs[0].runs[0].text = d['title']
    # 왼쪽 그림: 영역 안에 비율 유지, 세로 가운데
    pw, ph = d['size']; k = min(FW / pw, (BOTTOM - TOP - 0.35) / ph); w, h = pw * k, ph * k
    y = TOP + (BOTTOM - TOP - 0.35 - h) / 2
    s.shapes.add_picture(str(HERE / 'img' / d['img']), Inches(FX), Inches(y), Inches(w), Inches(h))
    text(s, FX, y + h + 0.1, FW, 0.25, [d['cap']], 9, FAINT)
    # 오른쪽 표
    pad = (BOTTOM - TOP - sum(len(v) for _, v in d['rows']) * LH) / len(d['rows'])
    y = TOP
    for key, vals in d['rows']:
        text(s, KX, y + pad / 2 - 0.02, KW, 0.3, [key], 12)
        text(s, VX, y + pad / 2, VW, len(vals) * LH, vals, 11)
        y += len(vals) * LH + pad
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(LX), Inches(y), Inches(LX + LW), Inches(y))
        ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)
    ns = s.notes_slide
    if ns.notes_text_frame is None:   # 노트 자리표시가 없으면 기초 정보 슬라이드의 노트 본문을 복사해서 씀
        body = next(sh for sh in base.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
        ns.shapes._spTree.append(copy.deepcopy(body._element))
    ns.notes_text_frame.text = d['notes']
    new.append(s)

# 새 슬라이드를 기초 정보 바로 뒤로 옮김
lst = prs.slides._sldIdLst; ids = list(lst)
for i, el in enumerate(ids[-len(new):]):
    lst.remove(el); lst.insert(base_idx + 1 + i, el)
prs.save(OUT)
print('saved', OUT.name, 'slides', len(prs.slides))
