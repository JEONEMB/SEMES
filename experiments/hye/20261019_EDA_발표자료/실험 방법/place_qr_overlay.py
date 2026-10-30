# 기초 정보 슬라이드 왼쪽 QR 이미지 묶음을 '구간당 한 장 겹침 그림'으로 바꾼다 (오른쪽 표, 제목 등은 그대로)
#   사용: python place_qr_overlay.py 입력.pptx 출력.pptx
import sys, pathlib, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

HERE = pathlib.Path(__file__).parent
F, OUT = HERE / sys.argv[1], HERE / sys.argv[2]
IDX = pd.read_csv(HERE / 'img' / 'qrov_index.csv').set_index('img')
INK, SUB, FAINT = (RGBColor.from_string(c) for c in ['111827', '4B5563', '9CA3AF'])
FONT, E = 'Noto Sans KR', 914400

prs = Presentation(F)
s = next(sl for sl in prs.slides if any(sh.has_text_frame and sh.text_frame.text == '기초 정보' for sh in sl.shapes))
grp = next(sh for sh in s.shapes if sh.shape_type == MSO_SHAPE_TYPE.GROUP)
gx, gy = grp.left / E, grp.top / E
grp._element.getparent().remove(grp._element)

def text(x, y, w, h, runs, size, align=PP_ALIGN.LEFT, anchor_mid=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    if anchor_mid:
        from pptx.enum.text import MSO_ANCHOR; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = align
    for t, color, bold, sz in runs:
        r = p.add_run(); r.text = t; r.font.size = Pt(sz or size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

labW, cw, gap = 1.05, 2.0, 0.18
ch = cw * 200 / 360
cols = [gx + labW, gx + labW + cw + gap]
text(cols[0], gy, cw, 0.28, [('-쪽', INK, True, None)], 11, PP_ALIGN.CENTER)
text(cols[1], gy, cw, 0.28, [('+쪽', INK, True, None)], 11, PP_ALIGN.CENTER)
y = gy + 0.38
for lab, key in [('Y 0.1 간격', 'Y01'), ('Y 0.05 홀수', 'Y005'), ('Θ', 'T')]:
    text(gx, y, labW, ch, [(lab, INK, True, None)], 11, anchor_mid=True)
    for j, sg in enumerate('mp'):
        img = f'qrov_{key}{sg}.png'; r = IDX.loc[img]
        s.shapes.add_picture(str(HERE / 'img' / img), Inches(cols[j]), Inches(y), Inches(cw), Inches(ch))
        v = lambda f: f[1:]   # 축은 행 이름에 있으니 값만
        text(cols[j], y + ch + 0.04, cw, 0.25,
             [(f'{v(r["first"])} → {v(r["last"])}', SUB, False, None), (f'   {r.first_dx_px:+.1f} → {r.last_dx_px:+.1f}px', FAINT, False, 8)],
             9.5, PP_ALIGN.CENTER)
    y += ch + 0.42
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith('파란 '):
        runs = sh.text_frame.paragraphs[0].runs
        runs[0].text = '파란 실선 = center의 QR 위치,  초록 점선 = 첫 이동 (center와 1px 안팎이라 거의 겹침),  빨강 실선 = 최대 이동,  px = 가로 이동량'
        for r in runs[1:]: r.text = ''
prs.save(OUT)
print('saved', OUT.name)
