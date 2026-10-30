# 정리본 v3 끝에 '부록 8. ECC 기준 패치: 고정 창 vs 라벨 기준' 추가 (ecc_compare.py 결과)
#   이미 있으면 내용만 다시 그림
import copy, pathlib, sys, numpy as np, pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent; IMG = HERE / 'img'
V3 = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DECK / 'SEMES_01_데이터EDA_정리본_v3.pptx'
SUB, INK, LINE, FAINT = (RGBColor.from_string(c) for c in ['4B5563', '111827', 'E5E7EB', '9CA3AF'])
FONT, E = 'Noto Sans KR', 914400
TITLE = '부록 8. ECC 기준 패치: 고정 창 vs 라벨 기준'
T = pd.read_csv(IMG / 'ecc_compare.csv')
A = pd.read_csv(IMG / 'ecc_motion_oldbox.csv').set_index('file'); B = pd.read_csv(IMG / 'ecc_motion.csv').set_index('file')
d = (B.dx - A.dx).drop('center.bmp'); dy = (B.dy - A.dy).drop('center.bmp')

prs = Presentation(V3); S = list(prs.slides)
title_of = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.top is not None and 0.8 < sh.top / E < 1.2 and not sh.text_frame.text.startswith('1-1')), '')
s = next((x for x in S if title_of(x) == TITLE), None)
if s is None:   # 부록 7 장의 머리글, 제목, 쪽 번호만 복사해 새 장
    src = next(x for x in S if title_of(x).startswith('부록 7'))
    s = prs.slides.add_slide(src.slide_layout)
    for sh in list(s.shapes): sh._element.getparent().remove(sh._element)
    for sh in src.shapes:
        if sh.is_placeholder or (sh.has_text_frame and sh.text_frame.text.startswith('1-1')): s.shapes._spTree.append(copy.deepcopy(sh._element))
for sh in list(s.shapes):
    if not sh.is_placeholder and not (sh.has_text_frame and sh.text_frame.text.startswith('1-1')): sh._element.getparent().remove(sh._element)
tt = next(sh for sh in s.shapes if sh.has_text_frame and sh.top is not None and 0.8 < sh.top / E < 1.2 and not sh.text_frame.text.startswith('1-1'))
rs = [r for p in tt.text_frame.paragraphs for r in p.runs]; rs[0].text = TITLE
for r in rs[1:]: r.text = ''

def text(x, y, w, h, lines, size, color=SUB, bold=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)
def line(x0, x1, y):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y), Inches(x1), Inches(y)); ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)

# 왼쪽: 사진별 차이 그림 + 결론
iw, ih = Image.open(IMG / 'ecc_compare.png').size; h = 6.0 * ih / iw
s.shapes.add_picture(str(IMG / 'ecc_compare.png'), Inches(0.8), Inches(2.03), Inches(6.0), Inches(h))
text(0.8, 2.03 + h + 0.05, 6.0, 0.3, ['사진 280장의 가로 이동 차이. 회색, 파랑은 특징점(모서리 630점)과의 차이, 주황은 두 패치 방식끼리의 차이.'], 9, FAINT)
text(0.8, 2.03 + h + 0.55, 6.0, 1.3, [
    f'두 패치 방식의 사진별 이동 차이: 가로 최대 {d.abs().max():.4f}px (σ {d.std():.4f}), 세로 최대 {dy.abs().max():.4f}px',
    '사진 공통 흔들림(0.013 ~ 0.032px)보다 한 자리 작음',
    '→ 패치를 라벨 기준으로 바꿔도 이동량, 기울기, 잔차는 그대로. 맞춤 점수(상관계수)만 오름'], 11, INK)
# 오른쪽: 항목별 표
X = (7.21, 9.71, 11.11); Wc = (2.4, 1.35, 1.42); top = 2.03
for x, w, k in zip(X, Wc, ['항목', '고정 창', '라벨 기준']): text(x, top, w, 0.3, [k], 11, INK, True)
line(7.21, 12.54, top + 0.34); y = top + 0.34; rh = (6.43 - y) / len(T)
for r in T.itertuples(index=False):
    for x, w, v in zip(X, Wc, r): text(x, y + (rh - 0.2) / 2, w, 0.2, [str(v)], 10)
    y += rh; line(7.21, 12.54, y)
ns = s.notes_slide
if ns.notes_text_frame is None:   # 노트 마스터에 본문 자리가 없어서 다른 장의 메모 본문 상자를 복사
    body = next(sh for x in S if x.has_notes_slide for sh in x.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
    ns.shapes._spTree.append(copy.deepcopy(body._element))
ns.notes_text_frame.text = '''[측정]
presentation_eda_20261019/v2/ecc_measure.py          -> v2/img/ecc_motion.csv (라벨 기준: center 사진의 QR 라벨 + 바깥 8px)
presentation_eda_20261019/v2/ecc_measure.py --oldbox -> v2/img/ecc_motion_oldbox.csv (고정 창: 예전 measure_qr.py 의 180x192px 좌표)

[비교, 그림]
v2/ecc_compare.py -> v2/img/ecc_compare.csv, v2/img/ecc_compare.png

[방법]
- 같은 원본 281장, 같은 ECC 설정(템플릿 매칭 ±130px -> ECC 아핀, 가운데 점 이동). 다른 것은 기준 패치 위치와 크기뿐
- 기울기, 직선 잔차는 이동 경로 장과 같은 방법. 특징점과 차이는 모서리 630점 평균(photo_motion.csv)과 사진별 비교
- 고정 창은 라벨 테두리가 한쪽만 들어가는 등 QR마다 어긋나 있었음. 라벨 기준은 세 QR 같은 규칙'''
prs.save(V3); print('slides', len(prs.slides))
