# 덱 맨 뒤에 '모서리 맞춤 전수 결과' 7장 추가: 통과/불확실 기준 1장 + trial 6장(지도 + 사진별 막대)
#   사용: python add_corner_slides.py 입력.pptx 출력.pptx   (덱 폴더 기준 경로)
import copy, sys, pathlib, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; OUT_DIR = HERE / 'out'; DECK = HERE.parent / 'presentation_eda_20261019'
F, OUT = DECK / sys.argv[1], DECK / sys.argv[2]
SUB, LINE, FAINT, INK = (RGBColor.from_string(c) for c in ['4B5563', 'E5E7EB', '9CA3AF', '111827'])
FONT, E = 'Noto Sans KR', 914400
S = pd.read_csv(OUT_DIR / 'trial_summary.csv')
D = pd.read_csv(OUT_DIR / 'corners_280.csv')
cen = D[D.file == 'center.bmp']; why = D[~D.valid].reason.str.split(';').explode().value_counts()

prs = Presentation(F)
base = next(sl for sl in prs.slides if any(sh.has_text_frame and '이동 관계' in sh.text_frame.text for sh in sl.shapes))

def new_slide(title):
    s = prs.slides.add_slide(base.slide_layout)
    for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
    for sh in base.shapes:
        if sh.has_text_frame and (sh.text_frame.text.startswith('1-1') or '이동 관계' in sh.text_frame.text) or (sh.is_placeholder and 'Slide Number' in sh.name):
            s.shapes._spTree.append(copy.deepcopy(sh._element))
    t = next(sh for sh in s.shapes if sh.has_text_frame and '이동 관계' in sh.text_frame.text)
    rr = t.text_frame.paragraphs[0].runs; rr[0].text = title
    for r in rr[1:]: r.text = ''
    return s

def text(s, x, y, w, h, lines, size, color=SUB, bold=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

def rows_table(s, rows, x=0.8, w=11.73, kw=2.6, top=2.03, bottom=6.6, lh=0.2):
    pad = (bottom - top - sum(len(v) for _, v in rows) * lh) / len(rows); y = top
    for k, vals in rows:
        text(s, x, y + pad / 2 - 0.02, kw, 0.3, [k], 12)
        text(s, x + kw, y + pad / 2, w - kw, len(vals) * lh, vals, 11)
        y += len(vals) * lh + pad
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x + w), Inches(y))
        ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)

def notes(s, t):
    ns = s.notes_slide
    if ns.notes_text_frame is None:
        body = next(sh for sh in base.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
        ns.shapes._spTree.append(copy.deepcopy(body._element))
    ns.notes_text_frame.text = t

# 1) 통과 / 불확실 기준
s = new_slide('QR 격자 모서리 맞춤: 통과와 불확실 기준')
diag = int(((cen.black_neighbors == 2) & ~cen.valid).sum())
rows_table(s, [
    ('방법', ['QR 디코딩으로 얻은 격자에서 검정과 흰 칸이 만나는 모서리 755점 (center에서 ID 고정)',
             '사진마다 같은 점을 추적: 모서리 주변 밝기에 흐려진 2×2 검정/흰 경계 모델을 서브픽셀로 맞춤']),
    ('통과 조건 (모두 만족)', ['최적화 수렴  |  밝기 모델 오차(RMSE) ≤ 15  |  대비 ≥ 45',
                       '9×9 창과 11×11 창 결과 차이 ≤ 0.65px  |  시작점에서 3.6px 이내  |  흐림 σ ≤ 2.6']),
    ('불확실 이유 (전체)', [f'밝기 모델 오차 큼 {why.get("photometric_rmse", 0):,}건,  창 크기에 따라 위치가 달라짐 {why.get("window_instability", 0):,}건,  최적화 미수렴 {why.get("optimizer", 0):,}건',
                     '(점 하나가 사진 한 장에서 걸린 횟수, 한 번에 여러 이유 가능)']),
    ('항상 불확실한 점', [f'{diag}점: 검정 칸 두 개가 꼭짓점만 맞닿은 "대각선 모서리". 인쇄가 번져 2×2 모델과 모양이 달라서 모든 사진에서 오차가 큼']),
    ('center 결과', [f'{int(cen.valid.sum())} / {len(cen)}점 통과  (10/15 결과 608점과 차이: 창을 맞춘 위치로 다시 옮겨 맞추도록 바꿈)']),
], kw=2.4)
notes(s, '모서리 755점 각각에 흐려진 검정 흰 경계 모델을 맞추고, 여섯 가지 진단을 모두 통과하면 통과, 하나라도 걸리면 불확실로 표시했습니다. 항상 불확실한 125점은 검정 칸이 꼭짓점만 맞닿은 대각선 모서리라 모델과 모양이 다릅니다.')

# 2) trial 6장
TITLE = {'Y 0.1 간격 +': 'Y 0.1 간격 +쪽', 'Y 0.1 간격 -': 'Y 0.1 간격 -쪽', 'Y 0.05 홀수 +': 'Y 0.05 홀수 +쪽', 'Y 0.05 홀수 -': 'Y 0.05 홀수 -쪽', 'Θ +': 'Θ +쪽', 'Θ -': 'Θ -쪽'}
for i, r in enumerate(S.itertuples(), 1):
    s = new_slide(f'모서리 맞춤 전수 결과: {TITLE[r.trial]} ({r.photos}장)')
    mw = 9.6; mh = mw * 654 / 2151; mx = (13.333 - mw) / 2
    s.shapes.add_picture(str(OUT_DIR / f'map_{i}.png'), Inches(mx), Inches(1.9), Inches(mw), Inches(mh))
    text(s, mx, 1.9 + mh + 0.08, mw, 0.3,
         [f'초록 = 이 trial 모든 사진에서 통과 {r.always_ok}점,   주황 = 한 장이라도 불확실 {r.sometimes_bad + r.always_bad}점 (항상 불확실 {r.always_bad}점, 가끔 {r.sometimes_bad}점)'], 11, INK)
    bw = 11.73; bh = bw * 1.35 / 11.6
    s.shapes.add_picture(str(OUT_DIR / f'bar_{i}.png'), Inches(0.8), Inches(5.55), Inches(bw), Inches(bh))
    text(s, 0.8, 5.33, bw, 0.22, [f'사진마다 통과 점 수 (전수 {r.photos}장): {r.per_photo_min} ~ {r.per_photo_max}점,  전체 통과율 {r.valid_rate}%'], 9, FAINT)
    notes(s, f'{TITLE[r.trial]} {r.photos}장 전부에 같은 모서리 755점을 맞춘 결과입니다. 모든 사진에서 통과한 점은 {r.always_ok}점이고, 사진마다 통과 점 수는 {r.per_photo_min}점에서 {r.per_photo_max}점 사이입니다.')

prs.save(OUT)
print('saved', OUT.name, len(prs.slides), 'slides')
