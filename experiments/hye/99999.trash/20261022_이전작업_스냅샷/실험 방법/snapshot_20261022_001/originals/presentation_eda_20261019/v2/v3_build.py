# 정리본 v3: 사용자가 고친 SEMES_01_데이터EDA_정리본_v2.pptx (읽기만) -> SEMES_01_데이터EDA_정리본_v3.pptx
#   ① '이동 경로 (특징점)', '이동 경로 (ECC)' 그림을 배율 고정 그림으로 교체 (path_fixed.py), ECC 장 표와 메모 채움
#   ② 모든 발표 메모: [제목] 묶음 사이에 빈 줄 하나, 한 줄에 붙어 있던 '[..] .., [..] ..' 은 줄을 나눔
import copy, re, pathlib, numpy as np, pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent; IMG = HERE / 'img'
SRC, OUT = DECK / 'SEMES_01_데이터EDA_정리본_v2.pptx', DECK / 'SEMES_01_데이터EDA_정리본_v3.pptx'
SUB, LINE, FAINT = (RGBColor.from_string(c) for c in ['4B5563', 'E5E7EB', '9CA3AF'])
FONT, E = 'Noto Sans KR', 914400
KX, KW, VX, VW, LW = 7.21, 1.3, 8.51, 4.03, 5.33
prs = Presentation(SRC); S = list(prs.slides)
title_of = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.top is not None and 0.8 < sh.top / E < 1.2 and not sh.text_frame.text.startswith('1-1')), '')
sc = next(s for s in S if '이동 경로 (특징점)' in title_of(s)); se = next(s for s in S if '이동 경로 (ECC)' in title_of(s))

def text(s, x, y, w, h, lines, size, color=SUB):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

def put_picture(s, path):
    for sh in list(s.shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE: sh._element.getparent().remove(sh._element)
    w = 6.0; iw, ih = Image.open(path).size; h = w * ih / iw
    s.shapes.add_picture(str(path), Inches(0.8), Inches(1.95), Inches(w), Inches(h))
    for sh in s.shapes:   # 그림 아래 설명 글을 그림 바로 아래로
        if sh.has_text_frame and sh.text_frame.text.startswith(('모서리 630점', 'ECC', '선은')) and sh.left / E < 1: sh.top = Inches(1.95 + h + 0.05)
    return 1.95 + h

def clear_right(s):
    for sh in list(s.shapes):
        if sh.left is not None and sh.left / E >= 6.95 and sh.top / E > 1.8 and not sh.is_placeholder and sh.shape_type != MSO_SHAPE_TYPE.PICTURE:
            sh._element.getparent().remove(sh._element)

def rows_table(s, rows, top=2.03, bottom=6.43, lh=0.2):
    pad = (bottom - top - sum(len(v) for _, v in rows) * lh) / len(rows); y = top
    for k, vals in rows:
        text(s, KX, y + pad / 2 - 0.02, KW, 0.3, [k], 12); text(s, VX, y + pad / 2, VW, len(vals) * lh, vals, 11)
        y += len(vals) * lh + pad
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(KX), Inches(y), Inches(KX + LW), Inches(y)); ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)

def set_notes(s, t):
    ns = s.notes_slide
    if ns.notes_text_frame is None:
        body = next(sh for sh in sc.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
        ns.shapes._spTree.append(copy.deepcopy(body._element))
    ns.notes_text_frame.text = t

# ---- 이동 경로 (특징점): 그림만 교체, 설명 글 고침
end = put_picture(sc, IMG / 'path_corner.png')
cap = [sh for sh in sc.shapes if sh.has_text_frame and sh.text_frame.text.startswith('모서리 630점')]
if cap:
    rr = cap[0].text_frame.paragraphs[0].runs; rr[0].text = '모서리 630점의 사진별 평균 이동. 가로 ×1, 세로 ×100 (ECC 장과 같은 축). 선은 명령값 크기 순 연결이고 촬영 시간 순 궤적이 아님.'
    for r in rr[1:]: r.text = ''
    cap[0].top = Inches(end + 0.05)

t6 = sc.notes_slide.notes_text_frame.text.replace('[그림] v2_figs.py -> img/path.png', '[그림] path_fixed.py -> img/path_corner.png (ECC 장과 같은 축, 같은 배율)')
sc.notes_slide.notes_text_frame.text = t6

# ---- 이동 경로 (ECC): 그림, 설명, 표
Z = pd.read_csv(IMG / 'path_ecc_stats.csv').set_index('trial'); y = Z[Z.axis == 'Y']; t = Z[Z.axis == 'Θ']
Q = pd.read_csv(DECK.parent / '14_데이터_QR마커위치.csv')
end = put_picture(se, IMG / 'path_ecc.png')
for sh in list(se.shapes):
    if sh.has_text_frame and sh.left / E < 1 and sh.top / E > 5.0 and not sh.text_frame.text.startswith('1-1'): sh._element.getparent().remove(sh._element)
text(se, 0.8, end + 0.05, 6.0, 0.4, ['ECC: QR 3개 패치를 center에 맞춘 이동의 평균. 가로 ×1, 세로 ×100 (특징점 장과 같은 축).'], 9, FAINT)
clear_right(se)
rows_table(se, [('Y 기울기', [f'trial별 {y.slope_px.min():.2f} ~ {y.slope_px.max():.2f}px / 1mm']),
                ('Θ 기울기', [f'+쪽 {Z.slope_px["Θ +"]:.2f}px, -쪽 {Z.slope_px["Θ -"]:.2f}px / Θ 1', f'(+쪽과 -쪽이 {100 * abs(Z.slope_px["Θ -"] / Z.slope_px["Θ +"] - 1):.1f}% 다름)']),
                ('직선 잔차', [f'Y {y.resid_px.min():.3f} ~ {y.resid_px.max():.3f}px,  Θ {t.resid_px.min():.3f} ~ {t.resid_px.max():.3f}px (σ)']),
                ('세로 이동', [f'가로 이동의 {100 * Z.dy_slope.iloc[0]:.2f}% (이동 방향 {np.degrees(np.arctan(Z.dy_slope.iloc[0])):.2f}° 기울어짐)']),
                ('특징점과 차이', ['사진마다 가로 최대 0.09px, 세로 최대 0.015px', '경로 모양과 숫자가 거의 같음'])])
set_notes(se, '''[숫자]
presentation_eda_20261019/v2/path_fixed.py -> img/path_ecc_stats.csv

[그림]
path_fixed.py -> img/path_ecc.png (특징점 장 img/path_corner.png 와 같은 축, 같은 배율)

[원본 측정]
14_데이터_QR마커위치.csv (analysis_agent/measure_qr.py 로 만든 ECC 측정)

[방법]
① center 사진에서 QR 3개를 고정 창(180x192px)으로 잘라 기준으로 둠
② 다른 사진마다 템플릿 매칭으로 대략 위치를 찾고, ECC 아핀으로 QR 패치 전체를 center 패치에 맞춤
③ QR 기준점을 그 변환으로 옮긴 양 = 그 QR의 이동 (dLx, dCx, dRx / dLy, dCy, dRy)
④ QR 3개 평균을 사진의 이동으로 보고, trial마다 명령값 크기 순으로 이어 그림
- 기울기, 직선 잔차, 세로 이동 비율은 특징점 장과 같은 방법으로 계산''')

# ---- 모든 메모: [제목] 묶음 사이 빈 줄
for s in prs.slides:
    if not s.has_notes_slide or s.notes_slide.notes_text_frame is None: continue
    t0 = s.notes_slide.notes_text_frame.text
    t1 = re.sub(r',\s*(\[[가-힣 ]+\])', r'\n\1', t0)                  # 한 줄에 붙은 '[숫자] .., [그림] ..' 나누기 (한글 제목만)
    t1 = re.sub(r'^(\[[가-힣 ]+\])[ \t]+', r'\1\n', t1, flags=re.M)  # 줄 맨 앞 '[제목] 내용' -> '[제목]' 다음 줄에 내용 (경로 속 [ ]는 그대로)
    lines = [l.rstrip() for l in t1.split('\n')]
    out = []
    for l in lines:
        if re.fullmatch(r'\[[가-힣 ]+\]', l) and out and out[-1] != '': out.append('')
        if l == '' and (not out or out[-1] == ''): continue
        out.append(l)
    t1 = '\n'.join(out).strip()
    if t1 != t0: s.notes_slide.notes_text_frame.text = t1
prs.save(OUT); print('saved', OUT.name, len(prs.slides))
