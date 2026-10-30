# 덱 맨 뒤에 '쉽게 보기: 280장 중 틀린 장수' 1장 추가 (그림은 easy_figure.py 의 슬라이드용)
#   사용: python add_easy_slide.py 입력.pptx 출력.pptx   (presentation_eda_20261019 폴더 기준)
import copy, sys, pathlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent / 'presentation_eda_20261019'
F, OUT = DECK / sys.argv[1], DECK / sys.argv[2]
SUB, INK = RGBColor.from_string('4B5563'), RGBColor.from_string('111827')
FONT = 'Noto Sans KR'

prs = Presentation(F)
base = next(sl for sl in prs.slides if any(sh.has_text_frame and '이동 관계' in sh.text_frame.text for sh in sl.shapes))
s = prs.slides.add_slide(base.slide_layout)
for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
for sh in base.shapes:
    if sh.has_text_frame and (sh.text_frame.text.startswith('1-1') or '이동 관계' in sh.text_frame.text) or (sh.is_placeholder and 'Slide Number' in sh.name):
        s.shapes._spTree.append(copy.deepcopy(sh._element))
t = next(sh for sh in s.shapes if sh.has_text_frame and '이동 관계' in sh.text_frame.text)
rr = t.text_frame.paragraphs[0].runs; rr[0].text = '사진 280장 중 Y/Θ를 틀린 장수'
for r in rr[1:]: r.text = ''

def text(x, y, w, h, lines, size, color=SUB, bold=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(6)
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

w = 7.4; h = w * 4.2 / 8
s.shapes.add_picture(str(HERE / '쉽게보기_틀린장수_슬라이드.png'), Inches(0.8), Inches(2.1), Inches(w), Inches(h))
text(8.7, 2.4, 3.83, 0.4, ['한 줄 결론'], 14, INK, True)
text(8.7, 2.95, 3.83, 2.6, ['측정 방법을 바꿔도 거의 다 6장 정도 틀림.',
                           '지금 한계는 측정 방법이 아니라 사진 자체가 조금씩 흔들리는 것.',
                           '그래도 쓴다면 모서리 630개(대각선 빼고)가 가장 나음.',
                           '모서리 755개를 전부 넣으면 오히려 나빠짐 (10.6장).'], 12)
text(0.8, 2.1 + h + 0.1, w, 0.3, ['새 사진으로 시험하는 걸 20번 반복한 평균 (반복 5분할). 막대가 짧을수록 좋음.'], 9.5, RGBColor.from_string('9CA3AF'))
ns = s.notes_slide
if ns.notes_text_frame is None:
    body = next(sh for sh in base.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
    ns.shapes._spTree.append(copy.deepcopy(body._element))
ns.notes_text_frame.text = ('앞 장의 결과를 한눈에 보면, 사진 280장에서 Y인지 Θ인지 틀리는 장수는 방법과 관계없이 6장 정도입니다. '
                            '대각선 모서리를 뺀 630개 모서리가 5.7장으로 가장 적고, 모서리 755개를 전부 넣으면 10.6장으로 오히려 나빠집니다. '
                            '측정 방법보다 사진 자체의 흔들림이 지금의 한계입니다.')
prs.save(OUT)
print('saved', OUT.name, len(prs.slides), 'slides')
