# 정리본 v3 갱신: 사용자가 v2 에 더 고친 내용을 v3 에 반영하고, 노이즈 4분할 2장을 제목 없이 꽉 채움
#   v3 4번  : 제목 '기준 위치 – 특징점' -> '이동 관계 – 특징점 분석' (v2 와 같게)
#   v3 4번 뒤: v2 5번 '이동 관계 – ECC 분석' 추가 (v2 처럼 4번 복제 후 제목만 바꿈)
#   한 칸 이동량 뒤: v2 10번 '노이즈 – Trial별 노이즈 분석' 표 추가 (쪽 번호는 새 순서로 고침)
#   노이즈 4분할 2장: 위쪽 '1-1 … | ③ 노이즈' 머리글 삭제, 4칸을 슬라이드 전체에 맞춰 키움
import copy, pathlib
from pptx import Presentation
from pptx.util import Inches
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent
V2, V3 = DECK / 'SEMES_01_데이터EDA_정리본_v2.pptx', DECK / 'SEMES_01_데이터EDA_정리본_v3.pptx'
E = 914400
src, prs = Presentation(V2), Presentation(V3)

def title_shape(s):
    return next(sh for sh in s.shapes if sh.has_text_frame and sh.top is not None and 0.8 < sh.top / E < 1.2 and not sh.text_frame.text.startswith('1-1'))
def set_text(sh, new):
    rs = [r for p in sh.text_frame.paragraphs for r in p.runs]
    rs[0].text = new
    for r in rs[1:]: r.text = ''
def find(deck, key):
    return next(s for s in deck.slides if key in title_shape_safe(s))
def title_shape_safe(s):
    try: return title_shape(s).text_frame.text
    except StopIteration: return ''

def clone(from_slide, layout):
    """from_slide 의 도형을 새 슬라이드로 복사 (그림 관계 포함). 다른 파일의 슬라이드도 가능."""
    new = prs.slides.add_slide(layout)
    for sh in list(new.shapes): sh._element.getparent().remove(sh._element)
    tree = new.shapes._spTree
    for el in from_slide.shapes._spTree.iterchildren():
        if el.tag in (qn('p:nvGrpSpPr'), qn('p:grpSpPr')): continue
        el = copy.deepcopy(el)
        for blip in el.iter(qn('a:blip')):
            rid = blip.get(qn('r:embed'))
            img = from_slide.part.related_part(rid)
            blip.set(qn('r:embed'), new.part.relate_to(img, RT.IMAGE))
        tree.append(el)
    if from_slide.has_notes_slide:  # 발표 메모는 도형째 복사 (노트 마스터에 본문 자리가 없음)
        nt = new.notes_slide.shapes._spTree
        for el in list(nt.iterchildren()):
            if el.tag not in (qn('p:nvGrpSpPr'), qn('p:grpSpPr')): nt.remove(el)
        for el in from_slide.notes_slide.shapes._spTree.iterchildren():
            if el.tag not in (qn('p:nvGrpSpPr'), qn('p:grpSpPr')): nt.append(copy.deepcopy(el))
    return new

def move(slide, after):
    ids = prs.slides._sldIdLst; el = ids[list(prs.slides).index(slide)]
    ids.remove(el); ids.insert(list(prs.slides).index(after) + 1, el)

# 1. 특징점 제목
s4 = find(prs, '특징점'); set_text(title_shape(s4), '이동 관계 – 특징점 분석')

# 2. ECC 분석 장 (v2 5번과 같게: 4번 복제 + 제목)
ecc = clone(s4, s4.slide_layout)
set_text(title_shape(ecc), '이동 관계 – ECC 분석'); move(ecc, s4)

# 3. 노이즈 표 장 (v2 10번 복사)
t_src = find(src, 'Trial별 노이즈')
layout = next(l for l in prs.slide_layouts if l.name == t_src.slide_layout.name)
tbl = clone(t_src, layout); move(tbl, find(prs, '한 칸 이동량'))
REF = {'0점 어긋남': '6쪽', '방향 바뀐 직후': '11쪽', '사진 공통 흔들림': '12쪽', '점 하나 흔들림': '12쪽',
       '밝기 변화': '11쪽', '세로 밀림': '12쪽', '센서 잡음': '11쪽'}
boxes = [sh for sh in tbl.shapes if sh.has_text_frame]
for sh in boxes:
    if sh.text_frame.text in REF:
        row = sh.top; ref = next(b for b in boxes if b.top == row and b.left / E > 11)
        set_text(ref, REF[sh.text_frame.text])

# 4. 노이즈 4분할 2장: 머리글 삭제 + 꽉 채우기
quads = [s for s in prs.slides if any(sh.has_text_frame and sh.text_frame.text in ('촬영 순서', '사진 공통 흔들림') for sh in s.shapes) and sum(sh.shape_type == 13 for sh in s.shapes) == 4]
W, H = 13.333, 7.5; MX, GAP, Y0, RGAP = 0.35, 0.30, 0.30, 0.25
CW = (W - 2 * MX - GAP) / 2
for s in quads:
    for sh in list(s.shapes):
        if sh.has_text_frame and sh.text_frame.text.startswith('1-1'): sh._element.getparent().remove(sh._element)
    label = [sh for sh in s.shapes if sh.has_text_frame and sh.height / E < 0.28 and not sh.is_placeholder]
    pics = [sh for sh in s.shapes if sh.shape_type == 13]
    caps = [sh for sh in s.shapes if sh.has_text_frame and not sh.is_placeholder and sh not in label]
    key = lambda sh: (round(sh.top / E, 0) // 2, sh.left)
    pics.sort(key=lambda sh: (sh.top, sh.left))
    ph = CW * pics[0].height / pics[0].width
    row_h = 0.30 + 0.32 + ph
    for i, pic in enumerate(pics):
        r, c = divmod(i, 2); x = MX + c * (CW + GAP); y = Y0 + r * (row_h + RGAP)
        # 같은 칸의 제목과 설명: 그림 바로 위에 있던 두 상자
        above = sorted([sh for sh in label + caps if abs(sh.left - pic.left) < Inches(0.05) and sh.top < pic.top and pic.top - sh.top < Inches(0.7)], key=lambda sh: sh.top)
        for sh, dy, h in zip(above, (0, 0.30), (0.28, 0.30)):
            sh.left, sh.top, sh.width, sh.height = Inches(x), Inches(y + dy), Inches(CW), Inches(h)
        pic.left, pic.top, pic.width, pic.height = Inches(x), Inches(y + 0.62), Inches(CW), Inches(ph)
    for sh in s.shapes:
        if sh.is_placeholder: sh.left, sh.top, sh.width, sh.height = Inches(11.93), Inches(7.00), Inches(0.60), Inches(0.30)

# 5. 슬라이드가 한 장 밀려서 0점 위치 참조 5쪽 -> 6쪽
for s in quads:
    for sh in s.shapes:
        if sh.has_text_frame and '5쪽 참고' in sh.text_frame.text:
            for r in (r for p in sh.text_frame.paragraphs for r in p.runs): r.text = r.text.replace('5쪽 참고', '6쪽 참고')

prs.save(V3)
print([title_shape_safe(s) or '(제목 없음)' for s in prs.slides])
