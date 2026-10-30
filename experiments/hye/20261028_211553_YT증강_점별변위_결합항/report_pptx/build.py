"""EDA 템플릿(2026-10-20_데이터EDA_v2.pptx)의 도형을 복제해 데이터 증강 보고서를 만든다."""
import copy, sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches
from pptx.dml.color import RGBColor
from lxml import etree

TPL, CH, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
prs = Presentation(TPL)
T = list(prs.slides)
def proto(si, name): return next(s for s in T[si - 1].shapes if s.name == name)
LAY = {l.name: l for l in prs.slide_layouts}
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

def clone(slide, src, x=None, y=None, w=None, h=None):
    el = copy.deepcopy(src._element); slide.shapes._spTree.append(el)
    sh = slide.shapes[-1]
    if x is not None: sh.left = Inches(x)
    if y is not None: sh.top = Inches(y)
    if w is not None: sh.width = Inches(w)
    if h is not None: sh.height = Inches(h)
    return sh

def set_text(sh, paras, color=None, bold_color=None):
    """paras: list of paragraphs; each paragraph a str or list of (text, bold). 첫 run 서식을 복제한다"""
    txb = sh.text_frame._txBody
    ps = txb.findall(A + "p"); base_p = ps[0]
    base_r = base_p.find(A + "r"); base_rpr = copy.deepcopy(base_r.find(A + "rPr")) if base_r is not None else etree.SubElement(etree.Element(A + "r"), A + "rPr")
    ppr = base_p.find(A + "pPr"); end = base_p.find(A + "endParaRPr")
    for p in ps: txb.remove(p)
    for para in paras:
        if isinstance(para, str): para = [(para, False)]
        p = etree.SubElement(txb, A + "p")
        if ppr is not None: p.append(copy.deepcopy(ppr))
        for text, bold in para:
            r = etree.SubElement(p, A + "r"); rpr = copy.deepcopy(base_rpr); rpr.set("b", "1" if bold else "0")
            c = bold_color if (bold and bold_color) else color
            if c:
                for f in rpr.findall(A + "solidFill"): rpr.remove(f)
                sf = etree.Element(A + "solidFill"); etree.SubElement(sf, A + "srgbClr", val=c)
                rpr.insert(0, sf)
            r.append(rpr); t = etree.SubElement(r, A + "t"); t.text = text
            if text != text.strip(): t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        if end is not None: p.append(copy.deepcopy(end))
    return sh

def rich(s):
    """'보통 **굵게** 보통' → [(보통, False), (굵게, True), ...]"""
    out, bold = [], False
    for part in s.split("**"):
        if part: out.append((part, bold))
        bold = not bold
    return out

INK, INK2, MUTED, BLUE = "111827", "4B5563", "9CA3AF", "0071E3"
page = [1]

def content(section, title, summary):
    s = prs.slides.add_slide(LAY["CONTENT_GRAY"]); page[0] += 1
    # 위치 표시줄, 쪽 번호
    set_text(clone(s, proto(3, "TextBox 2")), [[("2-1  ", False), (f"20261029  |  {section}", False)]])
    rs = s.shapes[-1].text_frame.paragraphs[0].runs; rs[0].font.color.rgb = RGBColor.from_string(MUTED); rs[1].font.color.rgb = RGBColor.from_string(INK)
    set_text(clone(s, proto(3, "TextBox 3")), [str(page[0])])
    clone(s, proto(3, "Rounded Rectangle 21"))
    set_text(clone(s, proto(3, "TextBox 22")), [[("정리  ", True)] + rich(summary)], color=INK)
    return s

def title_fix(s, title):
    ph = next(p for p in s.placeholders if "TITLE" in str(p.placeholder_format.type))
    ph._element.getparent().remove(ph._element)
    set_text(clone(s, proto(3, "Title 1")), [[(title, True)]])

def kv(s, items, x0=7.21, y0=2.22, step=0.63, keyw=1.31, valw=4.03):
    for i, (k, v) in enumerate(items):
        y = y0 + i * step
        set_text(clone(s, proto(3, "TextBox 6"), x0, y), [k])
        set_text(clone(s, proto(3, "TextBox 7"), x0 + 1.30, y + 0.02, valw), [rich(line) for line in v.split("\n")], color=INK2, bold_color=INK)
        clone(s, proto(3, "Connector 8"), x0, y + 0.53)

def picture(s, name, x=0.81, y=1.94, w=6.0):
    s.shapes.add_picture(str(CH / name), Inches(x), Inches(y), width=Inches(w))

def caption(s, text, y=6.17):
    set_text(clone(s, proto(3, "TextBox 5"), 0.81, y), [text])

# ---------- 1. 표지 (구분 슬라이드) ----------
d = prs.slides.add_slide(LAY["DIVIDER"])
for n in ("Text 1", "Text 2", "Shape 3", "Text 5", "Text 6"): clone(d, proto(1, n))
d.background.fill.solid(); d.background.fill.fore_color.rgb = T[0].background.fill.fore_color.rgb
sh = {x.name: x for x in d.shapes}
set_text(sh["Text 1"], [[("02", True)]]); set_text(sh["Text 2"], [[("데이터 증강", True)]])
r = sh["Text 5"].text_frame.paragraphs[0].runs; r[0].text = "2-1    "; r[1].text = "Y-T 평면 증강 (20261029)"
r = sh["Text 6"].text_frame.paragraphs[0].runs; r[0].text = "2-2    "; r[1].text = "새 자료로 사전 예측 비교"

# ---------- 2. 범위 ----------
s = content("① 범위와 생성식", "증강 범위 – 격자 16,281점 × 영점 15개 × 3회",
            "16,281 × 15 × 3 = **732,645개** 샘플을 측정값으로 만들어 두 모듈로 채점. 판 전체 평행이동이 아니라 **점별 변위**")
title_fix(s, "증강 범위 – 격자 16,281점 × 영점 15개 × 3회")
picture(s, "c1_grid.png")
kv(s, [("격자", "Y −2 ~ +2mm (0.05) 81개\nT −1 ~ +1 (0.01) 201개"),
       ("영점", "실측 11개 + 흔들리기 쉬운 방향 **4개**"),
       ("반복", "점마다 반복 촬영 잡음 3회\n총 **732,645개**"),
       ("형태", "사진이 아니라 **측정값**\njoint 102관측, control 630점 변위"),
       ("표기", "**가정 시뮬레이션**\n실측 성적으로 쓰지 않음")])
caption(s, "실측 혼합은 Y 0~1, T 0~1 한 사분면뿐. 나머지는 Y와 T 효과가 더해진다는 가정.")

# ---------- 3. 생성식 ----------
s = content("① 범위와 생성식", "생성식 – 점마다 영점 + 단일축 반응 + 잡음",
            "T 의 +/− 기울기가 달라 **따로 맞춰야** 함. 반응식 잔차 **0.050px** 는 반복 잡음(0.120px)보다 작음")
title_fix(s, "생성식 – 점마다 영점 + 단일축 반응 + 잡음")
picture(s, "c2_shape.png")
kv(s, [("생성식", "샘플 = 영점 필드 + 점별 반응 (Y, T+, T−)\n+ [YT 결합항, 기본 끔] + 반복 잡음"),
       ("반응식", "**Y 2차, T ± 따로 2차** (흐린 경계 측정)\njoint, control 측정 공간은 Y 3차, T ± 3차"),
       ("T ± 분리", "T 잔차 0.072 → **0.049px**\n같은 크기라도 + 쪽과 − 쪽 기울기가 다름"),
       ("검증", "1014 trial 안 4겹 교차검증\n잔차 **0.050px** (선택 536점)"),
       ("점", "고정 630점 (control 선택 536점으로 평가)")])
caption(s, "반응식은 1014 단일축 281장으로 맞춤. 판 전체가 아니라 점마다 따로 옮긴다.")

# ---------- 4. 영점 ----------
s = content("② 영점과 잡음", "영점 15개 – 실측 11개 + 흔들리기 쉬운 방향 4개",
            "영점은 연속 노이즈가 아니라 **실측 영점 위치에서 출발**. 가로는 **접근 방향(백래시)**, 세로는 **날짜**가 흔듦")
title_fix(s, "영점 15개 – 실측 11개 + 흔들리기 쉬운 방향 4개")
picture(s, "c3_origins.png")
kv(s, [("센터 사진", "1014 center (기준 0), 1020 center 3장\ndx −0.21 ~ −0.29, dy +0.41 ~ +0.49px"),
       ("trial 영점", "S1 ~ S6, 1014 단일축 trial 마다 맞춘 값\n+접근 trial 은 가로로 **−0.44 ~ −0.78px**"),
       ("혼합 세션", "1020 혼합 자세에서 맞춘 영점\ndx **−0.90**, dy +0.47px"),
       ("추가 후보", "실측 11개의 주성분 ±2σ\n1번 68.5% (가로), 2번 30.9% (세로)"),
       ("백래시", "센터 사진 영점에 접근 방향만큼 더함\n+접근 **−0.61px**, −접근 −0.17px")])
caption(s, "1014 center 기준, 선택 536점 평균. 접근 방향은 자세 부호로 가정 (촬영 순서 없음).")

# ---------- 5. 결합항 ----------
s = content("② 영점과 잡음", "결합항 – 지금 자료로는 잡음과 구분되지 않음",
            "Y > 0, T > 0, |Y| ≤ 1 에서는 **결합항이 잡음과 구분되지 않음**. 혼합 잔차는 **백래시 영점 차이**. 다른 사분면은 새 자료로 확인")
title_fix(s, "결합항 – 지금 자료로는 잡음과 구분되지 않음")
picture(s, "c4_coupling.png")
kv(s, [("시험", "1020 혼합 T ≤ 1, 45장 (15자세)\n자세 하나 빼고 맞춘 뒤 그 자세 예측"),
       ("센터 영점", "**0.486px**, 판 전체 가로 −0.66px 밀림\n크기가 1014 +/− 접근 차이와 같음"),
       ("세션 영점", "**0.135px**, 반복 잡음 0.120px 에 가까움"),
       ("결합항 추가", "YT 0.150px, 고차 0.201px\n넣으면 오히려 나빠짐"),
       ("처리", "결합항 기본 끔, 켜서 비교 가능\n전체 학습 YT 는 (1, 1)에서 +0.29px")])
caption(s, "T = 2 (12장)는 1014 범위(±1) 밖 외삽. 세션 영점으로도 0.305px.")

# ---------- 6. 반복 잡음 ----------
s = content("② 영점과 잡음", "반복 잡음 – 1020 반복과 1014 연속 두 가지",
            "1020 세션의 반복 흔들림은 1014 의 **약 4배** (0.108 vs 0.029px). 잡음은 **날짜, 세션마다 다름**")
title_fix(s, "반복 잡음 – 1020 반복과 1014 연속 두 가지")
picture(s, "c5_noise.png")
kv(s, [("1020 반복", "같은 자세 3장의 편차 48개 (T ≤ 1)\n**0.108px**, 대부분 판 전체 가로 흔들림"),
       ("1014 연속", "1014 281장의 반응식 잔차 (자유도 보정)\n**0.029px**. 같은 자세 반복이 없어 잔차로 추정"),
       ("T = 2 제외", "control 추적이 자세마다 37 ~ 76점씩 1px 넘게 튐\n반복 잡음이 아니라 **추적 실패** (최대 21px)"),
       ("촬영 순번", "1020 은 2번째 사진만 판 평균 +0.16px\n순번을 유지해서 뽑음"),
       ("뽑는 법", "점마다 따로가 아니라 **사진 한 장의 편차 필드**\njoint, control 이 같은 사진 편차를 씀")])
caption(s, "이전 결과(T = 2 포함, 0.513px)는 그대로 두고 새 결과를 따로 저장.")

# ---------- 7. 쌍둥이 ----------
s = content("③ 증강 자체 평가", "실측 사진과 증강 쌍둥이 – 추정값과 판정이 같음",
            "|T| ≤ 1 에서 증강 쌍둥이가 두 모듈의 **추정값과 통과 판정을 재현**. 두 모듈 모두 이 사진으로 학습해 실측 통과는 **재대입**")
title_fix(s, "실측 사진과 증강 쌍둥이 – 추정값과 판정이 같음")
picture(s, "c6_twin.png")
kv(s, [("방법", "사진마다 그 사진을 빼고 맞춘 생성기로\n같은 자세, 영점의 쌍둥이를 만들어 비교"),
       ("모듈", "채택 joint, control 의 C++ 측정\n추정은 Python 이식 (C++ 와 차이 1e-8 이하)"),
       ("joint Y 차이", "1014 **0.020**, 1020 **0.030mm** RMS\n최대 0.084mm"),
       ("판정 일치", "joint 280/280, 45/45\ncontrol 축과 판정 280/280, 45/45"),
       ("T = 2", "재현 안 됨 (Y 차이 0.80mm)\n생성 범위 밖")])
caption(s, "1014 는 trial 안 4겹, 1020 혼합은 자세 하나 빼기 (그 자세 잡음 표본도 제외).")

# ---------- 8. 잡음 퍼짐 표 ----------
s = content("③ 증강 자체 평가", "잡음 퍼짐 – 1014 사진엔 1014 잡음, 1020 사진엔 1020 잡음",
            "실측이 95% 구간에 95% 들어오면 맞는 잡음. **1014 연속은 1014 에** 맞고, 1020 반복으로도 1020 은 **82 ~ 87%**")
title_fix(s, "잡음 퍼짐 – 1014 사진엔 1014 잡음, 1020 사진엔 1020 잡음")
cols = [(0.81, 2.85), (3.71, 2.30), (6.06, 2.30), (8.41, 4.09)]
heads = ["항목", "1014 단일축 280장", "1020 혼합 T ≤ 1, 45장", "뜻"]
for (x, w), h in zip(cols, heads): set_text(clone(s, proto(8, "TextBox 4"), x, 1.90, w), [[(h, True)]])
clone(s, proto(8, "Connector 8"))
rows = [("joint 95% 안 (Y / T), 1020 반복", "99% / 99%", "84% / 87%", "1014 에는 조금 넓음"),
        ("joint 95% 안 (Y / T), 1014 연속", "**95% / 95%**", "82% / 76%", "**1014 에 딱 맞음**"),
        ("control 95% 안, 1020 반복", "100%", "**82%**", "1020 퍼짐을 다 설명하지 못함"),
        ("control 95% 안, 1014 연속", "93%", "**18%**", "1020 에는 **너무 좁음**"),
        ("joint 잡음 쌍둥이 통과 (실측)", "100%, 100% (100%)", "100%, 100% (100%)", "두 잡음 모두 실측과 같음"),
        ("control 잡음 쌍둥이 통과 (실측)", "100%, 100% (100%)", "39%, 40% (40%)", "두 잡음 모두 실측과 같음"),
        ("이전: T = 2 섞인 1020 잡음", "joint 98%, control 91%", "-", "실측 100% 보다 낮았던 원인은 **T = 2 추적 실패**")]
for i, row in enumerate(rows):
    y = 2.35 + i * 0.42
    for j, ((x, w), v) in enumerate(zip(cols, row)):
        src = proto(8, "TextBox 9") if j == 0 else proto(8, "TextBox 10")
        set_text(clone(s, src, x, y, w), [rich(v)], color=INK2, bold_color=INK)
    clone(s, proto(8, "Connector 13"), 0.81, y + 0.33)
caption(s, "통과 칸은 1020 반복, 1014 연속 순. 95% 구간 = 잡음 쌍둥이 추정값의 2.5 ~ 97.5% 범위.", y=5.5)

# ---------- 9. 오차 지도 ----------
s = content("④ Y-T 평면 오차 지도", "joint 오차 지도 – |Y| ≤ 1 에서 98 ~ 99% 통과",
            "잡음을 줄여도 **Y < 0, T < 0 사분면과 |Y| 큰 곳**은 그대로. 잡음이 아니라 **joint 의 계통 오차** (가정 시뮬레이션)")
title_fix(s, "joint 오차 지도 – |Y| ≤ 1 에서 98 ~ 99% 통과")
picture(s, "c7_map_1020.png", 1.0, 1.80, 4.6); picture(s, "c7_map_1014.png", 1.0, 4.10, 4.6)
kv(s, [("전체", "**95.4%** (1020 반복) / **96.8%** (1014 연속)"),
       ("|Y| ≤ 1", "98.1% / 99.0%"),
       ("센터 박스", "±0.09 안 99.2% / 99.8%"),
       ("사분면", "Y > 0, T > 0: 98.6% / 99.3%\nY < 0, T < 0: **90.1% / 92.2%**"),
       ("모서리", "(−2, −1): **29% / 42%**\n|Y| ≥ 1.5: 90.9% / 92.8%"),
       ("그림", "칸마다 영점 15개 × 3회 = 45개 통과율\n점선 상자 = 실측 혼합 범위, 빨간 점선 = Y ≈ −5.68T")], y0=1.95, step=0.7)

# ---------- 10. control ----------
s = content("④ Y-T 평면 오차 지도", "control – T = 0 줄 98%, Y = 0 줄은 영점 따라 72 ~ 99%",
            "control 은 쓰임새(1차 뒤 T ≈ 0)에서 **98%**. **날짜가 바뀐 영점**에서 T 판단이 약함")
title_fix(s, "control – T = 0 줄 98%, Y = 0 줄은 영점 따라 72 ~ 99%")
picture(s, "c8_control.png")
kv(s, [("출력", "Y 또는 T 하나만 냄\n혼합 자세는 원래 통과할 수 없음 (전체 4.8%)"),
       ("T = 0 줄", "Y만 움직인 자세\n**98.2%** (1020 반복) / **97.8%** (1014 연속)"),
       ("Y = 0 줄", "T만 움직인 자세\n87.0% / 87.0%"),
       ("영점별", "1014 center, S1 ~ S6: **91 ~ 99%**\n1020 센터, 세션, PC2: **72 ~ 79%**"),
       ("뜻", "Y = 0 줄 실패는 잡음이 아니라\n**날짜(세로) 영점** 때문")])
caption(s, "Y = 0 줄 201자세 × 3회 통과율. PC2 = 추가 후보 중 세로(날짜) 방향.")

# ---------- 11. 알게 된 것, 모르는 것 ----------
s = content("⑤ 정리와 다음 일", "이번 증강으로 알게 된 것, 모르는 것",
            "점별 증강이 |T| ≤ 1 실측을 재현함을 확인. **−쪽 혼합, |Y| > 1, 새 세션**은 아직 가정")
title_fix(s, "이번 증강으로 알게 된 것, 모르는 것")
for n in ("Shape 3", "Shape 4", "Text 5", "Shape 21", "Shape 22", "Text 23"): clone(s, proto(11, n))
know = [("점별 반응", "잔차 0.050px, 잡음보다 작음"), ("결합항", "+쪽 |Y| ≤ 1 에선 잡음과 구분 안 됨"),
        ("혼합 잔차", "결합항이 아니라 백래시 −0.66px"), ("반복 잡음", "1014 0.029, 1020 0.108px"), ("쌍둥이", "추정과 판정 재현 (|T| ≤ 1)")]
unk = [("−쪽 혼합", "결합항과 백래시 부호"), ("|Y| > 1 혼합", "실측 없음, 더하기 가정"), ("새 세션", "생성기 고정 사전 예측 필요"),
       ("이미지 증강", "측정값만 만듦, 사진은 아직"), ("T = 2 촬영", "사양 밖 자세를 찍은 이유")]
for col, items, dot, key, val in ((0, know, "Shape 6", "Text 7", "Text 8"), (1, unk, "Shape 24", "Text 25", "Text 26")):
    for i, (k, v) in enumerate(items):
        dy = i * 0.66
        b = proto(11, dot); clone(s, b, None, (b.top / 914400) + dy)
        kk = proto(11, key); set_text(clone(s, kk, None, kk.top / 914400 + dy), [[(k, True)]])
        vv = proto(11, val); set_text(clone(s, vv, None, vv.top / 914400 + dy), [v])

# ---------- 12. 다음 일 ----------
s = content("⑤ 정리와 다음 일", "다음 일: 필요한 것",
            "오늘 받는 자료로 **생성기를 고정한 사전 예측(A)**부터. 결합항을 확인하려면 **−쪽 혼합 자세(B)**가 필요")
title_fix(s, "다음 일: 필요한 것")
heads = [("TextBox 4", ""), ("TextBox 5", "중요도"), ("TextBox 6", "일"), ("TextBox 7", "어떻게"), ("TextBox 8", "지금")]
for n, h in heads: set_text(clone(s, proto(12, n)), [[(h, True)]])
clone(s, proto(12, "Connector 9"))
todo = [("A", True, "새 자료 사전 예측", "생성기를 고정한 채 새 사진과 비교 (evaluate_new.py)", "20261020 으로 점검만 끝남"),
        ("B", True, "−쪽, |Y| > 1 혼합 자세", "결합항과 백래시 부호 확인", "+쪽 15자세뿐"),
        ("C", True, "촬영 순서 기록", "사진마다 접근 방향, 순번", "백래시를 자세 부호로 가정"),
        ("D", False, "정지 반복 촬영", "1014 연속 잡음을 직접 측정", "반응식 잔차로 추정"),
        ("E", False, "이미지 증강 검증", "센터 사진을 휘어 실측 341자세 재현", "약 8.5GB, 계산 몇 분"),
        ("F", False, "C++ 생성기", "팀 전달용으로 옮기기", "지금은 Python 실험 코드")]
for i, (lab, must, what, how, now) in enumerate(todo):
    dy = i * 0.475
    bx = proto(12, "Rounded Rectangle 10" if must else "Rounded Rectangle 45"); clone(s, bx, None, proto(12, "Rounded Rectangle 10").top / 914400 + dy)
    t = proto(12, "TextBox 11"); set_text(clone(s, t, None, t.top / 914400 + dy), [[(lab, True)]])
    pr = proto(12, "TextBox 12" if must else "TextBox 47"); set_text(clone(s, pr, None, proto(12, "TextBox 12").top / 914400 + dy), [[("필수" if must else "권장", True)]])
    for n, v in (("TextBox 13", [(what, True)]), ("TextBox 14", [(how, False)]), ("TextBox 15", [(now, False)])):
        p0 = proto(12, n); set_text(clone(s, p0, None, p0.top / 914400 + dy), [v])
    ln = proto(12, "Connector 16"); clone(s, ln, None, ln.top / 914400 + dy)

# 템플릿 원래 슬라이드 지우기
sld = prs.slides._sldIdLst
for sid in list(sld)[:len(T)]:
    prs.part.drop_rel(sid.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")); sld.remove(sid)
prs.save(OUT); print("saved", OUT, len(prs.slides))
