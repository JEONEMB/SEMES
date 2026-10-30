# 기초 정보 슬라이드의 QR 확대 13장 배치는 그대로 두고, 그림 파일만 img/qrbox_*.png 최신본으로 바꾼다
#   사용: python swap_qr_images.py 입력.pptx 출력.pptx
import sys, pathlib
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

HERE = pathlib.Path(__file__).parent
F, OUT = HERE / sys.argv[1], HERE / sys.argv[2]
prs = Presentation(F)
s = next(sl for sl in prs.slides if any(sh.has_text_frame and sh.text_frame.text == '기초 정보' for sh in sl.shapes))
def walk(shapes):   # 그룹 안까지 모든 도형 (사용자가 그룹을 풀거나 다시 묶어도 동작)
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP: yield from walk(sh.shapes)
        else: yield sh
kids = list(walk(s.shapes))
pics = [k for k in kids if k.shape_type == MSO_SHAPE_TYPE.PICTURE]
caps = [k for k in kids if k.has_text_frame and k.text_frame.text.strip()]
done = []
for pic in pics:
    cap = min(caps, key=lambda c: abs(c.left - pic.left) + abs(c.top - (pic.top + pic.height)))
    name = cap.text_frame.text.strip()
    f = 'center' if name.startswith('center') else name.replace('Θ ', 'T').replace('Y ', 'Y').split()[0]
    _, rid = s.part.get_or_add_image_part(str(HERE / 'img' / f'qrbox_{f}.png'))
    pic._element.blipFill.blip.rEmbed = rid
    done.append(f)
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith('파란 '):
        runs = sh.text_frame.paragraphs[0].runs
        runs[0].text = '파란 점선 = center QR 위치,  초록 실선 = 최소 이동 QR 위치 (최대 이동 사진에서는 초록 점선),  빨강 실선 = 최대 이동 QR 위치,  px = 가로 이동량'
        for r in runs[1:]: r.text = ''
prs.save(OUT)
print('swapped', len(done), sorted(done))
