from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json
P=Path(__file__).resolve().parent
fontpath='C:/Windows/Fonts/malgun.ttf'
def font(size):return ImageFont.truetype(fontpath,size)
for source,target,title,caption in [
 ('all_subpixel.png','발표용_전체모서리.png','QR 격자 기반 모서리 · 실제 영상 경계 서브픽셀 보정','전체 755점 표시  |  연두: 적합·안정성 기준 통과 608점  |  주황: 불확실 147점  |  가로·세로 동일 5배'),
 ('finder_subpixel_detail.png','발표용_큰네모상세.png','큰 네모 · 실제 경계 보정','연두: 적합·안정성 기준 통과  |  가로·세로 동일 16배')]:
    im=Image.open(P/source).convert('RGB');w,h=im.size
    out=Image.new('RGB',(w,h+150),(22,24,27));out.paste(im,(0,90));d=ImageDraw.Draw(out)
    d.text((24,20),title,font=font(38 if w>1500 else 29),fill=(242,245,247))
    d.text((24,h+108),caption,font=font(29 if w>1500 else 20),fill=(211,218,225))
    out.save(P/target)
assert len(list(P.glob('QR*_subpixel.png')))==3
print('Presentation PNGs saved')
