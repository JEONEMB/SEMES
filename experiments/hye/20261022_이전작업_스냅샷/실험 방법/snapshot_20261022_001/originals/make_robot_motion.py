from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path

OUT = Path(__file__).with_name('robot_teaching_motion.gif')
W, H = 960, 560
font_path = r'C:\Windows\Fonts\malgun.ttf'
title_font = ImageFont.truetype(font_path, 29)
body_font = ImageFont.truetype(font_path, 21)
small_font = ImageFont.truetype(font_path, 16)

def lerp(a, b, t):
    return a + (b-a)*max(0, min(1, t))

def qr(draw, x, y, s=20):
    draw.rectangle((x, y, x+s, y+s), fill='white', outline='#263c56', width=1)
    for dx, dy in ((2,2),(s-7,2),(2,s-7)):
        draw.rectangle((x+dx,y+dy,x+dx+5,y+dy+5),outline='#17273a',width=2)
    for dx,dy in ((11,10),(14,14),(9,16)):
        draw.rectangle((x+dx,y+dy,x+dx+2,y+dy+2),fill='#17273a')

frames=[]
for i in range(70):
    t=i/10
    im=Image.new('RGB',(W,H),'#f4f8fc')
    d=ImageDraw.Draw(im)
    d.text((35,22),'클린 공정 로봇 티칭: 가능한 동작 흐름',font=title_font,fill='#17324d')
    d.text((36,61),'실제 장비 구조가 확인되지 않은 개념 애니메이션',font=small_font,fill='#6c8297')

    # Rack: narrow receiving slot and fixed fiducial strip.
    d.rounded_rectangle((625,112,935,421),radius=12,fill='#d7e2eb',outline='#748ba0',width=3)
    for sy in (145,225,345,405):
        d.rectangle((625,sy,935,sy+19),fill='#9aafc0',outline='#60788e',width=2)
    d.rectangle((646,247,906,276),fill='#a6b8c8',outline='#526d83',width=2)
    for qx in (685,765,845): qr(d,qx,251,20)
    d.text((662,181),'선반 사이의 좁은 공간',font=body_font,fill='#17324d')
    d.line((653,299,921,299),fill='#bcd2e0',width=3)

    # Robot geometry: arm moves first, then adjusts image-plane position/angle, then inserts.
    if t<1.8:
        phase='① 로봇팔이 선반 앞에 접근'
        x=lerp(324,550,t/1.8); y=322; angle=0
    elif t<3.1:
        phase='② 카메라가 QR 3개를 촬영'
        x=550; y=322; angle=0
    elif t<4.8:
        phase='③ 영상으로 Y·Θ 차이를 계산하고 보정'
        f=(t-3.1)/1.7
        x=550; y=lerp(322,309,f); angle=lerp(-7,0,f)
    else:
        phase='④ 위치가 맞으면 좁은 공간에 진입'
        x=lerp(550,705,(t-4.8)/1.8); y=309; angle=0
    d.rounded_rectangle((45,245,188,405),radius=18,fill='#7e9cb4',outline='#425e76',width=4)
    d.ellipse((157,290,210,343),fill='#546f87',outline='#233d57',width=3)
    d.line((183,317,292,258),fill='#557c9d',width=42)
    d.ellipse((268,235,316,283),fill='#355a7c',outline='#17324d',width=3)
    d.line((292,259,x-75,y),fill='#668bac',width=34)
    d.ellipse((x-88,y-14,x-60,y+14),fill='#355a7c',outline='#17324d',width=2)

    # End effector and wafer, with a small in-plane correction angle.
    rad=math.radians(angle)
    ex=x+85*math.cos(rad); ey=y+85*math.sin(rad)
    d.line((x-70,y,ex,ey),fill='#456c8e',width=10)
    d.ellipse((ex-50,ey-9,ex+50,ey+9),fill='#86d0dd',outline='#2b8394',width=2)
    d.text((112,425),'로봇팔: 대상물을 옮기는 역할 (추정)',font=small_font,fill='#3b5b76')

    # Camera travels with the arm, directed toward the QR strip.
    cx=x-18; cy=y-74
    d.rounded_rectangle((cx-23,cy-14,cx+23,cy+14),radius=5,fill='#263f5c',outline='#85b5d8',width=2)
    d.ellipse((cx+12,cy-7,cx+25,cy+7),fill='#7acbe7',outline='#102e49',width=2)
    if 1.8<=t<3.1:
        d.polygon([(cx+25,cy-10),(cx+25,cy+10),(754,255)],fill='#e0f8ff')
        d.line((cx+25,cy,754,255),fill='#40afd5',width=3)
        d.text((344,139),'QR 위치·기울기 측정',font=body_font,fill='#1787b1')
    elif 3.1<=t<4.8:
        d.line((x+40,y+40,x+40,y+8),fill='#e67943',width=5)
        d.polygon([(x+32,y+12),(x+48,y+12),(x+40,y)],fill='#e67943')
        d.arc((x-19,y-23,x+70,y+60),210,320,fill='#e67943',width=4)
        d.text((350,139),'Y 이동 + Θ 회전 보정',font=body_font,fill='#c46939')

    d.rounded_rectangle((35,472,925,538),radius=15,fill='#173f63')
    d.text((59,486),phase,font=title_font,fill='white')
    d.text((748,24),'개념도',font=small_font,fill='#7f92a4')
    frames.append(im)

frames[0].save(OUT,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
print(OUT)
