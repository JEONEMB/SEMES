"""Create an annotated, 2x-speed overview of the single-axis SEMES image set."""
from pathlib import Path
import subprocess
import cv2
import numpy as np

ROOT = Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')
OUT = Path(__file__).with_name('SSAFY_TestImageSet_2x_annotated.mp4')
FPS = 20
W, H = 1920, 1080
POINTS = [(1644, 1065), (1902, 1065), (2157, 1065)]
COLORS = {'ref': (255, 230, 0), 'cur': (240, 90, 255), 'text': (245, 245, 245)}

def read(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im

center = read(ROOT / 'center.bmp')
gray_center = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
templates = [gray_center[y-55:y+55, x-55:x+55] for x,y in POINTS]

def position_offsets(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    offsets = []
    for (x,y), template in zip(POINTS,templates):
        roi = gray[y-155:y+155, x-155:x+155]
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if score < .85:
            raise RuntimeError(f'Unreliable QR match: {score:.3f}')
        offsets.append((loc[0]-100, loc[1]-100, score))
    return offsets

def text(im, s, xy, scale, color=(245,245,245), weight=2):
    cv2.putText(im,s,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(8,12,18),weight+3,cv2.LINE_AA)
    cv2.putText(im,s,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,weight,cv2.LINE_AA)

def make_frame(image, axis, value, filename, idx, total):
    canvas = np.full((H,W,3),(28,31,37),np.uint8)
    crop = image[690:1530,1220:2620]
    canvas[120:960,30:1430] = crop
    offsets = position_offsets(image) if value != 0 else [(0,0,1)]*3
    # Each reference line is attached to one QR in the center photograph.
    for i, ((x,y),(dx,dy,_)) in enumerate(zip(POINTS,offsets)):
        bx, by = x-1220+30, y-690+120
        cx, cy = bx+dx, by+dy
        cv2.line(canvas,(bx,by-135),(bx,by+135),COLORS['ref'],2,cv2.LINE_AA)
        cv2.line(canvas,(cx,cy-115),(cx,cy+115),COLORS['cur'],2,cv2.LINE_AA)
        cv2.circle(canvas,(bx,by),7,COLORS['ref'],-1,cv2.LINE_AA)
        cv2.circle(canvas,(cx,cy),7,COLORS['cur'],-1,cv2.LINE_AA)
        if abs(dx)>3:
            cv2.arrowedLine(canvas,(bx,by+125),(cx,cy+125),COLORS['cur'],3,cv2.LINE_AA,tipLength=.16)
        text(canvas, f'{i+1}',(bx-12,by-155),.85,COLORS['ref'])
    cv2.rectangle(canvas,(30,120),(1430,960),(95,100,105),2)
    text(canvas,'QR MARKER MOTION  |  CENTER vs CURRENT',(30,75),1.1)
    cv2.rectangle(canvas,(1450,120),(1890,690),(42,47,55),-1)
    cv2.rectangle(canvas,(1450,120),(1890,690),(85,92,100),2)
    text(canvas,'THETA' if axis=='T' else 'Y POSITION',(1480,185),1.15,(180,185,190))
    text(canvas,f'{value:+.2f}',(1470,295),2.55,(255,255,255),5)
    text(canvas,'filename value',(1480,340),.75,(170,175,180))
    text(canvas,'QR SHIFT FROM CENTER',(1480,410),.8,COLORS['ref'])
    for i,(dx,dy,_) in enumerate(offsets):
        text(canvas,f'{i+1}   {dx:+4d} px',(1480,465+i*62),1.05,COLORS['cur'])
    text(canvas,'cyan = center / pink = current',(1480,670),.55,(210,210,210))
    overview = cv2.resize(image,(440,248),interpolation=cv2.INTER_AREA)
    canvas[720:968,1450:1890] = overview
    cv2.rectangle(canvas,(1450,720),(1890,968),(85,92,100),2)
    text(canvas,f'{filename}   [{idx}/{total}]',(30,1010),.78)
    text(canvas,'Single-axis images: theta and Y are separate sequences.',(30,1050),.70,(190,195,205))
    # 20 fps, one source image per frame. The prior preview used 10 fps.
    return canvas

def sequence():
    for j in range(-100,101):
        if j == 0:
            yield 'T',0.0,ROOT/'center.bmp'
        else:
            yield 'T',j/100,ROOT/'Theta'/f'T{j/100:+.2f}.bmp'
    for j in range(-40,41):
        if j == 0:
            yield 'Y',0.0,ROOT/'center.bmp'
        else:
            yield 'Y',j/20,ROOT/'Y'/f'Y{j/20:+.2f}.bmp'

def main():
    frames=list(sequence())
    # Add 0.45 s at each center frame, matching half the old 1 s pauses.
    expanded=[]
    for item in frames:
        expanded.extend([item] * (10 if item[1] == 0 else 1))
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
         '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','veryfast',
         '-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for idx,(axis,value,path) in enumerate(expanded,1):
            im=read(path)
            frame=make_frame(im,axis,value,path.name,idx,len(expanded))
            proc.stdin.write(frame.tobytes())
            if idx%50==0:
                print(f'{idx}/{len(expanded)}',flush=True)
    finally:
        proc.stdin.close()
    rc=proc.wait()
    if rc:
        raise RuntimeError(f'ffmpeg exited {rc}')
    print(OUT)

if __name__=='__main__':
    main()
