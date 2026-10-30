from pathlib import Path
import sys,json
BASE=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'deps'),str(BASE/'measurement')]
import cv2,numpy as np,pandas as pd
from PIL import Image,ImageDraw
from measure import ROOT,BOXES,CROP,read
from spread import select
swatch=Image.open(r'C:/Users/SSAFY/AppData/Local/Temp/codex-clipboard-db93e6e8-56f6-4db0-8caf-b82288fd3240.png').convert('RGB')
green=tuple(int(v) for v in np.median(np.array(swatch).reshape(-1,3),axis=0))
ref=read(ROOT/'center.bmp');points,regions,strengths=select(ref,4)
points=points+np.array(CROP[:2]);original=cv2.imdecode(np.fromfile(ROOT/'center.bmp',np.uint8),0)
rows=[];scale=4;gap=60;panelw=184*scale;panelh=198*scale
for grid in [False,True]:
    canvas=Image.new('RGB',(panelw*3+gap*2,panelh),(30,32,35));draw=ImageDraw.Draw(canvas)
    for k,(x0,y0,x1,y1) in enumerate(BOXES):
        left=k*(panelw+gap)
        tile=Image.fromarray(original[y0:y1,x0:x1]).convert('RGB').resize((panelw,panelh),Image.Resampling.NEAREST)
        canvas.paste(tile,(left,0))
        if grid:
            for xx,yy,xx2,yy2 in [(left+panelw//2,0,left+panelw//2,panelh),(left,panelh//2,left+panelw,panelh//2)]:
                # Broken line separates the four spatial regions without covering corners.
                length=max(abs(xx2-xx),abs(yy2-yy))
                for t in range(0,length,24):
                    a=t/max(length,1);b=min(t+11,length)/max(length,1)
                    draw.line((xx+(xx2-xx)*a,yy+(yy2-yy)*a,xx+(xx2-xx)*b,yy+(yy2-yy)*b),fill=green,width=2)
        for i in np.flatnonzero(regions==k):
            x,y=points[i];px=left+(x-x0)*panelw/(x1-x0);py=(y-y0)*panelh/(y1-y0)
            draw.ellipse((px-3,py-3,px+3,py+3),fill=green)
            if not grid:rows.append(dict(point_id=int(i),qr=k+1,x=float(x),y=float(y),quadrant=1+int(x>=(x0+x1)/2)+2*int(y>=(y0+y1)/2),strength=float(strengths[i])))
    canvas.save(OUT/('features_green_4regions.png' if grid else 'features_green.png'))
pd.DataFrame(rows).to_csv(OUT/'features.csv',index=False)
(OUT/'metadata.json').write_text(json.dumps(dict(source=str(ROOT/'center.bmp'),green_rgb=green,features=len(points),per_qr=np.bincount(regions).tolist(),detector='Existing spread.select; ShiTomasi quality .02, minDistance 4px, center only, subpixel refinement; 16px detection margin for tracking support',scope='Dense candidate map only, not the previously evaluated 205-point set; quadrants are spatial grouping, no per-quadrant quota or motion evaluation',scale_xy=4),indent=2),encoding='utf-8')
print('green',green,'points',len(points),'per QR',np.bincount(regions).tolist())
