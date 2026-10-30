from pathlib import Path
import sys,json
BASE=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'deps'),str(BASE/'measurement')]
import cv2,numpy as np,pandas as pd
from PIL import Image,ImageDraw
from measure import ROOT,CROP,BOXES,read
ref=read(ROOT/'center.bmp');det=cv2.QRCodeDetector()
ok,text,quads,grids=det.detectAndDecodeMulti(ref);assert ok
found={}
for t,q,g in zip(text,quads,grids):
    gx,gy=q.mean(axis=0)+CROP[:2]
    k=next(k for k,(a,b,c,d) in enumerate(BOXES) if a<gx<c and b<gy<d)
    found[k]=(t,q+np.array(CROP[:2]),g)
for k,(a,b,c,d) in enumerate(BOXES):
    if k in found:continue
    x0,y0=a-CROP[0]-5,b-CROP[1]-5
    crop=ref[y0:d-CROP[1]+5,x0:c-CROP[0]+5]
    for s in [1,2,3,4]:
        t,q,g=det.detectAndDecode(cv2.resize(crop,None,fx=s,fy=s))
        if t:
            found[k]=(t,q.reshape(4,2)/s+np.array([x0+CROP[0],y0+CROP[1]]),g);break
assert len(found)==3
rows=[];info=[]
for k,(t,q,g) in sorted(found.items()):
    n=len(g);black=(g==0).astype(np.uint8);padded=np.pad(black,1)
    H=cv2.getPerspectiveTransform(np.float32([[0,0],[n,0],[n,n],[0,n]]),q.astype(np.float32))
    vertices=[]
    for y in range(n+1):
        for x in range(n+1):
            patch=padded[y:y+2,x:x+2];count=int(patch.sum())
            if count in [1,3] or (count==2 and patch[0,0]==patch[1,1]):vertices.append((x,y))
    v=np.array(vertices,np.float32);mapped=cv2.perspectiveTransform(v[None],H)[0]
    for (u,v),(x,y) in zip(vertices,mapped):rows.append(dict(qr=k+1,payload=t,grid_x=u,grid_y=v,x=float(x),y=float(y)))
    Image.fromarray(g).resize((n*30,n*30),Image.Resampling.NEAREST).save(OUT/f'QR{k+1}_decoded_grid.png')
    info.append(dict(qr=k+1,payload=t,modules=n,corners=len(vertices),quad=q.tolist()))
original=cv2.imdecode(np.fromfile(ROOT/'center.bmp',np.uint8),0)
pw,ph,gap=920,990,55;green=(139,250,131)
for four in [False,True]:
    canvas=Image.new('RGB',(3*pw+2*gap,ph),(22,24,27));draw=ImageDraw.Draw(canvas)
    for k,(a,b,c,d) in enumerate(BOXES):
        left=k*(pw+gap)
        canvas.paste(Image.fromarray(original[b:d,a:c]).convert('RGB').resize((pw,ph),Image.Resampling.LANCZOS),(left,0))
        if four:
            for y in range(0,ph,32):draw.line((left+pw//2,y,left+pw//2,min(y+17,ph)),fill=green,width=3)
            for x in range(0,pw,32):draw.line((left+x,ph//2,left+min(x+17,pw),ph//2),fill=green,width=3)
        for r in rows:
            if r['qr']!=k+1:continue
            x=left+(r['x']-a)*pw/(c-a);y=(r['y']-b)*ph/(d-b)
            draw.rectangle((x-7,y-7,x+7,y+7),outline=(15,25,16),width=5)
            draw.rectangle((x-6,y-6,x+6,y+6),outline=green,width=3)
    canvas.save(OUT/('grid_corners_4regions.png' if four else 'grid_corners.png'))
    if not four:
        for k in range(3):canvas.crop((k*(pw+gap),0,k*(pw+gap)+pw,ph)).save(OUT/f'QR{k+1}_corners.png')
pd.DataFrame(rows).to_csv(OUT/'structural_corners.csv',index=False,encoding='utf-8-sig')
(OUT/'metadata.json').write_text(json.dumps(dict(qr=info,method='Decoded straight_qrcode module matrix. Corners at grid vertices with 1 or 3 black neighbors, plus diagonally touching 2-black vertices. Straight boundaries excluded. All structural corners shown, no selection quota.',scope='Expected structural positions projected by QR quadrilateral homography; NOT independently refined subpixel image measurements; not motion evaluation',source=str(ROOT/'center.bmp')),indent=2),encoding='utf-8')
print(info)
