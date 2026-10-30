"""Presentation-only representative contour corners; not evaluated tracking features."""
exec((__import__('pathlib').Path(__file__).parent/'build.py').read_text(encoding='utf-8').split('rows=[];scale=')[0])
scale=5;pw=920;ph=990;gap=55
rows=[]
for k,(x0,y0,x1,y1) in enumerate(BOXES):
    im=original[y0:y1,x0:x1];sm=cv2.GaussianBlur(im,(0,0),.9)
    _,bw=cv2.threshold(sm,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours,_=cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    eig=cv2.cornerMinEigenVal(sm,5);candidates=[]
    for c in contours:
        if cv2.contourArea(c)<18:continue
        p=cv2.approxPolyDP(c,1.4,True).reshape(-1,2).astype(float)
        if len(p)<4:continue
        for j,q in enumerate(p):
            x,y=q;a=p[j-1]-q;b=p[(j+1)%len(p)]-q
            if min(np.linalg.norm(a),np.linalg.norm(b))<4:continue
            cosine=abs(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
            if cosine>.32 or not (13<x<im.shape[1]-13 and 13<y<im.shape[0]-13):continue
            region=int(x>=im.shape[1]/2)+2*int(y>=im.shape[0]/2)
            candidates.append((float(eig[int(y),int(x)]),x,y,region))
    chosen=[]
    for quadrant in range(4):
        count=0
        for score,x,y,q in sorted(candidates,reverse=True):
            if q!=quadrant or any((x-a[1])**2+(y-a[2])**2<9**2 for a in chosen):continue
            chosen.append((score,x,y,q));count+=1
            if count==6:break
    for score,x,y,q in chosen:rows.append(dict(qr=k+1,x=x+x0,y=y+y0,quadrant=q+1,score=score))
for grid in [False,True]:
    canvas=Image.new('RGB',(3*pw+2*gap,ph),(22,24,27));d=ImageDraw.Draw(canvas)
    for k,(x0,y0,x1,y1) in enumerate(BOXES):
        left=k*(pw+gap)
        tile=Image.fromarray(original[y0:y1,x0:x1]).convert('RGB').resize((pw,ph),Image.Resampling.LANCZOS)
        canvas.paste(tile,(left,0))
        if grid:
            for t in range(0,ph,30):d.line((left+pw//2,t,left+pw//2,min(t+16,ph)),fill=green,width=4)
            for t in range(0,pw,30):d.line((left+t,ph//2,left+min(t+16,pw),ph//2),fill=green,width=4)
        for r in rows:
            if r['qr']!=k+1:continue
            x=left+(r['x']-x0)*pw/(x1-x0);y=(r['y']-y0)*ph/(y1-y0)
            d.ellipse((x-12,y-12,x+12,y+12),fill=(15,25,16))
            d.ellipse((x-9,y-9,x+9,y+9),fill=green)
    canvas.save(OUT/('slide_corners_4regions.png' if grid else 'slide_corners.png'))
    if not grid:
        for k in range(3):canvas.crop((k*(pw+gap),0,k*(pw+gap)+pw,ph)).save(OUT/f'slide_QR{k+1}.png')
pd.DataFrame(rows).to_csv(OUT/'slide_representative_corners.csv',index=False)
(OUT/'slide_metadata.json').write_text(json.dumps(dict(points=len(rows),selection='Otsu black contours, polygon simplification epsilon 1.4px, near-right angles, segment length >=4px, minimum separation 9px, top 6 responses per quadrant; representative illustration only; not tracking validation',green_rgb=green,marker_diameter_output_pixels=18,scale_xy=5),indent=2),encoding='utf-8')
print('representative corners',len(rows))
