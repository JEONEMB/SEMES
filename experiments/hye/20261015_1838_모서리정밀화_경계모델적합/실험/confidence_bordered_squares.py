from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent;BASE=P.parent
sys.path[:0]=[str(BASE/'deps'),str(BASE/'measurement')]
import numpy as np,pandas as pd,cv2
from PIL import Image,ImageDraw,ImageFont
from measure import ROOT,BOXES
OUT=P/'confidence_bordered_squares';OUT.mkdir(exist_ok=True)
df=pd.read_csv(P/'refined_corners.csv').fillna({'reason':''})
# Fixed quality scales, shared by all three QR images; not per-image normalization.
photometric=np.exp(-.5*(df.rmse/15)**2)
stability=np.exp(-.5*(df.window_stability_px/.65)**2)
contrast=np.clip(df.contrast/90,0,1)
df['quality_score']=photometric*stability*contrast
hard=df.reason.str.contains('optimizer|search_limit|blur_limit')
df.loc[hard,'quality_score']*=.25
df['quality_score']=df.quality_score.clip(0,1)
df.to_csv(OUT/'corners_with_quality.csv',index=False,encoding='utf-8-sig')
raw=cv2.imdecode(np.fromfile(ROOT/'center.bmp',np.uint8),0)
font=lambda s:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',s)
stops=np.array([[244,105,64],[255,216,81],[139,250,131]],float)
def color(q):
    pos=float(np.clip(q,0,1))*2;i=min(int(pos),1);f=pos-i
    return tuple(np.rint(stops[i]*(1-f)+stops[i+1]*f).astype(int))
def panel(k,box,size,grid=False):
    a,b,c,d=box;w,h=size
    im=Image.fromarray(raw[b:d,a:c]).convert('RGB').resize((w,h),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(im)
    if grid:
        for y in range(0,h,32):draw.line((w//2,y,w//2,min(y+16,h)),fill=(215,225,233),width=3)
        for x in range(0,w,32):draw.line((x,h//2,min(x+16,w),h//2),fill=(215,225,233),width=3)
    # Filled marker diameter is 4.6 source pixels, versus previously tiny hollow centers.
    radius=2.3*w/(c-a)
    for r in df[df.qr==k].itertuples():
        if not(a<=r.x<c and b<=r.y<d):continue
        x,y=(r.x-a)*w/(c-a),(r.y-b)*h/(d-b)

        draw.rectangle((x-radius,y-radius,x+radius,y+radius),fill=color(r.quality_score),outline=(0,0,0),width=max(2,int(round(w/(c-a)*.4))))
    return im
def decorate(im,title,subtitle):
    w,h=im.size;header=115;footer=155
    out=Image.new('RGB',(w,h+header+footer),(22,25,29));out.paste(im,(0,header));d=ImageDraw.Draw(out)
    fs=40 if w>1500 else 30
    d.text((24,16),title,font=font(fs),fill=(244,247,249))
    d.text((24,70),subtitle,font=font(27 if w>1500 else 23),fill=(205,215,224))
    bw=min(w-100,900);left=(w-bw)//2;top=header+h+24
    for x in range(bw):d.line((left+x,top,left+x,top+22),fill=color(x/(bw-1)),width=1)
    d.text((left,top+30),'낮음  0',font=font(23),fill=(228,233,238))
    d.text((left+bw-105,top+30),'1  높음',font=font(23),fill=(228,233,238))
    d.text((24,top+80),'확신도 = 경계 적합·위치 안정성·대비에 따른 품질 점수 (정확도 확률 아님)',font=font(26 if w>1500 else 20),fill=(205,215,224))
    return out
for four in [False,True]:
    w,h,gap=1104,1188,55
    canvas=Image.new('RGB',(3*w+2*gap,h),(22,25,29))
    for k,box in enumerate(BOXES,1):
        im=panel(k,box,(w,h),four);canvas.paste(im,((k-1)*(w+gap),0))
        if not four:
            im.save(OUT/f'QR{k}_filled_clean.png')
            decorate(im,f'QR {k} · 서브픽셀 모서리 확신도','전체 모서리 표시 · 가로·세로 동일 6배').save(OUT/f'QR{k}_filled_legend.png')
    canvas.save(OUT/('all_filled_4regions_clean.png' if four else 'all_filled_clean.png'))
    decorate(canvas,'실제 영상 경계 기반 · 특징점 확신도',f'전체 {len(df)}개 모서리 · 검정 테두리의 채운 네모 · 가로·세로 동일 6배').save(OUT/('발표용_확신도_4분할.png' if four else '발표용_확신도_전체.png'))
detail=panel(1,(1557,976,1623,1045),(1056,1104))
decorate(detail,'큰 네모 · 모서리 확신도','가로·세로 동일 16배 · 실제 경계 보정').save(OUT/'발표용_확신도_큰네모.png')
settings=dict(total=len(df),quality_score='exp(-0.5*(rmse/15)^2) * exp(-0.5*(window_stability_px/0.65)^2) * clip(contrast/90,0,1); multiply by .25 for optimizer/search_limit/blur_limit flags',meaning='Heuristic measurement quality; not calibrated probability or absolute accuracy. Original valid/reason columns preserved. No points hidden.',color_stops_rgb=stops.tolist(),filled_marker_source_diameter_px=4.6,quality_quantiles=df.quality_score.quantile([0,.25,.5,.75,1]).to_dict())
(OUT/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
print(json.dumps(settings['quality_quantiles']))


