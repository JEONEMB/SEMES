from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent;BASE=P.parent
sys.path[:0]=[str(BASE/'deps'),str(BASE/'measurement')]
import cv2,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection,PolyCollection
from PIL import Image,ImageDraw,ImageFont
from measure import ROOT
plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False})
z=np.load(P/'tracks.npz');pts=z['points'];D=z['D'];valid=z['valid'];conf=z['confidence']
entries=json.loads((P/'sequences.json').read_text(encoding='utf-8'));reps=pd.read_csv(P/'representative_ids.csv').point_id.to_numpy()
bg=cv2.imdecode(np.fromfile(ROOT/'center.bmp',np.uint8),0)
OUT=P/'slide_panels_bold';OUT.mkdir(exist_ok=True)
xa,xb,ya,yb=1480,2320,935,1215
for e in entries:
    ii=np.array(e['indices']);n=len(ii)
    for selection,ids in [('all755',np.arange(len(pts))),('representative36',reps)]:
        for amp in [1,100]:
            coords=pts[None]+D[ii]*[1,amp];allseg=[];allcolors=[];dashed=[]
            for j in ids:
                seg=np.stack([coords[:-1,j],coords[1:,j]],axis=1)
                ok=valid[ii[:-1],j]&valid[ii[1:],j]
                colors=plt.get_cmap('plasma')(np.linspace(0,1,n-1));colors[:,3]=(.55 if selection=='all755' else .95)+.04*conf[j]
                if e['extra_endpoint']:
                    if ok[-1]:dashed.append(seg[-1])
                    ok[-1]=False
                allseg.extend(seg[ok]);allcolors.extend(colors[ok])
            fig=plt.figure(figsize=(16,16*(yb-ya)/(xb-xa)),facecolor='#17191f');ax=fig.add_axes([0,0,1,1])
            ax.imshow(bg[ya:yb,xa:xb],cmap='gray',extent=[xa,xb,yb,ya],vmin=-30,vmax=320)
            ax.add_collection(LineCollection(allseg,colors=allcolors,linewidths=2.5 if selection=='all755' else 5.5))
            if dashed:ax.add_collection(LineCollection(dashed,colors='#ffe34e',linewidths=2.0 if selection=='all755' else 4.0,linestyles='dashed',alpha=.8))
            good=ids[valid[ii[0],ids]]
            ax.scatter(coords[0,good,0],coords[0,good,1],s=(18 if selection=='all755' else 105),marker='s',c='#8bfa83',edgecolors='black',linewidths=.5,zorder=5)
            arrows=[]
            for j in ids[valid[ii[-1],ids]]:
                tip=coords[-1,j];direction=coords[-1,j]-coords[0,j]
                unit=direction/max(np.linalg.norm(direction),1e-10);normal=np.array([-unit[1],unit[0]])
                size=5.5 if selection=='all755' else 10
                arrows.append([tip,tip-size*unit+size*.45*normal,tip-size*unit-size*.45*normal])
            ax.add_collection(PolyCollection(arrows,facecolors='#ffe34e',edgecolors='none',zorder=5))
            ax.set(xlim=(xa,xb),ylim=(yb,ya));ax.set_aspect('equal');ax.set_axis_off()
            target=OUT/f'{e["slug"]}_{selection}_dy{amp}.png'
            fig.savefig(target,dpi=180,facecolor='#17191f');plt.close(fig)
    # Large slide-ready two-panel composition, all new points by default.
    a=Image.open(OUT/f'{e["slug"]}_representative36_dy1.png').convert('RGB');b=Image.open(OUT/f'{e["slug"]}_representative36_dy100.png').convert('RGB')
    w,h=a.size;canvas=Image.new('RGB',(w,2*h+290),(23,25,31));draw=ImageDraw.Draw(canvas)
    font=lambda s:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',s)
    title=f'{e["axis"]} {e["start"]:+.2f} → {e["end"]:+.2f}  |  대표 36점 · 굵은 경로'
    draw.text((30,15),title,font=font(56),fill=(245,246,249))
    draw.text((30,90),'실제 변위: 가로 ×1 · 세로 ×1',font=font(37),fill=(215,220,227));canvas.paste(a,(0,145))
    draw.text((30,h+155),'확대 변위: 가로 ×1 · 세로 ×100 (길이·기울기 확대)',font=font(37),fill=(215,220,227));canvas.paste(b,(0,h+210))
    footer='대표점: center 품질·공간 간격으로 선택 · center 고정 추적 · 명령값 크기순 연결 · 보라 → 노랑 · 연두 네모: 시작 / 노랑: 끝'
    if e['extra_endpoint']:footer+=' · 1.95→2.00 별도 끝점 점선'
    draw.text((25,2*h+235),footer,font=font(29),fill=(215,220,227))
    canvas.save(OUT/f'{e["slug"]}_발표용.png')
    print('SLIDE',e['slug'],flush=True)

