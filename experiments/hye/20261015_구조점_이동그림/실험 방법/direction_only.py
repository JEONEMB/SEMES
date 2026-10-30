from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent;BASE=P.parent
sys.path[:0]=[str(BASE/'deps'),str(BASE/'measurement')]
import cv2,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from measure import ROOT
OUT=P/'positive_direction_only';OUT.mkdir(exist_ok=True)
z=np.load(P/'tracks.npz');names=z['names'].astype(str);pts=z['points'];D=z['D'];valid=z['valid']
entries=json.loads((P/'sequences.json').read_text(encoding='utf-8'))
bg=cv2.imdecode(np.fromfile(ROOT/'center.bmp',np.uint8),0)
records=[]
for axis,slug in [('Y','Y_plus_005'),('T','T_plus')]:
    e=next(e for e in entries if e['slug']==slug);ii=np.array(e['indices'])
    # Actual image-plane direction, endpoint minus start. No vertical exaggeration.
    delta=D[ii[-1]]-D[ii[0]];length=np.linalg.norm(delta,axis=1)
    good=valid[ii[-1]]&valid[ii[0]]&(length>1e-8)
    unit=delta/np.maximum(length[:,None],1e-8)
    assert len(pts)==755 and good.all()
    polygons=[]
    for j in range(len(pts)):
        start=pts[j];u=unit[j];normal=np.array([-u[1],u[0]])
        # Equal length arrows. Thick shaft and broad head, all points retained.
        L=5.5;neck=start+u*(L-2.1);tip=start+u*L
        polygons.append([start+normal*.55,neck+normal*.55,neck+normal*1.45,tip,neck-normal*1.45,neck-normal*.55,start-normal*.55])
        records.append(dict(axis=axis,point_id=j,center_x=start[0],center_y=start[1],direction_x=u[0],direction_y=u[1],valid=bool(good[j])))
    for mode,bounds in [('all',(1538,2270,953,1191)),('QR1',(1545,1748,960,1180)),('QR2',(1802,2006,962,1182)),('QR3',(2060,2265,964,1184))]:
        a,c,b,d=bounds
        fig=plt.figure(figsize=(18,18*(d-b)/(c-a)),facecolor='#17191f');ax=fig.add_axes([0,0,1,1])
        ax.imshow(bg[b:d,a:c],cmap='gray',extent=[a,c,d,b],vmin=-35,vmax=325)
        ax.add_collection(PolyCollection(polygons,facecolors='#D9FA56',edgecolors='none',zorder=4))
        ax.set(xlim=(a,c),ylim=(d,b));ax.set_aspect('equal');ax.set_axis_off()
        fig.savefig(OUT/f'{axis}_positive_direction_{mode}.png',dpi=180,facecolor='#17191f');plt.close(fig)
pd.DataFrame(records).to_csv(OUT/'directions.csv',index=False)
(OUT/'method.json').write_text(json.dumps(dict(points_per_axis=755,direction='Endpoint minus first photo in positive command sequence; unit vector in original image pixels; same center ID positions used as arrow anchors.',source_sequences=['Y_plus_005','T_plus'],arrow_length_display_source_px=5.5,anisotropic_scaling=False,meaning='Direction only; equal arrow length does not represent displacement magnitude; these are measured net image directions, not forced right-facing icons.'),indent=2),encoding='utf-8')
print('755 normalized arrows per axis; positive Y and T PNGs saved')
