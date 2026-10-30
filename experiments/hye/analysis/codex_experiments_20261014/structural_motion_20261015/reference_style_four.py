from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent;BASE=P.parent
sys.path[:0]=[str(BASE/'deps'),str(BASE/'measurement')]
import cv2,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection,PolyCollection
from measure import ROOT
OUT=P/'reference_style_four';OUT.mkdir(exist_ok=True)
z=np.load(P/'tracks.npz');pts=z['points'];D=z['D'];valid=z['valid'];names=z['names'].astype(str)
entries=json.loads((P/'sequences.json').read_text(encoding='utf-8'))
xa,xb,ya,yb=1540,2270,950,1193
records=[]
for axis,slug in [('Y','Y_plus_010'),('T','T_plus')]:
    e=next(e for e in entries if e['slug']==slug);ii=np.array(e['indices']);vals=np.array([float(names[i][1:-4]) for i in ii])
    d=D[ii]-D[ii[0]];P0=pts+D[ii[0]]
    fit=(np.vander(vals,3)@np.linalg.lstsq(np.vander(vals,3),d.reshape(len(vals),-1),rcond=None)[0]).reshape(d.shape)
    smooth=P0[None]+(fit-fit[0])*[1,100];raw=P0[None]+d*[1,100]
    im=cv2.imdecode(np.fromfile(ROOT/('Y' if axis=='Y' else 'Theta')/names[ii[0]],np.uint8),0)
    def canvas(vmin,vmax):
        fig=plt.figure(figsize=(18,18*(yb-ya)/(xb-xa)),facecolor='#16181d');ax=fig.add_axes([0,0,1,1])
        ax.imshow(im[ya:yb,xa:xb],cmap='gray',extent=[xa,xb,yb,ya],vmin=vmin,vmax=vmax)
        ax.set(xlim=(xa,xb),ylim=(yb,ya));ax.set_aspect('equal');ax.set_axis_off();return fig,ax
    norm=plt.Normalize(abs(vals[0]),abs(vals[-1]));cmap=plt.get_cmap('plasma')
    fig,ax=canvas(-60,420);segments=[];colors=[];arrows=[]
    for j in range(len(pts)):
        v=valid[ii,j];q=smooth[:,j];seg=np.stack([q[:-1],q[1:]],axis=1);good=v[:-1]&v[1:]
        segments.extend(seg[good]);colors.extend(cmap(norm(abs(vals[1:])))[good])
        end=q[-1];tangent=q[-1]-q[-4];u=tangent/max(np.linalg.norm(tangent),1e-12);normal=np.array([-u[1],u[0]])
        # Original arrow extension: 7 image-coordinate pixels beyond fitted endpoint.
        tip=end+7*u;arrows.append([tip,end+2.8*normal,end-2.8*normal])
    ax.add_collection(LineCollection(segments,colors=colors,linewidths=3.2,capstyle='round',zorder=3))
    ax.scatter(raw[:,:,0].ravel(),raw[:,:,1].ravel(),s=5,c=np.repeat(cmap(norm(abs(vals))),len(pts),axis=0),alpha=.35,edgecolors='none',zorder=2)
    ax.add_collection(PolyCollection(arrows,facecolors=[cmap(1.)],edgecolors='none',zorder=5))
    ax.scatter(P0[:,0],P0[:,1],s=34,c='white',edgecolors='black',linewidths=.8,zorder=6)
    fig.savefig(OUT/f'{axis}_paths_755.png',dpi=180,facecolor='#16181d');plt.close(fig)
    # Original residual definition: last displacement minus QR-wide median, x50.
    residual=d[-1]-np.median(d[-1],axis=0);mag=np.linalg.norm(residual,axis=1)
    n2=plt.Normalize(0,np.percentile(mag,98));cm2=plt.get_cmap('cool')
    fig,ax=canvas(-150,600)
    ax.scatter(P0[:,0],P0[:,1],s=9,c='#9EE4FA',edgecolors='none',zorder=2)
    ax.quiver(P0[:,0],P0[:,1],residual[:,0],residual[:,1],color=cm2(n2(mag)),angles='xy',scale_units='xy',scale=1/50,width=.0028,headwidth=3.5,headlength=4,zorder=3)
    fig.savefig(OUT/f'{axis}_residual_755.png',dpi=180,facecolor='#16181d');plt.close(fig)
    np.savez_compressed(OUT/f'{axis}_display_data.npz',point_ids=np.arange(755),P0=P0,measured=d,fitted=fit,displayed_paths=smooth,residual=residual,files=names[ii])
    records.append(dict(axis=axis,points=755,files=names[ii].tolist(),style_source='analysis_v7b/trial_tracks_ppt.py',path_display='quadratic trend, x1/y100, plasma command colors, faint raw dots, white/black starts',residual_display='last relative displacement minus per-component median across 755 QR points, x50, cool colormap normalized by 98th percentile',tracking='All coordinates measured directly from fixed center. Subtract first displacement for display only, no new reference tracking or model correction.',viewport=[xa,xb,ya,yb]))
(OUT/'settings.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('Four panels saved, 755 points per panel; original path/residual semantics preserved')
