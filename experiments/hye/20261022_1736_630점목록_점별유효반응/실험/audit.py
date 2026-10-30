"""Descriptive inventory only; no training or performance evaluation."""
from pathlib import Path
import csv,json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
corners=json.loads((ROOT/'joint_center_630_20261022/assets/corners.json').read_text())
z=np.load(ROOT/'joint_center_630_20261022/results/measurements.npz')
rows=list(csv.DictReader((ROOT/'joint_center_630_20261022/results/manifest.csv').open(encoding='utf-8-sig')))
D=z['D']; valid=z['valid']; q=z['quality']; points=z['points']
Y=np.array([float(r['Y']) for r in rows]); T=np.array([float(r['T']) for r in rows])
single=np.array([r['scope']=='single' for r in rows])
slopes=[]
for labels,other in [(Y,T),(T,Y)]:
    mask=single&(labels!=0)&(other==0)
    A=np.c_[np.ones(mask.sum()),labels[mask]]
    slopes.append(np.linalg.lstsq(A,D[mask].reshape(mask.sum(),-1),rcond=None)[0][1].reshape(630,2))
repeats=[]
for condition in sorted({r['condition'] for r in rows if r['scope']=='mixed'}):
    idx=[i for i,r in enumerate(rows) if r['scope']=='mixed' and r['condition']==condition]
    repeats.append(np.std(D[idx],axis=0,ddof=1))
repeat_rms=np.sqrt(np.mean(np.square(repeats),axis=0))
out=[]
for j,c in enumerate(corners):
    a,b=slopes[0][j],slopes[1][j]
    cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    patch=''.join(str(int(v)) for v in np.array(c['patch']).ravel())
    out.append(dict(id=int(z['ids'][j]),qr=c['qr'],gx=c['gx'],gy=c['gy'],x=points[j,0],y=points[j,1],patch=patch,valid_count=int(valid[:,j].sum()),valid_rate=float(valid[:,j].mean()),rmse_median=float(np.median(q[:,j,0])),stability_median=float(np.median(q[:,j,1])),contrast_median=float(np.median(q[:,j,2])),sigma_u_median=float(np.median(q[:,j,3])),sigma_v_median=float(np.median(q[:,j,4])),Y_dx=a[0],Y_dy=a[1],T_dx=b[0],T_dy=b[1],YT_cosine=cos,repeat_sd_x=repeat_rms[j,0],repeat_sd_y=repeat_rms[j,1]))
with (HERE/'points630.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
summary=dict(images=len(rows),points=len(corners),D_shape=list(D.shape),id_unique=len(set(z['ids'].tolist())),quality_shape=list(q.shape),valid_fraction=float(valid.mean()),always_valid=int(np.all(valid,axis=0).sum()),never_valid=int((~np.any(valid,axis=0)).sum()),per_image_valid_min=int(valid.sum(1).min()),per_image_valid_max=int(valid.sum(1).max()),per_point_valid_min=int(valid.sum(0).min()),per_point_valid_max=int(valid.sum(0).max()),median_abs_YT_cosine=float(np.median(np.abs([r['YT_cosine'] for r in out]))),repeat_sd_median_xy=np.median(repeat_rms,axis=0).tolist(),qr=[])
for qr in (1,2,3):
    idx=np.array([c['qr']==qr for c in corners]);sub=[r for r in out if r['qr']==qr]
    summary['qr'].append(dict(qr=qr,count=int(idx.sum()),x_range=[float(points[idx,0].min()),float(points[idx,0].max())],y_range=[float(points[idx,1].min()),float(points[idx,1].max())],patch_counts={p:sum(r['patch']==p for r in sub) for p in sorted({r['patch'] for r in sub})},valid_fraction=float(valid[:,idx].mean()),Y_slope_median=np.median(slopes[0][idx],axis=0).tolist(),T_slope_median=np.median(slopes[1][idx],axis=0).tolist()))
summary['quality_medians']=np.median(q,axis=(0,1)).tolist()
summary['worst_valid_points']=sorted(out,key=lambda r:r['valid_rate'])[:10]
(HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='worst_valid_points'},ensure_ascii=False,indent=2))
print('worst IDs',[(r['id'],r['qr'],r['valid_count']) for r in summary['worst_valid_points']])
