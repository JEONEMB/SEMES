"""Render measured point displacements for a reproducible 20-image selection.

Uses the fixed-model full280 prediction table and independently re-extracts
the selected original photos to verify the cached measurement arrays.
"""
from pathlib import Path
import csv,json,sys,time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
sys.path.insert(0,str(BASE/'measurement'))
from spread import SpreadPairExtractor
from measure import ROOT,BOXES

CROP=[1450,945,2370,1190]
COLORS=['#0086D1','#C32670','#008360']
REGION_NAMES=['Left QR','Center QR','Right QR']
cv2.setNumThreads(1)

def read_full(path):
    im=cv2.imdecode(np.fromfile(path,np.uint8),0)
    if im is None:raise ValueError(str(path))
    return im

def crop_image(im):
    x0,y0,x1,y1=CROP
    return im[y0:y1,x0:x1].copy()

def row_value(r,key):
    if key in ['axis_sign_correct','passes_all_requirements']:return r[key].lower()=='true'
    return float(r[key])

def figure(sample,points,regions,center_crop,current_crop):
    D=np.array(sample['D']);current=points+D;factor=sample['arrow_scale']
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.045,.957,f"{sample['index']:02d}   {sample['file']}  |  {sample['group_label']}",fontsize=20,fontweight='bold',color='#13253D')
    status='PASS' if sample['axis_sign_correct'] else 'FAIL'
    tolerance='PASS' if sample['passes_all_requirements'] else 'FAIL'
    subtitle=(f"Truth: {sample['axis']} {sample['value']:+.2f}     Prediction: {sample['pred_axis']} {sample['pred_value']:+.6f}"
              f"     |error Y| {sample['error_Y']:.6f}     |error T| {sample['error_T']:.6f}"
              f"     Axis/sign {status}  /  tolerance {tolerance}")
    fig.text(.045,.921,subtitle,fontsize=12.1,color='#24364A')
    for box,image,pts,title in [([.045,.675,.44,.21],center_crop,points,'CENTER  |  205 fixed reference points'),([.515,.675,.44,.21],current_crop,current,'CURRENT  |  205 measured point positions')]:
        ax=fig.add_axes(box);ax.imshow(image,cmap='gray',vmin=0,vmax=255,origin='upper',interpolation='nearest')
        for r,color in enumerate(COLORS):
            p=pts[regions==r]-np.array(CROP[:2]);ax.scatter(p[:,0],p[:,1],s=4.2,c=color,edgecolors='white',linewidths=.16,zorder=3)
        ax.set_xlim(0,CROP[2]-CROP[0]);ax.set_ylim(CROP[3]-CROP[1],0);ax.set_axis_off();ax.set_title(title,loc='left',fontsize=11,pad=7,fontweight='bold',color='#24364A')
    ax=fig.add_axes([.065,.255,.87,.355])
    for r,color in enumerate(COLORS):
        sel=regions==r;p=points[sel];d=D[sel]
        x0,y0,x1,y1=BOXES[r]
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#D0D7DE',lw=1))
        ax.scatter(p[:,0],p[:,1],s=4.0,color=color,alpha=.72,zorder=3)
        ax.quiver(p[:,0],p[:,1],d[:,0]*factor,d[:,1]*factor,angles='xy',scale_units='xy',scale=1,color=color,width=.0011,headwidth=3.1,headlength=4,headaxislength=3.5,alpha=.84,zorder=4)
        ax.text((x0+x1)/2,960,REGION_NAMES[r],color=color,ha='center',fontsize=11,fontweight='bold')
    ax.set_xlim(1530,2290);ax.set_ylim(1182,950);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Original image x coordinate [pixel]',fontsize=10);ax.set_ylabel('Original image y [pixel]',fontsize=10)
    ax.tick_params(labelsize=9);ax.grid(True,alpha=.17,lw=.6)
    ax.set_title(f'MEASURED DISPLACEMENT  |  all 205 vectors  |  arrows ×{factor:g}; original pixel units',loc='left',fontsize=11.5,pad=13,fontweight='bold',color='#24364A')
    for spine in ax.spines.values():spine.set_color('#C4CDD6')
    stats=fig.add_axes([.13,.073,.74,.114]);stats.axis('off')
    values=[]
    for q in sample['qr_means']:values.append([q['region_name'],str(q['point_count']),f"{q['mean_dx']:+.6f}",f"{q['mean_dy']:+.6f}"])
    values.append(['ALL POINTS','205',f"{sample['mean_dx']:+.6f}",f"{sample['mean_dy']:+.6f}"])
    table=stats.table(cellText=values,colLabels=['Region','Points','Mean dx [pixel]','Mean dy [pixel]'],loc='center',cellLoc='center',colWidths=[.28,.15,.285,.285])
    table.auto_set_font_size(False);table.set_fontsize(10.5);table.scale(1,1.13)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#DEE5EC');cell.set_linewidth(.65)
        if r==0:cell.set_facecolor('#EAF0F6');cell.set_text_props(weight='bold',color='#18314A')
        elif r==4:cell.set_facecolor('#F1F5F9');cell.set_text_props(weight='bold')
        elif c==0:cell.set_text_props(color=COLORS[r-1],weight='bold')
    fig.text(.045,.017,'Fixed-model replay of recorded calibration images; this is not an independent generalization test.  Error tolerance: Y ≤ 0.09 and T ≤ 0.09.',fontsize=9,color='#546373')
    fig.savefig(sample['png_path'],dpi=150,facecolor='white')
    plt.close(fig)

def main():
    start=time.time();HERE.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((BASE/'full280'/'predictions.csv').open(encoding='utf-8-sig')))
    near=[]
    for axis in ['Y','T']:
        near.extend(sorted([r for r in rows if r['true_axis']==axis],key=lambda r:(abs(float(r['true_value'])),-float(r['true_value'])))[:5])
    selected=list(near);near_names={r['file'] for r in near};rng=np.random.default_rng(20261014)
    for axis in ['Y','T']:
        pool=sorted([r for r in rows if r['true_axis']==axis and r['file'] not in near_names],key=lambda r:float(r['true_value']))
        selected.extend(pool[int(j)] for j in rng.choice(len(pool),size=5,replace=False))
    cache=np.load(BASE/'measurement'/'gn17_spread14.npz');cache_names=cache['names'].astype(str).tolist()
    points=cache['points'].astype(float);regions=cache['regions'].astype(int)
    assert len(set(r['file'] for r in selected))==20 and len(points)==205
    center_path=ROOT/'center.bmp';center_full=read_full(center_path);center_crop=crop_image(center_full)
    center_jpg=HERE/'center_crop.jpg';assert cv2.imwrite(str(center_jpg),center_crop,[cv2.IMWRITE_JPEG_QUALITY,65])
    extractor=SpreadPairExtractor(center_path,features=BASE/'measurement'/'gn17_spread14_features.npz')
    common=dict(points=points.tolist(),regions=regions.tolist(),crop=CROP,center_jpeg_path=str(center_jpg),samples=[])
    detail=dict(schema_version=1,selection={'near_zero':'Five smallest absolute commands per axis, positive first on ties; Y then T.','random':'Five remaining Y then five remaining T; pools numerically sorted, numpy default_rng sampling without replacement.','random_seed':20261014},center_original_path=str(center_path),center_shape_px=[3840,2160],crop_metadata={'xyxy_original_px':CROP,'width_px':CROP[2]-CROP[0],'height_px':CROP[3]-CROP[1],'jpeg_quality':65},source_predictions_csv=str(BASE/'full280'/'predictions.csv'),source_measurements_npz=str(BASE/'measurement'/'gn17_spread14.npz'),coordinate_convention='original-image pixel coordinates, x right/y down; D=current-center',quality_columns=['photometric_rmse_gray','last_step_px','gain','offset_gray','LK_forward_backward_error_px'],arrows_note='All 205 raw measured displacement vectors. A uniform annotated scale per image controls length; coordinates stay in original pixels. Arrow scale is not a physical motion magnification.',scaling_rule='Largest factor from [0.25,0.5,1,2,5,10,20,50,100] keeping the 95th-percentile displayed arrow length <=22 original-coordinate pixels.',samples=[])
    scales=np.array([.25,.5,1,2,5,10,20,50,100])
    for number,row in enumerate(selected,1):
        name=row['file'];path=ROOT/('Y' if name[0]=='Y' else 'Theta')/name;ci=cache_names.index(name)
        result=extractor.extract(path);D=result['D'].astype(float);cached=cache['D'][ci].astype(float)
        delta=float(np.max(np.abs(D-cached)));assert np.array_equal(result['points'],points),name
        assert delta<=1e-5,(name,delta)
        group='near-zero' if number<=10 else 'random';slug=f'{number:02d}_'+name[:-4].replace('+','p').replace('-','m').replace('.','_')
        current_crop=crop_image(read_full(path));current_jpg=HERE/(slug+'_current.jpg');assert cv2.imwrite(str(current_jpg),current_crop,[cv2.IMWRITE_JPEG_QUALITY,65])
        max_scale=22/max(float(np.quantile(np.linalg.norm(D,axis=1),.95)),1e-6);eligible=scales[scales<=max_scale];factor=float(eligible[-1] if len(eligible) else scales[0])
        sample=dict(index=number,group=group,group_label='NEAR ZERO' if group=='near-zero' else 'SEEDED RANDOM',file=name,axis=row['true_axis'],value=float(row['true_value']),pred_axis=row['pred_axis'],pred_value=float(row['pred_value']),error_Y=float(row['abs_error_Y']),error_T=float(row['abs_error_T']),axis_sign_correct=row_value(row,'axis_sign_correct'),passes_all_requirements=row_value(row,'passes_all_requirements'),mean_dx=float(D[:,0].mean()),mean_dy=float(D[:,1].mean()),D=D.tolist(),current_jpeg_path=str(current_jpg),png_path=str(HERE/(slug+'.png')),arrow_scale=factor,cache_max_abs_difference_px=delta,qr_means=[])
        for r in range(3):
            sel=regions==r;sample['qr_means'].append(dict(region=r,region_name=REGION_NAMES[r],point_count=int(sel.sum()),mean_dx=float(D[sel,0].mean()),mean_dy=float(D[sel,1].mean())))
        common['samples'].append(sample)
        detail['samples'].append(dict(**sample,source_current_path=str(path),source_center_path=str(center_path),point_coordinates_px=points.tolist(),point_regions=regions.tolist(),current_point_coordinates_px=(points+D).tolist(),quality=result['quality'].tolist(),valid=result['valid'].tolist(),valid_count=int(result['valid'].sum()),coarse_qr_ncc_scores=result['coarse_score'].tolist(),runtime_seconds=float(result['runtime_seconds']),prediction_truth={'true_Y':float(row['true_Y']),'true_T':float(row['true_T']),'pred_Y':float(row['pred_Y']),'pred_T':float(row['pred_T'])},crop_metadata=detail['crop_metadata']))
        # Save the machine-readable evidence as it becomes available for the gallery builder.
        (HERE/'compact.json').write_text(json.dumps(common,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
        (HERE/'data.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf-8')
        figure(sample,points,regions,center_crop,current_crop)
        print(number,name,'max_cache_diff',delta,'arrows',factor,'elapsed',round(time.time()-start,1),flush=True)
    summary=dict(n=20,near_zero=10,random=10,all_selected_reextracted=True,max_cache_abs_difference_px=max(s['cache_max_abs_difference_px'] for s in common['samples']),all_selected_valid_points=all(s['valid_count']==205 for s in detail['samples']),png_count=len(list(HERE.glob('*.png'))),current_jpeg_count=len(list(HERE.glob('*_current.jpg'))),runtime_seconds=time.time()-start)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(summary,flush=True)

if __name__=='__main__':main()
