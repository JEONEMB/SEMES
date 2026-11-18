"""Four reference images, ONE fit per upper/lower protruding structure.
Visibility gaps are masked before fitting; visible points share one slope AND one intercept.
No independent left/right line fits or angle averaging. No controller/calibration changes.
"""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'analysis/reference_slope_check_20261029/reference_slope_results.json'
HELPER=ROOT/'analysis/theta_line_hypothesis/measure_lines.py'
spec=importlib.util.spec_from_file_location('saved_line_measurement',HELPER)
helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
refs=json.loads(OLD.read_text(encoding='utf-8'))['rows']
assert len(refs)==4
cv2.setNumThreads(2)

# Frozen from first-reference geometry, before comparing four-image measurements.
# Pixel coordinates are original 3840x2160. Endpoints of exclusion intervals are inclusive.
# Each search band tracks the same dark-to-light (+vertical-gradient) boundary of the rail.
REGIONS={
    'upper':{'x_range':[24,3816],'search_y':[692,736],'exclude_x':[[1370,2450]],
             'visibility_reason':'Central connector and QR mounting assembly covers the upper rail'},
    'lower':{'x_range':[24,3816],'search_y':[1845,1905],'exclude_x':[[1560,2300]],
             'visibility_reason':'Central lower bracket/reflection covers the lower rail boundary'},
}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def extract_points(image,region):
    x0,x1=region['x_range']; low,high=region['search_y']
    y0,y1=low-8,high+9
    patch=image[y0:y1,x0:x1+1].astype(np.float32)
    blurred=cv2.GaussianBlur(patch,(0,0),1.0,borderType=cv2.BORDER_REFLECT101)
    gradient=np.gradient(blurred,axis=0)
    columns=np.arange(2,patch.shape[1]-2,2)
    visible=np.ones(len(columns),dtype=bool)
    for a,b in region['exclude_x']: visible &= ((x0+columns<a)|(x0+columns>b))
    columns=columns[visible]
    peak=np.argmax(gradient[8:-8,columns],axis=0)+8
    ga,gb,gc=gradient[peak-1,columns],gradient[peak,columns],gradient[peak+1,columns]
    denominator=ga-2*gb+gc
    offset=np.divide(.5*(ga-gc),denominator,out=np.zeros_like(gb),where=denominator < -1e-8)
    offset=np.clip(offset,-.5,.5)
    valid=(peak>8)&(peak<gradient.shape[0]-9)&(gb>0)
    sample_y=peak[valid][None,:]+np.arange(-5,6)[:,None]
    observed=blurred[sample_y,columns[valid][None,:]]
    initial=peak[valid]+offset[valid]
    refined,widths=helper.step_centers(observed,sample_y,initial)
    x=(x0+columns[valid]).astype(float); y=y0+refined
    assert len(x)>20 and np.isfinite(y).all() and np.isfinite(widths).all()
    return x,y,{'visible_candidate_columns':len(columns),'accepted_edge_columns':len(x),
        'excluded_boundary_or_nonpositive_gradient_columns':int(np.sum(~valid)),
        'step_center_at_bound_fraction':float(np.mean(np.abs(refined-initial)>=1.999)),
        'median_step_width_px':float(np.median(widths))}

records=[]; images=[]; points={}; legacy_differences=[]
for ref in refs:
    path=Path(ref['path']); raw=np.fromfile(path,dtype=np.uint8)
    digest=hashlib.sha256(raw).hexdigest(); assert digest==ref['sha256']
    image=cv2.imdecode(raw,0); assert image.shape==(2160,3840)
    images.append(image)
    # Same helper/engine must reproduce the old native measurements on identical old ROIs.
    for new_name,old_name in [('upper_left','rail_UL_edge'),('upper_right','rail_UR_edge')]:
        measured=helper.extract_line(image,helper.FEATURES[old_name])['angle_deg']
        difference=measured-ref['lines'][new_name]['image_angle_deg']
        assert abs(difference)<1e-6,(ref['file'],old_name,difference)
        legacy_differences.append({'file':ref['file'],'line':old_name,'python_minus_native_image_deg':difference})
    record={'file':ref['file'],'path':ref['path'],'dataset':ref['dataset'],'sha256':digest,'structures':{}}
    for name,region in REGIONS.items():
        x,y,diagnostics=extract_points(image,region)
        angle,slope,intercept,residual=helper.fit_line(x,y,robust=True)
        # The combined clouds from BOTH visible sides go into THIS ONE fit call.
        record['structures'][name]={'image_angle_deg':angle,'slope_dy_dx':slope,'intercept_y_px':intercept,
            'edge_fit_rms_px':float(np.sqrt(np.mean(residual**2))),
            'edge_fit_max_abs_residual_px':float(np.max(np.abs(residual))),
            'sample_x_min_max':[float(x.min()),float(x.max())],**diagnostics}
        points[(ref['file'],name)]=(x,y)
        assert len(region['exclude_x'])==1
        a,b=region['exclude_x'][0]
        assert np.any(x<a) and np.any(x>b) and not np.any((x>=a)&(x<=b))
        np.savez_compressed(HERE/(ref['file'].replace('.bmp','')+'_'+name+'_edge_points.npz'),x=x,y=y)
    assert sha(path)==digest
    records.append(record)

for record in records:
    for name,s in record['structures'].items():
        s['delta_from_first_center_image_deg']=s['image_angle_deg']-records[0]['structures'][name]['image_angle_deg']
ranges={}
for name in REGIONS:
    low=min(records,key=lambda r:r['structures'][name]['image_angle_deg'])
    high=max(records,key=lambda r:r['structures'][name]['image_angle_deg'])
    ranges[name]={'min_file':low['file'],'max_file':high['file'],
        'max_minus_min_image_deg':high['structures'][name]['image_angle_deg']-low['structures'][name]['image_angle_deg']}
pairs=[{'first':a['file'],'second':b['file'],
        'difference_image_deg':{name:b['structures'][name]['image_angle_deg']-a['structures'][name]['image_angle_deg'] for name in REGIONS}}
       for a,b in itertools.combinations(records,2)]
result={'scope':'Exactly four reference images; two whole-structure slopes per image (upper/lower)',
    'method':{'edge':'Existing sigma=1 Gaussian, vertical gradient, 11-sample Gaussian-blurred-step subpixel least-squares fit',
        'line':'ONE Huber line through ALL retained edge points per structure, shared slope and intercept',
        'sampling':'Every 2nd column; fixed bands and visibility masks identical for all four images',
        'coordinates':'Original pixels, x right / y down, atan(dy/dx) image degrees',
        'no_resampling':True,'no_local_angle_averaging':True,'no_robot_theta_conversion':True,
        'roi_selection':'First reference geometry only; excluded center denotes occlusion, not separate structure models'},
    'regions':REGIONS,'opencv_version':cv2.__version__,'helper_sha256':sha(HELPER),
    'legacy_native_crosscheck':legacy_differences,'rows':records,'ranges':ranges,'pairwise':pairs,
    'limitations':['These are raw IMAGE angles, not robot theta',
        'One line summarizes potentially curved/nonuniform visible boundaries; fit residual is included',
        'Single observation per saved reference, no inferential equivalence or uncertainty bound',
        'New upper/lower features do not inherit the former left/right robot-angle calibration',
        'No motion-image testing or algorithm/controller/config changes performed']}
(HERE/'full_width_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

colors={'upper':'#ff6438','lower':'#00aedb'}
fig,ax=plt.subplots(figsize=(14,8),layout='constrained')
ax.imshow(images[0],cmap='gray',vmin=0,vmax=255)
for name,region in REGIONS.items():
    color=colors[name]; x,y=points[(records[0]['file'],name)]; s=records[0]['structures'][name]
    x0,x1=region['x_range']; lo,hi=region['search_y']
    ax.add_patch(Rectangle((x0,lo),x1-x0,hi-lo,fill=False,edgecolor=color,lw=1.2))
    for a,b in region['exclude_x']:
        ax.add_patch(Rectangle((a,lo),b-a,hi-lo,facecolor='gray',alpha=.45,hatch='////',edgecolor='gray'))
    ax.scatter(x,y,s=1,color=color,alpha=.8)
    xx=np.array([x0,x1]); ax.plot(xx,s['slope_dy_dx']*xx+s['intercept_y_px'],color=color,lw=1.6,
        label=f"ONE {name} fit: {s['image_angle_deg']:.6f} deg")
    ax.text(40,lo-20,f'{name.upper()}: one shared line; center hatch = no samples',color=color,fontsize=10,
        bbox={'facecolor':'black','alpha':.5,'edgecolor':'none'})
ax.set_title('First center.bmp — upper/lower protruding structures, ONE fit each')
ax.set_xlabel('Original x (pixels)'); ax.set_ylabel('Original y (pixels)')
ax.legend(loc='upper right'); ax.set_xlim(0,3840); ax.set_ylim(2160,0)
fig.savefig(HERE/'measurement_regions.png',dpi=150); plt.close(fig)

fig,axes=plt.subplots(4,2,figsize=(16,10),layout='constrained')
for row,(record,image) in enumerate(zip(records,images)):
    for col,(name,region) in enumerate(REGIONS.items()):
        ax=axes[row,col]; x0,x1=region['x_range']; lo,hi=region['search_y']; s=record['structures'][name]
        ax.imshow(image[lo-8:hi+9],cmap='gray',vmin=0,vmax=255,extent=(0,3840,hi+9,lo-8),aspect='auto')
        for a,b in region['exclude_x']: ax.axvspan(a,b,color='gray',alpha=.45,hatch='//')
        x,y=points[(record['file'],name)]; ax.scatter(x,y,s=1,color='#00d8ff',label='edge points')
        xx=np.array([x0,x1]); ax.plot(xx,s['slope_dy_dx']*xx+s['intercept_y_px'],color='#ff633d',lw=1.2,label='one global fit')
        ax.set_title(f"{record['file']} | {name}: {s['image_angle_deg']:.6f} deg | fit RMS {s['edge_fit_rms_px']:.3f} px")
        ax.set_xlim(0,3840); ax.set_ylim(hi+9,lo-8); ax.set_xlabel('Original x (pixels)'); ax.set_ylabel('Original y (pixels)')
fig.suptitle('Four reference images: visible edge points and one fit per structure — vertical scale magnified')
fig.savefig(HERE/'four_reference_fit_details.png',dpi=130); plt.close(fig)

report=['# 기준 이미지 4장: 상단·하단 외부 구조물 전체 기울기 재측정','',
    '사용자 표시를 구조물 식별용으로 해석해, 상단 구조물 전체에 직선 하나, 하단 구조물 전체에 직선 하나를 맞췄습니다. 좌우 각도를 별도로 구해 평균내지 않았습니다. 중앙 장착물에 가려진 부분은 점 추출에서 제외하고, 양옆의 원본 경계점들을 한꺼번에 Huber 직선 맞춤에 넣었습니다.','',
    '경계 측정은 기존 Gaussian 평활 → 밝기 증가 경계 탐색 → 흐린 계단 모형의 서브픽셀 맞춤 방법을 유지했습니다. 상단·하단 각각 동일한 밝기 경계선을 추적했습니다. 네 장에 동일한 탐색 범위와 가림 마스크를 사용했습니다.','',
    '**아래는 영상 속 선의 각도입니다. 로봇 θ나 보정 명령으로 환산하지 않았습니다.**','',
    '| 기준 이미지 | 상단 전체 각도(°) | 하단 전체 각도(°) | 상단: 1차 대비 차이(°) | 하단: 1차 대비 차이(°) |','|---|---:|---:|---:|---:|']
for r in records:
    a,b=r['structures']['upper'],r['structures']['lower']
    report.append(f"| {r['dataset']} / {r['file']} | {a['image_angle_deg']:.6f} | {b['image_angle_deg']:.6f} | {a['delta_from_first_center_image_deg']:+.6f} | {b['delta_from_first_center_image_deg']:+.6f} |")
report+=['','## 네 장의 최대−최소 차이','']
for name,label in [('upper','상단'),('lower','하단')]:
    v=ranges[name]; report.append(f"- {label}: **{v['max_minus_min_image_deg']:.6f}°**, 최대 {v['max_file']}, 최소 {v['min_file']}")
report+=['','## 실제 사용 영역','',
    '- x=24~3816px의 넓은 띠를 사용했습니다. 2픽셀 간격으로 경계점을 추출했습니다.',
    '- 상단 경계 탐색 y=692~736px, 중앙 x=1370~2450px는 장착물 가림 때문에 제외했습니다.',
    '- 하단 경계 탐색 y=1845~1905px, 중앙 x=1560~2300px는 하단 장착물/반사부 가림 때문에 제외했습니다.',
    '- 범위는 첫 기준 사진의 구조물 위치를 보고 정했으며, 네 장의 결과 차이를 줄이도록 변경하지 않았습니다.',
    '- 마스크 양쪽 점은 **공통 기울기와 공통 절편을 갖는 하나의 직선**으로 맞췄습니다.',
    '', '![측정 영역 및 전체 직선](measurement_regions.png)','',
    '![네 기준 이미지의 경계점과 전체 직선](four_reference_fit_details.png)','',
    '## 재현과 한계','',
    f'- 실행 환경: Python / OpenCV {cv2.__version__}. 기존 경계 측정 함수를 그대로 재사용했습니다.',
    f'- 이전 좌우 ROI에 같은 함수를 적용한 8개 비교에서 기존 C++ 각도와의 최대 차이는 {max(abs(v["python_minus_native_image_deg"]) for v in legacy_differences):.12g}°입니다. 따라서 환경 변경으로 인한 차이와 영역 통합의 차이를 구분할 수 있습니다.',
    '- 모든 원본 이미지의 실행 전후 해시를 확인했습니다. 현재 C++ 알고리즘과 설정은 변경하지 않았습니다.',
    '- 이 비교는 기준 사진 네 장의 측정값 비교이며, 허용오차 합격·통계적 동등성·실장비 보정 성능을 검증한 것이 아닙니다.',
    '- 전체 직선은 보이는 경계의 대표 기울기입니다. 곡률·반사·부분적인 경계 형태는 직선 잔차로 남을 수 있습니다.',
    '- 새 상단/하단 기울기를 로봇 θ로 바꾸려면 그 특징에 대한 변화율이 필요합니다. 기존 좌우 선의 환산 계수를 적용하지 않았습니다.',
    '- full_width_results.json에 기울기 dy/dx, 절편, 잔차, 점 수, 6개 이미지 쌍의 차이를 저장했습니다. 각 NPZ 파일에는 실제 사용한 경계점 전체가 있습니다.',
    '- 재현: python measure_full_width.py']
(HERE/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print('\n'.join(report[:18]))
print('Legacy-native maximum angle difference:',max(abs(v['python_minus_native_image_deg']) for v in legacy_differences))
print('RESULTS:',json.dumps(ranges,ensure_ascii=False))
