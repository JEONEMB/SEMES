"""Infer nominal-zero QR tracking regions separately for six first-data runs.

Reuses saved ZNCC-initialized affine registrations; center.bmp is not a fit
sample. It supplies template coordinates and a display background only.
"""

import hashlib
from itertools import groupby
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MARKERS = ("L", "C", "R")
COLORS = ("#d04b45", "#b78313", "#36854c", "#8653aa", "#168aa6", "#275ac2")
NAMES = ("Y +0.10 → +2.00", "Y -0.10 → -2.00", "θ +0.01 → +1.00",
         "θ -0.01 → -1.00", "Y +0.05 → +1.95", "Y -1.95 → -0.05")
plt.rcParams.update({"font.family":"Malgun Gothic", "font.size":11,
                     "axes.unicode_minus":False, "figure.facecolor":"white",
                     "savefig.facecolor":"white", "svg.fonttype":"path"})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fit(labels, values, degree):
    design = np.polynomial.polynomial.polyvander(labels, degree)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    return {"degree":degree, "coefficients_ascending":coefficients.tolist(),
            "at_label_zero":coefficients[0].tolist(),
            "rms_residual_components":np.sqrt(np.mean(residual**2,axis=0)).tolist(),
            "max_abs_residual_components":np.max(np.abs(residual),axis=0).tolist()}


def region_geometry(roi):
    x0,y0,x1,y1 = roi
    center = np.array([(x0+x1-1)/2,(y0+y1-1)/2])
    # Corners denote centers of the outermost pixels in the original tracking ROI.
    corners = np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],dtype=float)
    return center,corners


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    measurements_path=BASE/"initial_error/measurements.json"
    metadata_path=BASE/"initial_error/capture_order.json"
    measurements=read_json(measurements_path)
    metadata=read_json(metadata_path)
    lookup={r['file']:r for r in measurements['measurements']}
    data_root=Path(measurements['data_root'])
    ordered=sorted(metadata['files_in_modification_order'],key=lambda r:(r['modified_ns'],r['file']))
    assert len(ordered)==len(lookup)==281 and set(lookup)=={r['file'] for r in ordered}
    for r in ordered:
        if (data_root/r['file']).stat().st_mtime_ns != r['modified_ns']:
            raise ValueError('Source modification time changed: '+r['file'])
    center_row=next(r for r in measurements['measurements'] if r['axis']=='center')
    center_data=(data_root/center_row['file']).read_bytes()
    assert hashlib.sha256(center_data).hexdigest()==center_row['sha256']
    center_image=cv2.imdecode(np.frombuffer(center_data,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    groups=[]
    source_points=[]
    groups_by_time=[(name,list(items)) for name,items in groupby(ordered,key=lambda r:r['group'])]
    for number,(name,meta_rows) in enumerate(groups_by_time,1):
        if name=='center':
            continue
        rows=[lookup[r['file']] for r in meta_rows]
        axis=rows[0]['axis']
        labels=np.array([r['label'] for r in rows],dtype=float)
        assert all(r['axis']==axis and r['file']!='center.bmp' for r in rows)
        result={"capture_group_number":number,"name":name,"axis":axis,
                "label_unit":"mm" if axis=='Y' else "degree", "image_count":len(rows),
                "first_file":meta_rows[0]['file'],"last_file":meta_rows[-1]['file'],
                "first_time_kst":meta_rows[0]['modified_kst'],"last_time_kst":meta_rows[-1]['modified_kst'],
                "labels_in_time_order":labels.tolist(),"markers":{}}
        for marker in MARKERS:
            roi=measurements['regions_xyxy'][marker]
            base_center,base_corners=region_geometry(roi)
            values=[]
            displacements=[]
            for row,meta in zip(rows,meta_rows):
                registration=row['regions'][marker]['affine']
                if 'error' in registration:
                    raise ValueError('Saved affine registration failed')
                delta=np.array([registration['dx'],registration['dy']])
                linear=np.asarray(registration['warp'],dtype=float)[:,:2]
                target_center=base_center+delta
                target_corners=(base_corners-base_center)@linear.T+target_center
                values.append(np.concatenate([target_center,target_corners.ravel()]))
                displacements.append(delta)
                source_points.append({"file":row['file'],"order":meta['modification_order'],
                                      "group_number":number,"marker":marker,"label":row['label'],
                                      "center_xy_px":target_center.tolist(),
                                      "roi_pixel_center_corners_xy_px":target_corners.tolist()})
            values=np.asarray(values)
            linear_fit=fit(labels,values,1)
            quadratic_fit=fit(labels,values,2)
            zero=np.array(linear_fit['at_label_zero'])
            quad_zero=np.array(quadratic_fit['at_label_zero'])
            predicted=zero[2:].reshape(4,2)
            delta=zero[:2]-base_center
            result['markers'][marker]={
                "reference_roi_xyxy_exclusive":roi,"reference_roi_center_xy_px":base_center.tolist(),
                "predicted_zero_center_xy_px":zero[:2].tolist(),
                "predicted_zero_center_minus_reference_px":delta.tolist(),
                "predicted_zero_roi_pixel_center_corners_xy_px":predicted.tolist(),
                "predicted_zero_roi_outer_edge_bbox_xyxy_px":[float(predicted[:,0].min()-.5),float(predicted[:,1].min()-.5),float(predicted[:,0].max()+.5),float(predicted[:,1].max()+.5)],
                "linear_fit":linear_fit,"quadratic_sensitivity_fit":quadratic_fit,
                "quadratic_minus_linear_zero_center_xy_px":(quad_zero[:2]-zero[:2]).tolist(),
                "observed_affine_displacements_xy_px":np.array(displacements).tolist()}
            assert np.allclose(predicted.mean(axis=0),zero[:2],atol=1e-8)
        groups.append(result)
    envelopes={}
    for m in MARKERS:
        centers=np.array([g['markers'][m]['predicted_zero_center_xy_px'] for g in groups])
        shifts=np.array([g['markers'][m]['predicted_zero_center_minus_reference_px'] for g in groups])
        envelopes[m]={"center_xy_min_px":centers.min(axis=0).tolist(),"center_xy_max_px":centers.max(axis=0).tolist(),
                      "center_xy_range_px":np.ptp(centers,axis=0).tolist(),
                      "max_distance_from_reference_roi_center_px":float(np.linalg.norm(shifts,axis=1).max())}
    result={"purpose":"Infer each acquisition run's nominal-label-zero QR tracking regions from nonzero images only",
            "sources":{"measurements":str(measurements_path),"metadata":str(metadata_path),"data_root":str(data_root),
                       "source_json_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (measurements_path,metadata_path)},
                       "original_modification_times_rechecked":281,"center_image_sha256_verified":True,
                       "all_nonzero_images_rehashed_this_run":False},
            "method":{"registration":"Reuse original per-QR ZNCC-initialized affine ECC measurements; no new raw-image registration",
                      "coordinate_convention":"original 3840x2160 pixel coordinates, origin at top-left pixel center, x right/y down",
                      "primary_model":"For each marker in each chronological run, independently fit position(q) = intercept + slope*q by unweighted least squares; evaluate at q=0",
                      "region_model":"Apply each saved affine linear part and measured center displacement to the existing ROI's four corner pixel centers; extrapolate corners at q=0",
                      "center_excluded_from_fits":True,"time_used_for":"run grouping and order, not a time-to-position model",
                      "sensitivity_check":"Quadratic fits on the same samples; compare q=0 predictions, without choosing a model by in-sample residual or assigning probabilities"},
            "limitations":[
                "These are extrapolated nominal-zero candidates for separate runs, not an identified true mechanical zero or a calibrated probability region.",
                "Labels are nominal; the other axis and return-to-zero state are unverified. No common origin is imposed across runs.",
                "Only differences determine slopes; the filename-label origin supplies q=0. An unknown shift of that origin cannot be recovered from increments alone.",
                "The three ROIs are existing tracking patches including margins, not newly detected exact QR code boundaries or QR geometric centers.",
                "center.bmp supplies a coordinate/template anchor, but its displacement is not included as a zero-valued fit sample.",
                "The primary model assumes approximately linear image response within each run. Quadratic sensitivity does not establish which model is physically correct.",
                "Candidate min-max ranges across six runs are not confidence intervals or acceptance criteria. Small in-sample residuals do not prove absolute accuracy."],
            "image_count_used":280,"run_count":len(groups),"marker_fit_count":len(groups)*3,
            "groups":groups,"candidate_envelopes":envelopes,"source_points":source_points}
    (HERE/'zero_region_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_report(result)
    plot_regions(result,center_image)
    plot_trends(result)
    print(json.dumps({"image_count":280,"groups":[{"number":g['capture_group_number'],"name":g['name'],
                      "centers":{m:g['markers'][m]['predicted_zero_center_xy_px'] for m in MARKERS}} for g in groups],
                      "envelopes":envelopes,
                      "max_abs_quadratic_intercept_change_px":float(max(np.max(np.abs(g['markers'][m]['quadratic_minus_linear_zero_center_xy_px'])) for g in groups for m in MARKERS))},ensure_ascii=False,indent=2))


def write_report(result):
    lines=["1차 데이터: 촬영 구간별 라벨 0의 QR 추적 영역 추정", "",
           "기존 ZNCC → 아핀 ECC 측정값을 재사용. center.bmp를 제외한 280장을 메타데이터순 6구간으로 나눴습니다.",
           "각 구간·각 QR에서 영상 위치 = a × 파일명 라벨 + b를 적합하고 라벨 0의 위치 b를 계산했습니다.",
           "시간은 촬영 묶음을 나누는 데 사용했으며 시간 자체를 위치로 환산하지 않았습니다.",
           "center.bmp는 좌표·템플릿 기준으로만 사용했으며 (라벨 0, 변위 0) 관측값으로 적합에 넣지 않았습니다.",
           "기존 ROI의 아핀 변형을 이용해 네 모서리도 함께 외삽했습니다. 정확한 QR 외곽을 새로 검출한 결과는 아닙니다.",
           "좌표: 원본 3840×2160, 좌상단 픽셀 중심 (0,0), x 오른쪽·y 아래쪽. 단위 px.",
           "각 구간의 라벨 0 후보이며, 실제 기계 영점 또는 하나의 공통 정답 영역을 확정한 결과가 아닙니다.",
           "변화량만으로는 절대 영점 위치가 정해지지 않습니다. 여기서는 파일명 라벨의 0을 외삽 목표로 사용했습니다.",
           "구간 간 최소~최대는 후보 범위이지 확률/신뢰구간이 아닙니다.", "",
           "구간 | QR | 기준 ROI 중심(x,y) | 예상 영점 중심(x,y) | 기준 대비(dx,dy) | 직선 적합 RMS(x,y) | 이차식-직선식 영점 차이(x,y)"]
    for g in result['groups']:
        for m in MARKERS:
            r=g['markers'][m]
            fmt=lambda v:'('+', '.join(f'{x:.6f}' for x in v)+')'
            lines.append(f"{g['capture_group_number']} {g['name']} | {m} | {fmt(r['reference_roi_center_xy_px'])} | {fmt(r['predicted_zero_center_xy_px'])} | {fmt(r['predicted_zero_center_minus_reference_px'])} | {fmt(r['linear_fit']['rms_residual_components'][:2])} | {fmt(r['quadratic_minus_linear_zero_center_xy_px'])}")
    lines += ["", "구간별 예상 영역 네 모서리 (픽셀 중심 좌표, 좌상→우상→우하→좌하)"]
    for g in result['groups']:
        lines += [f"구간 {g['capture_group_number']}: {g['first_time_kst']} ~ {g['last_time_kst']}"]
        for m in MARKERS:
            points=g['markers'][m]['predicted_zero_roi_pixel_center_corners_xy_px']
            lines.append(m+' | '+' / '.join(f'({x:.6f}, {y:.6f})' for x,y in points))
    lines += ["", "6구간 영점 후보의 중심 좌표 범위"]
    for m,e in result['candidate_envelopes'].items():
        lines.append(f"{m}: x {e['center_xy_min_px'][0]:.6f} ~ {e['center_xy_max_px'][0]:.6f}; y {e['center_xy_min_px'][1]:.6f} ~ {e['center_xy_max_px'][1]:.6f}; 폭(x,y) = {e['center_xy_range_px']}")
    (HERE/'zero_region_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot_regions(result, image):
    fig=plt.figure(figsize=(16,9.4))
    fig.text(.045,.95,"1차 데이터 | 라벨 0에서 예상되는 QR 3개 영역",fontsize=23,weight='bold',color='#182b43')
    fig.text(.045,.905,"center.bmp를 적합에서 제외 · 시간순 6구간 별도 추정 · 기존 QR 추적 영역 기준",fontsize=13,color='#52657b')
    top=fig.add_axes([.08,.54,.84,.31])
    top.imshow(image[920:1215,1490:2310],cmap='gray',extent=[1489.5,2309.5,1214.5,919.5],vmin=0,vmax=255)
    for m in MARKERS:
        base=result['groups'][0]['markers'][m]
        _,corners=region_geometry(base['reference_roi_xyxy_exclusive'])
        top.add_patch(Polygon(corners,closed=True,fill=False,color='#131d2b',ls='--',lw=1.3))
        for g,color in zip(result['groups'],COLORS):
            top.add_patch(Polygon(g['markers'][m]['predicted_zero_roi_pixel_center_corners_xy_px'],closed=True,fill=False,color=color,lw=.85,alpha=.85))
        center=base['reference_roi_center_xy_px']
        top.text(center[0],945,m,ha='center',fontsize=14,weight='bold',color='white',bbox=dict(fc='#182b43',ec='none',pad=3))
    top.set(xlim=(1490,2310),ylim=(1215,920),xlabel='원본 이미지 x (px)',ylabel='원본 이미지 y (px)')
    top.set_title('기준 이미지 위 예상 영역: 차이가 1px 안팎이어서 실제 배율에서는 거의 겹칩니다.',fontsize=11,pad=6)
    for column,m in enumerate(MARKERS):
        ax=fig.add_axes([.07+column*.31,.18,.26,.24])
        ax.axhline(0,color='#b8c5d1',lw=.8);ax.axvline(0,color='#b8c5d1',lw=.8)
        ax.scatter([0],[0],marker='*',s=130,color='#182b43',zorder=5)
        ax.annotate('기존 ROI 중심',(0,0),xytext=(-83,-21),textcoords='offset points',fontsize=9,color='#182b43')
        label_offsets={2:(-27,-15),3:(-10,11),4:(5,8),5:(6,-2),6:(-36,5),7:(8,-7)}
        for g,color in zip(result['groups'],COLORS):
            x,y=g['markers'][m]['predicted_zero_center_minus_reference_px']
            ax.scatter(x,y,s=45,color=color,zorder=6)
            ax.annotate(str(g['capture_group_number']),(x,y),xytext=label_offsets[g['capture_group_number']],textcoords='offset points',fontsize=10,color=color,weight='bold',arrowprops={'arrowstyle':'-','color':color,'lw':.7})
        ax.set(xlim=(-.98,.25),ylim=(.12,-.035),xlabel='기준 ROI 중심 대비 Δx (px)',ylabel='Δy (px)')
        ax.set_title(f'{m} QR · 영점 후보 확대',fontsize=12,weight='bold')
        ax.grid(alpha=.18);ax.set_axisbelow(True)
    legend=[Line2D([],[],marker='o',ls='',color=c,label=f'{g["capture_group_number"]}: {n}') for g,c,n in zip(result['groups'],COLORS,NAMES)]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,fontsize=10,columnspacing=2.5)
    fig.text(.045,.025,'번호는 메타데이터순 촬영 구간입니다. 하단은 y가 아래로 증가합니다. 후보 범위는 확률 구간이나 실제 기계 영점의 보증 범위가 아닙니다.',fontsize=10,color='#52657b')
    fig.savefig(HERE/'zero_region_candidates.png',dpi=150)
    fig.savefig(HERE/'zero_region_candidates.svg')
    plt.close(fig)


def plot_trends(result):
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    fig.subplots_adjust(left=.065,right=.97,bottom=.12,top=.84,hspace=.40,wspace=.25)
    marker_colors=('#c34b48','#287a90','#7a52a5')
    for ax,g,title in zip(axes.flat,result['groups'],NAMES):
        x=np.array(g['labels_in_time_order'])
        grid=np.linspace(min(0,x.min()),max(0,x.max()),200)
        for m,color in zip(MARKERS,marker_colors):
            r=g['markers'][m]
            observed=np.array(r['observed_affine_displacements_xy_px'])[:,0]
            coefficients=np.array(r['linear_fit']['coefficients_ascending'])
            baseline=r['reference_roi_center_xy_px'][0]
            ax.scatter(x,observed,s=7,color=color,alpha=.6)
            ax.plot(grid,coefficients[0,0]+coefficients[1,0]*grid-baseline,color=color,lw=.9,label=m)
            ax.scatter([0],[r['predicted_zero_center_minus_reference_px'][0]],s=35,marker='x',color=color,zorder=6)
        ax.axvline(0,color='#6a7988',lw=.8,ls='--');ax.axhline(0,color='#aebbc8',lw=.6)
        ax.set_title(f"{g['capture_group_number']}. {title}",fontsize=12,weight='bold')
        ax.set_xlabel('파일명 Y (mm)' if g['axis']=='Y' else '파일명 θ (°)')
        ax.set_ylabel('기존 ROI 중심 대비 Δx (px)');ax.grid(alpha=.18)
    axes.flat[0].legend(frameon=False,fontsize=10,ncol=3)
    fig.text(.045,.95,'추정 과정 | 각 구간의 영상 위치 추세를 라벨 0까지 연장',fontsize=22,weight='bold',color='#182b43')
    fig.text(.045,.90,'점: 기존 아핀 측정값  ·  선: 구간별 직선 적합  ·  ×: 라벨 0의 예측값  ·  center.bmp는 적합에 사용하지 않음',fontsize=12,color='#52657b')
    fig.text(.045,.045,'좌표 전체 변화가 커서 영점 후보 사이의 작은 차이는 별도 확대 그림에서 확인해야 합니다. 직선은 근사이며 이차식에 따른 민감도도 결과 파일에 기록했습니다.',fontsize=10,color='#52657b')
    fig.savefig(HERE/'zero_region_fit_process.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
