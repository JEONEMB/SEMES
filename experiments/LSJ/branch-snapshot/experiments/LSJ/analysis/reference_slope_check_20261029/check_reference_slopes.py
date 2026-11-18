"""Re-measure ONLY the four reference images using the unchanged native theta estimator.
Compare image-line slope/angle. No robot-angle regression, calibration update or equivalence test.
"""
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PAYLOAD=ROOT/'.tmp/theta_implementation_20261029/payload'
PRIOR=ROOT/'analysis/algorithm_benchmark_20261029'
EXE=ROOT/'.tmp/theta_implementation_20261029/review_fix_20261029/build/Release/main.exe'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(PRIOR/'test_manifest.json')
prior={r['id']:r for r in read(PRIOR/'native_results.json')['results']}
cases=[r for r in manifest['cases'] if r['is_reference']]
assert len(cases)==4
assert all(sha(PAYLOAD/rel)==digest for rel,digest in manifest['source_sha256'].items())
before={r['path']:sha(Path(r['path'])) for r in cases}
assert all(before[r['path']]==r['expected_sha256'] for r in cases)
(HERE/'input_paths.txt').write_text('\n'.join(r['path'] for r in cases)+'\n',encoding='utf-8')
command=[str(EXE),'--reference',manifest['reference'],'--config',manifest['theta_config'],
    '--input-list',str(HERE/'input_paths.txt'),'--output',str(HERE/'native_reference_results.json')]
run=subprocess.run(command,capture_output=True,cwd=HERE)
(HERE/'run.log').write_bytes(run.stdout+run.stderr)
assert run.returncode in (0,1),(run.returncode,run.stderr)
native=read(HERE/'native_reference_results.json')
output={r['file']:r for r in native['results']}
assert len(output)==4
records=[]
for case in cases:
    r=output[case['path']]
    assert r['valid']==1
    lines=r['lines']
    old=prior[case['id']]['decision']['theta']['lines']
    assert all(abs(lines[i]['angle_deg']-old[i]['angle_deg'])<1e-12 for i in range(2))
    # Current implementation rounds locator translation when constructing each edge ROI.
    rounded=[round(x) for x in r['shift_xy_px']]
    assert rounded==[0,0],('Different actual integer ROI',case['id'],rounded)
    records.append({'id':case['id'],'dataset':case['dataset'],'file':Path(case['path']).name,
        'path':case['path'],'time_metadata':case['time_metadata'],
        'sha256':before[case['path']],'locator_shift_xy_px':r['shift_xy_px'],
        'actual_integer_roi_shift_xy_px':rounded,
        'lines':{name:{'image_angle_deg':lines[i]['angle_deg'],
            'slope_dy_dx':math.tan(math.radians(lines[i]['angle_deg'])),
            'edge_fit_rms_px':lines[i]['edge_rms_px'],'edge_points':lines[i]['points']}
            for i,name in enumerate(['upper_left','upper_right'])}})
baseline=records[0]
for r in records:
    for name,v in r['lines'].items():
        v['delta_image_angle_from_first_center_deg']=v['image_angle_deg']-baseline['lines'][name]['image_angle_deg']
pairs=[]
for a,b in itertools.combinations(records,2):
    pairs.append({'first':a['file'],'second':b['file'],
        'delta_image_angle_deg':{name:b['lines'][name]['image_angle_deg']-a['lines'][name]['image_angle_deg']
                                for name in ['upper_left','upper_right']}})
ranges={}
for name in ['upper_left','upper_right']:
    low=min(records,key=lambda r:r['lines'][name]['image_angle_deg'])
    high=max(records,key=lambda r:r['lines'][name]['image_angle_deg'])
    ranges[name]={'min_file':low['file'],'max_file':high['file'],
        'range_image_angle_deg':high['lines'][name]['image_angle_deg']-low['lines'][name]['image_angle_deg'],
        'range_slope_dy_dx':high['lines'][name]['slope_dy_dx']-low['lines'][name]['slope_dy_dx']}
assert all(sha(Path(p))==digest for p,digest in before.items())
result={'scope':'Only first center.bmp and second center1.bmp/center2.bmp/center3.bmp',
    'method':'Unchanged native OpenCV 3.4.15: central QR ZNCC+affine ECC localization, blurred-step subpixel edge fitting, Huber line fit',
    'units':'Image line angle relative to horizontal in degrees; positive down to right. slope_dy_dx=tan(angle). NOT robot theta.',
    'comparable_regions':'Both structural edge ROIs use the same integer pixel coordinates in all four images; no affine resampling',
    'decision':'Measured image-line slopes differ. No population-equivalence or physical-pose conclusion.',
    'no_new_acceptance_threshold':True,'new_capture_replicates':False,
    'limitations':['One saved image per entry; re-running the same deterministic estimator does not measure uncertainty',
        'First and second reference images have different nominal origins',
        'Observed differences do not distinguish actual orientation from feature-measurement variability',
        'Robot theta tolerance must not be directly applied to image-line angles'],
    'native_return_code':run.returncode,'native_return_code_note':'1 means some theta commands unavailable, not line measurement failure',
    'all_4_inputs_unchanged_sha256':True,'matches_prior_341_image_run_angles_within_1e_minus_12_deg':True,
    'rows':records,'pairwise_differences':pairs,'ranges':ranges,
    'source_sha256':{'theta_alignment.cpp':sha(PAYLOAD/'src/theta_alignment.cpp'),
                     'theta_alignment.json':sha(PAYLOAD/'configs/theta_alignment.json'),
                     'executable':sha(EXE)}}
(HERE/'reference_slope_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
text=['# 기준 이미지 4장의 선 기울기 비교','',
    '**현재 측정 방법으로 얻은 네 장의 선 기울기는 서로 다릅니다.** 비교 대상은 1차 center.bmp와 2차 center1·2·3.bmp의 좌·우 상단 구조물 경계선입니다.','',
    '표의 각도는 사진 속 선이 가로축과 이루는 **영상 각도**이며 로봇 θ가 아닙니다. 현재 C++ 방법을 그대로 적용해 네 장만 다시 측정했습니다. 네 장 모두 정수 반올림한 ROI 이동은 (0,0)이어서 실제 측정한 픽셀 영역도 동일합니다.','',
    '| 기준 이미지 | 왼쪽 선 각도(°) | 오른쪽 선 각도(°) | 왼쪽: 1차 대비 차이(°) | 오른쪽: 1차 대비 차이(°) |',
    '|---|---:|---:|---:|---:|']
for r in records:
    l,rr=r['lines']['upper_left'],r['lines']['upper_right']
    text.append(f"| {r['dataset']} / {r['file']} | {l['image_angle_deg']:.6f} | {rr['image_angle_deg']:.6f} | {l['delta_image_angle_from_first_center_deg']:+.6f} | {rr['delta_image_angle_from_first_center_deg']:+.6f} |")
text+=['','## 최대−최소 차이','']
for name,label in [('upper_left','왼쪽'),('upper_right','오른쪽')]:
    v=ranges[name]; text.append(f"- {label}: **{v['range_image_angle_deg']:.6f}°**, {v['max_file']} − {v['min_file']}")
text+=['','두 선 모두 최댓값은 2차 center1, 최솟값은 2차 center3입니다. 따라서 이번 차이는 1차·2차 사이에만 존재하는 것이 아니라 2차 기준 사진 3장 사이에서도 관측됩니다.','',
    '## 기울기 dy/dx','',
    '| 기준 이미지 | 왼쪽 dy/dx | 오른쪽 dy/dx |','|---|---:|---:|']
for r in records: text.append(f"| {r['file']} | {r['lines']['upper_left']['slope_dy_dx']:.9f} | {r['lines']['upper_right']['slope_dy_dx']:.9f} |")
text+=['','## 이번 비교의 범위','',
    '- 경계점 → 연속 흐린 계단 모형 맞춤 → Huber 직선 측정까지 기존 코드를 사용했습니다. 알고리즘·설정·기준 이미지는 수정하지 않았습니다.',
    '- 기존 341장 시험의 네 기준 이미지 각도와 이번 재실행 값이 1e-12° 이내에서 일치했습니다. 이는 계산 재현 확인이며 측정 불확실성의 크기가 아닙니다.',
    '- 여기서 확인한 것은 측정값의 차이입니다. 사진마다 실제 자세가 다른지, 특징 측정에 차이가 생기는지는 이 비교만으로 분리할 수 없습니다.',
    '- 로봇 θ 허용오차 ±0.005°를 영상 선 각도에 직접 적용하지 않았습니다. 새 합격 기준이나 동등성 검정도 도입하지 않았습니다.',
    '- 입력 네 장의 해시가 실행 전후 동일합니다. 상세 값·6개 이미지 쌍의 차이·해시는 reference_slope_results.json에 있습니다.']
(HERE/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print('\n'.join(text[:18]))
print(json.dumps(ranges,ensure_ascii=False,indent=2))
