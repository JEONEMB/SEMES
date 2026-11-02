"""575 original images -> unchanged signed-Sobel maxima; no Gaussian or rejection."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import sys
import time
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, data):
    (HERE/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    paths = {'first_order': ANALYSIS/'initial_error/capture_order.json',
             'first_hashes': ANALYSIS/'initial_error/measurements.json',
             'second': ANALYSIS/'second_data_inspection/inventory.json',
             'third': ANALYSIS/'third_data_inspection/validation_selection.json',
             'corrections': ANALYSIS/'third_data_inspection/label_corrections.json',
             'targets': ANALYSIS/'reference_line_methods_108_20261030/configs/targets.json',
             'reference_provenance': ANALYSIS/'huber_convergence_20261101/provenance.json'}
    first, first_measure = read(paths['first_order']), read(paths['first_hashes'])
    second, third = read(paths['second']), read(paths['third'])
    old_prov = read(paths['reference_provenance'])
    assert sha(ANALYSIS/'third_data_inspection/label_corrections.json') == third['label_corrections_sha256']
    assert third['included_image_count'] == 234 and not third['excluded']
    expected_first = {r['file']: r['sha256'] for r in first_measure['measurements']}
    group_numbers = {g['group']: i for i,g in enumerate(first['chronological_groups'], 1)}
    manifest = []
    for r in first['files_in_modification_order']:
        axis = 'theta' if r['group'].startswith('Theta') else 'Y' if r['group'].startswith('Y') else None
        manifest.append({'id': f'D1_{r["modification_order"]:03}', 'dataset': 1,
                         'file': r['file'], 'path': str(Path(first['summary']['data_root'])/r['file']),
                         'expected_sha256': expected_first[r['file']], 'is_reference': axis is None,
                         'group': r['group'], 'capture_group': group_numbers[r['group']],
                         'axis': axis, 'source_label': r['label'],
                         'y_mm': float(r['label']) if axis=='Y' else None,
                         'theta_deg': float(r['label']) if axis=='theta' else None,
                         'repeat': None, 'time': r['modified_kst'],
                         'other_axis': 'Fixed within the existing single-axis capture segment; no absolute fixed coordinate inferred.'})
    for i,r in enumerate(sorted(second['images'], key=lambda r:(r['archive_modified_time_no_timezone'],r['file'])), 1):
        match = re.search(r'\((\d+)\)\.bmp$', r['file'])
        manifest.append({'id': f'D2_{i:03}', 'dataset': 2, 'file': r['file'],
                         'path': str(Path(second['summary']['extracted_root'])/r['file']),
                         'expected_sha256': r['sha256'], 'is_reference': r['is_reference'],
                         'group': r['condition'], 'capture_group': None, 'axis': None,
                         'y_mm': r['label_y_mm'], 'theta_deg': r['label_theta_degree'],
                         'repeat': int(match[1]) if match else None,
                         'time': r['archive_modified_time_no_timezone'],
                         'y_actual': r['Y_Actual'], 'theta_actual': r['T_Actual']})
    for i,r in enumerate(sorted(third['included'], key=lambda r:(r['mtime_local'],r['file'])), 1):
        manifest.append({'id': f'D3_{i:03}', 'dataset': 3, 'file': r['file'],
                         'path': str(Path(third['source_root'])/r['file']),
                         'expected_sha256': r['sha256'], 'is_reference': False,
                         'group': str(Path(r['file']).parent), 'capture_group': None, 'axis': 'theta',
                         'y_mm': r['validation_y_mm'], 'theta_deg': r['validation_theta_deg'],
                         'repeat': r['repeat'], 'time': r['mtime_local'],
                         'y_actual': r['effective_y_actual'], 'theta_actual': r['effective_theta_actual'],
                         'label_correction': r.get('label_correction')})
    assert Counter(r['dataset'] for r in manifest) == {1:281, 2:60, 3:234}
    targets = [r for r in read(paths['targets'])['targets'] if r['id'] in ['upper_structure','lower_structure']]
    assert len(targets)==2
    settings = {'targets': targets, 'range_convention': 'inclusive',
                'gradient': {'dx':0,'dy':1,'ksize':3,'scale':1,'depth':'CV_32F','border':'BORDER_REFLECT_101'},
                'selection': 'maximum signed positive Gy in the unchanged band per visible x; first y wins exact ties',
                'column_step_px': 1, 'gaussian': False, 'point_rejection': False,
                'minimum_point_count_filter': False, 'roi_tracking': False,
                'huber': {'t':1.345, 'residual':'vertical', 'scale_est':'mad', 'scale_center':0,
                          'update_scale':True,'maxiter':50,'tol':1e-8,'conv':'dev','start_params':'OLS','cov':'H1'},
                'axis_models': 'per original capture group / fixed coordinate, intercept plus one predictor, OLS',
                'replicates': 'keep every image; no averaging, no p-values or independence claim',
                'cross_Y_theta_sensitivity_comparison': 'second dataset additionally restricted to theta levels shared by every Y group; all images remain in full per-group analysis'}
    dump('settings.json', settings)
    dump('manifest.json', manifest)
    if '--manifest-only' in sys.argv:
        print('Manifest/settings prepared; image extraction not repeated.', flush=True)
        return
    old_points = np.load(ANALYSIS/'huber_convergence_20261101/inputs.npz', allow_pickle=False)
    ref_by_hash = {r['sha256']: r['id'] for r in old_prov['image_checks']}
    arrays, measured, refs = {}, [], []
    t0=time.perf_counter()
    for i,entry in enumerate(manifest,1):
        raw=Path(entry['path']).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        assert digest==entry['expected_sha256'], entry['path']
        gray=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        assert gray is not None and gray.shape==(2160,3840), entry['path']
        gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1,borderType=cv2.BORDER_REFLECT_101)
        for target in targets:
            key=entry['id']+'_'+target['id']
            xs=np.array([x for x in range(target['x_range'][0],target['x_range'][1]+1)
                         if not any(a<=x<=b for a,b in target['exclude_x'])])
            a,b=target['band_y']
            ys=np.argmax(gy[a:b+1,xs],axis=0)+a
            responses=gy[ys,xs]
            arrays[key]=np.column_stack([xs,ys]).astype(np.uint16)
            arrays[key+'_response']=responses.astype(np.int16)
            measured.append({'id':entry['id'],'target':target['id'],'n':len(xs),
                             'band_endpoint_points':int(np.count_nonzero((ys==a)|(ys==b))),
                             'nonpositive_response_points':int(np.count_nonzero(responses<=0)),
                             'response_min':float(responses.min()), 'response_max':float(responses.max()),
                             'y_min':int(ys.min()),'y_max':int(ys.max()),'sha256':digest})
            if digest in ref_by_hash:
                ref=ref_by_hash[digest]
                same=bool(np.array_equal(arrays[key],old_points[ref+'_'+target['id']]))
                assert same,(key,ref)
                refs.append({'id':entry['id'],'reference':ref,'target':target['id'],'points_bitwise_equal':same})
        if i%25==0 or i==len(manifest):
            print(f'Extracted {i}/{len(manifest)} images; elapsed {time.perf_counter()-t0:.1f}s',flush=True)
    assert len(refs)==8 and len(measured)==1150
    np.savez_compressed(HERE/'points.npz',**arrays)
    dump('extraction.json',measured)
    dump('provenance.json',{'sources':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
                          'opencv_version':cv2.__version__,'numpy_version':np.__version__,
                          'images':len(manifest),'lines':len(measured),
                          'point_count':sum(r['n'] for r in measured),
                          'all_image_hashes_match_prior':True,'reference_crosschecks':refs,
                          'points_sha256':sha(HERE/'points.npz'),'settings_sha256':sha(HERE/'settings.json'),
                          'elapsed_seconds':time.perf_counter()-t0,
                          'note':'Endpoint/nonpositive response counts are observations only; no point is removed or ROI changed.'})
    print('Extraction finished; all eight reference point sets exactly reproduced.',flush=True)


if __name__=='__main__':
    main()
