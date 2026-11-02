"""Check C++ Sobel/Huber against stored verified Python results, train images only."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parents[1]
SOURCE=HERE.parents[1]/'analysis/huber_axis_sensitivity_20261101'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    executable=Path(sys.argv[1]).resolve()
    out=HERE/'validation'
    out.mkdir(exist_ok=True)
    split=read(HERE/'calibration/data_split.json')
    train=split['calibration']
    ids={r['id'] for r in train}
    assert ids.isdisjoint(r['id'] for r in split['held_out'])
    points=np.load(SOURCE/'points.npz',allow_pickle=False)
    fixtures=[]
    for r in train:
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['expected_sha256']
        fixtures.append({'id':r['id'],'path':r['path'],**{
            side+'_points':points[r['id']+'_'+side+'_structure'].astype(int).tolist()
            for side in ('upper','lower')}})
    fixture=out/'native_fixture.json'
    fixture.write_text(json.dumps({'config_path':str(HERE/'config/calibration.json'),'images':fixtures},ensure_ascii=False,indent=1),encoding='utf-8')
    run=subprocess.run([str(executable),str(fixture)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    (out/'native_stderr.txt').write_text(run.stderr,encoding='utf-8')
    if run.returncode:
        raise RuntimeError(f'native probe failed ({run.returncode}): {run.stderr}')
    native=json.loads(run.stdout)
    (out/'native_lines.json').write_text(json.dumps(native,indent=2,allow_nan=False),encoding='utf-8')
    expected={(r['id'],r['target']):r for r in read(SOURCE/'measurements.json') if r['id'] in ids}
    mapping={'slope':'slope','intercept_px':'intercept_px','angle_deg':'image_angle_deg','scale_px':'scale_px'}
    differences=[]
    for r in native:
        p=expected[r['id'],r['target']+'_structure']
        differences.append({'id':r['id'],'target':r['target'],
            **{k:abs(r[k]-p[v]) for k,v in mapping.items()},
            'iterations_cpp':r['iterations'],'iterations_python':p['iteration']})
    maxima={k:max(r[k] for r in differences) for k in mapping}
    summary={'calibration_images':len(train),'lines':len(native),
        'point_count':sum(r['point_count'] for r in native),'all_points_exactly_equal':True,
        'maximum_absolute_difference':maxima,
        'iteration_count_differences':sum(r['iterations_cpp']!=r['iterations_python'] for r in differences),
        'held_out_images_executed':0,
        'skeleton_command_exactly_matches_detector':True,
        'comparison_purpose':'Floating-point implementation parity, not alignment accuracy.',
        'numeric_parity_limits':{'slope':1e-10,'intercept_px':1e-7,'angle_deg':1e-8,'scale_px':1e-7},
        'limits_note':'Engineering regression-test tolerances only, not image acceptance/rejection or alignment criteria.'}
    summary['passed']=all(maxima[k]<=v for k,v in summary['numeric_parity_limits'].items())
    (out/'parity_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'parity_differences.json').write_text(json.dumps(differences,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if not summary['passed']:
        raise RuntimeError('native/reference parity failed')


if __name__=='__main__':
    main()
