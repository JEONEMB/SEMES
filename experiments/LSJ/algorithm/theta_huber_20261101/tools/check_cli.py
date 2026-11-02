"""CLI success/error contract; uses one calibration image, no held-out evaluation."""
from pathlib import Path
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parents[1]


def main():
    exe=Path(sys.argv[1]).resolve()
    config=HERE/'config/calibration.json'
    validation=HERE/'validation'
    split=json.loads((HERE/'calibration/data_split.json').read_text(encoding='utf-8'))
    image=Path(split['calibration'][0]['path'])
    checks=[]

    def run(args):
        return subprocess.run([str(exe),*map(str,args)],capture_output=True,text=True,encoding='utf-8',errors='replace')

    good=run([config,image])
    assert good.returncode==0,good.stderr
    answer=json.loads(good.stdout)
    assert answer['y_command_available'] is False and answer['y_correction_mm']==0
    assert answer['theta_correction_deg']==-answer['theta_estimate_deg']
    assert answer['upper']['point_count']==2712 and answer['lower']['point_count']==3052
    checks.append('Unicode image path, JSON output, theta sign, no Y command')
    (validation/'cli_calibration_smoke.json').write_text(json.dumps(answer,indent=2,allow_nan=False),encoding='utf-8')

    cases=[('missing arguments',[]),('missing config',[validation/'absent.json',image]),
           ('missing image',[config,validation/'absent.bmp'])]
    bad_config=json.loads(config.read_text(encoding='utf-8'))
    del bad_config['model']['upper']['per_y_mm']
    invalid=validation/'invalid_config_for_test.json'
    invalid.write_text(json.dumps(bad_config,indent=2),encoding='utf-8')
    cases.append(('incomplete calibration',[invalid,image]))
    for name,args in cases:
        failure=run(args)
        assert failure.returncode!=0 and not failure.stdout.strip() and failure.stderr.strip(),(name,failure)
        checks.append(name+': nonzero exit and no correction output')
    summary={'passed':True,'checks':checks,'held_out_images_executed':0,
             'scope':'interface/implementation check only; no accuracy metric calculated'}
    (validation/'cli_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
