"""Audit oracle only. Never imported by the native runtime."""
import sys,json,importlib.util,subprocess
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[3]
final=root/'final';out=final/'results/direct_parity_opencv500_20261027_001';out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('original_direct',root/'codex_experiments_20261014/joint_yt_reboot_20261022/direct/extractor.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
observer=mod.DirectExtractor()
images=[final/'data/raw/hw/bmp/20261014/center.bmp',final/'data/raw/hw/bmp/20261014/Y/Y+0.50.bmp']
summary={'scope':'Development numerical comparison; no independent accuracy or hardware claim','criteria':{'max_displacement_difference_px':0.001,'success_exact':True},'images':[]}
for i,image in enumerate(images):
    destination=out/f'native_{i}'
    subprocess.run([str(final/'build/observer5/Release/direct_extract.exe'),str(images[0]),str(final/'data/reference/joint_center_630_20261022/assets/reference.npz'),str(image),str(destination)],check=True)
    actual=json.loads((destination/'observation.json').read_text());expected=observer.extract_bytes(image.read_bytes())
    entry={'image':image.relative_to(final).as_posix(),'candidates':{}}
    for name,p in expected['candidates'].items():
        c=actual['candidates'][name];a=np.array(c['D'],float);b=p['D'];difference=float(np.nanmax(np.abs(a-b)))
        entry['candidates'][name]={'max_displacement_difference_px':difference,'success_equal':c['success']==bool(p['success']),'passed':difference<=0.001 and c['success']==bool(p['success'])}
    summary['images'].append(entry)
summary['passed']=all(c['passed'] for e in summary['images'] for c in e['candidates'].values())
(out/'checks.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
sys.exit(0 if summary['passed'] else 1)

