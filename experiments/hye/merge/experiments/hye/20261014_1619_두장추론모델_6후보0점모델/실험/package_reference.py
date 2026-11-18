"""Store reference provenance and deployment center measurement without raw photos."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import zero_experiment as Z

def main():
    source=Z.ROOT/'center.bmp'
    (Z.HERE/'reference.json').write_text(json.dumps(dict(source_path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),image_shape=[2160,3840],Y_unit='mm (provided project convention)',T_unit='filename T unit; physical unit unconfirmed',allowed_motion='one nonzero axis at a time',Y_step=.05,T_step=.01,Y_tolerance=.09,T_tolerance=.09),ensure_ascii=False,indent=2),encoding='utf8')
    deployment=json.loads((Z.HERE/'modeling/deployment.json').read_text(encoding='utf8'))
    name=deployment['measurement'];data=np.load(Z.HERE/'measurement'/f'{name}.npz')
    D0=data['D'][data['names']=='center.bmp'][0]
    np.savez_compressed(Z.HERE/'modeling/deploy_reference.npz',D0=D0)
    # Record library versions from the actual interpreter that ran experiments.
    from importlib.metadata import version
    packages=['numpy','pandas','opencv-python-headless','scipy','scikit-learn','matplotlib','Pillow']
    (Z.HERE/'requirements.txt').write_text('\n'.join(f'{p}=={version(p)}' for p in packages)+'\n',encoding='ascii')
    print('Reference and deployment origin packaged for',name)

if __name__=='__main__':main()
