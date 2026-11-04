import json,joblib,sys,hashlib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from engine import *
from fullfield import field_features
from predict import predict
sys.path.insert(0,str(B));import zero_experiment as Z

def main():
    obs=load_observations();checks={};images=[]
    for name in ['Y+0.05.bmp','T-0.05.bmp']:
        result=predict((Z.ROOT/('Y' if name[0]=='Y' else 'Theta')/name).read_bytes(),(Z.ROOT/'center.bmp').read_bytes(),H/'fixedfeature_model.joblib')
        bundle=joblib.load(H/'fixedfeature_model.joblib');c=bundle['config'];i=list(obs['names']).index(name)
        x=extract(obs['points'],obs['D'][i],obs['q'][i],c['measurement'],c['feature']);expected=bundle['model'].predict(x[None])[0]
        feature_error=float(abs(x-result['feature']).max());prediction_error=float(abs(expected-[result['Y'],result['T']]).max())
        assert feature_error<1e-8 and prediction_error<1e-7
        images.append(dict(source=name,feature_max_delta=feature_error,prediction_max_delta=prediction_error,Y=result['Y'],T=result['T']))
    checks['fixedfeature_image_bytes']=images
    for tag in ['fixedfeature','coordinate','fullfield','linear']:
        bundle=joblib.load(H/f'{tag}_model.joblib');assert len(bundle['ids'])==630
        pp=pd.read_csv(H/(tag+'_oof.csv' if tag!='linear' else 'linear_oof.csv'))
        assert len(pp)==pp.file.nunique()==280 and np.isfinite(pp[['pred_Y','pred_T']]).all().all()
        checks[tag+'_model_load_and_oof']=True
    # Every preserved artifact still matches the experiment's preservation snapshot.
    audit=json.loads((H/'audit.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in audit['protected_sha256'].items())
    checks['original_artifacts_preserved']=len(audit['protected_sha256'])
    (H/'selected_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
