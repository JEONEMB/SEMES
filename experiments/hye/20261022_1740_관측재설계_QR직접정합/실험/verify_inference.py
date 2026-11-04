from pathlib import Path
import sys,json,hashlib,inspect
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'joint_yt_generalization_20261022'))
from data import load
sys.path.insert(0,str(H))
from metrics import metric
from predict import Predictor

def main():
    m,_=load();checks=[]
    assert metric(np.zeros((2,2)),np.full((2,2),np.nan))==dict(n=2,failures=2,passed=0,Y_MAE=None,Y_max=None,T_MAE=None,T_max=None)
    assert metric([[0,0],[0,0]],[[.09,.09],[np.nan,np.nan]])['passed']==1
    assert list(inspect.signature(Predictor.predict_bytes).parameters)==['self','raw']
    for method,filename in [('direct','direct_predictions.csv'),('shared','shared_predictions.csv'),('pixel_pls','pixel_predictions.csv'),('pixel_cnn','pixel_predictions.csv'),('external','parallax_predictions.csv'),('qr_external','parallax_predictions.csv')]:
        ex=Predictor(method);saved=pd.read_csv(H/filename);saved=saved[saved.protocol=='training_reapplication']
        if method in ['pixel_pls','pixel_cnn','external','qr_external']:saved=saved[saved.method==method]
        for i in ([0,80,309,336] if method=='qr_external' else [336]):
            row=m.iloc[i];raw=Path(row.path).read_bytes();r=ex.predict_bytes(raw);target=saved[saved['index']==i][['predY','predT']].to_numpy()[0];diff=float(np.max(abs(np.array([r['Y'],r['T']])-target)))
            assert diff<2e-5,(method,i,diff)
            checks.append(dict(method=method,index=i,image_sha256=hashlib.sha256(raw).hexdigest(),Y=r['Y'],T=r['T'],max_difference_from_saved_reapplication=diff,seconds=r['seconds']))
    (H/'inference_verification.json').write_text(json.dumps(dict(status='passed',purpose='image-only API equivalence and failure metric checks; NOT new-photo accuracy test',signature='predict_bytes(raw) only; path/labels absent',all_failed_metric_test='passed',threshold_and_failure_test='passed',checks=checks),indent=2))
    print('API checks passed',len(checks),'maxdifference',max(x['max_difference_from_saved_reapplication'] for x in checks))

if __name__=='__main__':
    with threadpool_limits(1):main()
