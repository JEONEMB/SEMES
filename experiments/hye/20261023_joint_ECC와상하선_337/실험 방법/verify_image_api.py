from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
from image_observations import ObservationExtractor

def main():
    m=pd.read_csv(H/'manifest.csv');cache=np.load(H/'features.npz');pred=pd.read_csv(H/'static_predictions.csv');ex=ObservationExtractor();rows=[]
    for i in [0,80,291,336]:
        raw=Path(m.iloc[i].path).read_bytes()
        for condition in ['qr','upper','lower','both']:
            start=time.perf_counter();r=ex.extract_bytes(raw,condition);feature_error=float(np.max(abs(r['X']-cache['X_'+condition][i])));assert r['success'];assert feature_error<1e-10
            model=joblib.load(H/(condition+'_model.joblib'))['model'];p=model.predict(r['X'][None])[0];stored=pred[(pred.condition_name==condition)&(pred.protocol=='training_reapplication')&(pred['index']==i)][['predY','predT']].to_numpy()[0];difference=float(np.max(abs(p-stored)));assert difference<1e-9
            rows.append(dict(index=i,condition=condition,feature_difference=feature_error,prediction_difference=difference,seconds=time.perf_counter()-start,image_sha256=r['image_sha256']))
    sys.path.insert(0,str(H/'control'))
    cal=joblib.load(H/'control/full_calibration_both.joblib')['calibration'];s=cal.project(cache['X_both'][0]);c=cal.controller('fixed_jacobian').start(s,np.zeros_like(s));u=c.next_action();assert np.allclose(u,0);c.observe(s,u);assert c.status=='visual_goal_confirmed' and c.turn==1
    result=dict(status='passed',cases=rows,scope='API equivalence only; existing photos, not new-photo validation',center_zero_move_confirmation='passed in one counted capture cycle; no hardware command',reference_center_is_dataset_index0=bool(np.max(abs(cache['X_both'][0]))<1e-6))
    (H/'image_api_verification.json').write_text(json.dumps(result,indent=2));print('verified',len(rows),'image-condition pairs')

if __name__=='__main__':
    with threadpool_limits(1):main()
