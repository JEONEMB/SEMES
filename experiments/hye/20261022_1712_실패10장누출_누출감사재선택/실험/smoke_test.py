"""Real-image execution consistency only; NOT a performance estimate."""
from pathlib import Path
import sys,json,joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'joint_yt_generalization_20261022'))
from data import load
sys.path.insert(0,str(H))
from predict import predict_bytes,DEFAULT_MODEL

def main():
    m,a=load();bundle=joblib.load(DEFAULT_MODEL);records=[]
    for i in [335,336]:
        row=m.iloc[i];raw=Path(row.path).read_bytes();p=predict_bytes(raw);q=bundle['model'].predict(a['D'][i:i+1])[0]
        assert np.allclose([p['Y'],p['T']],q,atol=1e-8,rtol=0)
        records.append(dict(index=i,image_path=row.path,result=p,cached_prediction=q.tolist(),difference=float(np.max(abs(np.array([p['Y'],p['T']])-q)))))
        print(i,p['Y'],p['T'],flush=True)
    (H/'smoke_test.json').write_text(json.dumps(dict(scope='Execution equivalence on TRAINING images, not new-photo performance',records=records),indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(1):main()
