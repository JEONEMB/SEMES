"""Guarded response updates on the same purged, static photo combinations."""
import json,itertools,hashlib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from evaluate import H,OLD,CONDITIONS,Calibration,paths,summarize
from motion import MotionController,Settings

def main():
    m=pd.read_csv(OLD/'manifest.csv');a=np.load(OLD/'features.npz');q=m[['Y','T']].to_numpy();rows=[];audit=[]
    (H/'recorded_guarded_protocol.json').write_text(json.dumps(dict(scope='Exploratory static combinations; same36paths,48source images, all repeats as online-only. NOT autonomous probe selection or actual closed-loop replay.',train='Start fold excluded and all12rectangle images additionally purged from projection/noise/prior fitting.',prior='TRAIN-only linear response; session updates use relative differences only.',increments='Historical grid differences, including third Y reversal. These need not equal the policy proposal and are accepted as acknowledged increments; not a simulation of an executed adaptive policy.',score='Current pose inferred from negative full correction; no observed endpoint after correction.',source_hash=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()),indent=2))
    for cond in CONDITIONS:
        X=a['X_'+cond];good=a['success_'+cond]
        for pid,path in enumerate(paths()):
            states=[np.flatnonzero((m.dataset.to_numpy()==20261020)&np.all(np.isclose(q,v,atol=1e-12),axis=1)) for v in path]
            excluded=np.concatenate(states);fold=int(m.iloc[states[0][0]].fold)
            tr=np.flatnonzero((m.fold.to_numpy()!=fold)&good);tr=tr[~np.isin(tr,excluded)]
            assert not set(m.iloc[tr].cv_group)&set(m.iloc[excluded].cv_group)
            cal=Calibration.fit(X[tr],q[tr],m.iloc[tr].dataset.to_numpy(),tr);S=cal.project(X);offsets=path-path[0]
            audit.append(dict(condition=cond,path=pid,train=tr.tolist(),excluded=excluded.tolist()))
            for frames in [3,4]:
                step=abs(path[1]-path[0])+abs(path[2]-path[1]);cfg=Settings(probe_y=float(step[0]),probe_t=float(step[1]),probe_count=frames-1,max_step_y=max(.25,float(step[0])),max_step_t=max(.25,float(step[1])),noise_rms=max(cal.noise_rms,1.))
                for ids in itertools.product(*states[:frames]):
                    c=MotionController(cfg,cal.jacobian).start(S[ids[0]],np.zeros(S.shape[1]))
                    for k in range(1,frames):c.next_action();c.observe(S[ids[k]],offsets[k]-offsets[k-1])
                    predicted=-c.correction;e=abs(predicted-path[frames-1]);rows.append(dict(condition=cond,path=pid,frames=frames,moves=frames-1,indices=','.join(map(str,ids)),current_Y=path[frames-1,0],current_T=path[frames-1,1],predY=predicted[0],predT=predicted[1],Y_abs=e[0],T_abs=e[1],passed=bool(np.isfinite(e).all() and np.all(e<=.09+1e-12)),online_updates=c.identification_updates,response_snr=c.diagnostics.get('response_snr')))
        print('recorded guarded',cond,'complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(H/'recorded_guarded_predictions.csv',index=False)
    pd.DataFrame([dict(condition=cond,frames=int(frames),**summarize(g)) for (cond,frames),g in df.groupby(['condition','frames'])]).to_csv(H/'recorded_guarded_summary.csv',index=False)
    (H/'recorded_guarded_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
