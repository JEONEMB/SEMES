"""Static-image multi-view inference at recorded relative command offsets.

These are NOT temporal trajectories or closed-loop robot experiments. All nine
adjacent rectangles in the pre-existing 4x4 mixed command grid, all four starts,
and all repeated-image combinations are included. Every rectangle condition is
purged from calibration before single- and multi-image comparisons.
"""
from pathlib import Path
import sys,json,itertools
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'joint_yt_reboot_20261022'))
from metrics import metric
sys.path.insert(0,str(H))
from models import ObservationModel

def paths():
    out=[]
    for ya,yb in zip([.05,.1,.5],[.1,.5,1.]):
        for ta,tb in zip([.01,.1,1.],[.1,1.,2.]):
            for y0,y1 in [(ya,yb),(yb,ya)]:
                for t0,t1 in [(ta,tb),(tb,ta)]:
                    out.append(np.array([[y0,t0],[y1,t0],[y1,t1],[y0,t1]]))
    return out

def main():
    m=pd.read_csv(H/'manifest.csv');a=np.load(H/'features.npz');q=m[['Y','T']].to_numpy();out=[];audit=[];allpaths=paths();cfg=dict(degree=2,shrink=.1,scope='all')
    protocol=dict(status='Static actual-image combination development study; NOT executed motion, NOT a control success rate',paths=len(allpaths),rectangle_grid='Y=.05,.1,.5,1; T=.01,.1,1,2; adjacent rectangles, all four corner starts; firstY thenT then undoY',policy='Enumeration is benchmark construction, not autonomous action selection; not choosing paths by observed error',step_limits='Available grid steps range Y=.05 to.5,T=.09 to1. These are NOT validated small operational probes.',reference='original fixed center',training='historical starting-condition outer training fold, additionally purge ALL four rectangle conditions and all repeats; identical training for1/3/4frames',decoder=cfg,known_inference_input='images plus cumulative relative command differences; absolute initial/endpoint labels only for external scoring',repeat_combinations='all3^frames combinations; correlated combinations of48existing images, not independent new trials',first_image_turn=0,frames_to_moves={'1':0,'3':2,'4':3})
    (H/'stencil_protocol.json').write_text(json.dumps(protocol,indent=2))
    for cond in ['qr','upper','lower','both']:
        X=a['X_'+cond];good=a['success_'+cond]
        for pid,path in enumerate(allpaths):
            states=[np.flatnonzero((m.dataset.to_numpy()==20261020)&np.all(np.isclose(q,v,atol=1e-12),axis=1)) for v in path]
            assert all(len(ids)==3 for ids in states)
            fold=int(m.iloc[states[0][0]].fold);excluded=np.concatenate(states);tr=np.flatnonzero((m.fold.to_numpy()!=fold)&good);tr=tr[~np.isin(tr,excluded)]
            assert not set(tr)&set(excluded)
            assert not set(m.iloc[tr].cv_group)&set(m.iloc[excluded].cv_group)
            model=ObservationModel(cfg).fit(X[tr],q[tr],m.iloc[tr]);u=path-path[0]
            audit.append(dict(condition=cond,path=pid,fold=fold,states=path.tolist(),relative_offsets=u.tolist(),training_indices=tr.tolist(),excluded_indices=excluded.tolist()))
            for frames in [1,3,4]:
                for ids in itertools.product(*states[:frames]):
                    ids=np.array(ids);p=np.array([np.nan,np.nan]);success=False;nfev=0
                    if good[ids].all():
                        if frames==1:p=model.predict(X[ids])[0];success=bool(np.isfinite(p).all())
                        else:
                            p,info=model.predict_sequence(X[ids],u[:frames]);success=info['success'] and np.isfinite(p).all();nfev=info['nfev']
                            if not success:p[:]=np.nan
                    e=abs(p-path[0]);out.append(dict(condition=cond,path=pid,frames=frames,moves=frames-1,indices=','.join(map(str,ids)),Y=path[0,0],T=path[0,1],predY=p[0],predT=p[1],eY=e[0],eT=e[1],passed=bool(success and np.all(e<=.09+1e-12)),success=bool(success),nfev=nfev))
            if pid%9==8:print(cond,'paths',pid+1,'/',len(allpaths),flush=True)
        pd.DataFrame(out).to_csv(H/'stencil_predictions_progress.csv',index=False)
    result=pd.DataFrame(out);result.to_csv(H/'stencil_predictions.csv',index=False);(H/'stencil_training_audit.json').write_text(json.dumps(audit,indent=2));rows=[]
    for (cond,frames),g in result.groupby(['condition','frames']):
        r=metric(g[['Y','T']],g[['predY','predT']]);rows.append(dict(condition=cond,frames=frames,moves=frames-1,**r,pass_fraction=r['passed']/r['n'],path_setups=g.path.nunique(),source_images=48))
    pd.DataFrame(rows).to_csv(H/'stencil_summary.csv',index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
