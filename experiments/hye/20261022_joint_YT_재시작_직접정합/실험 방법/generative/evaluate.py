"""Fixed, preregistered empirical projective candidates: adaptive development CV."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys, time
import joblib, numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parents[1]/'joint_yt_generalization_20261022'
sys.path.insert(0,str(PREVIOUS))
from data import load
from generative_model import GenerativeProjective, CONFIGS

def metrics(y,p):
    e=abs(np.asarray(y)-np.asarray(p));good=np.isfinite(e).all(1);valid=e[good]
    return dict(n=len(e),failures=int((~good).sum()),passed=int(np.sum(good & np.all(e<=.09+1e-12,1))),Y_MAE=float(valid[:,0].mean()) if len(valid) else None,Y_max=float(valid[:,0].max()) if len(valid) else None,T_MAE=float(valid[:,1].mean()) if len(valid) else None,T_max=float(valid[:,1].max()) if len(valid) else None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observation',choices=['fixed630','ecc3','ecc5'],default='ecc3');parser.add_argument('--fit-scope',choices=['all','new'],default='all');args=parser.parse_args()
    output=HERE/(args.observation+('_new' if args.fit_scope=='new' else ''));output.mkdir(parents=True,exist_ok=False)
    configs=[dict(c,fit_scope=args.fit_scope) for c in CONFIGS]
    m,a=load();y=m[['Y','T']].to_numpy();prototype=GenerativeProjective(CONFIGS[0],a['points'],a['regions'])
    if args.observation=='fixed630':
        H=prototype.measure_homographies(a['D']);source=PREVIOUS/'measurements/features.npz'
    else:
        source=HERE.parent/'direct/features.npz';raw=np.load(source);key='homography_g'+args.observation[-1]
        H=raw['H_'+key].copy();H[~raw['success_'+key].astype(bool)]=np.nan
    np.savez_compressed(output/'observed_homographies.npz',H=H,index=m['index'].to_numpy())
    protocol=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),scope='ADAPTIVE DEVELOPMENT: every current341 image has already influenced prior model development; no independent test claim',observation=args.observation,observation_source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),configs=configs,auxiliary_after_initial_g3_result=args.fit_scope=='new' or args.observation=='ecc5',selection='Fixed-candidate comparison: all four candidates reported, no outer-test score used to fit/select parameters within a candidate. Any post-hoc family preference is exploratory.',folds=m.fold.tolist(),rules='Same original four folds; mixed three repeats grouped; three new centers all fold0. All origins, forward maps and covariance fit on outer training rows only. Failures remain in341 denominator.',model='Command->predicted projective image geometry, then two continuous coordinates are estimated by nonlinear inverse optimization; no metric camera calibration or physical angle claim',initialization='Training-forward derivative at zero; start from that image-only estimate and zero, choose lower geometric cost.',sources='research.json')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];audit=[];summaries=[]
    for candidate,cfg in enumerate(configs):
        for fold in range(4):
            started=time.monotonic();tr=np.flatnonzero(m.fold.to_numpy()!=fold);te=np.flatnonzero(m.fold.to_numpy()==fold)
            assert not set(m.iloc[tr].cv_group)&set(m.iloc[te].cv_group)
            model=GenerativeProjective(cfg,a['points'],a['regions']).fit_homographies(H[tr],y[tr],m.iloc[tr]);pred=model.predict_homographies(H[te])
            r=m.iloc[te].copy();r['candidate']=candidate;r['observation']=args.observation;r['predY']=pred[:,0];r['predT']=pred[:,1];r['eY']=abs(r.predY-r.Y);r['eT']=abs(r.predT-r['T']);r['passed']=(r.eY<=.09+1e-12)&(r.eT<=.09+1e-12);r['protocol']='fixed_candidate_development_oof';rows.append(r)
            info=dict(candidate=candidate,fold=fold,config=cfg,train_indices=tr.tolist(),test_indices=te.tolist(),fit=model.audit,prediction_diagnostics=model.prediction_diagnostics,runtime=time.monotonic()-started);audit.append(info)
            joblib.dump(dict(model=model,config=cfg,train_indices=tr,points=a['points'],regions=a['regions'],ids=a['ids']),output/f'candidate{candidate}_fold{fold}.joblib')
            p=pd.concat(rows);p.to_csv(output/'predictions.csv',index=False,encoding='utf-8-sig');(output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
            new=m.iloc[te].dataset.to_numpy()==20261020
            print(args.observation,'candidate',candidate,'fold',fold,metrics(y[te][new],pred[new]),'seconds',round(time.monotonic()-started,1),flush=True)
        whole=pd.concat([r for r in rows if int(r.candidate.iloc[0])==candidate])
        for dataset in ['all',20261014,20261020]:
            g=whole if dataset=='all' else whole[whole.dataset==dataset];summaries.append(dict(candidate=candidate,config=json.dumps(cfg),dataset=dataset,**metrics(g[['Y','T']],g[['predY','predT']])))
        pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
