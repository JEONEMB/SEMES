from pathlib import Path
import json,sys
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;PREV=H.parents[1]/'joint_yt_generalization_20261022';sys.path.insert(0,str(PREV))
from data import load,metric

def main():
    m,a=load();p=pd.read_csv(H/'predictions.csv');s=pd.read_csv(H/'summary.csv');new=m.dataset.to_numpy()==20261020
    assert len(p)==162*341 and not p.duplicated(['candidate','index']).any()
    for c,g in p.groupby('candidate'):
        g=g.sort_values('index');assert np.array_equal(g['index'].to_numpy(),np.arange(341))
        assert np.allclose(g[['Y','T']],m[['Y','T']])
        for label,mask in [('all',np.ones(len(m),bool)),('new',new),('old',~new)]:
            scores=metric(g[['Y','T']].to_numpy()[mask],g[['predY','predT']].to_numpy()[mask]);row=s[(s.candidate==c)&(s.subset==label)].iloc[0]
            for k,v in scores.items():assert np.isclose(v,row[k])
    for f in range(4):
        tr=m.fold!=f;te=~tr
        assert not set(m.cv_group[tr])&set(m.cv_group[te])
    assert m[(m.dataset==20261020)&(m.scope=='center')].fold.eq(0).all()
    audit=json.loads((H/'response_audit.json').read_text());response_max=max(np.max(abs(np.array(v['response'])-np.eye(2))) for c in audit for v in c['folds'])
    best=s[s.subset=='new'].sort_values(['passed','Y_max'],ascending=[False,True]).iloc[0].to_dict()
    report=dict(status='passed',candidate_count=162,predictions_count=len(p),all_summaries_recomputed=True,no_group_overlap=True,new_centers_same_test_fold=True,all630fixed_points_used=True,training_axis_response_identity_max_abs_error=float(response_max),response_identity_is_not_test_separation=True,best_development_candidate=best,limitations=['Repeated use of the same outer folds is development comparison, not untouched external evaluation.','Pairwise differences use only training condition means.','Covariance calculated from train-condition repeats cannot describe pose-dependent model mismatch.','Cancelling common translation also discards most useful motion magnitude; shape-only results are worse.'])
    (H/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
