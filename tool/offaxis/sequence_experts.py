"""Metadata-based sequence experts with coordinate-held-out image-only gating.

Sequence boundaries use only confirmed filesystem mtime (60/120/300 s gaps).
No threshold is chosen by validation accuracy. Same coordinate repetitions are
always kept together, including repetitions spanning a time-gap boundary.
"""
import sys
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.analysis_packages'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.analysis_packages/mplconfig'))
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from yaw_identifiability import load_data,transform,TICKS

OUT=ROOT/'docs/offaxis-eda/yaw-separation/sequence-experts'
BASE=ROOT/'docs/offaxis-eda/ppt-method-v2'


def sequences(meta,gap):
    # Include references while detecting pauses, then exclude them from models.
    chronological=meta.sort_values('mtime_ns',kind='stable')
    groups=(chronological.mtime_ns.diff().fillna(np.inf)/1e9>gap).cumsum().astype(int)
    return groups.reindex(meta.index).to_numpy()


def split_coordinates(meta,which):
    test=np.zeros(len(meta),bool)
    for cat in ['Y-only','theta-only','mixed']:
        ix=np.flatnonzero(meta.category.eq(cat))
        coords=np.unique(meta.iloc[ix][['y_mm','theta_deg']].to_numpy(),axis=0)
        if which=='interleaved':held=coords[2::3]
        else:held=coords[len(coords)//3:2*len(coords)//3]
        for c in held:
            test|=meta.y_mm.eq(c[0]).to_numpy()&meta.theta_deg.eq(c[1]).to_numpy()
    return test


def pose_basis(q):
    u,s,vt=np.linalg.svd(q-q.mean(0),full_matrices=False)
    if len(s)==0 or s[0]<1e-10:return np.empty((2,0))
    rank=int(np.sum(s>max(1e-9,s[0]*1e-9)))
    return vt[:rank].T


def shared_slopes(x,q,seq):
    xx=x.copy();qq=q.copy()
    for k in np.unique(seq):
        mask=seq==k;xx[mask]-=xx[mask].mean(0);qq[mask]-=qq[mask].mean(0)
    return np.linalg.lstsq(qq,xx,rcond=None)[0].T


def make_expert(x,q,seq_id,prior=None,pool=0.):
    if len(np.unique(q,axis=0))<3:return None
    qb=q.mean(0);xb=x.mean(0);V=pose_basis(q)
    if V.shape[1]==0:return None
    r=(q-qb)@V
    B=np.linalg.lstsq(r,x-xb,rcond=None)[0].T
    if prior is not None:B=pool*(prior@V)+(1-pool)*B
    inverse=np.linalg.pinv(B,rcond=1e-10)
    predicted=xb+r@B.T
    # A common heuristic temperature for soft gating, not a calibrated posterior.
    residual=np.sqrt(np.mean((x-predicted)**2,axis=1))
    return dict(seq=int(seq_id),qbar=qb,xbar=xb,V=V,B=B,inverse=inverse,
        rmin=r.min(0),rmax=r.max(0),rank=V.shape[1],train_frames=len(x),
        train_poses=len(np.unique(q,axis=0)),train_residual_median=float(np.median(residual)))


def predict(expert,x,bounded=True):
    r=(x-expert['xbar'])@expert['inverse'].T
    # Calibration bounds are learned solely from training coordinates.
    if bounded:r=np.clip(r,expert['rmin'],expert['rmax'])
    q=expert['qbar']+r@expert['V'].T
    fitted=expert['xbar']+r@expert['B'].T
    cost=np.mean((x-fitted)**2,axis=1)
    return q,cost


def metrics(name,frame,truth,pred,valid,extra):
    if not valid.any():return dict(model=name,images=0,**extra)
    e=pred[valid]-truth[valid]
    return dict(model=name,images=int(valid.sum()),Y_rmse_ticks=float(np.sqrt(np.mean(e[:,0]**2))),
        yaw_rmse_ticks=float(np.sqrt(np.mean(e[:,1]**2))),
        Y_mae_ticks=float(np.mean(abs(e[:,0]))),yaw_mae_ticks=float(np.mean(abs(e[:,1]))),
        both_within_tick=float(np.mean(np.max(abs(e),axis=1)<=1)),**extra)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta,raw,ids=load_data()
    active=meta.category.isin(['Y-only','theta-only','mixed']).to_numpy()
    q=meta[['y_mm','theta_deg']].to_numpy()/TICKS
    summaries=[];predrows=[];seqrows=[];fits=[];gaterows=[];bundles={}
    assignments=meta.reset_index()[['path','category','mtime_ns','time_kst','y_mm','theta_deg']].copy()
    actual_mtime=[]
    for path,row in meta.iterrows():
        ns=(ROOT/'data/SSAFY Test Image Set'/path).stat().st_mtime_ns
        assert abs(ns-int(row.mtime_ns))<=1000,path
        actual_mtime.append(ns)
    assignments['current_mtime_ns']=actual_mtime
    for gap in [60,120,300]:
        seq=sequences(meta,gap);assignments[f'sequence_gap_{gap}s']=seq
        for k in np.unique(seq[active]):
            ix=np.flatnonzero(active&(seq==k));g=meta.iloc[ix]
            rank=pose_basis(q[ix]).shape[1]
            seqrows.append(dict(gap_seconds=gap,sequence=int(k),images=len(ix),coordinates=len(np.unique(q[ix],axis=0)),
                pose_rank=rank,start_kst=g.sort_values('mtime_ns').time_kst.iloc[0],end_kst=g.sort_values('mtime_ns').time_kst.iloc[-1],
                categories=','.join(g.category.unique())))
        for split in ['interleaved','middle_block']:
            test=split_coordinates(meta,split)&active;train=active&~test
            traincoords=set(map(tuple,q[train]));testcoords=set(map(tuple,q[test]))
            assert not traincoords&testcoords
            testix=np.flatnonzero(test);truth=q[testix];testseq=seq[testix]
            for feature in ['L630_full','three_QR_means']:
                x=transform(raw,feature);xt=x[testix]
                prior=shared_slopes(x[train],q[train],seq[train])
                global_expert=make_expert(x[train],q[train],-1)
                global_prediction,global_cost=predict(global_expert,xt,False)
                # A direct six-channel inverse regression is an additional strong
                # baseline, not the same model as a forward geometric expert.
                if feature=='three_QR_means':
                    mean=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-12)
                    design=np.column_stack([np.ones(train.sum()),(x[train]-mean)/scale])
                    coef=np.linalg.lstsq(design,q[train],rcond=None)[0]
                    inverse_prediction=np.column_stack([np.ones(len(xt)),(xt-mean)/scale])@coef
                for family in ['shared_slope','independent','pooled_25']:
                    experts=[]
                    for k in np.unique(seq[train]):
                        ix=train&(seq==k)
                        pool=1. if family=='shared_slope' else .25 if family=='pooled_25' else 0.
                        model=make_expert(x[ix],q[ix],k,prior,pool)
                        if model is not None:experts.append(model)
                    assert experts
                    Q=[];C=[];Qfree=[];Cfree=[]
                    for expert in experts:
                        p,c=predict(expert,xt);Q.append(p);C.append(c)
                        p,c=predict(expert,xt,False);Qfree.append(p);Cfree.append(c)
                    Q=np.stack(Q,axis=1);C=np.stack(C,axis=1)
                    Qfree=np.stack(Qfree,axis=1);Cfree=np.stack(Cfree,axis=1)
                    expert_ids=np.array([e['seq'] for e in experts]);ranks=np.array([e['rank'] for e in experts])
                    best=C.argmin(1);hard=Q[np.arange(len(xt)),best]
                    bestfree=Cfree.argmin(1);hardfree=Qfree[np.arange(len(xt)),bestfree]
                    temperature=max(.005,float(np.median([e['train_residual_median'] for e in experts])))
                    logits=-(C-C.min(1,keepdims=True))/(2*temperature**2)
                    weights=np.exp(np.maximum(logits,-700));weights/=weights.sum(1,keepdims=True)
                    soft=np.einsum('nk,nkj->nj',weights,Q)
                    lookup={e['seq']:i for i,e in enumerate(experts)}
                    oracle=np.full_like(truth,np.nan);available=np.array([int(k) in lookup for k in testseq])
                    global_conditioned=np.full_like(truth,np.nan)
                    for i,k in enumerate(testseq):
                        if int(k) in lookup:
                            oracle[i]=Qfree[i,lookup[int(k)]]
                            # Separate the benefit of knowing a 1D sweep from
                            # the benefit of sequence-specific fitted parameters.
                            local=experts[lookup[int(k)]]
                            globalJ=global_expert['B']@global_expert['V'].T
                            conditioned=local.copy()
                            conditioned['B']=globalJ@local['V']
                            conditioned['inverse']=np.linalg.pinv(conditioned['B'],rcond=1e-10)
                            conditioned['xbar']=global_expert['xbar']+globalJ@(local['qbar']-global_expert['qbar'])
                            global_conditioned[i]=predict(conditioned,xt[i:i+1],False)[0][0]
                    # All comparisons use exactly the cases with a usable known
                    # sequence expert; excluded cases are separately counted.
                    modes={'global_forward':global_prediction,'known_sequence':oracle,'image_hard_gate':hard,
                           'image_unbounded_gate':hardfree,'image_soft_blend':soft,
                           'global_parameters_known_sequence':global_conditioned}
                    if feature=='three_QR_means':modes['global_inverse_6ch']=inverse_prediction
                    for name,pred in modes.items():
                        for subset,mask in [('all_supported',np.ones(len(xt),bool)),('axis_supported',meta.iloc[testix].category.isin(['Y-only','theta-only']).to_numpy()),
                                            ('mixed_supported',meta.iloc[testix].category.eq('mixed').to_numpy())]:
                            valid=available&mask
                            summaries.append(metrics(name,meta.iloc[testix],truth,pred,valid,
                                dict(gap_seconds=gap,split=split,feature=feature,family=family,subset=subset,
                                    requested_test_images=int(mask.sum()),unsupported_test_images=int((mask&~available).sum()))))
                        for i,idx in enumerate(testix):
                            chosen=bestfree[i] if name=='image_unbounded_gate' else best[i]
                            if name in ['known_sequence','global_parameters_known_sequence']:
                                selected=int(testseq[i]) if available[i] else -1
                                selected_rank=experts[lookup[int(testseq[i])]]['rank'] if available[i] else -1
                            elif name.startswith('image_'):
                                selected=int(expert_ids[chosen]);selected_rank=int(ranks[chosen])
                            else:selected=-1;selected_rank=-1
                            predrows.append(dict(gap_seconds=gap,split=split,feature=feature,family=family,model=name,
                                path=meta.index[idx],category=meta.category.iloc[idx],sequence=int(testseq[i]),
                                supported=bool(available[i]),Y_true_ticks=truth[i,0],yaw_true_ticks=truth[i,1],
                                Y_pred_ticks=pred[i,0],yaw_pred_ticks=pred[i,1],
                                gate_sequence=selected,gate_rank=selected_rank))
                    for subset,mask in [('all',available),('axis',available&meta.iloc[testix].category.isin(['Y-only','theta-only']).to_numpy()),
                                       ('mixed',available&meta.iloc[testix].category.eq('mixed').to_numpy())]:
                        if mask.any():gaterows.append(dict(gap_seconds=gap,split=split,feature=feature,family=family,subset=subset,
                            images=int(mask.sum()),exact_sequence_accuracy=float(np.mean(expert_ids[best[mask]]==testseq[mask])),
                            chosen_rank1_fraction=float(np.mean(ranks[best[mask]]==1)),
                            median_best_residual_px=float(np.median(np.sqrt(C[mask].min(1)))),temperature_px=temperature))
                    for expert in experts:
                        trainlocal=train&(seq==expert['seq']);testlocal=test&(seq==expert['seq'])
                        for partition,ix in [('train',np.flatnonzero(trainlocal)),('test',np.flatnonzero(testlocal))]:
                            if len(ix)==0:continue
                            # Evaluate forward prediction at known command for
                            # held-out linearity diagnostics, not for routing.
                            rr=(q[ix]-expert['qbar'])@expert['V']
                            xp=expert['xbar']+rr@expert['B'].T
                            err=x[ix]-xp
                            fits.append(dict(gap_seconds=gap,split=split,feature=feature,family=family,
                                sequence=expert['seq'],rank=expert['rank'],partition=partition,images=len(ix),
                                reconstruction_rms_px=float(np.sqrt(np.mean(err**2)))))
                    if gap==120 and split=='interleaved' and feature=='L630_full':
                        bundles[family]=[dict(sequence=e['seq'],qbar=e['qbar'].tolist(),xbar=e['xbar'].tolist(),
                            basis=e['V'].tolist(),forward=e['B'].tolist(),rank=e['rank'],
                            calibration_min=e['rmin'].tolist(),calibration_max=e['rmax'].tolist(),
                            train_frames=e['train_frames'],train_coordinates=e['train_poses']) for e in experts]
    assignments.to_csv(OUT/'sequence_assignments.csv',index=False)
    pd.DataFrame(seqrows).to_csv(OUT/'sequence_inventory.csv',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'validation_summary.csv',index=False)
    pd.DataFrame(predrows).to_csv(OUT/'heldout_predictions.csv',index=False)
    pd.DataFrame(fits).to_csv(OUT/'linearity_validation.csv',index=False)
    pd.DataFrame(gaterows).to_csv(OUT/'image_gate_diagnostics.csv',index=False)
    (OUT/'primary_experts.json').write_text(json.dumps(bundles,indent=2),encoding='utf-8')
    sources=[Path(__file__),Path(__file__).with_name('yaw_identifiability.py'),BASE/'frames.csv',BASE/'points.csv.gz']
    manifest=dict(time_metadata='filesystem mtime; user confirmed as sequence cue',
        gap_threshold_seconds=[60,120,300],primary_gap_seconds=120,
        threshold_selected_using_errors=False,coordinate_repetitions_kept_together=True,
        scopes='within observed sequences; not held-out new-session generalization',
        primary_features='L630 dx/dy; extraction unchanged',pool_fraction=.25,
        gate_uses_true_label=False,gate_uses_timestamp=False,metadata_oracle_uses_sequence_id=True,
        known_sequence_extrapolation_allowed=True,image_hard_gate_bounded=True,
        note='Rank-1 experts only reconstruct their observed pose line; this is not unrestricted two-axis calibration.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    figures(meta,raw,assignments,summary)
    view=summary[summary.gap_seconds.eq(120)&summary.split.eq('interleaved')&summary.feature.eq('L630_full')&summary.subset.eq('all_supported')]
    print(view[['family','model','images','unsupported_test_images','Y_rmse_ticks','yaw_rmse_ticks','both_within_tick']].to_string(index=False),flush=True)
    print(pd.DataFrame(seqrows).query('gap_seconds == 120').to_string(index=False),flush=True)


def figures(meta,raw,assignment,summary):
    seq=assignment.sequence_gap_120s.to_numpy();active=meta.category.isin(['Y-only','theta-only','mixed']).to_numpy()
    mean=raw.reshape(len(raw),630,2).mean(1);mean-=mean[meta.index.get_loc('center.bmp')]
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    colors=plt.get_cmap('tab20')
    for ax,cat,col,title in [(axes[0],'Y-only','y_mm','Y-only sequences'),(axes[1],'theta-only','theta_deg','Yaw-only sequences')]:
        mask=meta.category.eq(cat).to_numpy();q=meta[col].to_numpy()
        base=np.linalg.lstsq(np.column_stack([np.ones(mask.sum()),q[mask]]),mean[mask,0],rcond=None)[0]
        y=mean[:,0]-base[1]*q
        for k in np.unique(seq[mask]):
            ix=np.flatnonzero(mask&(seq==k));x=q[ix];z=y[ix]
            ax.scatter(x,z,s=10,color=colors(k%20),label=f'S{k} (n={len(ix)})')
            if len(np.unique(x))>=3:
                line=np.polyfit(x,z,1);xx=np.linspace(x.min(),x.max(),50)
                ax.plot(xx,np.polyval(line,xx),c=colors(k%20))
        ax.set_title(title);ax.set_xlabel(col);ax.set_ylabel('Mean dx minus common slope (px)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Confirmed mtime sequences: residual lines after removing one common slope')
    fig.savefig(OUT/'01_sequence_lines.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4),layout='constrained')
    ordered=meta.reset_index().assign(sequence=seq).sort_values('mtime_ns')
    for k,g in ordered[ordered.category.isin(['Y-only','theta-only','mixed'])].groupby('sequence'):
        times=pd.to_datetime(g.mtime_ns,unit='ns',utc=True).dt.tz_convert('Asia/Seoul')
        ax.plot(times,np.full(len(g),k),'o-',ms=3,lw=2,color=colors(k%20))
    ax.set_ylabel('Sequence ID (gap >120 s)');ax.set_xlabel('KST filesystem mtime');ax.grid(alpha=.2);ax.set_title('Sequence boundaries fixed before accuracy evaluation')
    fig.savefig(OUT/'02_time_sequences.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    chosen=['global_forward','known_sequence','image_hard_gate','image_soft_blend']
    for ax,metric,title in [(axes[0],'Y_rmse_ticks','Y'),(axes[1],'yaw_rmse_ticks','yaw')]:
        g=summary[summary.gap_seconds.eq(120)&summary.split.eq('interleaved')&summary.feature.eq('L630_full')&summary.family.eq('independent')&summary.subset.eq('all_supported')].set_index('model')
        ax.bar(range(4),g.loc[chosen,metric],color=['#8495a5','#2c9564','#3b7cb2','#dd913c'])
        ax.set_xticks(range(4),['Global\nforward','Known\nsequence','Image\nhard gate','Image\nsoft blend']);ax.set_ylabel('Held-coordinate RMSE (ticks)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same held images: known sequence and image-only selection are different tests')
    fig.savefig(OUT/'03_gating_comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
