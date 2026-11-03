from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
P=Path(__file__).resolve().parent;ROOT=P.parents[1];V2=ROOT/'analysis/full_scene_origin_20261102'
sys.path.insert(0,str(V2))
from model_scene import SceneModel,blocks,block_data,calibration_mask
P.mkdir(parents=True,exist_ok=True)
def save(name,x):
    (P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)),encoding='utf-8')
def load():
    old=ROOT/'analysis/session_eda_20261102'
    d=pd.read_csv(old/'remeasured/observations.csv').merge(pd.read_csv(old/'image_inventory.csv')[['path','session','date','modified_utc','actual_Y','actual_T']],on='path',validate='one_to_one').copy()
    d['row_id']=np.arange(len(d));d['group']=d.session;d.loc[(d.session==3)&(d['T']<0),'group']=15
    d['suspect']=d.path.isin(set(pd.read_csv(old/'label_anomalies.csv').path))
    d['pose']=d.Y.round(6).astype(str)+':'+d['T'].round(6).astype(str)
    features=pd.read_csv(V2/'feature_partitions.csv')
    meas=np.load(V2/'measurements.npz');quality=np.load(V2/'quality.npz')
    arrays={'v':meas['values'],'valid':quality['valid'],'quality':quality['quality'],'line_valid':np.load(V2/'line_valid.npy')}
    return d,features,arrays

def local_label_audit(d,f,a):
    # Constant-Y local sequences support ordinal label checks without choosing a global zero.
    targets=d[d.path.str.contains('Y+0.05 T+0.32/Y+0.05 T+0.33',regex=False) | ((d.Y==.05)&np.isclose(d['T'],.57))]
    rows=[]
    for _,r in targets.iterrows():
        expected=.32 if 'T+0.32/' in r.path else .57
        train=d[(d.Y==.05)&(d['T'].between(expected-.05,expected+.05))&(~d.suspect)&(~np.isclose(d['T'],expected))]
        for label,ids in [('QR',None),('A_left',f[f.partition=='A_left'].feature_id.to_list()),('B_right',f[f.partition=='B_right'].feature_id.to_list())]:
            estimates=[]
            if ids is None:
                for col in ['f0','f2','f4']:
                    c=np.polyfit(train['T']-expected,train[col],2)
                    slope=c[1];estimates.append(expected+(r[col]-c[2])/slope)
            else:
                for fid in ids:
                    valid=a['valid'][train.row_id.to_numpy(int),fid]
                    if valid.mean()<.85 or not a['valid'][int(r.row_id),fid]:continue
                    tr=train.iloc[np.flatnonzero(valid)];x=a['v'][tr.row_id.to_numpy(int),fid,0]
                    c=np.polyfit(tr['T']-expected,x,2)
                    if abs(c[1])<20:continue
                    estimates.append(expected+(a['v'][int(r.row_id),fid,0]-c[2])/c[1])
            if estimates:rows.append({'path':r.path,'field':label,'file_T':r['T'],'folder_candidate_T':expected,
                'local_sequence_T':float(np.median(estimates)),'n_features':len(estimates),
                'deviation_from_expected_ticks':float((np.median(estimates)-expected)*100),
                'actual_log_T':r.actual_T,'interpretation':'local sequence coordinate, not physical T ground truth'})
    pd.DataFrame(rows).to_csv(P/'label_crosschecks.csv',index=False)

def repeat_vectors(d,f,a,partition='A_left'):
    # One median vector per scene region; do not pretend many patches are independent shots.
    records=[]
    for key,g in d[(d.cohort!='reference')].groupby('pose'):
        if len(g)<3:continue
        for ix,r in g.iterrows():
            other=g[g.row_id!=r.row_id];ri=int(r.row_id);oi=other.row_id.to_numpy(int)
            qr=np.array([r[['f0','f2','f4']].mean()-other[['f0','f2','f4']].mean().mean(),
                         r[['f1','f3','f5']].mean()-other[['f1','f3','f5']].mean().mean()])
            vectors=[qr];family_vectors=[]
            ff=f[f.partition==partition]
            for family,sub in ff.groupby('family'):
                diffs=[]
                for fid in sub.feature_id:
                    vo=a['valid'][oi,fid]
                    if not a['valid'][ri,fid] or vo.sum()<1:continue
                    diffs.append(a['v'][ri,fid,:2]-np.median(a['v'][oi[vo],fid,:2],axis=0))
                if len(diffs)>=2:
                    z=np.median(diffs,axis=0);vectors.append(z);family_vectors.append((family,z))
            common=np.median(vectors,axis=0);dispersion=np.median([np.linalg.norm(v-common) for v in vectors])
            records.append({'row_id':ri,'path':r.path,'pose':key,'session':r.session,'common_dx':common[0],'common_dy':common[1],
                'common_norm':float(np.linalg.norm(common)),'region_disagreement':dispersion,'regions':len(vectors),
                'QR_dx':qr[0],'QR_dy':qr[1]})
    result=pd.DataFrame(records)
    if len(result):
        med=result.common_norm.median();mad=1.4826*np.median(np.abs(result.common_norm-med));threshold=med+4*max(mad,.02)
        result['large_repeat_deviation']=result.common_norm>threshold
        result['threshold_px']=threshold
        result['coherent_motion_not_bad_patch']=result.large_repeat_deviation&(result.region_disagreement<np.maximum(.08,result.common_norm*.35))
    return result

def main():
    d,f,a=load();local_label_audit(d,f,a)
    audit=d.copy();audit.loc[audit.path.str.contains('Y+0.05 T+0.32/Y+0.05 T+0.33',regex=False),'T']=.32
    audit['pose']=audit.Y.round(6).astype(str)+':'+audit['T'].round(6).astype(str)
    repeat=repeat_vectors(audit,f,a);repeat.to_csv(P/'repeat_consistency.csv',index=False)
    print('REPEAT CANDIDATES');print(repeat.nlargest(12,'common_norm')[['path','common_norm','region_disagreement','coherent_motion_not_bad_patch']].to_string(index=False),flush=True)
    # Reproduce v2 OOF residuals by feature and region, preserving the old exclusions.
    cal=calibration_mask(d);usable=~d.suspect.to_numpy()
    folds=np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8],16)%5 for k in d.pose])
    residuals=[]
    for fold in range(5):
        train=d[usable&((folds!=fold)|cal)].copy();test=d[usable&(folds==fold)&~cal].copy()
        m=SceneModel(train,f,a,True,False,True).fit()
        specs,_=blocks(m.train,f,a,None,True)
        A=m.basis(m.latent(m.train))[0];At=m.basis(m.latent(test))[0]
        for s in specs:
            Z,valid,_=block_data(s,m.train,a,f);w=valid*s['quality_weight'];H=A.T@(w[:,None]*A)+np.diag(m.reg)
            rhs=A.T@(w[:,None]*Z);O=Z[m.refs&valid].mean(0);rhs[0]+=m.reg[0]*O
            C=np.linalg.solve(H,rhs);T,mask,_=block_data(s,test,a,f);raw=(At@C-T)@np.linalg.inv(s['L'])
            if s['fid'] is None:family='QR' if s['name']=='QR' else s['name']
            else:family=f.iloc[s['fid']].family
            for ii,(_,r) in enumerate(test.iterrows()):
                if not mask[ii]:continue
                residuals.append({'path':r.path,'row_id':int(r.row_id),'pose':r.pose,'Y':r.Y,'T':r['T'],'session':r.session,'fold':fold,
                    'feature':s['name'],'feature_id':s['fid'],'family':family,'dx':raw[ii,0],'dy':raw[ii,1],
                    'component_RMS':float(np.sqrt(np.mean(raw[ii]**2)))})
        print('REGION AUDIT',fold,flush=True)
    res=pd.DataFrame(residuals);res.to_csv(P/'v2_feature_OOF_residuals.csv',index=False)
    perimage=res.groupby(['path','session','Y','T','family'],as_index=False).agg(dx=('dx','median'),dy=('dy','median'),RMS=('component_RMS','median'),features=('feature','size'))
    perimage.to_csv(P/'v2_region_OOF_residuals.csv',index=False)
    perimage.groupby(['session','family']).agg(images=('path','size'),median_RMS=('RMS','median'),signed_dx=('dx','median'),signed_dy=('dy','median')).to_csv(P/'connection_break_by_session_region.csv')
    print('LABEL CHECK');print(pd.read_csv(P/'label_crosschecks.csv').to_string(index=False))
    print('DONE')
if __name__=='__main__':main()
