"""Paired same-column descriptive comparisons; no line fit or physical-angle inference."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'reference_line_methods_108_20261030'
COUNTS = np.arange(3,22,2)
STARTS = [11,5,21]
DTYPE = [('ref','U3'),('target','U20'),('x','i4'),('coarse_y','i4'),('peak','f8'),
         ('n','i4'),('initial_sigma','i4'),('A','f8'),('mu','f8'),('sigma','f8'),
         ('sse','f8'),('iterations','i4'),('converged','i4'),('finite','i4'),
         ('legacy_selected','i4'),('status','U30')]

def distribution(v):
    v=np.asarray(v,dtype=float)
    if not v.size: return {'count':0,'median':None,'p95':None,'max':None}
    assert np.isfinite(v).all()
    return {'count':len(v),'median':float(np.median(v)),
            'p95':float(np.quantile(v,.95)),'max':float(v.max())}

def main():
    data=np.genfromtxt(HERE/'fits.csv',delimiter=',',skip_header=1,dtype=DTYPE,encoding='utf-8')
    profiles=json.loads((HERE/'profiles.json').read_text())
    provenance=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
    unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in provenance['source_hashes'].items()}
    assert all(unchanged.values()),'Original source/results changed during experiment'
    summary={'design':provenance['design'],'source_files_unchanged':unchanged,
             'total_attempts':int(data.size),'groups':[],'parity':[]}
    archive={}
    for ref in ['R01','R02','R03','R04']:
        for target in ['upper_structure','lower_structure']:
            key=ref+'_'+target
            group=data[(data['ref']==ref)&(data['target']==target)]
            xs=np.unique(group['x'])
            matrices={name:np.empty((len(xs),10,3),dtype=float) for name in
                      ['mu','A','sigma','sse','iterations','converged','finite','legacy_selected']}
            for ni,n in enumerate(COUNTS):
                for si,start in enumerate(STARTS):
                    rows=np.sort(group[(group['n']==n)&(group['initial_sigma']==start)],order='x')
                    assert np.array_equal(rows['x'],xs)
                    for name,mat in matrices.items(): mat[:,ni,si]=rows[name]
            selected=matrices['legacy_selected'][:,:,0].astype(bool)
            valid=(matrices['finite'][:,:,0]>0)&(matrices['converged'][:,:,0]>0)&(matrices['A'][:,:,0]>0)
            mu=matrices['mu'][:,:,0]
            common=selected[:,1:].all(axis=1)
            raw_common=valid[:,1:].all(axis=1)
            rows11=np.sort(group[(group['n']==11)&(group['initial_sigma']==11)],order='x')
            assert not np.any(np.isnan(mu[selected]))
            old=json.loads((BASE/'results/intermediates'/ref/'E01'/(target+'.json')).read_text(encoding='utf-8'))['intermediates']
            oldcols=old['columns']
            assert [int(r[0]) for r in oldcols]==list(xs)
            assert [int(r[1]) for r in oldcols]==list(rows11['coarse_y'])
            old_selected=np.array([r[2]=='selected' for r in oldcols])
            assert np.array_equal(old_selected,rows11['legacy_selected'].astype(bool))
            rounddiff=np.abs(np.array([r[3] for r in oldcols],dtype=float)[old_selected]-rows11['mu'][old_selected])
            rawdiff=[]; profilediff=[]
            profiles_group=[p for p in profiles if p['ref']==ref and p['target']==target]
            pmap={p['x']:p for p in profiles_group}
            rmap={int(r['x']):r for r in rows11}
            for e in old['profile_examples']:
                r=rmap[e['x']]
                rawdiff.append(abs(e['mu']-float(r['mu'])))
                profilediff.append(float(np.max(np.abs(np.array(e['sobel_samples'])-np.array(pmap[e['x']]['sobel_samples_m10_p10'])[5:16]))))
            assert max(profilediff,default=0)==0
            assert max(rawdiff,default=0)<1e-9
            assert max(rounddiff,default=0)<=5.0001e-6
            parity={'ref':ref,'target':target,'coarse_points_equal':True,
                    'legacy_selected_mask_equal':True,'n_raw_examples':len(rawdiff),
                    'max_example_mu_diff_px':max(rawdiff,default=0),
                    'max_sobel_diff':max(profilediff,default=0),
                    'max_mu_diff_from_5decimal_archive':float(max(rounddiff,default=0))}
            summary['parity'].append(parity)
            g={'ref':ref,'target':target,'visible_columns':len(profiles_group),'eligible_columns':len(xs),
               'common_selected_5_to_21':int(common.sum()),'raw_common_converged_5_to_21':int(raw_common.sum()),
               'span_5_to_21_px':distribution(np.ptp(mu[common,1:],axis=1)),
               'raw_span_5_to_21_px':distribution(np.ptp(mu[raw_common,1:],axis=1)),
               'span_11_to_15_same_common_px':distribution(np.ptp(mu[common,4:7],axis=1)),
               'span_11_to_21_same_common_px':distribution(np.ptp(mu[common,4:],axis=1)),
               'span_17_to_21_same_common_px':distribution(np.ptp(mu[common,7:],axis=1)),
               'windows':[],'initialization_audit':[]}
            for ni,n in enumerate(COUNTS):
                r=group[(group['n']==n)&(group['initial_sigma']==11)]
                comparison_mask=common if n>=5 else (common & selected[:,ni])
                delta=mu[comparison_mask,ni]-mu[comparison_mask,4]
                pair=selected[:,ni]&selected[:,4]
                status,ct=np.unique(r['status'],return_counts=True)
                w={'n':int(n),'half_width_px':int((n-1)//2),'statuses':dict(zip(status,[int(c) for c in ct])),
                   'selected_count':int(selected[:,ni].sum()),'selected_percent':float(100*selected[:,ni].mean()),
                   'converged_positive_count':int(valid[:,ni].sum()),
                   'abs_delta_from_11_common_px':distribution(abs(delta)),
                   'signed_median_delta_from_11_common_px':float(np.median(delta)) if len(delta) else None,
                   'abs_delta_from_11_pair_selected_px':distribution(abs(mu[pair,ni]-mu[pair,4])),
                   'median_sigma_selected_px':float(np.median(matrices['sigma'][selected[:,ni],ni,0])) if selected[:,ni].any() else None}
                if ni>=2:
                    w['abs_adjacent_delta_common_px']=distribution(abs(mu[common,ni]-mu[common,ni-1]))
                g['windows'].append(w)
                start_common=matrices['legacy_selected'][:,ni,:].all(axis=1)
                raw_start_common=(matrices['finite'][:,ni,:]>0).all(axis=1)&(matrices['converged'][:,ni,:]>0).all(axis=1)&(matrices['A'][:,ni,:]>0).all(axis=1)
                g['initialization_audit'].append({'n':int(n),
                    'three_start_mu_span_selected_px':distribution(np.ptp(matrices['mu'][start_common,ni,:],axis=1)),
                    'three_start_mu_span_raw_converged_px':distribution(np.ptp(matrices['mu'][raw_start_common,ni,:],axis=1)),
                    'three_start_status_disagreements':int(np.any(matrices['legacy_selected'][:,ni,:]!=matrices['legacy_selected'][:,ni,0,None],axis=1).sum())})
            summary['groups'].append(g)
            archive[key+'_xs']=xs
            archive[key+'_coarse_y']=rows11['coarse_y']
            archive[key+'_common']=common
            for name,mat in matrices.items(): archive[key+'_'+name]=mat
    np.savez_compressed(HERE/'paired_arrays.npz',**archive)
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for g in summary['groups']:
        print(g['ref'],g['target'],'common',g['common_selected_5_to_21'],'/',g['eligible_columns'],
              'span',g['span_5_to_21_px'],'tail',g['span_17_to_21_same_common_px'])
        print('selected',[(w['n'],w['selected_count']) for w in g['windows']])
        print('adjacent P95',[(w['n'],round(w['abs_adjacent_delta_common_px']['p95'],6)) for w in g['windows'] if w.get('abs_adjacent_delta_common_px',{}).get('count')])
        print('init max',max(w['three_start_mu_span_selected_px']['max'] or 0 for w in g['initialization_audit'][1:]))
    print('All 8 baseline parity checks passed; sources unchanged.')

if __name__=='__main__': main()
