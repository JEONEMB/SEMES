from pathlib import Path
import csv,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
M=json.loads((HERE/'metadata.json').read_text())
N=np.array(M['sample_counts'])
STARTS=M['initializations'];STOPS=[x['id'] for x in M['stopping']]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):
    v=np.asarray(v).ravel();bad=int((~np.isfinite(v)).sum())
    return dict(count=len(v),nonfinite=bad,min=None if bad else float(v.min()),median=None if bad else float(np.median(v)),max=None if bad else float(v.max()))
def safe(v):
    if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
    if isinstance(v,list):return [safe(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def map_group(g):return np.memmap(HERE/g['file'],dtype='<f8',mode='r',shape=tuple(g['shape']))
COMPARE=[('cap_800_to_1600',0,1),('cap_1600_to_3200',1,2),('cap_3200_to_6400',2,3),
         ('function_only_to_dual_1e-10',3,6),('dual_1e-6_to_1e-8',4,5),('dual_1e-8_to_1e-10',5,6)]
RANGES=[(3,21),(7,11),(9,11),(11,15),(11,21),(13,21),(21,31),(31,41),(41,51),(51,61),(59,61)]
summary={'groups':[],'pooled':{},'comparisons':COMPARE,'ranges':RANGES,'all_returns_retained':True,'witnesses':[]}
old=np.load(HERE.parent/'gaussian_window_no_rejection_20261101/arrays.npz')
checks={'checkpoint_selftest_exact_cases':96,'baseline_800_exact_fits':0,'raw_returns':0,'all_visible_columns':True}
for g in M['groups']:
    d=map_group(g);key=g['ref']+'_'+g['target'];prior=old[key+'_data']
    assert d.shape[0]==len(g['x']) and d.shape[0]==prior.shape[0]
    assert np.array_equal(g['x'],prior[:,0,0])
    for ni,oi in [(0,7),(1,8),(2,9),(3,10),(4,11),(5,12)]:assert np.array_equal(d[:,:,3,0,ni],prior[:,:,oi]),(key,ni)
    checks['baseline_800_exact_fits']+=len(d)*30
    checks['raw_returns']+=int(np.prod(d.shape[:-1]))
    summary['groups'].append({'ref':g['ref'],'target':g['target'],'profiles':g['profiles'],
       'nonfinite_parameter_or_sse_returns':int((~np.isfinite(d[:,:,:,:,:4]).all(axis=-1)).sum()),
       'nonconverged_by_start_and_stop':(d[:,:,:,:,5]==0).sum(axis=(0,1)).tolist()})

with (HERE/'numerical_comparison.csv').open('w',newline='',encoding='utf-8-sig') as file:
    writer=csv.writer(file);writer.writerow(['target','n','initialization','comparison','count','nonfinite','mu_change_min','mu_change_median','mu_change_max','sse_change_max'])
    for target in ['upper_structure','lower_structure']:
        selected=[g for g in M['groups'] if g['target']==target]
        # Process each target separately; no return is dropped based on status or position.
        d=np.concatenate([map_group(g) for g in selected],axis=0)
        refs=np.concatenate([np.repeat(g['ref'],g['profiles']) for g in selected])
        xs=np.concatenate([g['x'] for g in selected]);cy=np.concatenate([g['coarse_y'] for g in selected])
        rows=[];pooled={'profiles':len(d),'by_n':rows,'window_ranges':[]}
        for ni,n in enumerate(N):
            dn=d[:,ni];mu=dn[:,:,:,1];sse=dn[:,:,:,3]
            row={'n':int(n),'nonconverged':(dn[:,:,:,5]==0).sum(axis=0).tolist(),
                 'nonfinite':(~np.isfinite(dn[:,:,:,:4]).all(axis=-1)).sum(axis=0).tolist(),
                 'initialization_spread':{},'numerical_comparisons':{},'window_adjacent':{}}
            # Initial-width effect: only fixed_5/11/21 (A0, mu0, simplex and data identical).
            for ci,case in enumerate(STOPS):
                row['initialization_spread'][case]={'mu':stats(np.ptp(mu[:,:3,ci],axis=1)),
                    'sse':stats(np.ptp(sse[:,:3,ci],axis=1)),
                    'all_three_converged':int((dn[:,:3,ci,5]==1).all(axis=1).sum())}
            for si,initial in enumerate(STARTS):
                row['numerical_comparisons'][initial]={}
                for name,a,b in COMPARE:
                    dm=np.abs(mu[:,si,b]-mu[:,si,a]);df=np.abs(sse[:,si,b]-sse[:,si,a])
                    result={'mu':stats(dm),'sse':stats(df),'exact_mu_unchanged':int((dm==0).sum()),
                            'objective_improved':int((sse[:,si,b]<sse[:,si,a]).sum())}
                    row['numerical_comparisons'][initial][name]=result
                    v=result['mu'];writer.writerow([target,n,initial,name,v['count'],v['nonfinite'],v['min'],v['median'],v['max'],result['sse']['max']])
                row['window_adjacent'][initial]={}
                if ni:
                    for ci in [0,5,6]:
                        row['window_adjacent'][initial][STOPS[ci]]=stats(np.abs(d[:,ni,si,ci,1]-d[:,ni-1,si,ci,1]))
            rows.append(row)
        for lo,hi in RANGES:
            for si,initial in enumerate(STARTS):
                for ci in [0,5,6]:
                    selected_n=(N>=lo)&(N<=hi)
                    pooled['window_ranges'].append({'lo':lo,'hi':hi,'initial':initial,'stop':STOPS[ci],
                        'mu_range':stats(np.ptp(d[:,selected_n,si,ci,1],axis=1))})
        # Explicit extreme examples; every underlying row also remains available in raw files.
        for n in [11,21,59,61]:
            ni=(n-3)//2;span=np.ptp(d[:,ni,:3,6,1],axis=1);idx=int(np.argmax(span))
            summary['witnesses'].append({'type':'initialization_spread','target':target,'ref':refs[idx],
                'x':int(xs[idx]),'coarse_y':int(cy[idx]),'n':n,'mu_span':float(span[idx]),
                'rows':[{"initial":STARTS[si],"stop":STOPS[ci],**{name:float(d[idx,ni,si,ci,fi]) for fi,name in enumerate(M['fields'])}}
                        for si in range(4) for ci in range(7)]})
        summary['pooled'][target]=pooled
        print(target,'N / initial-width mu max (dual tight) / unfinished counts at 3 starts / N-2 to N max by 3 starts')
        for row in rows:
            if row['n'] in [7,9,11,13,15,17,19,21,31,41,51,59,61]:
                print(row['n'],row['initialization_spread']['dual_1e-10']['mu']['max'],
                      [r[6] for r in row['nonconverged'][:3]],
                      [row['window_adjacent'][s].get('dual_1e-10',{}).get('max') for s in STARTS[:3]])
        del d
prov=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
checks['unchanged_files']={p:sha(p)==h for p,h in prov['unchanged_files'].items()}
checks['unchanged_images']={r['reference_id']:sha(r['path'])==r['sha256'] for r in prov['inputs']['references']}
checks['experiment_sources_unchanged']={p:sha(HERE/p)==h for p,h in prov['experiment_sources'].items()}
assert checks['raw_returns']==M['stored_returns']==19367040
assert checks['baseline_800_exact_fits']==691680
assert all(checks['unchanged_files'].values()) and all(checks['unchanged_images'].values()) and all(checks['experiment_sources_unchanged'].values())
(HERE/'summary.json').write_text(json.dumps(safe(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(HERE/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved all-position comparisons and verification; no threshold-based selection or exclusions.')
