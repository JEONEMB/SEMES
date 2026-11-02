from pathlib import Path
import csv,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
NS=np.arange(3,62,2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def stats(v):
    if not np.isfinite(v).all():return {'count':len(v),'nonfinite':int((~np.isfinite(v)).sum()),'min':None,'median':None,'max':None}
    return {'count':len(v),'nonfinite':0,'min':float(v.min()),'median':float(np.median(v)),'max':float(v.max())}
def main():
    prov=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
    runtime=json.loads((HERE/'runtime.json').read_text())
    targets=json.loads((HERE.parent/'reference_line_methods_108_20261030/configs/targets.json').read_text(encoding='utf-8'))['targets']
    tmap={t['id']:t for t in targets}
    profiles=[json.loads(line) for line in (HERE/'profiles.jsonl').read_text().splitlines()]
    old_profiles=json.loads((HERE.parent/'gaussian_window_sensitivity_20261101/profiles.json').read_text())
    omap={(p['ref'],p['target'],p['x']):p for p in old_profiles}
    all_g_exact=True;all_coarse_exact=True;compared=0
    for p in profiles:
        op=omap.get((p['ref'],p['target'],p['x']))
        if op is not None:
            compared+=1
            all_g_exact &= p['sobel_samples'][20:41]==op['sobel_samples_m10_p10']
            all_coarse_exact &= p['coarse_y']==op['coarse_y']
    d=np.loadtxt(HERE/'fits.csv',delimiter=',',skiprows=1,usecols=tuple(range(2,16)))
    assert d.shape==(23056*30,14),d.shape
    archive={};summary={'sample_counts':NS.tolist(),'no_rejections':True,'total_columns':len(profiles),'total_fits':len(d),'groups':[],'pooled':{}}
    pos=0
    all_arrays={}
    with (HERE/'window_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['ref','target','n','columns','nonconverged','nonfinite','mu_shift_min','mu_shift_median','mu_shift_max','adjacent_abs_delta_min','adjacent_abs_delta_median','adjacent_abs_delta_max'])
        for group in runtime['groups']:
            rid,target,count=group['ref'],group['target'],group['profiles']
            t=tmap[target]
            xs=np.array([x for x in range(t['x_range'][0],t['x_range'][1]+1) if not any(a<=x<=b for a,b in t['exclude_x'])])
            assert count==len(xs)
            g=d[pos:pos+count*len(NS)].reshape(count,len(NS),14);pos+=count*len(NS)
            assert np.array_equal(g[:,:,0],np.broadcast_to(xs[:,None],(count,len(NS))))
            assert np.array_equal(g[:,:,3],np.broadcast_to(NS[None,:],(count,len(NS))))
            assert np.array_equal(g[:,:,6],g[:,:,3])
            key=rid+'_'+target
            mu=g[:,:,8];cy=g[:,0,1];shift=mu-cy[:,None]
            archive[key+'_data']=g;archive[key+'_xs']=xs
            all_arrays[key]=g
            result={'ref':rid,'target':target,'columns':count,'by_n':[],'ranges':{}}
            for ni,n in enumerate(NS):
                delta=np.abs(mu[:,ni]-mu[:,ni-1]) if ni else None
                row={'n':int(n),'nonconverged':int((g[:,ni,12]==0).sum()),'nonfinite':int((g[:,ni,13]==0).sum()),'mu_minus_coarse':stats(shift[:,ni]),'adjacent_abs_delta':stats(delta) if ni else None}
                result['by_n'].append(row)
                a=row['mu_minus_coarse'];b=row['adjacent_abs_delta'] or {'min':None,'median':None,'max':None}
                w.writerow([rid,target,n,count,row['nonconverged'],row['nonfinite'],a['min'],a['median'],a['max'],b['min'],b['median'],b['max']])
            for lo,hi in [(3,21),(9,21),(11,15),(11,21),(13,21),(21,31),(31,41),(41,51),(51,61),(59,61)]:
                part=mu[:,(NS>=lo)&(NS<=hi)]
                result['ranges'][f'{lo}-{hi}']=stats(np.ptp(part,axis=1))
            summary['groups'].append(result)
    assert pos==len(d)
    for target in ['upper_structure','lower_structure']:
        g=np.concatenate([all_arrays[r+'_'+target] for r in ['R01','R02','R03','R04']])
        mu=g[:,:,8];rows=[]
        for ni,n in enumerate(NS):
            delta=np.abs(mu[:,ni]-mu[:,ni-1]) if ni else None
            rows.append({'n':int(n),'columns':len(g),'nonconverged':int((g[:,ni,12]==0).sum()),'nonfinite':int((g[:,ni,13]==0).sum()),'delta':stats(delta) if ni else None})
        summary['pooled'][target]=rows
    summary['selection_assessment']={
      'upper_structure':{'candidate_n':21,'status':'provisional_window_candidate_not_numerical_validation','basis':'All 10848 positions: N=13..21 span max 0.0007534482917890273 px. N=21 is within this observed plateau and is the count used in the paper downstream. This is an explicit candidate choice, not a paper-prescribed smallest-stable or universal accuracy rule. 52/10848 returns at N=21 did not satisfy the inherited solver stopping criterion; they remain in all summaries.'},
      'lower_structure':{'candidate_n':None,'status':'not_determined','basis':'No all-position plateau supports the previous N=59 choice across the tested range. N=59..61 includes multi-pixel changes and a very large out-of-image numerical return. No row was removed to choose a sample count.'}}
    np.savez_compressed(HERE/'arrays.npz',ns=NS,**archive)
    (HERE/'summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    source_ok={p:sha(p)==h for p,h in prov['unchanged_source_hashes'].items()}
    image_ok={r['reference_id']:sha(r['path'])==r['sha256'] for r in prov['design']['inputs']['references']}
    checks={'all_visible_columns_every_n':True,'rows_retained':len(d),'expected_rows':23056*30,'prior_profile_comparisons':compared,'prior_sobel_samples_exact':all_g_exact,'prior_coarse_positions_exact':all_coarse_exact,'unchanged_sources':source_ok,'unchanged_images':image_ok,'solver_source_exact':sha(HERE/'original_nelder_mead.hpp')==sha(HERE.parent/'gaussian_lower_window_extension_20261101/original_nelder_mead.hpp')}
    assert all_g_exact and all_coarse_exact and all(source_ok.values()) and all(image_ok.values()) and checks['solver_source_exact']
    (HERE/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    for t,rows in summary['pooled'].items():
        print(t,'N / unfinished / adjacent median / adjacent max')
        for row in rows:
            print(row['n'],row['nonconverged'],None if row['delta'] is None else (row['delta']['median'],row['delta']['max']))
    print('Verification passed; all rows retained, old Sobel/coarse results match exactly.')
if __name__=='__main__':main()
