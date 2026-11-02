"""Inspect saved artifacts and source provenance only; no detection, refitting or statistical tests."""
import collections
import difflib
import hashlib
import html
import json
import math
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / 'analysis/reference_line_methods_108_20261030'
SEL = ROOT / 'analysis/reference_line_methods_selection_20261030'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    lm = read(SRC/'results/line_measurements.json')
    rows = lm['records']
    cfg = read(SRC/'configs/methods.json')
    targets = {t['id']:t for t in read(SRC/'configs/targets.json')['targets']}
    manifest = read(SRC/'configs/inputs_manifest.json')
    issues = []
    observations = []
    for r in rows:
        mid = r['measurement_id']
        ip = SRC/'results'/r['intermediate_json']
        itop = read(ip)
        if itop['measurement_id'] != mid:
            issues.append([mid, 'intermediate ID mismatch'])
        it = itop['intermediates']
        points = it.get('fit_points', [])
        flags = it.get('fit_points_inlier_flag', [])
        finite = all(isinstance(v, (int,float)) and math.isfinite(v) for p in points for v in p)
        o = {'measurement_id':mid,'method_id':r['method_id'], 'target_id':r['target_id'],
             'status':r['status'], 'failure_reason':r['failure_reason'],
             'saved_fit_points':len(points), 'saved_flags':len(flags), 'finite_saved_points':finite}
        if r['status']=='ok':
            for k in ('slope_dy_dx','intercept_y_px','image_angle_deg'):
                if not isinstance(r[k],(float,int)) or not math.isfinite(r[k]): issues.append([mid,'invalid '+k])
            if points:
                if len(points)!=len(flags): issues.append([mid,'flag length mismatch'])
                if sum(flags)!=r['n_selected']: issues.append([mid,'n_selected differs from final flags'])
                active = [p for p,f in zip(points,flags) if f]
                o['active_saved_points'] = len(active)
                o['below_minimum_after_final_flags'] = len(active)<cfg['common']['fit']['min_points'][r['target_kind']]
                t=targets[r['target_id']]
                o['active_outside_band_count'] = sum(not(t['band_y'][0]-.5<=y<=t['band_y'][1]+.5) for x,y in active)
                o['active_outside_x_count'] = sum(not(t['x_range'][0]-.5<=x<=t['x_range'][1]+.5) for x,y in active)
                o['active_in_exclusion_count'] = sum(any(a-.5<=x<=b+.5 for a,b in t['exclude_x']) for x,y in active)
                if r['target_kind']=='external':
                    a,b=t['exclude_x'][0]
                    o['left_inlier_count']=sum(x<a-.5 for x,y in active)
                    o['right_inlier_count']=sum(x>b+.5 for x,y in active)
                    o['both_sides_have_support'] = bool(o['left_inlier_count'] and o['right_inlier_count'])
                # Structural duplicate accounting, not statistical independence inference.
                o['duplicate_saved_coordinates'] = len(points)-len(set(map(tuple,points)))
            if not finite: issues.append([mid,'nonfinite stored fit point'])
        o['segments_selected']=r.get('diagnostics',{}).get('segments_selected')
        o['segments_passing_filters']=r.get('diagnostics',{}).get('segments_passing_filters')
        o['support_list_aligned']=r.get('diagnostics',{}).get('support_list_aligned')
        o['nm_nonconverged_columns']=r.get('diagnostics',{}).get('nelder_mead_not_converged')
        if r['method_id']=='Q10':
            o['contour_offset_note']=r['diagnostics'].get('contour_offset_note')
        observations.append(o)

    selected = read(SEL/'results/line_measurements.json')['records']
    original = {r['measurement_id']:r for r in rows}
    source_hashes = read(SRC/'logs/hashes.json')
    logged = []
    for group,files in source_hashes.items():
        for rel,digest in files.items():
            path=SRC/rel
            logged.append({'group':group,'path':rel,'exists':path.exists(),
                           'matches_execution_log':path.exists() and sha(path)==digest})
    combos=read(SRC/'results/combinations.json')['combinations']
    combo_errors=[]
    for c in combos:
        refs=c['measurement_refs'].values()
        if not all(x in original for x in refs): combo_errors.append(c['combination_id']); continue
        n=sum(original[x]['status']=='ok' for x in refs)
        if n!=c['lines_ok'] or c['status']!=('complete' if n==5 else 'failed' if n==0 else 'partial'):
            combo_errors.append(c['combination_id'])
    checks={'record_count':len(rows),'unique_ids':len(original),
        'method_count':len(set(r['method_id'] for r in rows)),
        'status_counts':dict(collections.Counter(r['status'] for r in rows)),
        'selected_rows_equal_original':all(original[r['measurement_id']]==r for r in selected),
        'selected_count':len(selected),'combination_count':len(combos),'combination_errors':combo_errors,
        'stored_record_consistency_issues':issues,
        'runtime_source_hashes':logged,
        'input_hashes':[{'reference_id':r['reference_id'],'path':r['path'],'matches_manifest':sha(Path(r['path']))==r['sha256']} for r in manifest['references']],
        'scope':'Saved-artifact consistency and source comparison only. No image measurement, parameter sweep, model fitting or statistical test.'}
    write('artifact_checks.json',checks)
    write('stored_observations.json',observations)

    # Compare bundled detector source to the archived pristine source AND archive members.
    tp=SRC/'third_party'
    audit=[]
    diff_dir=HERE/'source_diffs';diff_dir.mkdir(exist_ok=True)
    groups={'lsd_1.6':['lsd.c','lsd.h'], 'devernay_1.0':['devernay.c','devernay.h'],
            'ED_Lib':['ED.cpp','ED.h','EDLines.cpp','EDLines.h','EDColor.cpp','EDColor.h','NFA.cpp','NFA.h'],
            'ELSED':['ELSED.cpp','ELSED.h','EdgeDrawer.cpp','EdgeDrawer.h','FullSegmentInfo.cpp','FullSegmentInfo.h','Utils.h']}
    zipmap={'lsd_1.6':'www.ipol.im_pub_art_2012_gjmr-lsd_lsd_1.6.zip',
            'devernay_1.0':'www.ipol.im_pub_art_2017_216_devernay_1.0.zip',
            'ED_Lib':'codeload.github.com_CihanTopal_ED_Lib_zip_refs_heads_master',
            'ELSED':'codeload.github.com_iago-suarez_ELSED_zip_refs_heads_main'}
    for group,files in groups.items():
        with zipfile.ZipFile(tp/'_downloads'/zipmap[group]) as z:
            for name in files:
                used=tp/group/name;pristine=tp/'_pristine'/group/name
                a=pristine.read_bytes();b=used.read_bytes()
                candidates=[x for x in z.namelist() if x.endswith('/'+name)]
                archive_equal=any(z.read(x)==a for x in candidates)
                row={'group':group,'file':name,'used_sha256':sha(used),'pristine_sha256':sha(pristine),
                     'unchanged_from_pristine':a==b,'pristine_equals_cached_official_archive':archive_equal}
                if a!=b:
                    diff=''.join(difflib.unified_diff(a.decode('utf-8-sig').splitlines(True),b.decode('utf-8-sig').splitlines(True),fromfile='pristine/'+group+'/'+name,tofile='used/'+group+'/'+name))
                    dest=diff_dir/(group+'_'+name+'.diff');dest.write_text(diff,encoding='utf-8');row['actual_diff']=dest.relative_to(HERE).as_posix()
                audit.append(row)
    fld=tp/'fld_port'
    for usedname,origname,dlname in [('fld_port.cpp','fast_line_detector.orig.cpp','raw.githubusercontent.com_opencv_opencv_contrib_3.4_modules_ximgproc_src_fast_line_detector.cpp')]:
        a=(fld/origname).read_bytes();b=(fld/usedname).read_bytes()
        dest=diff_dir/'fld_port_actual.diff'
        dest.write_text(''.join(difflib.unified_diff(a.decode('utf-8-sig').splitlines(True),b.decode('utf-8-sig').splitlines(True),fromfile=origname,tofile=usedname)),encoding='utf-8')
        audit.append({'group':'fld_port','file':usedname,'used_sha256':sha(fld/usedname),
                      'unchanged_from_pristine':a==b,'pristine_equals_cached_official_download':a==(tp/'_downloads'/dlname).read_bytes(),
                      'actual_diff':dest.relative_to(HERE).as_posix()})
    write('third_party_comparison.json',audit)
    # Numbered text snapshots are review aids, not changed measurement code.
    snaps=HERE/'source_views';snaps.mkdir(exist_ok=True)
    paths=list((SRC/'cpp').rglob('*.cpp'))+list((SRC/'cpp').rglob('*.hpp'))+list((SRC/'configs').glob('*.json'))+list((SRC/'tools').glob('*.py'))
    paths += [SEL/'select_existing_results.py', SEL/'write_screening_report.py',
              ROOT/'claude_코드검증용/20261030_108조합_기준4장_직선추출_Claude작업프롬프트.txt']
    source_views=[]
    for path in paths:
        rel=path.relative_to(ROOT).as_posix()
        fn=rel.replace('/','__')+'.html'
        lines=path.read_text(encoding='utf-8-sig').splitlines()
        body=''.join(f'<div id="L{i}"><a href="#L{i}">{i:4}</a> {html.escape(s)}</div>' for i,s in enumerate(lines,1))
        doc='<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+html.escape(path.name)+'</title><style>body{font-family:Consolas,monospace;font-size:13px}pre{white-space:pre-wrap}a{color:#777;text-decoration:none}div:target{background:#fff0a0}</style><h2>'+html.escape(rel)+'</h2><pre>'+body+'</pre></html>'
        (snaps/fn).write_text(doc,encoding='utf-8')
        source_views.append({'source':rel,'sha256':sha(path),'view':(Path('source_views')/fn).as_posix(),'line_count':len(lines)})
    write('source_views.json',source_views)
    print(json.dumps({'methods':checks['method_count'],'rows':len(rows),'statuses':checks['status_counts'],
        'record_issues':len(issues),'old_hash_mismatches':[x['path'] for x in logged if not x['matches_execution_log']],
        'single_side_external':[x['measurement_id'] for x in observations if x.get('both_sides_have_support') is False],
        'outside_band':[{'id':x['measurement_id'],'n':x['active_outside_band_count']} for x in observations if x.get('active_outside_band_count',0)>0],
        'final_minimum_violations':[x['measurement_id'] for x in observations if x.get('below_minimum_after_final_flags')],
        'changed_vendor_files':[x['group']+'/'+x['file'] for x in audit if not x['unchanged_from_pristine']]},ensure_ascii=False))

if __name__=='__main__':main()
