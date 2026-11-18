"""Wire comparison, precision labels and stage-specific provenance without running generators."""
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'reference_line_methods_108_20261030/tools'
def edit(name,old,new,count=1):
    p=BASE/name;s=p.read_text(encoding='utf-8-sig')
    assert s.count(old)==count,(name,old[:80],s.count(old))
    p.write_text(s.replace(old,new),encoding='utf-8')

p=BASE/'run_pipeline.py';s=p.read_text(encoding='utf-8-sig')
start=s.index('def numeric_fields(');end=s.index('def tree_hashes(',start)
s=s[:start]+s[end:]
s=s.replace('from pathlib import Path','from pathlib import Path\nfrom run_comparison import compare_runs, expected_measurement_ids\nfrom generation_provenance import GenerationRecord, files_under')
s=s.replace('def main():','def run_steps(args):',1)
s=s.replace('    ap = argparse.ArgumentParser()\n    ap.add_argument("--no-build", action="store_true")\n    args = ap.parse_args()\n','',1)
s=s.replace('    rep = compare_runs(results / "line_measurements.json", repeat / "line_measurements.json")',
'''    targets=json.loads((ROOT/'configs/targets.json').read_text(encoding='utf-8'))
    methods_config=json.loads((ROOT/'configs/methods.json').read_text(encoding='utf-8'))
    full_ids=expected_measurement_ids(manifest,targets,methods_config)
    rep = compare_runs(results / "line_measurements.json", repeat / "line_measurements.json",mode='full',expected_ids=full_ids)''')
s=s.replace('    print("repeat identical:", rep["identical"], "differences:", len(rep["differences"]))',
'''    print("repeat identical:", rep["identical"], "differences:", len(rep["differences"]))
    if not rep['passed']:raise RuntimeError('full repeat comparison failed: see repeatability.json')''')
s=s.replace('    sm = compare_runs(results / "line_measurements.json", single / "line_measurements.json")',
'''    subset_ids=expected_measurement_ids(manifest,targets,methods_config,methods={'E04','Q12'},refs={'R02'})
    sm = compare_runs(results / "line_measurements.json", single / "line_measurements.json",mode='subset',expected_ids=subset_ids)''')
s=s.replace('    print("single-method run matches full run:", sm["identical"], "ids:", sm["compared_ids"])',
'''    print("single-method subset matches full run:", sm["passed"], "ids:", sm["compared_ids"])
    if not sm['passed']:raise RuntimeError('declared subset comparison failed: see single_vs_full.json')''')
anchor='if __name__ == "__main__":\n    main()'
assert anchor in s
s=s.replace(anchor,'''def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--no-build',action='store_true')
    args=ap.parse_args()
    manifest=json.loads((ROOT/'configs/inputs_manifest.json').read_text(encoding='utf-8'))
    sources=[p for area in ('cpp','third_party','tools') for p in files_under(ROOT/area)
             if p.suffix.lower() in ('.cpp','.c','.hpp','.h','.py') or p.name=='CMakeLists.txt']
    inputs=list((ROOT/'configs').glob('*.json'))+[Path(r['path']) for r in manifest['references']]
    if args.no_build:sources.append(EXE)
    with GenerationRecord(ROOT,'measurement',sources=sources,inputs=inputs,
                          metadata={'argv':sys.argv,'no_build':args.no_build}) as provenance:
        run_steps(args)
        outputs=files_under(ROOT/'results')+[EXE]
        outputs += [p for p in LOGS.iterdir() if p.is_file() and p.suffix in ('.json','.txt')]
        outputs += files_under(LOGS/'single_method_example')
        outputs += [LOGS/'repeat_run/line_measurements.json']
        provenance.finish(outputs)


if __name__ == "__main__":
    main()''')
p.write_text(s,encoding='utf-8')

edit('fit_point_views.py',"    return dict(points=points, used=used, final_inlier=final, display_mask=display,\n                basis=basis, label=label, excluded_label=excluded)",
'''    precision=intermediates.get('fit_points_precision')
    precision_label=('계산 좌표 원정밀 저장(표시 반올림 별도)' if precision and precision.get('storage')=='binary64_json_roundtrip'
                     else '과거 좌표: 원정밀 저장 미확인·반올림 가능')
    return dict(points=points, used=used, final_inlier=final, display_mask=display,
                basis=basis, label=label, excluded_label=excluded,precision=precision,precision_label=precision_label)''')

p=BASE/'make_figures.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace('from fit_point_views import fit_point_view, rms_label','from fit_point_views import fit_point_view, rms_label\nfrom generation_provenance import GenerationRecord, files_under')
start=s.index('def main():');end=s.index('\n\nif __name__',start)
s=s[:start]+'''def main():
    manifest=json.loads((ROOT/'configs/inputs_manifest.json').read_text(encoding='utf-8'))
    sources=[Path(__file__),ROOT/'tools/fit_point_views.py',ROOT/'tools/generation_provenance.py']
    inputs=files_under(RES)+[ROOT/'configs/targets.json',ROOT/'configs/inputs_manifest.json']
    inputs += [Path(r['path']) for r in manifest['references']]
    with GenerationRecord(ROOT,'figures',sources=sources,inputs=inputs,
                          metadata={'argv':sys.argv,'opencv':cv2.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__}) as provenance:
        lm,recs,targets,manifest,images=load()
        FIG.mkdir(exist_ok=True)
        outputs=list(targets_figure(targets,images))
        crops=viewer_assets(images)
        (FIG/'viewer/crops.json').write_text(json.dumps(crops),encoding='utf-8')
        outputs.append(FIG/'viewer/crops.json')
        for rid in images:
            outputs.append(FIG/'viewer'/f'{rid}_preview.jpg')
            outputs += [FIG/'viewer'/f'{rid}_{name}.png' for name in crops]
        only=sys.argv[1:]
        for mid in sorted({r['method_id'] for r in recs.values()}):
            if only and mid not in only:continue
            for rid in ['R01','R02','R03','R04']:
                outputs.append(stage_figure(rid,mid,recs,targets,images[rid]))
            print('figures',mid,flush=True)
        for r in manifest['references']:
            assert sha(r['path'])==r['sha256']
        provenance.finish(outputs)
        print('done; inputs unchanged; provenance:',provenance.finish_path)
'''+s[end:]
p.write_text(s,encoding='utf-8')

p=BASE/'make_report.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace('from fit_point_views import fit_point_view, convergence_label, rms_label',
'''from fit_point_views import fit_point_view, convergence_label, rms_label
from run_comparison import comparison_label
from generation_provenance import GenerationRecord, files_under''')
anchor='RES = ROOT / "results"'
s=s.replace(anchor,anchor+'''

# If generation stops before finish(), the start record explicitly remains incomplete.
# Historical producer versions are never inferred from this script's current hash.
_report_inputs=files_under(RES)+files_under(ROOT/'figures')+list((ROOT/'configs').glob('*.json'))
_report_inputs += [ROOT/'logs'/name for name in ('environment.json','repeatability.json','single_vs_full.json','hashes.json')]
_legacy_finder=ROOT.parent/'reference_qr_finder_20261029/finder_results.json'
if _legacy_finder.exists():_report_inputs.append(_legacy_finder)
report_generation=GenerationRecord(ROOT,'report',sources=[Path(__file__),ROOT/'tools/fit_point_views.py',
    ROOT/'tools/run_comparison.py',ROOT/'tools/generation_provenance.py'],inputs=_report_inputs).start()
''')
s=s.replace('    return ([[round(a, 2), round(b, 2), int(c)]',
'''    # Viewer payload only. Original precision remains in results/<intermediate_json>.
    return ([[round(a, 2), round(b, 2), int(c)]''')
s=s.replace('DATA = {"crops": crops,',
'''DATA = {"point_display_precision":"viewer pts rounded to 2 decimals for display only; use intermediate_json fit_points for calculation coordinates", "crops": crops,''')
s=s.replace('"point_label": point_label, "rms_label": rms_label(r),',
'''"point_label": point_label, "raw_points_record": "results/"+r["intermediate_json"], "rms_label": rms_label(r),''')
s=s.replace('윤곽점은 정수 화소입니다.",',
'''윤곽점은 가장 바깥 밝은 화소의 중심입니다. 실제 경계와의 차이·기울기 영향은 별도 검증 대상이며 일정한 0.5px 차이를 가정하지 않습니다.",''')
s=s.replace('{"완전히 동일" if rep["identical"] else f\'차이 {len(rep["differences"])}건\'}','{e(comparison_label(rep))}')
s=s.replace('{"동일" if single["identical"] else "차이 있음"}','{e(comparison_label(single))}')
s=s.replace("{'동일' if rep['identical'] else '차이 있음'}",'{comparison_label(rep)}')
s=s.replace("{'동일' if single['identical'] else '차이'}",'{comparison_label(single)}')
anchor='<h2 id="s7">7. 재현·검증</h2>'
s=s.replace(anchor,anchor+'''
<p>이번 보고서 생성 기록: <code>{e(str(report_generation.finish_path.relative_to(ROOT)))}</code>.
측정·그림·보고서는 <code>logs/provenance/</code>의 별도 실행 기록으로 추적합니다.
과거 산출물에 일치하는 생성 기록이 없으면 생성 코드 버전은 미확인입니다.</p>
<p>뷰어에 넣은 점 좌표는 표시용 소수점 2자리 값입니다. 새 측정의 계산용 좌표는
<code>results/intermediates/…</code>의 <code>fit_points</code>에 원정밀로 저장되며,
<code>fit_points_precision</code>으로 확인합니다. 과거 반올림 기록에서 원래 자릿수를 복원하지 않습니다.</p>''')
s=s.replace(' → 11/11 PASS (<code>logs/03_selftest.stdout.txt</code>)',' → 해당 실행 로그 확인 (<code>logs/03_selftest.stdout.txt</code>)')
s=s.replace('f"- 자체 시험 11/11 PASS, 재실행 수치','f"- 자체 시험은 logs/03_selftest.stdout.txt 확인, 재실행 수치')
s += '''
report_generation.finish([ROOT/'report.html',ROOT/'report.md'])
print('report provenance:',report_generation.finish_path)
'''
p.write_text(s,encoding='utf-8')
print('tool edits applied')
