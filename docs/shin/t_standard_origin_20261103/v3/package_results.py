from pathlib import Path
import json,zipfile,hashlib
from pypdf import PdfReader
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
pdf=ROOT/'output/pdf/T_axis_standard_origin_v3_model_evaluation_20261103.pdf'
assert len(PdfReader(str(pdf)).pages)==10
qa=json.loads((P/'pdf_validation.json').read_text());qa['render_inspection']='PASS: all 10 pages rendered and visually inspected; final modified pages 1 and 2 re-rendered and inspected.'
qa['standalone_packaged_scene_CLI']='PASS: direct execution with one original image; local packaged reference and DLLs used.'
(P/'pdf_validation.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
extensions={'.py','.cpp','.hpp','.json','.csv','.md','.ps1','.png','.dll','.exe','.txt'}
files=[f for f in P.rglob('*') if f.is_file() and f.name!='SHA256_manifest.json' and f.suffix.lower() in extensions and not any(v in f.relative_to(P).parts for v in ['build','__pycache__'])]
manifest={f.relative_to(P).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
manifest['report.pdf']=hashlib.sha256(pdf.read_bytes()).hexdigest()
(P/'SHA256_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
bundle=P/'t_standard_origin_v3_model_and_evaluation.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in files:z.write(f,f.relative_to(P).as_posix())
    z.write(P/'SHA256_manifest.json','SHA256_manifest.json');z.write(pdf,'report.pdf')
    for name in ['HANDOFF_model_v3.md','standard_origin_v3.json','coordinates_standard_v3.csv','result_summary_v3.json','export_checks.json']:
        z.write(ROOT/'analysis/frame_link_audit_20261103'/name,'reference_origin/'+name)
    for name in ['predictions_method7.csv','metrics_method7.csv','configs_method7.csv']:
        z.write(P.parent/'v2/results'/name,'prior_v2/'+name)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    names=set(z.namelist())
    assert {'report.pdf','deployment/qr_model.json','deployment/scene_model.json','deployment/runtime/v3_estimate.exe','deployment/runtime/v3_feature_extract.dll'}<=names
    for rel,h in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==h,rel
print('PASS: PDF, standalone assets, bundle integrity and',len(manifest),'SHA256 entries; ZIP bytes',bundle.stat().st_size)
