from pathlib import Path
import sys,json,io,time,urllib.request
from zipfile import ZipFile,ZIP_DEFLATED
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import batch_jobs as B
BASE='http://127.0.0.1:8765'
OUT=Path(__file__).resolve().parent

def api(path,data=None,raw=None):
    request=urllib.request.Request(BASE+path,data=raw if raw is not None else json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/octet-stream' if raw is not None else 'application/json'})
    return json.load(urllib.request.urlopen(request,timeout=120))

if __name__=='__main__':
    assert B.truth_label('Y+0.05.bmp')['scope']=='single'
    assert B.truth_label('Y+0.05 T+0.01 (2).bmp')==dict(Y=.05,T=.01,axis='MIXED',scope='mixed')
    assert B.truth_label('center3.bmp')['scope']=='center'
    assert B.truth_label('Y+0.05_wrong.bmp') is None
    e=B.evaluate('Y+0.05.bmp',dict(axis='T',value=.01,Y=0,T=.01));assert e['eY']==.05 and e['eT']==.01 and e['passed'] is False
    e=B.evaluate('Y+0.05 T+0.01 (1).bmp',dict(axis='Y',value=.05,Y=.05,T=0));assert e['scope']=='mixed' and e['passed'] is None and e['eT']==.01
    sources=json.loads((OUT/'sources.json').read_text())
    assert len(sources)==341
    label_counts={}
    for s in sources:
        scope=(B.truth_label(s['name']) or {}).get('scope','unknown');label_counts[scope]=label_counts.get(scope,0)+1
    print('archive label counts',label_counts,flush=True)
    raw=B.read_source(next(s for s in sources if s['name']=='Y+0.05.bmp'))
    zip_data=io.BytesIO()
    with ZipFile(zip_data,'w',ZIP_DEFLATED) as z:z.writestr('nested/Y+0.05.bmp',raw);z.writestr('nested/Y+0.10.bmp',b'invalid')
    mixed=B.read_source(next(s for s in sources if 'Y+0.05 T+0.01 (1)' in s['name']))
    job=api('/api/batch/create',{});identity=job['id']
    from urllib.parse import urlencode
    for name,data in [('sample.zip',zip_data.getvalue()),('Y+0.05 T+0.01 (1).bmp',mixed)]:
        api('/api/batch/upload?'+urlencode(dict(id=identity,name=name,relative='test/'+name)),raw=data)
    api('/api/batch/start',dict(id=identity,workers=1))
    time.sleep(2);api('/api/batch/cancel',dict(id=identity))
    for _ in range(180):
        snap=api('/api/batch?id='+identity)
        if snap['state'] not in ('running','preparing','cancelling'):break
        time.sleep(1)
    assert snap['state']=='cancelled',snap
    assert snap['summary']['processed']<3
    print('cancelled',snap['summary'],flush=True)
    api('/api/batch/start',dict(id=identity,workers=2))
    for _ in range(180):
        snap=api('/api/batch?id='+identity)
        if snap['state'] not in ('running','preparing','cancelling'):break
        time.sleep(1)
    assert snap['state']=='complete',snap
    s=snap['summary'];assert s['total']==3 and s['processed']==3 and s['single']==2 and s['mixed']==1 and s['errors']==1 and s['single_inference_failures']==1
    for row in snap['rows']:
        if row.get('prediction'):
            detail=api('/api/batch/result?'+urlencode(dict(id=identity,index=row['index'])))
            assert detail['total']==630 and len(detail['trace']['ids'])==630
    assert any(r['evaluation']['passed'] is False for r in snap['rows'] if r.get('evaluation') and r['evaluation']['scope']=='single')
    assert Path(B.RUNS/identity/'job.json').exists()
    for resource in ['/batch','/batch.js','/batch.css','/?batch='+identity+'&item=0']:
        assert urllib.request.urlopen(BASE+resource).status==200
    checks=dict(passed=True,job=identity,archive_images=341,label_counts=label_counts,summary=s,checks=['actual two nested split ZIP inventories','upload ZIP plus mixed image','cancel and resume','labeled inference failure kept in denominator','mixed labels excluded from single score','630-point saved detail','HTTP routes and persistent results'])
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
