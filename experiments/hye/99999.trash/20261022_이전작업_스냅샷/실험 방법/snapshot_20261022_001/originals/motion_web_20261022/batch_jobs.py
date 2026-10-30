"""Persistent batch orchestration. Labels are evaluated only after image inference."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from zipfile import ZipFile
import json,uuid,threading,re,time,os
from datetime import datetime,timezone,timedelta
from archives import archive_sources,IMAGE_EXT
HERE=Path(__file__).resolve().parent
RUNS=HERE/'runs';RUNS.mkdir(exist_ok=True)
KNOWN=HERE.parent/'hw-analysis/data/hw/raw/zip'
JOBS={};LOCK=threading.RLock()

def truth_label(name):
    stem=Path(name).stem
    if re.fullmatch(r'center\d*',stem,re.I):return dict(Y=0.,T=0.,axis='CENTER',scope='center')
    stem=re.sub(r'\s*\(\d+\)$','',stem)
    parts=list(re.finditer(r'([YT])\s*([+−-])\s*(\d+(?:\.\d*)?|\.\d+)',stem,re.I))
    if not parts or re.sub(r'\s+','',re.sub(r'([YT])\s*([+−-])\s*(\d+(?:\.\d*)?|\.\d+)','',stem,flags=re.I)):
        return None
    values={'Y':0.,'T':0.};seen=set()
    for p in parts:
        axis=p[1].upper()
        if axis in seen:return None
        seen.add(axis);values[axis]=float(p[3])*(1 if p[2]=='+' else -1)
    active=[a for a in values if values[a]!=0]
    return dict(**values,axis=active[0] if len(active)==1 else 'MIXED' if active else 'CENTER',scope='single' if len(active)==1 else 'mixed' if active else 'center')

def evaluate(name,pred):
    label=truth_label(name)
    if not label:return None
    eY=abs(pred['Y']-label['Y']);eT=abs(pred['T']-label['T'])
    supported=label['scope']!='mixed'
    axis_match=label['axis']==pred['axis'] if supported else None
    val=label.get(label['axis'],0.)
    sign_match=(axis_match and (val>0)-(val<0)==(pred['value']>0)-(pred['value']<0)) if supported else None
    return dict(truth=label,eY=eY,eT=eT,axis_match=axis_match,sign_match=sign_match,within_tolerance=eY<=.09+1e-12 and eT<=.09+1e-12,passed=bool(sign_match and eY<=.09+1e-12 and eT<=.09+1e-12) if supported else None,scope=label['scope'])

def read_source(source):
    if source['kind']=='file':return Path(source['path']).read_bytes()
    with ZipFile(source['archive']) as z:
        info=z.infolist()[source['member_index']]
        if info.file_size>32*1024**2:raise ValueError('32MB 초과 이미지: 처리 실패로 기록합니다.')
        return z.read(info)

def worker(source):
    # Neither name nor folder label is passed into inference.
    from server import infer
    return infer(read_source(source))

def persist(job):
    with LOCK:
        dest=RUNS/job['id'];dest.mkdir(exist_ok=True)
        data={k:v for k,v in job.items() if k not in ('stop',)}
        temp=dest/'job.tmp';temp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8');temp.replace(dest/'job.json')

def summary(rows,total):
    singles=[r for r in rows if (r.get('evaluation') or {}).get('scope')=='single' or (r.get('label') or {}).get('scope')=='single']
    finished=[r for r in singles if r.get('prediction') and r.get('evaluation')]
    return dict(total=total,processed=len(rows),single=len(singles),single_success=len(finished),single_inference_failures=len(singles)-len(finished),passed=sum(bool(r['evaluation']['passed']) for r in finished),axis_correct=sum(bool(r['evaluation']['axis_match']) for r in finished),sign_correct=sum(bool(r['evaluation']['sign_match']) for r in finished),Y_max=max([r['evaluation']['eY'] for r in finished],default=None),T_max=max([r['evaluation']['eT'] for r in finished],default=None),mixed=sum((r.get('label') or {}).get('scope')=='mixed' for r in rows),centers=sum((r.get('label') or {}).get('scope')=='center' for r in rows),unknown=sum(r.get('label') is None for r in rows),errors=sum('error' in r for r in rows))

def snapshot(job):
    with LOCK:
        return dict(id=job['id'],state=job['state'],message=job.get('message',''),created=job['created'],rows=list(job['rows']),summary=summary(job['rows'],len(job['sources'])),workers=job.get('workers',2))

def new_job():
    identity=uuid.uuid4().hex
    job=dict(id=identity,state='uploading',created=datetime.now(timezone(timedelta(hours=9))).isoformat(),sources=[],rows=[],uploads=[],workers=2,stop=False,message='파일 준비 중')
    with LOCK:JOBS[identity]=job;persist(job)
    return job

def get_job(identity):
    if not re.fullmatch('[a-f0-9]{32}',identity):raise ValueError('잘못된 실행 ID')
    with LOCK:
        if identity not in JOBS:
            saved=RUNS/identity/'job.json'
            if not saved.exists():raise ValueError('실행을 찾을 수 없습니다.')
            job=json.loads(saved.read_text(encoding='utf-8'));job['stop']=False
            if job['state'] in ('running','preparing','cancelling'):job['state']='interrupted';job['message']='서버 재시작으로 중단됨. 완료 결과는 보존되었습니다.'
            JOBS[identity]=job
        return JOBS[identity]

def add_upload(job,stream,size,name,relative):
    if job['state']!='uploading':raise ValueError('이 실행은 파일을 더 받을 수 없습니다.')
    suffix=Path(name).suffix.lower()
    if suffix not in IMAGE_EXT|{'.zip'}:raise ValueError('이미지 또는 ZIP 파일만 지원합니다.')
    if size<=0 or size>2*1024**3:raise ValueError('파일 크기는 2GB 이하이어야 합니다.')
    if suffix in IMAGE_EXT and size>32*1024**2:raise ValueError('이미지는 32MB 이하이어야 합니다.')
    folder=RUNS/job['id']/'uploads';folder.mkdir(exist_ok=True);path=folder/(uuid.uuid4().hex+suffix)
    left=size
    with path.open('wb') as f:
        while left:
            block=stream.read(min(left,8*1024**2))
            if not block:raise ValueError('파일 전송이 중단되었습니다.')
            f.write(block);left-=len(block)
    with LOCK:job['uploads'].append(dict(path=str(path.resolve()),name=name,relative=relative));persist(job)

def run_job(job,known=False):
    try:
        with LOCK:job['state']='preparing';job['message']='ZIP/이미지 목록 준비 중';persist(job)
        work=RUNS/job['id']/'archives';work.mkdir(exist_ok=True)
        sources=list(job['sources'])
        if not sources:
            uploads=([dict(path=str(p.resolve()),name=p.name,relative=p.name) for p in sorted(KNOWN.glob('*.zip'))] if known else job['uploads'])
            for item in uploads:
                if Path(item['path']).suffix.lower()=='.zip':sources+=archive_sources(Path(item['path']),work,item['relative']+'/')
                else:sources.append(dict(kind='file',**item))
            with LOCK:job['sources']=sources;persist(job)
        if not sources:raise ValueError('지원하는 이미지를 찾지 못했습니다.')
        done={r['index'] for r in job['rows']}
        pending=iter((i,s) for i,s in enumerate(sources) if i not in done)
        with LOCK:job['state']='running';job['message']='630점 이미지 분석 중';persist(job)
        with ProcessPoolExecutor(max_workers=job['workers']) as pool:
            futures={}
            def submit_next():
                if job['stop']:return
                item=next(pending,None)
                if item:futures[pool.submit(worker,item[1])]=item
            for _ in range(job['workers']):submit_next()
            while futures:
                completed,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
                for future in completed:
                    i,source=futures.pop(future)
                    row=dict(index=i,name=source['name'],relative=source['relative'],label=truth_label(source['name']))
                    try:
                        prediction=future.result()
                        result_path=RUNS/job['id']/f'result_{i}.json'
                        result_path.write_text(json.dumps(prediction,ensure_ascii=False,allow_nan=False),encoding='utf-8')
                        row['prediction']={k:prediction[k] for k in ['axis','value','Y','T','valid','total','seconds','model']}
                        row['evaluation']=evaluate(source['name'],prediction)
                    except Exception as exc:row['error']=str(exc)
                    with LOCK:job['rows'].append(row);persist(job)
                    submit_next()
        with LOCK:job['state']='cancelled' if job['stop'] else 'complete';job['message']='중단됨 · 완료 결과 보존' if job['stop'] else '전체 처리 완료';persist(job)
    except Exception as exc:
        with LOCK:job['state']='error';job['message']=str(exc);persist(job)

def start(job,known=False,workers=2):
    if job['state'] not in ('uploading','interrupted','cancelled','error'):raise ValueError('이미 실행 중이거나 완료된 작업입니다.')
    job['workers']=min(max(int(workers),1),4);job['stop']=False;job['state']='preparing'
    threading.Thread(target=run_job,args=(job,known),daemon=True).start()
