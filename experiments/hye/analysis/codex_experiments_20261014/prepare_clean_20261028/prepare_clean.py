"""Lossless raw->clean preparation. Never filters samples or changes pixels/labels."""
from pathlib import Path,PurePosixPath
from zipfile import ZipFile
from collections import Counter
import csv,hashlib,json,re,shutil,sys,tempfile,os
import cv2,numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO=ROOT/'S15P31A507'
RAW=REPO/'data/raw'
CLEAN=REPO/'data/clean'
sys.path.insert(0,str(ROOT/'motion_web_20261022'))
from archives import normalize_split

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_csv(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def safe_member(name):
    name=name.replace('\\','/');p=PurePosixPath(name)
    if p.is_absolute() or any(s in ('..','') or ':' in s for s in p.parts):raise ValueError(f'Unsafe ZIP path: {name!r}')
    return p

def join_raw(raw,work):
    with ZipFile(raw) as z:
        parts=[i for i in z.infolist() if not i.is_dir()]
        assert all(re.search(r'\.(z\d+|zip)$',i.filename,re.I) for i in parts)
        last=[i for i in parts if i.filename.lower().endswith('.zip')];assert len(last)==1
        segments=sorted([i for i in parts if i not in last],key=lambda i:int(re.search(r'\.z(\d+)$',i.filename,re.I)[1]))
        assert [int(re.search(r'\.z(\d+)$',i.filename,re.I)[1]) for i in segments]==list(range(1,len(segments)+1))
        paths=[];inventory=[]
        for n,info in enumerate(segments+last):
            p=work/f'part-{n:02d}'
            with z.open(info) as source,p.open('xb') as target:shutil.copyfileobj(source,target,8*1024*1024)
            assert p.stat().st_size==info.file_size
            paths.append(p);inventory.append(dict(member=info.filename,bytes=info.file_size,crc32=f'{info.CRC:08x}',sha256=digest(p)))
    joined=work/'joined.zip'
    if len(paths)>1:normalize_split(paths,joined)
    else:shutil.copyfile(paths[0],joined)
    return joined,inventory

def parse_command(stem):
    if stem.lower().startswith('center'):return '0','0','center',re.sub(r'\D','',stem) or '1'
    y=re.search(r'Y\s*([+-]\d+(?:\.\d+)?)',stem,re.I)
    t=re.search(r'T\s*([+-]\d+(?:\.\d+)?)',stem,re.I)
    if not y and not t:raise ValueError(f'Unrecognized command name: {stem}')
    repeat=re.search(r'\((\d+)\)',stem)
    return y[1] if y else '0',t[1] if t else '0','mixed' if y and t else 'Y' if y else 'T',repeat[1] if repeat else '1'

def sidecar(image,stage):
    choices=[image.with_suffix('.txt'),Path(str(image.with_suffix('.txt'))+'.txt')]
    files=[p for p in choices if p.exists()]
    if not files:return dict(sidecar_path='',sidecar_Y_Actual_raw='',sidecar_T_Actual_raw='',sidecar_encoding='')
    assert len(files)==1
    p=files[0];b=p.read_bytes()
    try:t=b.decode('utf-8-sig');encoding='UTF-8/ASCII'
    except UnicodeDecodeError:t=b.decode('cp949');encoding='CP949'
    tokens=t.split();assert tokens[:2]==['Y_Actual','T_Actual'] and len(tokens)==4,(p,t)
    float(tokens[2]);float(tokens[3])
    return dict(sidecar_path=p.relative_to(stage).as_posix(),sidecar_Y_Actual_raw=tokens[2],sidecar_T_Actual_raw=tokens[3],sidecar_encoding=encoding)

def process(date,legacy):
    raw=RAW/f'zip_{date}.zip';target=CLEAN/f'clean_{date}'
    assert raw.is_file()
    assert target.resolve().is_relative_to(CLEAN.resolve()) and target.name==f'clean_{date}'
    if target.exists() and any(target.iterdir()):raise ValueError(f'Refusing to replace existing clean data: {target}')
    stage=HERE/f'.staging_{date}'
    if stage.exists():raise ValueError(f'Unfinished staging exists: {stage}')
    stage.mkdir();before=digest(raw);work=Path(tempfile.mkdtemp(prefix=f'archive_{date}_',dir=HERE))
    try:
        print(date,'joining nested split archive',flush=True)
        joined,parts=join_raw(raw,work)
        joined_hash=digest(joined);inventory=[]
        with ZipFile(joined) as z:
            members=[i for i in z.infolist() if not i.is_dir()]
            paths=[safe_member(i.filename) for i in members]
            prefixes={p.parts[0] for p in paths};strip_one=len(prefixes)==1 and all(len(p.parts)>1 for p in paths)
            seen=set()
            for info,p in zip(members,paths):
                relative=PurePosixPath(*p.parts[1:]) if strip_one else p
                key=str(relative).casefold()
                assert key not in seen,f'Colliding destination: {relative}'
                seen.add(key);dest=stage.joinpath(*relative.parts)
                assert dest.resolve().is_relative_to(stage.resolve())
                dest.parent.mkdir(parents=True,exist_ok=True)
                b=z.read(info) # ZIP CRC is checked during read.
                with dest.open('xb') as f:f.write(b)
                inventory.append(dict(relative_path=relative.as_posix(),source_member=info.filename,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),zip_crc32=f'{info.CRC:08x}'))
                if len(inventory)%40==0:print(date,'extracted files',len(inventory),'/',len(members),flush=True)
        rows=[]
        indexed={r['relative_path']:r for r in inventory}
        images=sorted(stage.rglob('*.bmp'))
        for n,image in enumerate(images):
            b=image.read_bytes();r=indexed[image.relative_to(stage).as_posix()];assert hashlib.sha256(b).hexdigest()==r['sha256']
            pixels=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED)
            assert pixels is not None and pixels.shape[:2]==(2160,3840),(image,pixels.shape if pixels is not None else None)
            y,t,kind,repeat=parse_command(image.stem)
            match=r['sha256'] in legacy[date]
            assert match,f'Photo differs from earlier evaluation corpus: {image}'
            rows.append(dict(relative_path=r['relative_path'],source_member=r['source_member'],bytes=r['bytes'],sha256=r['sha256'],width=pixels.shape[1],height=pixels.shape[0],channels=1 if pixels.ndim==2 else pixels.shape[2],filename_Y_command=y,filename_T_command=t,capture_repeat=repeat,condition=kind,**sidecar(image,stage),matches_previous_evaluation_bytes=match))
            if (n+1)%60==0:print(date,'verified images',n+1,'/',len(images),flush=True)
        assert len(images)==(281 if date=='20261014' else 60)
        # Check every output, including sidecars, against its original ZIP member.
        for r in inventory:assert digest(stage/r['relative_path'])==r['sha256']
        assert Counter(r['sha256'] for r in rows)==legacy[date]
        assert digest(raw)==before,'Raw source changed'
        meta=stage/'_metadata';meta.mkdir()
        write_csv(meta/'manifest.csv',rows);write_csv(meta/'file_inventory.csv',inventory)
        counts=Counter(r['condition'] for r in rows);hashes=Counter(r['sha256'] for r in rows)
        provenance=dict(dataset=date,status='verified',source_archive=f'../../raw/{raw.name}',source_archive_sha256=before,source_archive_bytes=raw.stat().st_size,source_archive_unchanged=True,nested_parts=parts,normalized_joined_sha256=joined_hash,removed_common_wrapper=next(iter(prefixes)) if strip_one else None,transformation='Extract exact member bytes and remove one shared wrapper directory. No pixel processing, renaming, sample selection, deduplication, label conversion, or train/test split.',images=len(rows),sidecar_txt=sum(Path(r['relative_path']).suffix.lower()=='.txt' for r in inventory),all_source_files=len(inventory),conditions=dict(counts),duplicate_image_hash_groups=sum(v>1 for v in hashes.values()),duplicate_images_retained=True,all_image_and_sidecar_bytes_match_source=True,all_images_decode=True,image_size=[3840,2160],all_image_bytes_match_previous_evaluation_corpus=True,labels='Filename command values and sidecar Y_Actual/T_Actual are recorded separately. Sidecar physical meaning/offset/units are not inferred. Labels are evaluation metadata, never image-inference input.',evaluation_status='Previously exposed development corpus, not new independent test data.')
        (meta/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
        description=f'''clean_{date}

원본: data/raw/{raw.name}
이미지: {len(rows)}장, BMP 원본 바이트 그대로, 3840x2160
TXT: {provenance['sidecar_txt']}개 원문 그대로

정리 방법
- 중첩 분할 ZIP을 결합하여 해제하고, 공통 최상위 포장 폴더 한 개만 제거했습니다.
- 영상의 크기, 색상, 밝기, 파일명, 촬영별 하위 폴더를 변경하지 않았습니다.
- 실패/불안정 사진이나 반복 사진도 제외하지 않았습니다.
- 원본 raw 파일은 변경하지 않았습니다.

파일 안내
- _metadata/manifest.csv: 사진별 경로, 해시, 이미지 크기, 파일명 명령, 동봉 TXT 원시값
- _metadata/file_inventory.csv: 이미지와 TXT 전체의 원본 ZIP 멤버 대응과 해시
- _metadata/provenance.json: 원본 ZIP 해시, 분할 조각, 검증 결과와 개수

정답 해석
파일명의 Y/T 명령과 TXT의 Y_Actual/T_Actual 값은 별개 열로 기록했습니다.
TXT 값의 좌표계, 원점, 단위를 임의 변환하지 않았습니다.
기존 실험은 파일명 명령을 평가 정답으로 사용했습니다.
파일명/명령값/시행 번호/TXT는 영상 추론기의 입력이 아닙니다.
이 자료는 기존에 개발/평가에 사용한 동일 자료이며 독립 신규 시험 자료가 아닙니다.

센터 기준
20261014/center.bmp가 기존 고정 영상 기준입니다.
20261020의 기준 위치/center1~3.bmp는 반복 촬영입니다. 이들을 자동으로 새 센터 기준으로 교체하지 않습니다.
'''
        (stage/'README.txt').write_text(description,encoding='utf-8')
        # Both the source and explicitly named destination are checked before move.
        assert stage.resolve().is_relative_to(HERE.resolve()) and target.resolve().is_relative_to(CLEAN.resolve())
        if target.exists():target.rmdir() # Only an empty, user-specified directory.
        stage.rename(target)
        print(date,'published',target,'images',len(rows),'sidecars',provenance['sidecar_txt'],flush=True)
        return dict(dataset=date,target=str(target),images=len(rows),txt=provenance['sidecar_txt'],conditions=dict(counts),source_sha256=before,verified=True)
    finally:
        assert work.resolve().is_relative_to(HERE.resolve()) and work.name.startswith(f'archive_{date}_')
        shutil.rmtree(work) # Only this invocation's private split-ZIP scratch files.

def main():
    import pandas as pd
    manifest=pd.read_csv(ROOT/'codex_experiments_20261014/joint_yt_visual_feedback_20261023/manifest.csv')
    legacy={str(d):Counter(g.sha256) for d,g in manifest.groupby('dataset')}
    rows=[process(date,legacy) for date in ['20261014','20261020']]
    (HERE/'result.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
