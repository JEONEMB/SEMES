"""원본 ZIP을 손실 없이 data/clean/clean_날짜/에 정리하는 독립 실행 도구.

실행(저장소 루트):
    python tool/01_prepare_clean.py
    python tool/01_prepare_clean.py --date 20261020
    python tool/01_prepare_clean.py --verify-only

현재 작업 폴더와 무관하게 이 파일의 위치로 저장소 루트를 찾습니다.
입력: data/raw/zip_20261014.zip, data/raw/zip_20261020.zip
출력: data/clean/clean_20261014, data/clean/clean_20261020
이미 존재하는 clean은 원본 ZIP/파일 해시/크기/영상 디코딩을 검증하고 보존합니다.
기존 데이터가 다르면 덮어쓰지 않고 오류를 보고합니다.

설치 의존성: python -m pip install numpy opencv-python
나머지는 Python 3.10+ 표준 라이브러리입니다. 외부 실험 폴더에 의존하지 않습니다.
이미지 가공/제외/중복 제거/정답 변환/학습·평가 분할은 수행하지 않습니다.
"""
from pathlib import Path,PurePosixPath
from zipfile import ZipFile
from collections import Counter
import argparse,csv,hashlib,json,re,shutil,struct,tempfile
try:
    import cv2,numpy as np
except ImportError as error:
    raise SystemExit('필요 패키지 설치: python -m pip install numpy opencv-python') from error

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
RAW=REPO/'data/raw'
CLEAN=REPO/'data/clean'
EXPECTED_COUNTS={'20261014':281,'20261020':60}

def normalize_split(parts,output):
    """분할 ZIP의 데이터는 그대로 결합하고 ZIP 내부 볼륨/오프셋만 정규화합니다.

    중앙 디렉터리의 각 파일 위치를 단일 ZIP 기준으로 바꿉니다.
    BMP/TXT의 압축 데이터·원본 파일 바이트는 변경하지 않습니다.
    """
    offsets=[];position=0
    with output.open('xb') as dest:
        for part in parts:
            offsets.append(position)
            with part.open('rb') as source:shutil.copyfileobj(source,dest,8*1024*1024)
            position+=part.stat().st_size
    with output.open('r+b') as f:
        f.seek(max(0,position-70000));tail=f.read();end_index=tail.rfind(b'PK\x05\x06')
        if end_index<0:raise ValueError('분할 ZIP의 종료 레코드가 없습니다.')
        end_index+=position-len(tail);f.seek(end_index);end=bytearray(f.read(22))
        if len(end)!=22:raise ValueError('잘린 ZIP 종료 레코드입니다.')
        disk,cd_disk,on_disk,total,cdsize,cdoff=struct.unpack_from('<4H2L',end,4)
        if max(disk,cd_disk)>=len(offsets):raise ValueError('누락된 분할 ZIP 조각이 있습니다.')
        if total==0xffff or cdoff==0xffffffff:raise ValueError('대형 ZIP64 중앙 디렉터리는 지원하지 않습니다.')
        cd_absolute=offsets[cd_disk]+cdoff;cursor=cd_absolute
        for _ in range(total):
            f.seek(cursor);head=bytearray(f.read(46))
            if len(head)!=46 or head[:4]!=b'PK\x01\x02':raise ValueError('분할 ZIP 중앙 디렉터리가 손상되었습니다.')
            name_len,extra_len,comment_len,volume=struct.unpack_from('<4H',head,28)
            local=struct.unpack_from('<L',head,42)[0]
            if volume>=len(offsets) or local==0xffffffff:raise ValueError('지원하지 않는 ZIP 파일 위치입니다.')
            absolute=offsets[volume]+local
            if absolute>0xffffffff:raise ValueError('4GB 이상 ZIP 오프셋은 지원하지 않습니다.')
            struct.pack_into('<H',head,34,0);struct.pack_into('<L',head,42,absolute)
            f.seek(cursor);f.write(head);cursor+=46+name_len+extra_len+comment_len
        struct.pack_into('<4H',end,4,0,0,total,total);struct.pack_into('<L',end,16,cd_absolute)
        f.seek(end_index);f.write(end)
        # 현재 자료는 ZIP64 locator가 있지만 크기/항목 수는 32비트 안에 있습니다.
        if end_index>=20:
            f.seek(end_index-20);locator=bytearray(f.read(20))
            if locator[:4]==b'PK\x06\x07':
                ldisk,loff,disks=struct.unpack_from('<LQL',locator,4)
                if ldisk>=len(offsets):raise ValueError('ZIP64 볼륨 정보가 잘못되었습니다.')
                absolute=offsets[ldisk]+loff
                struct.pack_into('<LQL',locator,4,0,absolute,1);f.seek(end_index-20);f.write(locator)
                f.seek(absolute);z64=bytearray(f.read(56))
                if len(z64)!=56 or z64[:4]!=b'PK\x06\x06':raise ValueError('ZIP64 종료 레코드가 손상되었습니다.')
                struct.pack_into('<2L',z64,16,0,0);struct.pack_into('<Q',z64,48,cd_absolute)
                f.seek(absolute);f.write(z64)
    return output

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
    if not p.parts or p.is_absolute() or any(s in ('..','') or ':' in s for s in p.parts):raise ValueError(f'Unsafe ZIP path: {name!r}')
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

def verify_existing(date,raw,target):
    """재실행 시 기존 파일을 변경하지 않고 원본 및 전체 출력과 대조합니다."""
    meta=target/'_metadata'
    if not all((meta/name).is_file() for name in ['provenance.json','file_inventory.csv','manifest.csv']):
        raise ValueError(f'기존 폴더의 검증 목록이 없습니다. 자동으로 덮어쓰지 않습니다: {target}')
    provenance=json.loads((meta/'provenance.json').read_text(encoding='utf-8'))
    raw_hash=digest(raw)
    if raw_hash!=provenance['source_archive_sha256']:
        raise ValueError(f'기존 clean과 현재 원본 ZIP이 다릅니다: {raw}')
    with (meta/'file_inventory.csv').open(encoding='utf-8-sig',newline='') as f:inventory=list(csv.DictReader(f))
    with (meta/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    indexed={r['relative_path']:r for r in inventory}
    if len(indexed)!=len(inventory):raise ValueError('파일 목록에 중복 경로가 있습니다.')
    expected=set()
    for n,r in enumerate(inventory,1):
        relative=safe_member(r['relative_path']);p=target.joinpath(*relative.parts)
        if not p.resolve().is_relative_to(target.resolve()):raise ValueError(f'잘못된 clean 경로: {p}')
        if not p.is_file() or p.stat().st_size!=int(r['bytes']) or digest(p)!=r['sha256']:
            raise ValueError(f'누락되거나 변경된 파일입니다: {p}')
        expected.add(relative.as_posix())
        if n%60==0:print(date,'기존 파일 해시 검증',n,'/',len(inventory),flush=True)
    actual={p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file() and '_metadata' not in p.relative_to(target).parts and p.relative_to(target).as_posix()!='README.txt'}
    if actual!=expected:raise ValueError(f'파일 목록과 실제 파일이 다릅니다: {target}')
    if len(rows)!=EXPECTED_COUNTS[date] or len(rows)!=int(provenance['images']):raise ValueError('이미지 개수가 기대값과 다릅니다.')
    seen=set()
    for r in rows:
        relative=safe_member(r['relative_path']);p=target.joinpath(*relative.parts)
        if r['relative_path'] in seen or r['relative_path'] not in indexed:raise ValueError('이미지 목록이 올바르지 않습니다.')
        seen.add(r['relative_path'])
        if r['sha256']!=indexed[r['relative_path']]['sha256']:raise ValueError('사진/파일 목록 해시가 다릅니다.')
        im=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[:2]!=(int(r['height']),int(r['width'])) or im.shape[:2]!=(2160,3840):raise ValueError(f'영상 검증 실패: {p}')
        y,t,kind,repeat=parse_command(p.stem)
        if (y,t,kind,repeat)!=(r['filename_Y_command'],r['filename_T_command'],r['condition'],r['capture_repeat']):raise ValueError(f'명령값 목록이 다릅니다: {p}')
        info=sidecar(p,target)
        if any(info[k]!=r[k] for k in info):raise ValueError(f'TXT 목록이 다릅니다: {p}')
    print(date,'기존 clean 검증 완료, 파일 변경 없음',flush=True)
    return dict(dataset=date,target=str(target),images=len(rows),txt=provenance['sidecar_txt'],source_sha256=raw_hash,verified=True,action='verified_existing')

def process(date,verify_only=False):
    raw=RAW/f'zip_{date}.zip';target=CLEAN/f'clean_{date}'
    if date not in EXPECTED_COUNTS:raise ValueError(f'지원하지 않는 날짜입니다: {date}')
    if not raw.is_file():raise FileNotFoundError(f'원본 ZIP을 찾을 수 없습니다: {raw}')
    if not target.resolve().is_relative_to(CLEAN.resolve()) or target.name!=f'clean_{date}':raise ValueError('출력 경로가 올바르지 않습니다.')
    if target.exists() and any(target.iterdir()):return verify_existing(date,raw,target)
    if verify_only:raise FileNotFoundError(f'검증할 clean 데이터가 없습니다: {target}')
    CLEAN.mkdir(parents=True,exist_ok=True)
    before=digest(raw);work=Path(tempfile.mkdtemp(prefix=f'.prepare_{date}_',dir=CLEAN));stage=work/'dataset';stage.mkdir()
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
            rows.append(dict(relative_path=r['relative_path'],source_member=r['source_member'],bytes=r['bytes'],sha256=r['sha256'],width=pixels.shape[1],height=pixels.shape[0],channels=1 if pixels.ndim==2 else pixels.shape[2],filename_Y_command=y,filename_T_command=t,capture_repeat=repeat,condition=kind,**sidecar(image,stage),matches_source_archive_bytes=True))
            if (n+1)%60==0:print(date,'verified images',n+1,'/',len(images),flush=True)
        if len(images)!=EXPECTED_COUNTS[date]:raise ValueError(f'{date} 이미지 개수 오류: {len(images)}')
        # Check every output, including sidecars, against its original ZIP member.
        for r in inventory:assert digest(stage/r['relative_path'])==r['sha256']
        assert digest(raw)==before,'Raw source changed'
        meta=stage/'_metadata';meta.mkdir()
        write_csv(meta/'manifest.csv',rows);write_csv(meta/'file_inventory.csv',inventory)
        counts=Counter(r['condition'] for r in rows);hashes=Counter(r['sha256'] for r in rows)
        provenance=dict(dataset=date,status='verified',preparation_script='tool/01_prepare_clean.py',source_archive=f'data/raw/{raw.name}',source_archive_path_basis='repository_root',source_archive_sha256=before,source_archive_bytes=raw.stat().st_size,source_archive_unchanged=True,nested_parts=parts,normalized_joined_sha256=joined_hash,removed_common_wrapper=next(iter(prefixes)) if strip_one else None,transformation='Extract exact member bytes and remove one shared wrapper directory. No pixel processing, renaming, sample selection, deduplication, label conversion, or train/test split.',images=len(rows),sidecar_txt=sum(Path(r['relative_path']).suffix.lower()=='.txt' for r in inventory),all_source_files=len(inventory),conditions=dict(counts),duplicate_image_hash_groups=sum(v>1 for v in hashes.values()),duplicate_images_retained=True,all_image_and_sidecar_bytes_match_source=True,all_images_decode=True,image_size=[3840,2160],all_image_bytes_match_source_archive=True,labels='Filename command values and sidecar Y_Actual/T_Actual are recorded separately. Sidecar physical meaning/offset/units are not inferred. Labels are evaluation metadata, never image-inference input.',evaluation_status='Previously exposed development corpus, not new independent test data.')
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
clean_20261014/center.bmp가 기존 고정 영상 기준입니다.
clean_20261020의 기준 위치/center1~3.bmp는 반복 촬영입니다. 이들을 자동으로 새 센터 기준으로 교체하지 않습니다.
'''
        (stage/'README.txt').write_text(description,encoding='utf-8')
        # Both the source and explicitly named destination are checked before move.
        if not stage.resolve().is_relative_to(work.resolve()) or not target.resolve().is_relative_to(CLEAN.resolve()):raise ValueError('게시 경로가 범위를 벗어났습니다.')
        if target.exists():target.rmdir() # Only an empty, user-specified directory.
        stage.rename(target)
        print(date,'published',target,'images',len(rows),'sidecars',provenance['sidecar_txt'],flush=True)
        return dict(dataset=date,target=str(target),images=len(rows),txt=provenance['sidecar_txt'],conditions=dict(counts),source_sha256=before,verified=True,action='created')
    finally:
        if not work.resolve().is_relative_to(CLEAN.resolve()) or not work.name.startswith(f'.prepare_{date}_'):raise ValueError('임시 폴더 정리 경로가 범위를 벗어났습니다.')
        shutil.rmtree(work) # Only this invocation's private split-ZIP scratch files.

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--date',choices=['all',*EXPECTED_COUNTS],default='all',help='처리할 날짜, 기본값: 두 날짜 모두')
    parser.add_argument('--verify-only',action='store_true',help='기존 clean 검증만 수행, 생성/수정하지 않음')
    args=parser.parse_args()
    dates=list(EXPECTED_COUNTS) if args.date=='all' else [args.date]
    rows=[process(date,args.verify_only) for date in dates]
    print(json.dumps(rows,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,AssertionError) as error:raise SystemExit(f'정제 실패: {error}') from error
