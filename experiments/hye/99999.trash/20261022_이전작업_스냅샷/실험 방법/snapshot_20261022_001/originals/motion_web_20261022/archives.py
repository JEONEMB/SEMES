"""Read ordinary / nested split ZIPs without extracting paths supplied by archives."""
from pathlib import Path
from zipfile import ZipFile
import struct,shutil,re,uuid
IMAGE_EXT={'.bmp','.png','.jpg','.jpeg'}

def normalize_split(parts,output):
    offsets=[];position=0
    with output.open('wb') as dest:
        for part in parts:
            offsets.append(position)
            with part.open('rb') as source:shutil.copyfileobj(source,dest,8*1024*1024)
            position+=part.stat().st_size
    with output.open('r+b') as f:
        f.seek(max(0,position-70000));tail=f.read();e=tail.rfind(b'PK\x05\x06')
        if e<0:raise ValueError('ZIP 종료 레코드를 찾을 수 없습니다.')
        e+=position-len(tail);f.seek(e);end=bytearray(f.read(22))
        disk,cd_disk,ndisk,total,cdsize,cdoff=struct.unpack_from('<4H2L',end,4)
        cdoff_abs=offsets[cd_disk]+cdoff
        # Rewrite central directory local offsets and disk IDs into a single archive.
        cursor=cdoff_abs
        for _ in range(total):
            f.seek(cursor);head=bytearray(f.read(46))
            if head[:4]!=b'PK\x01\x02':raise ValueError('지원하지 않는 분할 ZIP 중앙 디렉터리입니다.')
            name_len,extra_len,comment_len,volume=struct.unpack_from('<4H',head,28)
            local=struct.unpack_from('<L',head,42)[0]
            if local==0xffffffff:raise ValueError('4GB 이상 ZIP64 멤버는 지원하지 않습니다.')
            struct.pack_into('<H',head,34,0);struct.pack_into('<L',head,42,offsets[volume]+local)
            f.seek(cursor);f.write(head);cursor+=46+name_len+extra_len+comment_len
        struct.pack_into('<4H',end,4,0,0,total,total);struct.pack_into('<L',end,16,cdoff_abs)
        f.seek(e);f.write(end)
        # A ZIP64 EOCD may be present even when sizes fit in 32 bits.
        f.seek(e-20);locator=bytearray(f.read(20))
        if locator[:4]==b'PK\x06\x07':
            ldisk,loff,disks=struct.unpack_from('<LQL',locator,4);absolute=offsets[ldisk]+loff
            struct.pack_into('<LQL',locator,4,0,absolute,1);f.seek(e-20);f.write(locator)
            f.seek(absolute);z64=bytearray(f.read(56))
            if z64[:4]!=b'PK\x06\x06':raise ValueError('ZIP64 종료 레코드 오류')
            struct.pack_into('<2L',z64,16,0,0);struct.pack_into('<Q',z64,48,cdoff_abs);f.seek(absolute);f.write(z64)
    return output

def archive_sources(path,work,label_prefix='',depth=0):
    if depth>2:raise ValueError('중첩 압축은 2단계까지만 지원합니다.')
    sources=[]
    with ZipFile(path) as z:
        infos=z.infolist();splits={}
        for info in infos:
            if info.is_dir():continue
            ext=Path(info.filename).suffix.lower()
            if ext in IMAGE_EXT:
                sources.append(dict(kind='zip',archive=str(path.resolve()),member_index=infos.index(info),name=Path(info.filename).name,relative=label_prefix+info.filename))
            elif re.search(r'\.z\d+$',info.filename,re.I) or ext=='.zip':
                stem=re.sub(r'\.(?:z\d+|zip)$','',info.filename,flags=re.I)
                splits.setdefault(stem,[]).append(info)
        for stem,group in splits.items():
            final=next((i for i in group if i.filename.lower().endswith('.zip')),None)
            if not final:raise ValueError('분할 ZIP의 마지막 .zip 파일이 없습니다.')
            token=uuid.uuid4().hex;folder=work/token;folder.mkdir(parents=True)
            parts=[]
            ordered=sorted([i for i in group if i is not final],key=lambda i:int(re.search(r'\.z(\d+)$',i.filename,re.I)[1]))+[final]
            for n,info in enumerate(ordered):
                dest=folder/str(n)
                with z.open(info) as src,dest.open('wb') as out:shutil.copyfileobj(src,out,8*1024*1024)
                parts.append(dest)
            if len(parts)>1:
                combined=normalize_split(parts,folder/'joined.zip')
                for part in parts:part.unlink()  # Only files just created in this private job folder.
            else:combined=parts[0]
            sources+=archive_sources(combined,work,label_prefix+Path(stem).name+'/',depth+1)
    return sources
