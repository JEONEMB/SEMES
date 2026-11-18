"""Read-only source inventory; hashes decoded grayscale pixels used by inference."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,hashlib,json,zipfile
import cv2,numpy as np,pandas as pd

H=Path(__file__).resolve().parent;ROOT=H.parents[2]
PREV=ROOT/'codex_experiments_20261014/joint_yt_generalization_20261022'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def image_info(path):
    path=Path(path);raw=path.read_bytes();img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
    if img is None:raise ValueError(str(path))
    return dict(path=str(path.resolve()),relative=str(path.relative_to(ROOT)),bytes=len(raw),file_sha256=digest(raw),grayscale_pixel_sha256=digest(str(img.shape).encode()+img.tobytes()),height=img.shape[0],width=img.shape[1],all_pixels_zero=bool(not img.any()))

def main():
    cmd=['rg','--files','-uuu','-g','!**/.git/**','-g','!**/.codex/**','-g','!**/.aws/**','-g','*.bmp','-g','*.tif','-g','*.tiff','-g','*.png','-g','*.jpg','-g','*.jpeg','-g','*.zip','-g','*.7z','-g','*.rar']
    names=subprocess.check_output(cmd,cwd=ROOT,text=True,encoding='utf-8').splitlines();files=[ROOT/name for name in names]
    m=pd.read_csv(PREV/'measurements/manifest.csv');known_paths={str(Path(p).resolve()):i for i,p in enumerate(m.path)}
    raw_images=[p for p in files if p.suffix.lower() in ['.bmp','.tif','.tiff']]
    converted=[p for p in files if str(p).startswith(str(ROOT/'hw-analysis/data/hw/png')) and p.suffix.lower() in ['.png','.jpg','.jpeg']]
    infos=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(image_info,raw_images+converted)):
            infos.append(row)
            if i%100==0:print('hashed',i+1,'/',len(raw_images+converted),flush=True)
    frame=pd.DataFrame(infos);known=frame[frame.path.isin(known_paths)].copy()
    assert len(known)==341
    expected=m.set_index('path').sha256.to_dict()
    assert all(expected[row.path]==row.file_sha256 for row in known.itertuples())
    kbytes=set(known.file_sha256);kpixels=set(known.grayscale_pixel_sha256)
    frame['source341']=frame.path.isin(known_paths)
    frame['byte_matches_development341']=frame.file_sha256.isin(kbytes)
    frame['pixels_match_development341']=frame.grayscale_pixel_sha256.isin(kpixels)
    frame['status']=np.where(frame.source341,'known_development_image',np.where(frame.pixels_match_development341,'copy_or_format_conversion_of_development','unmatched_requires_review'))
    controls=frame.relative.str.contains(r'final\\build\\test-work\\',regex=True)&frame.relative.str.endswith('wrong_center.bmp')&(frame.height==40)&(frame.width==40)&frame.all_pixels_zero
    frame.loc[controls,'status']='synthetic_negative_test_control'
    frame.to_csv(H/'image_inventory.csv',index=False,encoding='utf-8-sig')
    exposed=known[['path','relative','file_sha256','grayscale_pixel_sha256','height','width']].copy()
    exposed['development_status']='exposed: used in training, OOF comparisons or failure-driven method development'
    exposed.to_csv(H/'exposed_images.csv',index=False,encoding='utf-8-sig')
    ext=pd.DataFrame([dict(path=str(p.relative_to(ROOT)),extension=p.suffix.lower()) for p in files]);ext.groupby('extension').size().rename('count').to_csv(H/'inventory_extension_counts.csv')
    raw_archives=[]
    for p in sorted((ROOT/'hw-analysis/data/hw/raw').rglob('*')):
        if p.is_file():raw_archives.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=digest(p.read_bytes())))
    archives=[]
    for p in files:
        if p.suffix.lower()!='.zip':continue
        try:
            with zipfile.ZipFile(p) as z:
                members=[dict(name=x.filename,bytes=x.file_size,crc32=f'{x.CRC:08x}') for x in z.infolist() if not x.is_dir()]
                images=[x for x in members if Path(x['name']).suffix.lower() in ['.bmp','.tif','.tiff']]
                archives.append(dict(path=str(p.relative_to(ROOT)),members=len(members),source_image_members=len(images),source_image_names=[x['name'] for x in images],hash_scope='ZIP members inventoried by names/size/CRC only; not all archive payloads SHA-rehashed'))
        except Exception as exc:archives.append(dict(path=str(p.relative_to(ROOT)),error=str(exc)))
    (H/'archive_inventory.json').write_text(json.dumps(archives,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(scope='Entire project filename inventory; all BMP/TIFF files plus hw-analysis converted PNG/JPEG decoded and SHA-256 checked. Report/presentation PNG/JPEG not treated as independently captured robot data.',project=str(ROOT),files_enumerated=len(files),raw_images_hashed=len(raw_images),converted_images_hashed=len(converted),development_images=341,unique_development_byte_hashes=len(kbytes),unique_development_grayscale_pixel_hashes=len(kpixels),all_known_source_sha_matches=True,unmatched_count=int((frame.status=='unmatched_requires_review').sum()),unmatched=frame.loc[frame.status=='unmatched_requires_review',['path','bytes','height','width','file_sha256','grayscale_pixel_sha256']].to_dict('records'),synthetic_negative_controls=int(controls.sum()),synthetic_control_source='final/tests/engine_checks.cpp lines71-73: cv::Mat::zeros(40,40,CV_8U) encoded as wrong_center.bmp',raw_archives=raw_archives,known_corpus_all_exposed=True,resplit_is_pristine_test=False,independent_new_capture_confirmed=False,limitations=['Hash mismatch alone cannot establish independent capture or historical non-use.','Generated images, crops, conversions, renamed copies, augmentation, and legacy snapshots are not new independent captures.','No search outside the requested project workspace was performed.','Archive members were inventoried but not every duplicate archive payload was decoded.'])
    (H/'source_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['files_enumerated','raw_images_hashed','converted_images_hashed','development_images','unique_development_byte_hashes','unique_development_grayscale_pixel_hashes','unmatched_count','independent_new_capture_confirmed']},indent=2))

if __name__=='__main__':main()
