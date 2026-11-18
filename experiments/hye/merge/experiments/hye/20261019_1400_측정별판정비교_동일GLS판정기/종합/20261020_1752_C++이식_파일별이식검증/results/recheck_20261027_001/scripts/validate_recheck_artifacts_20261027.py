"""Decode generated media and check OOXML package integrity without modifying files."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image
import cv2

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/recheck_20261027_001'

def main():
    roots=[OUT]
    for name in ('eda_commands.json','eda_fix_commands.json'):
        p=OUT/name
        if p.exists():
            for r in json.loads(p.read_text(encoding='utf-8')):
                roots.extend(ROOT/x for x in r.get('output_directories',[]))
    records=[]
    for root in roots:
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in {'.png','.gif','.mp4','.xlsx','.pptx'}:continue
            r={'file':str(p.relative_to(ROOT))}
            try:
                if p.suffix.lower() in {'.png','.gif'}:
                    with Image.open(p) as im:
                        frames=getattr(im,'n_frames',1);duration=0
                        for i in range(frames):
                            im.seek(i);im.load();duration+=im.info.get('duration',0)
                        r.update(width=im.width,height=im.height,decoded_frames=frames,duration_ms=duration)
                elif p.suffix.lower()=='.mp4':
                    cap=cv2.VideoCapture(str(p));frames=0
                    fps=cap.get(cv2.CAP_PROP_FPS)
                    while True:
                        ok,frame=cap.read()
                        if not ok:break
                        frames+=1
                    cap.release()
                    if not frames:raise RuntimeError('No decoded video frames')
                    r.update(decoded_frames=frames,fps=fps,duration_seconds=frames/fps)
                else:
                    with zipfile.ZipFile(p) as z:
                        bad=z.testzip()
                        if bad:raise RuntimeError('Bad CRC '+bad)
                        for n in z.namelist():
                            if n.endswith(('.xml','.rels')):ET.fromstring(z.read(n))
                        r.update(zip_members=len(z.namelist()))
                r['passed']=True
            except Exception as e:r.update(passed=False,error=str(e))
            records.append(r)
    (OUT/'artifact_integrity.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('artifacts',len(records),'failed',sum(not r['passed'] for r in records))

if __name__=='__main__':main()
