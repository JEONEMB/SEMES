"""Independent QR intensity observations: no 630-corner measurement is read.

Reference boxes fixed on original center. Integer coarse displacement retained,
along with the normalized current/reference intensity difference at 96x96.
"""
from pathlib import Path
import sys,json,hashlib,time
import cv2,numpy as np,pandas as pd
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
GENERAL=H.parent/'joint_yt_generalization_20261022'
sys.path.insert(0,str(GENERAL))
from data import load
BOXES=[(1552,969,1736,1167),(1809,971,1993,1169),(2067,973,2252,1171)]
CENTER=ROOT/'hw-analysis/data/hw/bmp/20261014/center.bmp'
CENTER_SHA='01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685'

def normalize(patch):
    a=cv2.resize(patch,(96,96),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    return (a-a.mean())/max(float(a.std()),.05)

class PixelExtractor:
    def __init__(self):
        raw=CENTER.read_bytes();assert hashlib.sha256(raw).hexdigest()==CENTER_SHA
        self.center=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        self.reference=np.array([normalize(self.center[y0:y1,x0:x1]) for x0,y0,x1,y1 in BOXES])
    def extract(self,image_bytes):
        img=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape!=self.center.shape:raise ValueError('Image shape mismatch.')
        for pad in [130,260,512]:
            shifts=[];scores=[]
            for x0,y0,x1,y1 in BOXES:
                corr=cv2.matchTemplate(img[y0-pad:y1+pad,x0-pad:x1+pad],self.center[y0:y1,x0:x1],cv2.TM_CCOEFF_NORMED)
                _,score,_,loc=cv2.minMaxLoc(corr);shifts.append((loc[0]-pad,loc[1]-pad));scores.append(score)
            if min(scores)>=.6:break
        if min(scores)<.6:raise ValueError('All-area coarse search failed.')
        current=[]
        for (x0,y0,x1,y1),(dx,dy) in zip(BOXES,shifts):current.append(normalize(img[y0+dy:y1+dy,x0+dx:x1+dx]))
        current=np.array(current);return dict(pixels=current-self.reference,shift=np.array(shifts,dtype=np.float32),ncc=np.array(scores),pad=pad)

def main():
    m,_=load();ex=PixelExtractor();pix=np.full((len(m),3,96,96),np.nan,np.float32);shift=np.full((len(m),3,2),np.nan,np.float32);rows=[]
    protocol=dict(scope='New raw-pixel branch; ADAPTIVE DEVELOPMENT data only',measurement='Fixed original center, same3 QR boxes; integer NCC coarse alignment; per-image contrast normalization;96x96 AREA resize; retain all6 coarse coordinates',uses_630_coordinates=False,fit_on_test_data=False,inference='image bytes only',all_images=len(m),reference_sha256=CENTER_SHA)
    (H/'pixel_measurement_protocol.json').write_text(json.dumps(protocol,indent=2))
    for i,row in m.iterrows():
        start=time.perf_counter()
        try:
            raw=Path(row.path).read_bytes();assert hashlib.sha256(raw).hexdigest()==row.sha256
            r=ex.extract(raw);pix[i]=r['pixels'];shift[i]=r['shift'];rec=dict(index=i,success=True,ncc_min=float(r['ncc'].min()),pad=r['pad'],error='')
        except Exception as e:rec=dict(index=i,success=False,error=str(e))
        rec['seconds']=time.perf_counter()-start;rows.append(rec)
        if i%50==0:print(i,flush=True)
    np.savez_compressed(H/'pixel_features.npz',pixels=pix,shift=shift,reference=ex.reference,success=np.isfinite(pix).all(axis=(1,2,3)))
    pd.DataFrame(rows).to_csv(H/'pixel_measurement_manifest.csv',index=False);print('done',sum(r['success'] for r in rows),flush=True)

if __name__=='__main__':main()
