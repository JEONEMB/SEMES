"""Experimental image-only inference. No robot commands are issued.

python predict.py IMAGE --method qr_external
The path is used only to read bytes; filenames and command labels are not parsed.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent

def qr_grid(Hs):
    boxes=[(1552,969,1736,1167),(1809,971,1993,1169),(2067,973,2252,1171)];ds=[]
    for q,(x0,y0,x1,y1) in enumerate(boxes):
        p=np.array([(x0+a*(x1-x0),y0+b*(y1-y0)) for b in [.2,.5,.8] for a in [.2,.5,.8]])
        z=np.c_[p,np.ones(9)]@Hs[q].T;ds.append(z[:,:2]/z[:,2:]-p)
    return np.concatenate(ds)

class Predictor:
    def __init__(self,method):
        self.method=method;self.bundle=joblib.load(H/(method+'_model.joblib'))
        self.extractors={}
        if method in ['direct','qr_external']:
            from direct.extractor import DirectExtractor
            self.extractors['direct']=DirectExtractor()
        if method=='shared':
            from direct.shared.shared_extractor import SharedExtractor
            self.extractors['shared']=SharedExtractor()
        if method in ['external','qr_external']:
            from parallax.extractor import ParallaxExtractor
            self.extractors['external']=ParallaxExtractor()
        if method in ['pixel_pls','pixel_cnn']:
            from raw_pixels import PixelExtractor
            self.extractors['pixels']=PixelExtractor()
    def predict_bytes(self,raw):
        start=time.perf_counter();model=self.bundle['model'];name=self.method
        if name in ['pixel_pls','pixel_cnn']:
            r=self.extractors['pixels'].extract(raw);p=model.predict(r['pixels'][None],r['shift'][None])[0]
        elif name in ['direct','shared']:
            r=self.extractors[name].extract_bytes(raw)['candidates'][self.bundle['config']['measurement']]
            if not r['success']:raise ValueError('Measurement failure: '+str(r.get('reasons')))
            p=model.predict(r['D'][None])[0]
        else:
            r=self.extractors['external'].extract_bytes(raw)['candidates'][self.bundle['config']['measurement']]
            if not r['success']:raise ValueError('External measurement failure: '+str(r.get('reasons')))
            D=r['D']
            if name=='qr_external':
                qr=self.extractors['direct'].extract_bytes(raw)['candidates']['homography_g5']
                if not qr['success']:raise ValueError('QR measurement failure: '+str(qr.get('reasons')))
                D=np.concatenate([qr_grid(qr['H']),D])
            p=model.predict(D[None])[0]
        if not np.isfinite(p).all():raise ValueError('Nonfinite prediction')
        return dict(Y=float(p[0]),T=float(p[1]),method=name,image_sha256=hashlib.sha256(raw).hexdigest(),seconds=time.perf_counter()-start,status='experimental; independent tolerance performance unverified',T_unit='command units; physical angular unit unconfirmed')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('image',type=Path);parser.add_argument('--method',choices=['direct','shared','pixel_pls','pixel_cnn','external','qr_external'],required=True);args=parser.parse_args()
    raw=args.image.read_bytes()
    try:result=Predictor(args.method).predict_bytes(raw)
    except (ValueError,RuntimeError) as exc:result=dict(status='measurement_or_prediction_failed',error=str(exc),image_sha256=hashlib.sha256(raw).hexdigest(),method=args.method)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':
    with threadpool_limits(1):main()
