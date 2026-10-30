"""특징점 위치 재기 변형: 같은 점(17px 세트 250개: QR 232 + 나사 18)에 대해 흐림 σ, 템플릿 크기를 바꿔 추적.
python e4_track.py <hp_template> <sigma> [featureset_hp]"""
import sys, time; sys.path.insert(0,'../analysis_v6')
from multiprocessing import Pool
import common as C
from common import *
HP=int(sys.argv[1]); SG=float(sys.argv[2]); FS=int(sys.argv[3]) if len(sys.argv)>3 else 8
def bl(im): return cv2.GaussianBlur(im,(0,0),SG) if SG>0 else im
F=pd.read_csv(f'../analysis_v6/features_qr_hp{FS}.csv')
cen=read_raw('center.bmp'); Tr=cut(cen,F,HP); Tb=cut(bl(cen),F,HP)
FP=F.assign(x=F.x+HPC,y=F.y+HPC); BIG=cut(np.pad(cen,HPC,mode='edge'),FP,HPC)
def job(n):
    img=read_raw(n); c0,_=coarse_init(np.pad(img,HPC,mode='edge'),BIG,FP)
    tm,sc=track_tm(img,Tr,F,HP,init=c0); return track_gn(bl(img),Tb,F,HP,tm)
if __name__=='__main__':
    t0=time.time()
    with Pool(16) as p: out=p.map(job,NAMES)
    np.savez_compressed(f'tr_fs{FS}_hp{HP}_s{SG:g}.npz',D=np.stack(out),names=np.array(NAMES)); print(HP,SG,f'{time.time()-t0:.0f}s')
