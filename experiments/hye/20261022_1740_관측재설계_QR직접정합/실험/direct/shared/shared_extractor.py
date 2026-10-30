"""One dense transform shared across three QR regions; common-plane hypothesis."""
from pathlib import Path
import importlib.util,time,hashlib
import cv2,numpy as np

HERE=Path(__file__).resolve().parent;DIRECT=HERE.parent
spec=importlib.util.spec_from_file_location('reboot_dense_pixel_base',DIRECT/'extractor.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
CANDIDATES=(('shared_affine_g3',cv2.MOTION_AFFINE,3),('shared_homography_g3',cv2.MOTION_HOMOGRAPHY,3),('shared_affine_g5',cv2.MOTION_AFFINE,5),('shared_homography_g5',cv2.MOTION_HOMOGRAPHY,5))
BOXES=base.BOXES
BBOX=(min(b[0] for b in BOXES),min(b[1] for b in BOXES),max(b[2] for b in BOXES),max(b[3] for b in BOXES))
MARGIN=20

class SharedExtractor(base.DirectExtractor):
    def __init__(self):
        super().__init__();x0,y0,x1,y1=BBOX;self.joint_template=self.center[y0:y1,x0:x1].copy();self.joint_mask=np.zeros(self.joint_template.shape,np.uint8);self.qr_masks=[]
        for a,b,c,d in BOXES:
            mask=np.zeros(self.joint_template.shape,np.uint8);k=base.TEMPLATE_INSET;mask[b-y0+k:d-y0-k,a-x0+k:c-x0-k]=255;self.joint_mask|=mask;self.qr_masks.append(mask>0)

    def extract_bytes(self,image_bytes):
        started=time.perf_counter();image=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None:raise ValueError('Cannot decode image')
        if image.shape!=self.center.shape:raise ValueError('Dimensions differ from fixed center')
        tn=time.perf_counter();shifts,scores,pad,attempts=self.coarse(image);ncc_seconds=time.perf_counter()-tn
        result=dict(image_sha256=hashlib.sha256(image_bytes).hexdigest(),ncc_shifts=shifts,ncc_scores=scores,ncc_pad=pad,ncc_attempts=attempts,ncc_seconds=ncc_seconds,candidates={})
        x0,y0,x1,y1=BBOX;dx,dy=np.rint(np.median(shifts,axis=0)).astype(int);left=x0+dx-MARGIN;top=y0+dy-MARGIN;right=x1+dx+MARGIN;bottom=y1+dy+MARGIN
        for name,motion,gaussian in CANDIDATES:
            began=time.perf_counter();D=np.full(self.points.shape,np.nan);HH=np.full((3,3),np.nan);why=[];ecc=np.full(3,np.nan);rmse=np.full(3,np.nan);gain=np.full(3,np.nan);offset=np.full(3,np.nan)
            try:
                if min(scores)<base.NCC_MIN:raise ValueError('NCC below fixed threshold')
                if min(left,top)<0 or right>image.shape[1] or bottom>image.shape[0]:raise ValueError('Shared current crop outside image')
                crop=image[top:bottom,left:right].copy();warp=base.translate(MARGIN,MARGIN).astype(np.float32)
                if motion!=cv2.MOTION_HOMOGRAPHY:warp=warp[:2]
                cc,local=cv2.findTransformECCWithMask(self.joint_template,crop,self.joint_mask,np.full(crop.shape,255,np.uint8),warp,motion,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,base.MAX_ITERATIONS,base.EPSILON),gaussian)
                localH=np.vstack([local,[0.,0.,1.]]) if local.shape==(2,3) else local.astype(float)
                HH=base.translate(left,top)@localH@base.translate(-x0,-y0);HH/=HH[2,2];pred=base.project(HH,self.points);measured=pred-self.points;expected=shifts[self.regions]
                if not np.isfinite(measured).all():why.append('nonfinite warp')
                if cc<base.ECC_MIN:why.append('ECC below fixed0.8')
                if np.max(abs(measured-expected))>10:why.append('refinement more than10px from perQR NCC')
                yy,xx=np.indices(self.joint_template.shape,dtype=np.float32);p=base.project(localH,np.c_[xx.ravel(),yy.ravel()]);aligned=cv2.remap(crop.astype(np.float32),p[:,0].reshape(xx.shape).astype(np.float32),p[:,1].reshape(xx.shape).astype(np.float32),cv2.INTER_LINEAR)
                for qr,valid in enumerate(self.qr_masks):
                    A=np.c_[aligned[valid],np.ones(valid.sum())];fit=np.linalg.lstsq(A,self.joint_template[valid].astype(float),rcond=None)[0];rmse[qr]=np.sqrt(np.mean((A@fit-self.joint_template[valid])**2));gain[qr],offset[qr]=fit;ecc[qr]=cc
                if not why:D=measured
            except (cv2.error,ValueError,np.linalg.LinAlgError) as exc:why=[type(exc).__name__+': '+str(exc)]
            result['candidates'][name]=dict(D=D,H=np.repeat(HH[None],3,axis=0),H_shared=HH,ecc=ecc,photometric_rmse=rmse,brightness_gain=gain,brightness_offset=offset,valid_pixels=np.array([m.sum() for m in self.qr_masks]),qr_success=np.array([not why]*3),success=not why,reasons=[why]*3,seconds=time.perf_counter()-began)
        result['seconds']=time.perf_counter()-started;return result

def extract_bytes(image_bytes):return SharedExtractor().extract_bytes(image_bytes)
