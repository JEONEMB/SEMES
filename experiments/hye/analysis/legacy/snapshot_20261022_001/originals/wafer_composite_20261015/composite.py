from pathlib import Path
import sys,json,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'codex_experiments_20261014/deps'))
import cv2,numpy as np
P=Path(__file__).resolve().parent
texture=cv2.imread(str(P/'image1.png'),cv2.IMREAD_UNCHANGED)
cs,_=cv2.findContours((texture[:,:,3]>127).astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
source=cv2.fitEllipse(max(cs,key=cv2.contourArea))
def matrix(ellipse,rotation=0):
    (x,y),(a,b),angle=ellipse;t=np.deg2rad(angle);r=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    q=np.deg2rad(rotation);rot=np.array([[np.cos(q),-np.sin(q)],[np.sin(q),np.cos(q)]])
    m=np.eye(3);m[:2,:2]=r@np.diag([a/2,b/2])@rot;m[:2,2]=[x,y];return m
S=matrix(source)
checks=[]
for clip,description in [(1,'02페이지_영점보정불량'),(2,'03페이지_영점보정정상')]:
    cap=cv2.VideoCapture(str(P/f'media{clip}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4))
    output=P/(description+'_무지개웨이퍼.mp4')
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-i',str(P/f'media{clip}.mp4'),'-map','0:v:0','-map','1:a?','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    perframe=[]
    for i in range(N):
        ok,frame=cap.read();assert ok
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([90,25,150],np.uint8),np.array([130,150,250],np.uint8))
        # Only the moving wafer scene and the lower top-view inset, not QR panel/text.
        roi=np.zeros_like(mask);roi[175:950,:1260]=255;roi[520:850,1360:1850]=255;mask=cv2.bitwise_and(mask,roi)
        clean=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        cs,_=cv2.findContours(clean,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        objects=[]
        for c in cs:
            area=cv2.contourArea(c)
            if area<3500 or len(c)<20:continue
            ell=cv2.fitEllipse(c);(cx,cy),(a,b),angle=ell
            if min(a,b)<110 or max(a,b)<150 or max(a,b)>360 or max(a,b)/min(a,b)>2:continue
            contourmask=np.zeros_like(mask);cv2.drawContours(contourmask,[c],-1,255,-1)
            ellipse_mask=np.zeros_like(mask);cv2.ellipse(ellipse_mask,ell,255,-1)
            overlap=np.count_nonzero(cv2.bitwise_and(ellipse_mask,contourmask))/max(np.count_nonzero(ellipse_mask),1)
            if overlap<.7:continue
            # Texture follows the visible plane. During spin, both views rotate together.
            spin=max(0,i-80)*18 if 80<=i<=135 else (55*18 if i>135 else 0)
            dst=matrix(ell,spin);M=(dst@np.linalg.inv(S))[:2]
            warped=cv2.warpAffine(texture[:,:,:3],M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            # Segment support preserves foreground labels, nozzle, red error overlays, and shadows.
            coverage=cv2.bitwise_and(contourmask,mask).astype(np.float32)/255
            coverage=cv2.GaussianBlur(coverage,(3,3),.45)
            coverage*=cv2.GaussianBlur(ellipse_mask.astype(np.float32)/255,(3,3),.45)
            # Keep the original surface lighting and water tint at restrained strength.
            lum=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(float)
            mean=float(np.median(lum[coverage>.8]))
            shade=np.clip(lum/max(mean,1),.78,1.15)
            replacement=np.clip(warped.astype(float)*shade[:,:,None],0,255)
            a3=(coverage*.97)[:,:,None]
            frame=np.rint(frame*(1-a3)+replacement*a3).astype(np.uint8)
            objects.append(dict(center=[cx,cy],axes=[a,b],coverage=overlap))
        encoder.stdin.write(frame.tobytes());perframe.append(len(objects))
        if i in [0,35,70,110,152]:cv2.imwrite(str(P/f'composite{clip}_{i:03d}.jpg'),frame)
        if i%40==0:print('VIDEO',clip,i+1,N,flush=True)
    cap.release();encoder.stdin.close();assert encoder.wait()==0
    check=cv2.VideoCapture(str(output));assert int(check.get(7))==N and int(check.get(3))==w and int(check.get(4))==h;check.release()
    checks.append(dict(output=output.name,frames=N,fps=fps,width=w,height=h,visible_wafer_counts=perframe))
(P/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',flush=True)
