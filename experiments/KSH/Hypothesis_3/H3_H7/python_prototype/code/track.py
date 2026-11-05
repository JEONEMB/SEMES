# 1단계 : QR 띠 모서리를 기준(center.bmp)에서 뽑고, 모든 영상에서 LK 로 추적 → 대응점 저장 (전체 화면 좌표)
import cv2,numpy as np,glob,os,re,time
SP='/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad'
ROW0=860                                   # 잘라 둔 띠의 원래 행 위치
X0,X1,Y0,Y1=1530,2280,95,330               # QR 3개 영역 (띠 좌표)
ref=cv2.imread(f'{SP}/run/img/center.bmp',0)
crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,50,1e-4)
c=cv2.goodFeaturesToTrack(ref[Y0:Y1,X0:X1],3000,0.02,6)
P0=cv2.cornerSubPix(ref[Y0:Y1,X0:X1],c,(5,5),(-1,-1),crit).reshape(-1,2)+[X0,Y0]
print('ref points',len(P0))
def coarse(g):
    a=ref[Y0:Y1,X0-260:X1+260].astype(np.float32); b=g[Y0:Y1,X0-260:X1+260].astype(np.float32)
    (dx,dy),_=cv2.phaseCorrelate(a,b,cv2.createHanningWindow(a.shape[::-1],cv2.CV_32F)); return dx,dy
lkc=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,100,1e-5)
def track(g):
    dx,dy=coarse(g)
    p0=P0.astype(np.float32).reshape(-1,1,2); p1=(P0+[dx,dy]).astype(np.float32).reshape(-1,1,2)
    q,st,_=cv2.calcOpticalFlowPyrLK(ref,g,p0,p1.copy(),winSize=(15,15),maxLevel=0,criteria=lkc,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    b,st2,_=cv2.calcOpticalFlowPyrLK(g,ref,q,p0.copy(),winSize=(15,15),maxLevel=0,criteria=lkc,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    fb=np.linalg.norm(b.reshape(-1,2)-P0,axis=1)
    ok=(st.ravel()==1)&(st2.ravel()==1)&(fb<0.1)
    return q.reshape(-1,2),ok
def truth(n):
    s=os.path.splitext(n)[0]
    y=re.search(r'Y\s*([+-][\d.]+)',s); t=re.search(r'T\s*([+-][\d.]+)',s); sh=re.search(r'\((\d)\)',s)
    return (float(y.group(1)) if y else 0.0),(float(t.group(1)) if t else 0.0),(int(sh.group(1)) if sh else 0)
files=[('center.bmp',f'{SP}/run/img/center.bmp',1)]
files+=[(os.path.basename(p),p,1) for p in sorted(glob.glob(f'{SP}/run/img/Y/*.bmp'))+sorted(glob.glob(f'{SP}/run/img/Theta/*.bmp'))]
for p in sorted(glob.glob(f'{SP}/new/img2/*/*.png')):
    d=os.path.basename(os.path.dirname(p)); b=os.path.splitext(os.path.basename(p))[0]
    name=(b+'.bmp') if d=='기준 위치' else f'{d} {b[-3:]}.bmp' if re.search(r'\(\d\)$',b) else b+'.bmp'
    files.append((name,p,2))
names=[];Q=[];OK=[];info=[];t0=time.time()
for n,p,s in files:
    g=cv2.imread(p,0); q,ok=track(g); names.append(n); Q.append(q); OK.append(ok)
    y,t,sh=truth(n); info.append((s,y,t,sh))
Q=np.array(Q); OK=np.array(OK)
P0a=P0+[0,ROW0]; Qa=Q+[0,ROW0]
np.savez('corr.npz',P0=P0a,Q=Qa,OK=OK,names=np.array(names),info=np.array(info))
print(len(names),'images', f'{1000*(time.time()-t0)/len(names):.0f} ms/img', 'valid pts median',np.median(OK.sum(1)),'min',OK.sum(1).min())
print(names[:3],names[-5:])
