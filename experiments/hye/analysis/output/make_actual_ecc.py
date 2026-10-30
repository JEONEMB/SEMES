from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).parent/'actual_qr_ecc'
OUT.mkdir(exist_ok=True)
ROOT=Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')
def read(p): return cv2.imdecode(np.fromfile(p,np.uint8),0)
ref=read(ROOT/'center.bmp'); current=read(ROOT/'Y'/'Y+2.00.bmp')
z=np.load(Path(__file__).parents[1]/'codex_experiments_20261014/center_fixed_20261015/center_once_tracks.npz')
pts=z['points']; regions=z['regions']
boxes=[(1552,969,1736,1167),(1809,971,1993,1169),(2067,973,2252,1171)]
aligned=[]; metrics=[]; before=[]
for q,(x0,y0,x1,y1) in enumerate(boxes):
    template=ref[y0:y1,x0:x1].copy()
    pad=100
    search=current[y0-pad:y1+pad,x0-pad:x1+pad]
    _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED))
    dx,dy=loc[0]-pad,loc[1]-pad
    target=current[y0+dy:y1+dy,x0+dx:x1+dx].copy()
    W=np.eye(2,3,dtype=np.float32)
    cc,W=cv2.findTransformECC(template,target,W,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,300,1e-8),None,3)
    total=W.copy();total[:,2]+=[dx,dy]
    raw=current[y0:y1,x0:x1].copy()
    # Warp source is the original full-resolution current frame. This samples
    # the affine mapping directly and avoids a second interpolation.
    sourceW=total.copy();sourceW[:,2]+=[x0,y0]
    registered=cv2.warpAffine(current,sourceW,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    before.append(raw);aligned.append(registered)
    sl=np.s_[8:-8,8:-8]
    corr=lambda b: float(np.corrcoef(template[sl].ravel(),b[sl].ravel())[0,1])
    metrics.append(dict(qr=q+1,ecc_score=float(cc),coarse_dx=dx,coarse_dy=dy,warp_reference_to_current=total.tolist(),correlation_before=corr(raw),correlation_after=corr(registered)))
    for name,a in [('reference',template),('before',raw),('after',registered)]: cv2.imwrite(str(OUT/f'qr{q+1}_{name}.png'),a)

plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'svg.fonttype':'none'})
BG='#192544'; FG='#FFFFFF'; MUTED='#BECADD'; CYAN='#50E6EB'
fig=plt.figure(figsize=(16,8.4),facecolor=BG)
fig.text(.035,.944,'실제 QR 영상의 특징점과 ECC 정합 전·후',color=FG,fontsize=25,weight='bold')
fig.text(.035,.9,'고정 기준: center.bmp   |   입력 예시: Y+2.00.bmp   |   실제 촬영 영상 사용',color=MUTED,fontsize=12)
ax=fig.add_axes([.035,.54,.93,.305]);xa,xb,ya,yb=1515,2285,950,1190
ax.imshow(ref[ya:yb,xa:xb],cmap='gray',vmin=0,vmax=255,extent=[xa,xb,yb,ya])
colors=['#50E6EB','#FFCC66','#B899FF']
for q,c in enumerate(colors):
    pp=pts[regions==q];ax.scatter(pp[:,0],pp[:,1],s=10,facecolors='none',edgecolors=c,linewidths=.75)
ax.set(xlim=(xa,xb),ylim=(yb,ya));ax.set_aspect('equal');ax.axis('off')
fig.text(.035,.855,'① 기준 QR 3개 — 고정 특징점 205개 전체 표시',color=FG,fontsize=15,weight='bold')
fig.text(.035,.505,'② 왼쪽 QR 확대 — 세 패널 모두 가로·세로 동일 배율 (원본 대비 약 3배)',color=FG,fontsize=15,weight='bold')
q=0;x0,y0,x1,y1=boxes[q];template=ref[y0:y1,x0:x1]
def overlay(a,b):
    a=a.astype(float)/255;b=b.astype(float)/255
    return np.stack([a,b,a],axis=-1)
panels=[np.repeat(template[:,:,None],3,axis=2)/255,overlay(template,before[q]),overlay(template,aligned[q])]
titles=['기준 영상','ECC 정합 전','ECC 정합 후']
for i,(im,title) in enumerate(zip(panels,titles)):
    ax=fig.add_axes([.045+i*.32,.115,.27,.315]);ax.imshow(im,interpolation='nearest');ax.axis('off');ax.set_aspect('equal')
    ax.set_title(title,color=FG,fontsize=16,weight='bold',pad=10)
    if i==0:
        pp=pts[regions==q]-[x0,y0];ax.scatter(pp[:,0],pp[:,1],s=14,facecolors='none',edgecolors=CYAN,linewidths=.8)
    else:
        key='correlation_before' if i==1 else 'correlation_after'
        fig.text(.18+i*.32,.075,f'영상 상관계수  {metrics[q][key]:.4f}',ha='center',color=MUTED,fontsize=12)
fig.text(.035,.025,'겹침 색상: 기준 = 자홍 / 입력 = 초록 / 일치 = 회색·흰색   ·   QR별 affine ECC + 거친 위치 정합   ·   영상 정합 수치는 Y·Θ 추정 정확도가 아님',color=MUTED,fontsize=10)
fig.savefig(OUT/'actual_qr_features_ecc.png',dpi=220,facecolor=BG)
fig.savefig(OUT/'actual_qr_features_ecc.svg',facecolor=BG)
(OUT/'metrics.json').write_text(json.dumps(dict(reference='center.bmp',current='Y+2.00.bmp',features=205,method='coarse template localization + affine ECC; inverse warp',metrics=metrics),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(metrics,indent=2))
