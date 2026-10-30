from pathlib import Path
import sys,json,time
BASE=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'deps'),str(BASE/'measurement')]
import cv2,numpy as np,pandas as pd
from scipy.optimize import least_squares
from scipy.special import erf
from PIL import Image,ImageDraw,ImageFont
from measure import ROOT,BOXES
SOURCE=BASE/'decoded_grid_corners_20261015'
meta=json.loads((SOURCE/'metadata.json').read_text(encoding='utf-8'))
original=cv2.imdecode(np.fromfile(ROOT/'center.bmp',np.uint8),0)
initial=pd.read_csv(SOURCE/'structural_corners.csv')

def field(xy,center,axes,sigma,patch):
    local=(xy-center)@np.linalg.inv(axes).T
    U=.5*(1+erf(local[:,0]/(np.sqrt(2)*sigma[0])))
    V=.5*(1+erf(local[:,1]/(np.sqrt(2)*sigma[1])))
    return patch[0,0]*(1-U)*(1-V)+patch[0,1]*U*(1-V)+patch[1,0]*(1-U)*V+patch[1,1]*U*V

def fit(xy,I,start,axes,patch):
    low,high=np.percentile(I,[10,90]);contrast=max(high-low,20)
    p0=[*start,1.,1.,low,contrast]
    lo=[start[0]-3.8,start[1]-3.8,.35,.35,0,10]
    hi=[start[0]+3.8,start[1]+3.8,2.7,2.7,240,255]
    def residual(p):return p[4]+p[5]*(1-field(xy,p[:2],axes,p[2:4],patch))-I
    result=least_squares(residual,p0,bounds=(lo,hi),loss='soft_l1',f_scale=6,max_nfev=85,xtol=1e-7,ftol=1e-7,gtol=1e-7)
    p=result.x;rmse=float(np.sqrt(np.mean(residual(p)**2)))
    return p,rmse,result.success

def synthetic_check():
    rows=[];yy,xx=np.mgrid[-5:6,-5:6];xy=np.c_[xx.ravel(),yy.ravel()].astype(float)
    for patch in [np.array([[1,0],[0,0]]),np.array([[0,1],[1,1]]),np.array([[1,0],[0,1]])]:
        for truth in [[.25,-.35],[-.6,.4]]:
            I=40+150*(1-field(xy,np.array(truth),np.eye(2),[.9,1.1],patch))
            p,rmse,ok=fit(xy,I,np.array([0.,0.]),np.eye(2),patch)
            err=float(np.linalg.norm(p[:2]-truth));assert err<1e-3
            rows.append(dict(error_px=err,rmse=rmse,converged=ok))
    return rows

validation=synthetic_check();records=[];start_time=time.time()
for item in meta['qr']:
    k=item['qr'];n=item['modules'];q=np.array(item['quad'],np.float32)
    H=cv2.getPerspectiveTransform(np.float32([[0,0],[n,0],[n,n],[0,n]]),q)
    grid=Image.open(SOURCE/f'QR{k}_decoded_grid.png').convert('L').resize((n,n),Image.Resampling.NEAREST)
    black=np.pad((np.array(grid)<128).astype(float),1)
    for r in initial[initial.qr==k].itertuples():
        u,v=int(r.grid_x),int(r.grid_y);center=np.array([r.x,r.y])
        projections=cv2.perspectiveTransform(np.float32([[[u,v],[u+.1,v],[u,v+.1]]]),H)[0]
        axes=(projections[1:]-projections[0]).T
        axes/=np.linalg.norm(axes,axis=0)
        patch=black[v:v+2,u:u+2]
        fits=[]
        for radius in [4,5]:
            xx0,yy0=np.rint(center).astype(int)
            yy,xx=np.mgrid[yy0-radius:yy0+radius+1,xx0-radius:xx0+radius+1]
            xy=np.c_[xx.ravel(),yy.ravel()].astype(float);I=original[yy,xx].ravel().astype(float)
            # Fit is strictly confined to this QR panel, including outer finder edges.
            a,b,c,d=BOXES[k-1];inside=(xy[:,0]>=a)&(xy[:,0]<c)&(xy[:,1]>=b)&(xy[:,1]<d)
            fits.append(fit(xy[inside],I[inside],center,axes,patch))
        p,rmse,ok=fits[0];p2,rmse2,ok2=fits[1]
        stability=float(np.linalg.norm(p[:2]-p2[:2]));shift=float(np.linalg.norm(p[:2]-center))
        # Fixed diagnostic thresholds; no labels or manual relocations.
        reasons=[]
        if not(ok and ok2):reasons.append('optimizer')
        if rmse>15:reasons.append('photometric_rmse')
        if p[5]<45:reasons.append('low_contrast')
        if stability>.65:reasons.append('window_instability')
        if np.max(abs(p[:2]-center))>3.6:reasons.append('search_limit')
        if max(p[2:4])>2.6:reasons.append('blur_limit')
        records.append(dict(qr=k,grid_x=u,grid_y=v,initial_x=r.x,initial_y=r.y,x=p[0],y=p[1],shift_px=shift,rmse=rmse,window_stability_px=stability,sigma_x=p[2],sigma_y=p[3],contrast=p[5],valid=not reasons,reason=';'.join(reasons),black_neighbors=int(patch.sum())))
    print('QR',k,'completed',round(time.time()-start_time,1),flush=True)
df=pd.DataFrame(records);df.to_csv(OUT/'refined_corners.csv',index=False,encoding='utf-8-sig')
df[~df.valid].to_csv(OUT/'uncertain_corners.csv',index=False,encoding='utf-8-sig')
GREEN=(139,250,131);AMBER=(255,180,65)
def render(k,bounds,size,compare=False,grid=False):
    a,b,c,d=bounds;w,h=size
    im=Image.fromarray(original[b:d,a:c]).convert('RGB').resize((w,h),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(im)
    sx,sy=w/(c-a),h/(d-b)
    if grid:
        for y in range(0,h,30):draw.line((w//2,y,w//2,min(y+14,h)),fill=GREEN,width=3)
        for x in range(0,w,30):draw.line((x,h//2,min(x+14,w),h//2),fill=GREEN,width=3)
    for r in df[df.qr==k].itertuples():
        if not(a<=r.x<c and b<=r.y<d):continue
        x,y=(r.x-a)*sx,(r.y-b)*sy
        if compare:
            ix,iy=(r.initial_x-a)*sx,(r.initial_y-b)*sy
            draw.line((ix-5,iy,ix+5,iy),fill=(208,124,255),width=2)
            draw.line((ix,iy-5,ix,iy+5),fill=(208,124,255),width=2)
            draw.line((ix,iy,x,y),fill=(208,124,255),width=2)
        color=GREEN if r.valid else AMBER
        rad=7 if sx<=6 else 10
        draw.rectangle((x-rad-1,y-rad-1,x+rad+1,y+rad+1),outline=(12,20,12),width=5)
        draw.rectangle((x-rad,y-rad,x+rad,y+rad),outline=color,width=3)
        draw.point((x,y),fill=color)
    return im
for grid in [False,True]:
    w,h,gap=920,990,55;canvas=Image.new('RGB',(3*w+2*gap,h),(22,24,27))
    for k,box in enumerate(BOXES,1):
        im=render(k,box,(w,h),grid=grid);canvas.paste(im,((k-1)*(w+gap),0))
        if not grid:im.save(OUT/f'QR{k}_subpixel.png')
    canvas.save(OUT/('all_subpixel_4regions.png' if grid else 'all_subpixel.png'))
# Detailed first finder, both axes enlarged by 16; no anisotropic displacement amplification.
detail=(1557,976,1625,1048)
render(1,detail,(1088,1152)).save(OUT/'finder_subpixel_detail.png')
render(1,detail,(1088,1152),compare=True).save(OUT/'finder_before_after.png')
summary=dict(total=len(df),valid=int(df.valid.sum()),uncertain=int((~df.valid).sum()),per_qr=df.groupby('qr').agg(total=('valid','size'),valid=('valid','sum'),median_shift_px=('shift_px','median'),median_rmse=('rmse','median')).reset_index().to_dict('records'),median_shift_px=float(df.shift_px.median()),max_shift_px=float(df.shift_px.max()),median_rmse=float(df.rmse.median()),synthetic_validation=validation,runtime_seconds=time.time()-start_time,method='Local 2x2 decoded module topology; Gaussian blurred quadrant photometric model, fixed local projective axes, robust soft-L1 least-squares of x/y, blur and brightness. Fit original pixels in 9x9 and 11x11 windows; accept by residual, contrast, convergence, window stability, bound checks.',limits='Subpixel fitted coordinates, not independently measured absolute accuracy. Rounded print corners define intersection under the fitted blur model. All structural vertices retained; diagonal contacts may be ambiguous. No motion accuracy claim. Green=diagnostics passed, amber=uncertain.',source=str(ROOT/'center.bmp'))
(OUT/'checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
html='<meta charset="utf-8"><title>QR 실제 영상 경계 서브픽셀 보정</title><style>body{font-family:Malgun Gothic;background:#16191d;color:#eee;margin:30px}img{max-width:100%}a{color:#8bfa83}td,th{padding:8px;border:1px solid #555}table{border-collapse:collapse}</style><h1>QR 실제 영상 경계 서브픽셀 보정</h1>'
html+=f'<p>전체 {len(df)}점 중 진단 통과 {int(df.valid.sum())}점(연두), 불확실 {int((~df.valid).sum())}점(주황). 실제 원본 영상으로 국소 경계 모델을 맞춤. 초기 격자 투영만의 그림과 구분. 서브픽셀 수치가 독립 계측 정확도를 보증하지는 않음.</p>'
html+='<p><a href="../../ALGORITHM_RULES.md">공통 규칙</a> · <a href="refined_corners.csv">전체 좌표와 진단</a> · <a href="uncertain_corners.csv">불확실점</a> · <a href="checks.json">방법 및 검증</a></p><h2>전체 모서리 · 가로 세로 동일 5배</h2><img src="all_subpixel.png"><h2>큰 네모 상세 · 가로 세로 동일 16배</h2><img src="finder_subpixel_detail.png"><h2>초기 위치와 보정 위치 비교</h2><p>보라 십자=초기 위치, 연두=영상 적합 위치, 주황=불확실.</p><img src="finder_before_after.png">'
html+=pd.DataFrame(summary['per_qr']).to_html(index=False)
(OUT/'report.html').write_text(html,encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k in ['total','valid','uncertain','per_qr','median_shift_px','max_shift_px','median_rmse','runtime_seconds']}),flush=True)
