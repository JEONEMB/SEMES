"""증폭 화살표 시각화: center 대비 현재 사진의 특징점 이동 (33px σ2 추적, QR 232점).
줄1 전체 이동 x k1, 줄2 평행이동(판 전체 평균)을 뺀 나머지 x k2, 줄3 잡음 없는 기대 무늬(학습 모델) x k2, 줄4 줄2-줄3 = 잡음 x k2."""
from judge7 import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
D,F=load('tr_fs8_hp16_s2.npz'); qr=F.region.str.startswith('QR').values; D=D[:,qr]; Fq=F[qr].reset_index(drop=True)
px,py=Fq.x.values.astype(float),Fq.y.values.astype(float)
cen=read_raw('center.bmp'); x0,x1,y0,y1=1400,2420,930,1210; bg=cen[y0:y1,x0:x1]
# 기대 무늬: 축마다 점별 d = b + s v + c v² (명령값, center 제외 학습 사진 전부), 기대값 = s v + c v²
MOD={}
for a in ['Y','Θ']:
    m=AXA==a; v=CMD[m]; A=np.c_[np.ones(m.sum()),v,v*v]; co=np.linalg.lstsq(A,D[m].reshape(m.sum(),-1),rcond=None)[0]; MOD[a]=co
def expect(a,v): co=MOD[a]; return (co[1]*v+co[2]*v*v).reshape(-1,2)
def shape(d): return d-d.mean(0,keepdims=True)
def panel(ax,d,k,col,title,sub):
    ax.imshow(bg,cmap='gray',vmin=0,vmax=255,extent=(x0,x1,y1,y0),alpha=0.55)
    ax.quiver(px,py,d[:,0]*k,d[:,1]*k,angles='xy',scale_units='xy',scale=1,color=col,width=0.0022,headwidth=4,headlength=5)
    ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title,loc='left',fontsize=10); ax.text(x0+8,y1-10,sub,fontsize=8.5,color='#0b0b0b',bbox=dict(fc='#fcfcfb',ec='none',alpha=0.85))
BL,OR='#2a78d6','#eb6834'
for tag,(fy,ft) in {'큰값':('Y+1.00.bmp','T+0.18.bmp'),'작은값':('Y+0.15.bmp','T+0.03.bmp')}.items():
    k1=40 if tag=='큰값' else 200; k2=2000 if tag=='큰값' else 2000
    fig,axs=plt.subplots(4,2,figsize=(17,11.5))
    for j,(fn,a,col) in enumerate([(fy,'Y',OR),(ft,'Θ',BL)]):
        i=NAMES.index(fn); d=D[i]; e=expect(a,CMD[i]); s=shape(d); se=shape(e); nz=s-se
        rms=lambda z: np.sqrt((z**2).sum(1).mean())
        panel(axs[0,j],d,k1,col,f'{fn}  ① 전체 이동 x{k1}',f'평균 이동 ({d[:,0].mean():+.2f}, {d[:,1].mean():+.2f}) px')
        panel(axs[1,j],s,k2,col,f'② 판 전체 평행이동을 뺀 나머지 x{k2}  (측정값 = 신호 + 잡음)',f'점마다 크기 RMS {rms(s):.4f} px')
        panel(axs[2,j],se,k2,col,f'③ 잡음 없는 기대 무늬 x{k2}  (= 신호, {a} 모델)',f'RMS {rms(se):.4f} px')
        panel(axs[3,j],nz,k2,'#e34948',f'④ ② - ③ = 잡음 x{k2}',f'RMS {rms(nz):.4f} px  (신호/잡음 {rms(se)/rms(nz):.2f})')
    # 두 사진의 기대 무늬 차이 = 구분 단서
    iy,it=NAMES.index(fy),NAMES.index(ft); dd=shape(expect('Y',CMD[iy]))-shape(expect('Θ',CMD[it]))
    fig.suptitle(f'{tag} 쌍: {fy} vs {ft} (평균 이동이 거의 같음).  Y와 Θ 기대 무늬 차이(구분 단서) RMS {np.sqrt((dd**2).sum(1).mean()):.4f} px  vs  점 하나 잡음 RMS 약 {np.sqrt(((shape(D[iy])-shape(expect("Y",CMD[iy])))**2).sum(1).mean()):.3f} px',x=0.01,ha='left',fontsize=12)
    fig.tight_layout(); fig.savefig(f'../63_v7_증폭화살표_{tag}.png',dpi=110); plt.close(fig)
# 요약 그림: 같은 크기 증폭으로 Y 무늬, Θ 무늬, 차이, 잡음
fig,axs=plt.subplots(2,2,figsize=(17,6))
v=0.15; ey,et=shape(expect('Y',v)),shape(expect('Θ',v*14.39/80.66))
i=NAMES.index('Y+0.15.bmp'); nz=shape(D[i])-shape(expect('Y',CMD[i]))
for ax,d,c,t in [(axs[0,0],ey,OR,'Y 0.15mm 기대 무늬 (평행이동 뺌) x3000'),(axs[0,1],et,BL,'같은 평균 이동의 Θ 0.027 기대 무늬 x3000'),
                 (axs[1,0],ey-et,'#1baf7a','둘의 차이 = Y/Θ를 가르는 유일한 단서 x3000'),(axs[1,1],nz,'#e34948','Y+0.15 사진 실제 잡음 x3000')]:
    r=np.sqrt((d**2).sum(1).mean()); panel(ax,d,3000,c,t,f'RMS {r:.4f} px')
fig.suptitle('왜 증폭이 소용없나: 단서(초록)와 잡음(빨강)이 같은 화살표 안에 섞여 있어서 같이 커짐',x=0.01,ha='left',fontsize=12)
fig.tight_layout(); fig.savefig('../64_v7_단서vs잡음_Y015.png',dpi=110)
