import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Polygon, FancyBboxPatch, Rectangle, Circle
plt.rcParams['font.family']='Noto Sans CJK KR'; plt.rcParams['axes.unicode_minus']=False
INK='#14213D'; SUB='#5B6B82'; NEAR='#2F6FED'; FAR='#E8833A'; BG='#F7F8F5'; GHOST='#B8C2CF'
fig=plt.figure(figsize=(19.2,10.8),dpi=100); fig.patch.set_facecolor(BG)
fig.text(0.5,0.935,'가까운 것은 크게, 먼 것은 작게 움직인다 — 운동 시차',ha='center',fontsize=40,fontweight='bold',color=INK)
fig.text(0.5,0.885,'카메라가 옆으로 움직이면 QR과 뒷벽이 다르게 밀리고, 제자리에서 돌면 똑같이 밀린다',ha='center',fontsize=22,color=SUB)
def cam(ax,x,y,ang=0,color=INK,alpha=1,ls='-'):
    c,s=np.cos(np.radians(ang)),np.sin(np.radians(ang)); R=lambda p:(x+p[0]*c-p[1]*s,y+p[0]*s+p[1]*c)
    body=[R(p) for p in [(-0.32,-0.25),(0.32,-0.25),(0.32,0.12),(-0.32,0.12)]]; lens=[R(p) for p in [(-0.14,0.12),(0.14,0.12),(0.2,0.3),(-0.2,0.3)]]
    ax.add_patch(Polygon(body,closed=True,fc=color if alpha==1 else 'none',ec=color,lw=2,alpha=1 if alpha==1 else .9,ls=ls)); ax.add_patch(Polygon(lens,closed=True,fc=color if alpha==1 else 'none',ec=color,lw=2,ls=ls))
    return R((0,0.3))
def scene(ax,title,mode):
    ax.set_xlim(-3.2,3.2); ax.set_ylim(-0.9,6.2); ax.axis('off')
    ax.text(0,6.0,title,ha='center',fontsize=26,fontweight='bold',color=INK)
    ax.plot([-3,3],[5.2,5.2],color=FAR,lw=6,solid_capstyle='butt'); ax.text(-3.0,5.45,'뒷벽 (멀다)',fontsize=16,color=FAR)
    ax.plot([-1.6,1.6],[2.9,2.9],color=NEAR,lw=6,solid_capstyle='butt'); ax.text(-1.6,3.15,'QR 판 (가깝다)',fontsize=16,color=NEAR)
    qr=(0.0,2.9); sc=(0.0,5.2)
    ax.add_patch(Rectangle((qr[0]-0.18,qr[1]-0.18),0.36,0.36,fc='white',ec=NEAR,lw=3)); ax.add_patch(Circle(sc,0.14,fc=FAR,ec='white',lw=2))
    if mode=='move':
        o0=cam(ax,0,0,color=GHOST,alpha=0,ls='--'); o1=cam(ax,1.1,0,color=INK)
        for o,c,ls in [(o0,GHOST,'--'),(o1,INK,'-')]:
            ax.plot([o[0],qr[0]],[o[1],qr[1]],color=NEAR if ls=='-' else GHOST,lw=2.2,ls=ls); ax.plot([o[0],sc[0]],[o[1],sc[1]],color=FAR if ls=='-' else GHOST,lw=2.2,ls=ls)
        ax.annotate('',xy=(1.0,-0.42),xytext=(0.1,-0.42),arrowprops=dict(arrowstyle='-|>',lw=3,color=INK,mutation_scale=25)); ax.text(0.55,-0.95,'옆으로 이동 (Y)',ha='center',fontsize=17,color=INK)
        a_n=np.degrees(np.arctan2(1.1,2.9-0.3)); a_f=np.degrees(np.arctan2(1.1,5.2-0.3))
        return a_n,a_f
    else:
        o0=cam(ax,0,0,ang=0,color=GHOST,alpha=0,ls='--'); o1=cam(ax,0,0,ang=-12,color=INK)
        for ang,c,ls in [(0,GHOST,'--'),(-12,INK,'-')]:
            d=np.array([np.sin(np.radians(-ang)),np.cos(np.radians(-ang))]); ax.plot([0,d[0]*5.6],[0.3,0.3+d[1]*5.6],color=c,lw=2.2,ls=ls)
        from matplotlib.patches import Arc
        ax.annotate('',xy=(0.42,2.05),xytext=(0.0,2.1),arrowprops=dict(arrowstyle='-|>',lw=3.5,color=INK,mutation_scale=26,connectionstyle='arc3,rad=-0.25'))
        ax.text(0,-0.95,'제자리에서 회전 (Θ)',ha='center',fontsize=17,color=INK)
        return 12,12
axL=fig.add_axes([0.03,0.29,0.45,0.53]); axR=fig.add_axes([0.52,0.29,0.45,0.53])
nL,fL=scene(axL,'평행이동 Y','move'); nR,fR=scene(axR,'회전 Θ','rot')
def screen(rect,dq,dw,cap,capc):
    ax=fig.add_axes(rect); ax.set_xlim(-6,6); ax.set_ylim(-1.3,1.6); ax.axis('off')
    ax.add_patch(FancyBboxPatch((-5.8,-1.1),11.6,2.5,boxstyle='round,pad=0.02,rounding_size=0.25',fc='#1B2436',ec='none'))
    ax.text(-5.5,1.05,'카메라 화면',fontsize=13,color='#B8C2CF')
    for x0,dx,col,shape,y in [(-1.8,dq,NEAR,'sq',0.15),(2.0,dw,FAR,'ci',0.15)]:
        if shape=='sq': ax.add_patch(Rectangle((x0-0.3,y-0.3),0.6,0.6,fc='none',ec=GHOST,lw=2,ls='--')); ax.add_patch(Rectangle((x0+dx-0.3,y-0.3),0.6,0.6,fc='white',ec=col,lw=3))
        else: ax.add_patch(Circle((x0,y),0.24,fc='none',ec=GHOST,lw=2,ls='--')); ax.add_patch(Circle((x0+dx,y),0.24,fc=col))
        ax.annotate('',xy=(x0+dx,y-0.62),xytext=(x0,y-0.62),arrowprops=dict(arrowstyle='-|>',lw=3.5,color=col,mutation_scale=22))
    ax.text(0,-1.62,cap,ha='center',fontsize=20,fontweight='bold',color=capc)
screen([0.06,0.08,0.39,0.20],-3.2,-1.8,'QR이 더 많이 밀림 → 차이 d 생김','#C2410C')
screen([0.55,0.08,0.39,0.20],-2.4,-2.4,'QR과 뒷벽이 똑같이 밀림 → 차이 d ≈ 0',NEAR)
fig.savefig('시차_원리_16x9.png',facecolor=BG)
