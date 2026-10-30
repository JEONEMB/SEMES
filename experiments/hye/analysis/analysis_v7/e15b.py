from judge7 import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'
cen=read_raw('center.bmp'); F=F8; qr=F.region.str.startswith('QR').values; Fq=F[qr].reset_index(drop=True)
o=np.load('e15_top.npy')
sat33=np.array([(cen[y-16:y+17,x-16:x+17]>=250).mean() for x,y in zip(Fq.x,Fq.y)])
sat17=np.array([(cen[y-8:y+9,x-8:x+9]>=250).mean() for x,y in zip(Fq.x,Fq.y)])
print('33px 창에 포화 픽셀 있는 QR 점:',int((sat33>0).sum()),'개 /',len(Fq)); print(Fq.assign(sat33=sat33)[sat33>0][['x','y','region']].assign(sat33=sat33[sat33>0].round(3)).to_string())
# 점별 모양 잡음 (평행이동 뺀 잔차) 33px vs 17px
for fn,lab in [('tr_fs8_hp16_s2.npz','33px'),('v6','17px')]:
    D,_=load(fn); D=D[:,qr]; m=AXA=='Y'; v=CMD[m]; A=np.c_[np.ones(m.sum()),v,v*v]
    R=(D[m].reshape(m.sum(),-1)-A@np.linalg.lstsq(A,D[m].reshape(m.sum(),-1),rcond=None)[0]).reshape(m.sum(),-1,2); R-=R.mean(1,keepdims=True)
    sd=np.sqrt((R**2).mean(0)).max(1); print(lab,'모양 잡음 중앙',np.median(sd).round(4),' 상위 점:',[(int(Fq.x[i]),int(Fq.y[i]),round(sd[i],3)) for i in o[:4]])
fig,axs=plt.subplots(1,4,figsize=(14,3.8))
for ax,i in zip(axs,o[:4]):
    x,y=int(Fq.x[i]),int(Fq.y[i]); ax.imshow(cen[y-40:y+41,x-40:x+41],cmap='gray',vmin=0,vmax=255,extent=(x-40,x+40,y+40,y-40))
    for h,c in [(8,'#1baf7a'),(16,'#eb6834')]: ax.add_patch(plt.Rectangle((x-h,y-h),2*h+1,2*h+1,fill=False,ec=c,lw=2))
    s=cen[y-40:y+41,x-40:x+41]>=250; yy,xx=np.where(s); ax.scatter(xx+x-40,yy+y-40,s=1,c='#e34948')
    ax.plot(x,y,'+',c='#2a78d6',ms=12,mew=2); ax.set_title(f'({x},{y}) 33px 포화 {sat33[i]*100:.0f}%',fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('큰 화살표 점: 초록=17px 창(점 고를 때 검사), 주황=33px 창(추적에 사용), 빨강=포화 픽셀(255)',x=0.01,ha='left')
fig.tight_layout(); fig.savefig('../65_v7_이상점_확인.png',dpi=120)
