from judge7 import *
D,F=load('tr_fs8_hp16_s2.npz'); qr=F.region.str.startswith('QR').values; D=D[:,qr]; Fq=F[qr].reset_index(drop=True)
MOD={}
for a in ['Y','Θ']:
    m=AXA==a; v=CMD[m]; A=np.c_[np.ones(m.sum()),v,v*v]; co=np.linalg.lstsq(A,D[m].reshape(m.sum(),-1),rcond=None)[0]; MOD[a]=co
    MOD[a+'res']=(D[m].reshape(m.sum(),-1)-A@co).reshape(m.sum(),-1,2)
e=(MOD['Y'][1]*0.15+MOD['Y'][2]*0.0225).reshape(-1,2); s=e-e.mean(0); mag=np.hypot(*s.T)
sd=np.sqrt((MOD['Yres']**2).mean(0)).max(1)
o=np.argsort(-mag)[:8]
print(pd.DataFrame(dict(x=Fq.x[o],y=Fq.y[o],region=Fq.region[o],shape_px=mag[o].round(4),noise_sd=sd[o].round(3),sat=Fq.sat[o])).to_string())
print('전체 중앙 shape',np.median(mag).round(5),'noise',np.median(sd).round(3))
np.save('e15_top.npy',o)
