"""데이터 증폭 실험: 이동 벡터의 방향은 그대로, 길이만 키운다.
(a) 전체 k배  (b) 사진마다 판 전체 평행이동은 그대로 두고 나머지(점마다 다른 이동 = 원근 단서)만 k배  (c) 비선형: 길이^p (방향 유지)
판정: 최종 파이프라인 (QR마다 아핀 요약 + 2차 매핑 + center 0점 + 공분산 GLS) 과 점별 가중 없음 판정. 짝/홀, 명령값 라벨.
신호/잡음: Y 1mm 무늬와 같은 평균 이동 Θ 무늬의 차이를 잡음으로 나눈 값 (σ/mm)."""
import judge7 as J
from judge7 import *
D0,F=load('tr_fs8_hp16_s2.npz'); qr=F.region.str.startswith('QR').values; D0=D0[:,qr]; Fq=F[qr].reset_index(drop=True)
def amp(D,kind,k):
    if kind=='전체 k배': return D*k
    if kind=='모양 차이만 k배':
        m=D.mean(1,keepdims=True); return m+(D-m)*k
    if kind=='비선형 (길이^p)':
        L=np.linalg.norm(D,axis=2,keepdims=True); return D/np.maximum(L,1e-9)*L**k
rows=[]
for kind,ks in [('전체 k배',[1,10,100]),('모양 차이만 k배',[1,10,100]),('비선형 (길이^p)',[1,2,3])]:
    for k in ks:
        D=amp(D0,kind,k); X,cols=summarize(D,Fq,'affine')
        n,ymin,_=floor(X,NAMES)
        e1=evaluate(run(X,True,'center','raw','full'),'cmd')
        Xp=D.reshape(len(NAMES),-1); old=J.cov_est; J.cov_est=lambda R,kk: np.eye(R.shape[1])*R.var(); e2=evaluate(run(Xp,True,'center','raw','x'),'cmd'); J.cov_est=old
        rows.append(dict(증폭=kind,배율=k,신호대잡음_sigma_per_mm=n,최소Y_3sigma=ymin,최종방법_축정확도=e1['축정확도'],최종방법_틀림=e1['축틀림_Y']+e1['축틀림_Θ'],
                         점별가중없음_축정확도=e2['축정확도'],점별가중없음_틀림=e2['축틀림_Y']+e2['축틀림_Θ']))
        print(rows[-1],flush=True)
pd.DataFrame(rows).round(3).to_csv('../62_v7_데이터증폭.csv',index=False,encoding='utf-8-sig')
