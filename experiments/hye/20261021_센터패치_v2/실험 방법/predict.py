"""Fixed-center experimental v2. No motion labels or trial ID at inference."""
from run import HERE,Z,SpreadPairExtractor,spatial,np,json,Path
import argparse,hashlib

def predict(center,current):
    expected=hashlib.sha256((Z.ROOT/'center.bmp').read_bytes()).hexdigest()
    if hashlib.sha256(Path(center).read_bytes()).hexdigest()!=expected:
        raise ValueError('This calibration requires the saved center; a new installation needs feature initialization and calibration validation.')
    ext=SpreadPairExtractor(center,str(HERE/'features.npz'))
    r=ext.extract(current);a=np.load(HERE/'tracks.npz');d=r['D'].astype(float)-a['D'][0]
    q=r['quality'];w=np.maximum(1/(1+(q[:,0]/6)**2+(q[:,4]/.15)**2)*np.where(r['valid'],1,.05),.001)
    X=spatial(r['points'],d[None],w[None])[0]
    params=json.loads((HERE/'v2_patch630_quality_model.json').read_text());W=np.array(params['W']);choices=[]
    for key,(axis,b,s,c) in params['branches'].items():
        z=W@(X-np.array(b));s=W@np.array(s);c=W@np.array(c);v=float(z@s/(s@s))
        for _ in range(8):
            J=s+2*v*c;v+=float(J@(z-v*s-v*v*c)/max(J@J,1e-20))
        choices.append(dict(axis=axis,value=v,cost=float(np.sum((z-v*s-v*v*c)**2))))
    choices.sort(key=lambda x:x['cost']);best=choices[0]
    return dict(**best,Y=best['value'] if best['axis']=='Y' else 0.,T=best['value'] if best['axis']=='T' else 0.,valid_points=int(r['valid'].sum()),candidates=choices,actual_error=None,version='v2_patch630_quality_experimental')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--center',required=True);ap.add_argument('--current',required=True);ap.add_argument('--out');a=ap.parse_args()
    result=predict(a.center,a.current);text=json.dumps(result,indent=2)
    if a.out:Path(a.out).write_text(text)
    print(text)
