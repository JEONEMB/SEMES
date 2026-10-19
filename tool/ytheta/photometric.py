"""Optional integrated pixel residual for calibrated physical Y/yaw models.
Uses original RGB->gray pixels and cubic interpolation. No OpenCV Python dependency.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from ytheta_model import physical,gauss_newton,estimate

def gray(path):
    rgb=np.asarray(Image.open(path).convert('RGB'),float)
    return rgb@np.array([.299,.587,.114])

def cubic_weights(t):
    return np.stack([-.5*t+t*t-.5*t**3,1-2.5*t*t+1.5*t**3,
                     .5*t+2*t*t-1.5*t**3,-.5*t*t+.5*t**3],axis=-1)

def sample(image,points):
    points=np.asarray(points,float);base=np.floor(points).astype(int);frac=points-base
    if np.any(base<1) or np.any(base[:,0]+2>=image.shape[1]) or np.any(base[:,1]+2>=image.shape[0]):
        raise ValueError('warp outside sampling bounds')
    wx=cubic_weights(frac[:,0]);wy=cubic_weights(frac[:,1]);values=np.zeros(len(points))
    for j in range(4):
        for i in range(4):values+=wx[:,i]*wy[:,j]*image[base[:,1]+j-1,base[:,0]+i-1]
    return values

def register(model,reference,current,session=None,initial=None):
    if model['kind']!='physical':raise ValueError('physical model required')
    if reference.shape!=current.shape:raise ValueError('image size mismatch')
    grid=np.asarray(model['reference']);locations=[]
    for qr in np.split(grid,3):
        low=np.ceil(qr.min(0)).astype(int);high=np.floor(qr.max(0)).astype(int)
        xx,yy=np.meshgrid(np.arange(low[0],high[0],3),np.arange(low[1],high[1],3))
        locations.append(np.c_[xx.ravel(),yy.ravel()])
    q=np.concatenate(locations);iy=q[:,1];ix=q[:,0]
    gradient=np.hypot(reference[iy,ix+1]-reference[iy,ix-1],reference[iy+1,ix]-reference[iy-1,ix])
    q=q[gradient>15];template=reference[q[:,1],q[:,0]]
    if len(q)<100:raise ValueError('insufficient image gradients')
    offset=np.zeros(2) if session is None else np.asarray(model['offsets'][session])
    if offset.shape!=(2,):raise ValueError('pixel model requires global 2D session offset')
    def residual(pose):
        coords=physical(q,pose,model['parameters'],model['fx'],model['fy'])+offset
        values=sample(current,coords)
        # Variable projection: eliminate linear brightness and contrast at every pose.
        A=np.c_[template,np.ones(len(template))]
        photometry=np.linalg.lstsq(A,values,rcond=None)[0]
        return (values-A@photometry)/255.
    initial=np.zeros(2) if initial is None else np.asarray(initial,float)
    from ytheta_model import physical_pose_jacobian
    J=physical_pose_jacobian(q,initial,model['parameters'],model['fx'],model['fy'])
    ratio=J[:,0]@J[:,1]/(J[:,0]@J[:,0])
    guesses=[gauss_newton(residual,initial+[-ratio*t,t],[-2,-1],[2,1],iterations=40) for t in [-.05,0,.05]]
    best=min(guesses,key=lambda x:x['rms'])
    return {'y_mm':float(best['x'][0]),'theta_deg':float(best['x'][1]),
            'normalized_intensity_rms':best['rms'],'converged':best['converged'],
            'pixels':len(q),'absolute_machine_zero_verified':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('reference');parser.add_argument('current');parser.add_argument('--session');parser.add_argument('--initial',nargs=2,type=float,metavar=('Y_MM','THETA_DEG'));args=parser.parse_args()
    print(json.dumps(register(json.load(open(args.model,encoding='utf-8')),gray(args.reference),gray(args.current),args.session,args.initial),indent=2))
