"""Inference on coherent QR pixel warps; no independent corner measurement."""
from pathlib import Path
import sys,numpy as np
H=Path(__file__).resolve().parent
for p in [H.parent/'joint_yt_generalization_20261022',H.parent/'joint_yt_erroraudit_20261022/robust']:sys.path.insert(0,str(p))
from forward_model import ForwardGLS
from robust_model import RobustBaggedForward

MEASUREMENTS=['affine_g3','homography_g3','affine_g5','homography_g5']
CONFIGS=[dict(measurement=m,head=h) for m in MEASUREMENTS for h in ['linear','quadratic','balanced']]

class DirectDecoder:
    def __init__(self,config):self.config=dict(config)
    def features(self,D):
        d=np.asarray(D);d=d[None] if d.ndim==2 else d;raw=d.reshape(len(d),-1);h=d[:,:,0].mean(axis=1,keepdims=True)/100
        return np.c_[raw,raw*h]
    def fit(self,D,y,metadata):
        self.training_indices=metadata['index'].to_numpy();head=self.config['head']
        if head=='balanced':self.model=RobustBaggedForward(dict(fit_mode='new_condition_means',shrink=.01,huber=None,bags=1,seed=20261022)).fit(D,y,metadata)
        else:self.model=ForwardGLS(degree=1 if head=='linear' else 2,shrink=.01 if head=='linear' else .1).fit(self.features(D),y)
        return self
    def predict(self,D):
        if self.config['head']=='balanced':return self.model.predict(D)
        return self.model.predict(self.features(D))
