"""Separate experimental calibration model; fit metadata never enters predict."""
from pathlib import Path
import sys,numpy as np
H=Path(__file__).resolve().parent;PREV=H.parent/'joint_yt_generalization_20261022'
for folder in [PREV,H/'geometric',H/'conditional']:
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
from data import features
from joint_model import JointRegressor
from geometric_models import RepeatGLS,geometry_features
from conditional_model import ConditionalDecoder
from paired_model import PairedRidge
from neural_model import PairResidualNet

class LearnedSeparation:
    def __init__(self,config,points,regions):self.config=dict(config);self.points=points;self.regions=regions
    def bank(self,D):return features(dict(D=D,points=self.points,regions=self.regions))
    def fit(self,D,y,metadata):
        c=self.config;self.training_indices=metadata['index'].to_numpy();group=metadata.cv_group.to_numpy();dates=metadata.dataset.to_numpy();new=dates==20261020
        if c['method']=='baseline':self.model=JointRegressor(c['params']).fit(D,y);return self
        bank=self.bank(D)
        if c['method']=='repeat':
            self.model=RepeatGLS(**c['params']).fit(bank[c['feature']],y,group,new)
        elif c['method']=='paired':self.model=PairedRidge(c['params']).fit(bank[c['feature']],y,dates)
        elif c['method']=='neural':self.model=PairResidualNet(c['params']).fit(bank[c['feature']],y,dates,group)
        elif c['method']=='conditional':
            strata=np.where(new,'new_'+metadata.scope.astype(str),'old')
            self.model=ConditionalDecoder(c['params']).fit(bank[c['feature']],bank['raw_shift'],y,group,strata)
        return self
    def predict(self,D):
        D=np.asarray(D);D=D[None] if D.ndim==2 else D;c=self.config
        if c['method']=='baseline':return self.model.predict(D)
        bank=self.bank(D)
        if c['method']=='conditional':return self.model.predict(bank[c['feature']],bank['raw_shift'])
        return self.model.predict(bank[c['feature']])
