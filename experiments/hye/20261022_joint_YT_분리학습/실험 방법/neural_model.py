"""Small residual Y/T decoder with supervised matched-axis pair losses.
Inspired by grouped-observation supervision, not a reproduction of a VAE paper.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from paired_model import graph_operator

class PairResidualNet:
    def __init__(self,config):self.config=dict(config)
    def fit(self,X,y,dates,groups):
        from forward_model import ForwardGLS
        c=self.config;torch.manual_seed(20261022);np.random.seed(20261022)
        self.base=ForwardGLS(1,.01).fit(X,y);bp=self.base.predict(X)
        if c['crossfit']:
            for tr,va in GroupKFold(3).split(X,groups=groups):bp[va]=ForwardGLS(1,.01).fit(X[tr],y[tr]).predict(X[va])
        self.mean=X.mean(0);self.std=np.maximum(X.std(0),1e-6)
        self.pca=PCA(n_components=min(c['components'],len(X)-2),whiten=True,svd_solver='full').fit((X-self.mean)/self.std)
        Z=np.c_[self.pca.transform((X-self.mean)/self.std),bp]
        self.zscale=np.maximum(Z.std(0),1e-5);Z=Z/self.zscale
        device='cuda' if torch.cuda.is_available() else 'cpu';self.device=device
        net=torch.nn.Sequential(torch.nn.Linear(Z.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,2)).to(device)
        torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
        z=torch.tensor(Z,dtype=torch.float32,device=device);b=torch.tensor(bp,dtype=torch.float32,device=device);target=torch.tensor(y,dtype=torch.float32,device=device)
        ly=torch.tensor(graph_operator(y,dates,'sameT'),dtype=torch.float32,device=device);lt=torch.tensor(graph_operator(y,dates,'sameY'),dtype=torch.float32,device=device)
        weight=torch.tensor(np.where(np.asarray(dates)==20261020,3.,1.),dtype=torch.float32,device=device)[:,None]
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.05)
        losses=[]
        for epoch in range(800):
            opt.zero_grad();delta=.2*net(z);res=b+delta-target
            loss=(weight*res.square()).mean()+.05*delta.square().mean()
            if c['pair_weight']:
                loss=loss+c['pair_weight']*(res[:,0]@(ly@res[:,0])+res[:,1]@(lt@res[:,1]))/(2*len(y))
            loss.backward();opt.step()
            if epoch%200==0:losses.append(float(loss.detach().cpu()))
        self.weights={k:v.detach().cpu().numpy() for k,v in net.state_dict().items()};self.losses=losses
        return self
    def predict(self,X):
        bp=self.base.predict(X);z=np.c_[self.pca.transform((X-self.mean)/self.std),bp]/self.zscale
        w=self.weights;hidden=np.tanh(z@w['0.weight'].T+w['0.bias']);delta=.2*(hidden@w['2.weight'].T+w['2.bias'])
        return bp+delta
