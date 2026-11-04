"""Raw-intensity learning baselines, no corner coordinates in fit or inference."""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import torch
from torch import nn

class PixelPLS:
    def __init__(self,components):self.components=components
    def features(self,pixels,shift):return np.c_[pixels[:,:,::2,::2].reshape(len(pixels),-1),shift.reshape(len(shift),-1)/100]
    def fit(self,pixels,shift,y,metadata):
        x=self.features(pixels,shift);self.model=PLSRegression(n_components=self.components,scale=True,max_iter=1000,tol=1e-7).fit(x,y);self.training_indices=metadata['index'].to_numpy();return self
    def predict(self,pixels,shift):return self.model.predict(self.features(pixels,shift))

class SmallConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,8,5,stride=2,padding=2),nn.SiLU(),nn.Conv2d(8,12,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((6,6)))
        self.head=nn.Sequential(nn.Linear(3*12*6*6+6,32),nn.SiLU(),nn.Linear(32,2))
        self.skip=nn.Linear(6,2)
    def forward(self,pixels,shift):
        n=len(pixels);h=self.conv(pixels.reshape(n*3,1,48,48)).reshape(n,-1)
        return self.skip(shift)+self.head(torch.cat([h,shift],dim=1))

class PixelCNN:
    def __init__(self,epochs=350):self.epochs=epochs
    def fit(self,pixels,shift,y,metadata):
        torch.manual_seed(20261022);np.random.seed(20261022);torch.set_num_threads(1)
        device='cuda' if torch.cuda.is_available() else 'cpu';net=SmallConv().to(device)
        self.pixel_scale=max(float(np.std(pixels)),.02)
        x=torch.tensor(pixels[:,:,::2,::2]/self.pixel_scale,dtype=torch.float32,device=device)
        self.shift_mean=shift.reshape(len(shift),-1).mean(0);self.shift_scale=np.maximum(shift.reshape(len(shift),-1).std(0),1.)
        s=torch.tensor((shift.reshape(len(shift),-1)-self.shift_mean)/self.shift_scale,dtype=torch.float32,device=device);yy=torch.tensor(y,dtype=torch.float32,device=device)
        # Fixed weighting, train metadata only. Each date has equal initial mass.
        dates=metadata.dataset.to_numpy();w=np.array([1/np.sum(dates==date) for date in dates]);w=w/w.mean();w=torch.tensor(w,dtype=torch.float32,device=device)[:,None]
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);losses=[]
        for epoch in range(self.epochs):
            opt.zero_grad();p=net(x,s);loss=(w*(p-yy).square()).mean();loss.backward();opt.step()
            if epoch%100==0:losses.append(float(loss.detach().cpu()))
        self.state={k:v.detach().cpu() for k,v in net.state_dict().items()};self.losses=losses;self.training_indices=metadata['index'].to_numpy();self.device_used=device
        del x,s,yy,net
        if device=='cuda':torch.cuda.empty_cache()
        return self
    def predict(self,pixels,shift):
        net=SmallConv();net.load_state_dict(self.state);net.eval()
        with torch.no_grad():
            p=net(torch.tensor(pixels[:,:,::2,::2]/self.pixel_scale,dtype=torch.float32),torch.tensor((shift.reshape(len(shift),-1)-self.shift_mean)/self.shift_scale,dtype=torch.float32)).numpy()
        return p.astype(float)
