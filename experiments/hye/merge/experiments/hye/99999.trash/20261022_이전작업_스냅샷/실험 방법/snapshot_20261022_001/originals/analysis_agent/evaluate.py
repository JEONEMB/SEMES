from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

out = Path(__file__).parent
d = pd.read_csv(out / 'qr_measurements.csv')
d = d[d.axis != 'center'].copy()
for i in range(3):
    w, h = 180, 192
    d[f'x{i}'] = d[f'q{i}_dx'] + (d[f'q{i}_a00'] - 1)*w/2 + d[f'q{i}_a01']*h/2
    d[f'y{i}'] = d[f'q{i}_dy'] + d[f'q{i}_a10']*w/2 + (d[f'q{i}_a11'] - 1)*h/2
d['shift_x'] = d[[f'x{i}' for i in range(3)]].mean(axis=1)
d['outer_spacing_change'] = d.x2-d.x0
d['middle_bend_x'] = d.x1-(d.x0+d.x2)/2
d['outer_vertical_change'] = d.y2-d.y0
d['scale_gradient'] = d.q2_a00-d.q0_a00
d['shear_gradient'] = d.q2_a01-d.q0_a01

features = ['shift_x','middle_bend_x']
# Interleave magnitudes to prevent adjacent settings from being in the same fold.
d['step'] = np.where(d.axis == 'T', np.rint(np.abs(d.value)*100), np.rint(np.abs(d.value)*20)).astype(int)
test = (d.step % 2 == 0)
targets = np.column_stack([np.where(d.axis == 'T',d.value,0), np.where(d.axis == 'Y',d.value,0)])
X = d[features].to_numpy()
model = make_pipeline(StandardScaler(), Ridge(alpha=0))
model.fit(X[~test],targets[~test])
pred = model.predict(X[test])
truth = targets[test]
evals = d[test].copy()
evals['theta_true'] = truth[:,0]; evals['y_true'] = truth[:,1]
evals['theta_pred'] = pred[:,0]; evals['y_pred'] = pred[:,1]
evals.to_csv(out/'heldout_predictions.csv',index=False)
print('all test n',len(evals),'T MAE',mean_absolute_error(truth[:,0],pred[:,0]),'Y MAE',mean_absolute_error(truth[:,1],pred[:,1]))
for maxshift in [2,5,10,15,30,80]:
    q = evals[np.abs(evals.shift_x)<=maxshift]
    if len(q):
        print('maxshift',maxshift,'n',len(q),'T MAE',mean_absolute_error(q.theta_true,q.theta_pred),'Y MAE',mean_absolute_error(q.y_true,q.y_pred))
for axis in ['T','Y']:
    q = d[d.axis==axis]
    for f in features:
        coef = np.polyfit(q.value,q[f],1)
        residual = q[f] - np.polyval(coef,q.value)
        print(axis,f,'slope',coef[0],'intercept',coef[1],'resid_sd',residual.std())

fig,axs=plt.subplots(2,2,figsize=(12,8))
colors={'T':'#d95f02','Y':'#1b9e77'}
for axis in ['T','Y']:
    q=d[d.axis==axis]
    axs[0,0].scatter(q.shift_x,q.middle_bend_x,s=9,c=colors[axis],label=axis)
    axs[0,1].scatter(q.shift_x,q.outer_vertical_change,s=9,c=colors[axis],label=axis)
    axs[1,0].scatter(q.shift_x,q.scale_gradient,s=9,c=colors[axis],label=axis)
axs[1,1].scatter(evals.theta_true,evals.theta_pred,s=10,c=[colors[a] for a in evals.axis])
axs[0,0].set(xlabel='mean QR horizontal shift (px)',ylabel='middle QR bend (px)')
axs[0,1].set(xlabel='mean QR horizontal shift (px)',ylabel='outer QR vertical difference (px)')
axs[1,0].set(xlabel='mean QR horizontal shift (px)',ylabel='outer QR scale gradient')
axs[1,1].set(xlabel='true theta',ylabel='predicted theta')
for ax in axs.flat: ax.grid(alpha=.2)
axs[0,0].legend()
fig.tight_layout(); fig.savefig(out/'qr_geometry.png',dpi=160)
