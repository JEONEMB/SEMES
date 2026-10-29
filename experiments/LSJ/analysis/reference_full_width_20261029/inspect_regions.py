import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
root=here.parents[1]
refs=json.loads((root/'analysis/reference_slope_check_20261029/reference_slope_results.json').read_text(encoding='utf-8'))['rows']
image=cv2.imdecode(np.fromfile(refs[0]['path'],dtype=np.uint8),0)
fig,axes=plt.subplots(2,1,figsize=(18,5),layout='constrained')
for ax,(lo,hi,title) in zip(axes,[(670,795,'UPPER protruding rail: original image coordinates'),(1800,1940,'LOWER protruding rail: original image coordinates')]):
    ax.imshow(image[lo:hi],cmap='gray',vmin=0,vmax=255,extent=(0,image.shape[1],hi,lo),aspect='auto')
    ax.set_title(title); ax.set_xlabel('x (original pixels)'); ax.set_ylabel('y (original pixels)')
    ax.set_xticks(np.arange(0,3841,200)); ax.grid(axis='x',alpha=.2)
fig.savefig(here/'reference_bands_inspection.png',dpi=150)
plt.close(fig)
prior=json.loads((root/'analysis/theta_line_hypothesis/line_measurements.json').read_text(encoding='utf-8'))
for r in prior['measurements']:
    if r['dataset']=='first' and r['file']=='center.bmp':
        for name in ['rail_UL_edge','rail_UR_edge','rail_BL_edge','rail_BR_edge']:
            print(name,r['features'][name]['endpoints_xy'])
