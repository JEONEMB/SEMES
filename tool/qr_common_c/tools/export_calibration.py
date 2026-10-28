"""Freeze Python reference/model values; runtime never imports Python.

Usage: python tools/export_calibration.py --analysis-root PATH_TO_S15P31A507
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis-root',type=Path,required=True);a=p.parse_args()
    root=a.analysis_root.resolve();module=Path(__file__).resolve().parents[1];out=module/'calibration';out.mkdir(exist_ok=True)
    sys.path[:0]=[str(root/'.analysis_packages'),str(root/'tool/offaxis')]
    import numpy as np
    import pandas as pd
    import cv2
    from plate_edges import fit_ridge,tune
    base=root/'docs/offaxis-eda';source=base/'common-background-edges';data=root/'data/SSAFY Test Image Set'
    meta=pd.read_csv(base/'ppt-method-v2/frames.csv').set_index('path');mixed=meta[meta.category.eq('mixed')].sort_index()
    refs=meta[meta.category.eq('new-reference')];nominal=mixed[['y_mm','theta_deg']].to_numpy()
    y=(mixed[['y_actual','theta_actual']].to_numpy()-refs[['y_actual','theta_actual']].mean().to_numpy())/np.array([.05,.01])
    coords=np.unique(nominal,axis=0);groups=np.array([np.flatnonzero(np.all(coords==q,axis=1))[0] for q in nominal])
    qr=pd.read_csv(base/'ppt-method-v2/qr_displacements.csv').pivot(index='path',columns='qr',values=['dx','dy'])
    edges=pd.read_csv(source/'edge_features.csv').set_index('path');cols=[e+'_'+c for e in ['upper','lower'] for c in ['position','slope']]
    edge_ref=edges.loc['기준 위치/center1.bmp',cols].to_numpy(dtype=float)
    x=np.c_[qr.loc[mixed.index,[(d,s) for s in 'LCR' for d in ['dx','dy']]].to_numpy(),edges.loc[mixed.index,cols].to_numpy()-edge_ref]
    def array(z):
        z=np.asarray(z)
        if z.ndim==0:return format(float(z),'.17g')
        return '{'+','.join(array(v) for v in z)+'}'
    def model(m):return '{'+','.join(array(m[k]) for k in ['mean','scale','ymean','coef'])+'}'
    deploy=json.loads((source/'models.json').read_text())['QR6_common_shared4']
    foldmodels=[]
    for g in range(19):
        train=groups!=g;alpha=tune(x[train],y[train],groups[train])[0];foldmodels.append(fit_ridge(x[train],y[train],alpha))
    (out/'model_data.h').write_text('/* Generated double precision models. Evaluation folds must never be selected from an unknown image. */\n#ifndef QC_MODEL_DATA_H\n#define QC_MODEL_DATA_H\n'
        +'static const qc_model qc_deployment_model='+model(deploy)+';\nstatic const qc_model qc_fold_models[19]={\n'+',\n'.join(model(m) for m in foldmodels)+'};\n#endif\n')
    specs=json.loads((base/'ppt-method-v2/reference.json').read_text());old=json.loads((base/'raw/reference.json').read_text())
    pts=pd.read_csv(base/'ppt-method-v2/points.csv.gz');refpoints=pts[pts.path.eq('기준 위치/center1.bmp')&~pts.diagonal].set_index(['qr','id'])
    rows=[]
    for q,s in enumerate(specs):
        for f in s['features']:
            if f['diagonal']:continue
            rr=refpoints.loc[(s['slot'],f['id'])]
            rows.append('{'+str(q)+','+array(f['seed'])+','+array(np.array(f['bits']).ravel())+','+array(np.array(f['basis']).ravel())+','+array([rr.x,rr.y])+'}')
    bounds=[];oldimage=cv2.imdecode(np.fromfile(data/'center.bmp',np.uint8),cv2.IMREAD_GRAYSCALE)
    for q,s in enumerate(old):
        bounds.append(s['bounds']);xx,yy,w,h=s['bounds'];cv2.imwrite(str(out/f'template_{q}.png'),oldimage[yy:yy+h,xx:xx+w])
    bp=refpoints.reset_index().groupby('qr').agg(x0=('x','min'),x1=('x','max'),y0=('y','min'),y1=('y','max'))
    qc=[(bp.x0.min()+bp.x1.max())/2,(bp.y0.min()+bp.y1.max())/2]
    (out/'reference_data.h').write_text('/* Generated from the reference image only; no test image lookup. */\n#ifndef QC_REFERENCE_DATA_H\n#define QC_REFERENCE_DATA_H\n'
        +'typedef struct {int qr;double seed[2],bits[4],basis[4],reference[2];} qc_corner_spec;\n'
        +'static const qc_corner_spec qc_corner_specs[630]={\n'+',\n'.join(rows)+'};\n'
        +'static const int qc_patch_bounds[3][4]='+array(bounds)+';\n'
        +'static const double qc_reference_qr_center[2]='+array(qc)+';\n'
        +'static const double qc_reference_edges[4]='+array(edge_ref)+';\n#endif\n')
    samples=pd.DataFrame(x,columns=['L_dx','L_dy','C_dx','C_dy','R_dx','R_dy','upper_position','upper_slope','lower_position','lower_slope'])
    samples.insert(0,'path',mixed.index);samples.insert(1,'fold',groups);samples['Y_mm']=y[:,0]*.05;samples['yaw_deg']=y[:,1]*.01
    golden=pd.read_csv(source/'coordinate_holdout.csv');golden=golden[golden.model.eq('QR6_common_shared4')].set_index('path')
    samples['python_Y_mm']=golden.loc[mixed.index,'predicted_y_mm'].to_numpy();samples['python_yaw_deg']=golden.loc[mixed.index,'predicted_yaw_deg'].to_numpy()
    samples.to_csv(module/'tests/golden_features.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(dict(model='QR6_common_shared4',features=10,training_images=57,coordinate_groups=19,
        ticks=dict(Y_mm=.05,yaw_deg=.01),image_size=[3840,2160],reference='기준 위치/center1.bmp',
        source_hashes={str(s.relative_to(root)):hashlib.sha256(s.read_bytes()).hexdigest() for s in [source/'models.json',source/'edge_features.csv',base/'ppt-method-v2/points.csv.gz']},
        scope='Dataset-specific calibration; no cross-device generalization or closed-loop convergence validated.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Exported 630 reference corners, 3 templates, 1 deployment +19 fold models, 57 validation rows.')


if __name__=='__main__':main()
