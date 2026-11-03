from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
REPO=ROOT/'qrwithbar-lanczos-src'
EXP=REPO/'experiments/hye/20261102_173432_t_six_methods'
ORIGIN=ROOT/'analysis/full_scene_origin_20261102'

def main():
    raw=pd.read_csv(EXP/'observations.csv')
    coords=pd.read_csv(ORIGIN/'coordinates_standard_v2.csv')
    d=raw.merge(coords[['path','Y_standard','T_standard','suspect','group']],on='path',validate='one_to_one')
    assert len(d)==575 and d.path.tolist()==raw.path.tolist()
    for variant in ['v2','nominal']:
        out=P/variant
        (out/'results/models').mkdir(parents=True,exist_ok=True)
        raw.to_csv(out/'original_observations.csv',index=False,float_format='%.17g')
        z=raw.copy()
        if variant=='v2':
            z['Y']=d.Y_standard;z['T']=d.T_standard
        z.to_csv(out/'observations.csv',index=False,float_format='%.17g')
    d.to_csv(P/'coordinate_manifest.csv',index=False,float_format='%.17g')
    # Preserve original pose folds BEFORE replacing coordinates. Features are unchanged.
    compare=(EXP/'tools/compare.cpp').read_text(encoding='utf-8')
    compare=compare.replace('auto rows=six::read_rows(experiment/"observations.csv");',
        'auto rows=six::read_rows(experiment/"original_observations.csv"); auto corrected=six::read_rows(experiment/"observations.csv"); if(rows.size()!=corrected.size())throw std::runtime_error("Coordinate count mismatch"); for(size_t i=0;i<rows.size();i++){if(rows[i].path!=corrected[i].path)throw std::runtime_error("Coordinate path mismatch");rows[i].y=corrected[i].y;rows[i].t=corrected[i].t;}')
    compare=compare.replace('auto ref=six::load_frame(experiment/"cache/reference.yml.gz");auto reference=qrwithbar::detail::read_gray(assets/"reference.png");','')
    start=compare.index('if(method==4){auto frame=')
    end=compare.index('predicted=p.t;',start)
    compare=compare[:start]+'p=six::predict(model,row.x,row.time);'+compare[end:]
    compare=compare.replace('for(int method=first;method<=last;method++){','for(int method=first;method<=last;method++){ if(method==4||method==5)continue;')
    (P/'train.cpp').write_text(compare,encoding='utf-8')
    combo=(EXP/'tools/combine.cpp').read_text(encoding='utf-8').replace('for(int family:{7,8})','for(int family:{7})')
    combo=combo.replace('auto rows=six::read_rows(root/"observations.csv");',
        'auto rows=six::read_rows(root/"original_observations.csv");auto corrected=six::read_rows(root/"observations.csv");if(rows.size()!=corrected.size())throw std::runtime_error("Coordinate count mismatch");for(size_t i=0;i<rows.size();i++){if(rows[i].path!=corrected[i].path)throw std::runtime_error("Coordinate path mismatch");rows[i].y=corrected[i].y;rows[i].t=corrected[i].t;}')
    # Ensemble range must follow corrected training labels.
    combo=combo.replace('ensemble.min_t=-1;ensemble.max_t=2;', 'ensemble.min_t=1e30;ensemble.max_t=-1e30;for(int i:train){ensemble.min_t=std::min(ensemble.min_t,rows[i].t);ensemble.max_t=std::max(ensemble.max_t,rows[i].t);}')
    combo=combo.replace('deployment.min_t=-1;deployment.max_t=2;', 'deployment.min_t=1e30;deployment.max_t=-1e30;for(int i:all){deployment.min_t=std::min(deployment.min_t,rows[i].t);deployment.max_t=std::max(deployment.max_t,rows[i].t);}')
    (P/'combine.cpp').write_text(combo,encoding='utf-8')
    files=[EXP/'observations.csv',EXP/'assets/best_image_ensemble.json',ORIGIN/'standard_origin_v2.json',ORIGIN/'coordinates_standard_v2.csv']
    (P/'input_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
    print('Prepared 575 rows; 571 evaluation images; original-pose folds retained; v2 fixed as supplied.')

if __name__=='__main__':main()
