"""Check serialized model with independently extracted image-pair inputs."""
import sys,json,pickle
import numpy as np
from pathlib import Path
from experiment import HERE
sys.path.insert(0,str(HERE.parent/'measurement'))
from measure import ROOT as IMAGE_ROOT
from extract_pair import PairExtractor
from spread import SpreadPairExtractor
from unified import independent_data

def main():
    bundle=pickle.loads((HERE/'deploy_model.pkl').read_bytes());name=bundle['measurement'];model=bundle['model']
    features=HERE.parent/bundle['features_file']
    if name.startswith('gn17_spread'):extractor=SpreadPairExtractor(IMAGE_ROOT/'center.bmp',features=features)
    else:extractor=PairExtractor(IMAGE_ROOT/'center.bmp',method=name,features=features,include_screws=False)
    data=independent_data(name);rows=[]
    for filename in ['T+0.01.bmp','T-0.05.bmp','Y+0.05.bmp','Y-0.10.bmp']:
        current=IMAGE_ROOT/('Theta' if filename.startswith('T') else 'Y')/filename
        pair=extractor.extract(current);idx=list(data['names']).index(filename)
        # Required coordinates must appear in the identical trained order.
        assert np.allclose(pair['points'],bundle['reference_points'][:len(pair['points'])],atol=1e-4)
        p=model.predict(pair['D'][None]);pc=model.predict(data['D'][idx:idx+1])
        qr=np.arange(len(pair['points']));difference=float(np.max(abs(pair['D']-data['D'][idx,qr])))
        rows.append(dict(file=filename,pred_axis=str(p['pred_axis'][0]),pred_value=float(p['pred_value'][0]),batch_pred_axis=str(pc['pred_axis'][0]),batch_pred_value=float(pc['pred_value'][0]),max_pair_batch_feature_difference_px=difference,runtime_seconds=float(pair['runtime_seconds']),valid_fraction=float(pair['valid'].mean())))
        assert p['pred_axis'][0]==pc['pred_axis'][0]
        assert abs(p['pred_value'][0]-pc['pred_value'][0])<1e-4
    out=dict(measurement=name,all_prediction_equivalent=True,cases=rows)
    (HERE/'deployment_smoke.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
