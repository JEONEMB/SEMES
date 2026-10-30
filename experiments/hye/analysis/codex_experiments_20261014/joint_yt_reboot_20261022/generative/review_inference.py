"""Read-only review of the final byte-only API on cached DEVELOPMENT examples."""
from pathlib import Path
import hashlib,importlib.util,inspect,json,sys
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits

H=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(H))
from predict import Predictor

def main():
    spec=importlib.util.spec_from_file_location('_byte_audit_guard',H/'generative/raw_parallax_audit/audit.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    manifest=pd.read_csv(H/'parallax/manifest.csv')
    predictions=pd.read_csv(H/'parallax_predictions.csv')
    rows=[]
    for method in ['external','qr_external']:
        model=Predictor(method)
        assert list(inspect.signature(model.predict_bytes).parameters)==['raw']
        for index in [0,281,340]:
            raw=Path(manifest.iloc[index].path).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==manifest.iloc[index].sha256
            with guard.prohibit_python_reads():
                result=model.predict_bytes(raw)
            expected=predictions[(predictions.method==method)&(predictions.protocol=='training_reapplication')&(predictions['index']==index)][['predY','predT']].to_numpy()[0]
            delta=float(np.max(abs(np.array([result['Y'],result['T']])-expected)))
            assert delta<1e-8
            rows.append(dict(method=method,index=index,maximum_saved_prediction_difference=delta,no_filesystem_reads_during_inference=True))
        rejected=False
        try:model.predict_bytes(b'not an image')
        except ValueError:rejected=True
        assert rejected
    result=dict(status='passed',purpose='Verify inference API; samples are development resubstitution, not independent accuracy evidence',current_input_signature='predict_bytes(raw)',allowed_fixed_inputs='Previously loaded model, fixed original center image and fixed extractor protocol',labels_and_filename_supplied_to_predict=False,cases=rows,invalid_image_rejected_both_methods=True,source_sha256={name:hashlib.sha256((H/name).read_bytes()).hexdigest() for name in ['predict.py','parallax/extractor.py','direct/extractor.py','direct_models.py']})
    (H/'generative/inference_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    report=json.loads((H/'review.json').read_text(encoding='utf-8'))
    report['raw_inference_api_review']=result
    report['reviewed_source_sha256']['predict.py']=result['source_sha256']['predict.py']
    (H/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
