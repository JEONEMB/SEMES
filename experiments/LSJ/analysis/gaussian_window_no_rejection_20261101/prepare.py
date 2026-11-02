from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_line_methods_108_20261030'
OLD=HERE.parent/'gaussian_lower_window_extension_20261101'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((BASE/'configs/inputs_manifest.json').read_text(encoding='utf-8'))
for ref in manifest['references']:
    assert sha(ref['path'])==ref['sha256'],ref['reference_id']
(HERE/'original_nelder_mead.hpp').write_bytes((OLD/'original_nelder_mead.hpp').read_bytes())
config={'maximum_sample_count':61,'column_step_px':1,'solver_max_iterations':800,'solver_relative_objective_spread_tolerance':1e-10}
(HERE/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
tracked=[BASE/'cpp/src/methods_points.cpp',BASE/'configs/methods.json',BASE/'configs/targets.json',BASE/'configs/inputs_manifest.json',OLD/'summary.json']
design={
 'scope':'Only Gaussian neighborhood-size comparison on 4 reference images, upper/lower structure; no Huber/line/theta tests.',
 'paper_comparison':'Fabijanska 2015 Fig.3/Table1 compares fixed-position fits across odd N=3..21. Report every position rather than selecting one example to decide for the whole structure.',
 'expanded_sample_counts':'N=23..61 extends the previously discussed range. This upper endpoint is a tested range, not a stability criterion or universal bound.',
 'coarse_selection':'Existing signed-positive Gy maximum in each existing search band. Earliest y resolves exact ties. All visible columns at 1px spacing.',
 'unchanged_spatial_scope':'Same x range, y band, and central occlusion masks as the previous experiment. Samples may extend beyond the coarse-search band; all are real source pixels.',
 'gradient':'Native OpenCV 3.4.15; full-image raw grayscale Sobel dy=1,dx=0,ksize=3,scale=1,border=REFLECT101; polarity=+1.',
 'model':'A exp(-(y-mu)^2/(2 sigma^2)), raw signed samples, unweighted SSE, no offset. Existing numerical abs(sigma)+1e-12 retained.',
 'initialization':'2014 predecessor III.B: A=max response of the fitted neighborhood, mu=coarse location, sigma=N. Unlike the earlier experiment, initial sigma is not fixed at 11.',
 'solver':'Existing exact Nelder-Mead source; initial simplex steps=.2*A,1,.5*N; max800 and relative objective-spread tolerance1e-10. These numerical settings are inherited implementation choices, not specified/validated by the 2015 paper. All returns retained.',
 'rejections_removed':['minimum response 8','coarse maximum at band endpoint','nonconvergence rejection','nonpositive fitted A rejection','abs(mu-coarse)>2px rejection','previous common-accepted cohort'],
 'reporting':'Every attempted fit saved, including nonfinite and nonconverged results; solver status is recorded, not used to hide rows. No extra position bounds, clipping, outlier removal, peak-count test, or residual gate.',
 'selection':'No invented px threshold, acceptance percentage, or smallest-stable-N rule. Full tables and all-position changes are inspected. A sample count will not be forced if the unfiltered results do not support it.',
 'sources':['https://www.researchgate.net/publication/283556095_Subpixel_edge_detection_in_blurry_and_noisy_images','https://www.annals-csis.org/Volume_2/pliks/136.pdf'],
 'config':config,'inputs':manifest}
(HERE/'provenance.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'design':design,'unchanged_source_hashes':{str(p.resolve()):sha(p) for p in tracked},'source_hashes':{p:sha(HERE/p) for p in ['experiment.cpp','CMakeLists.txt','original_nelder_mead.hpp','run_config.json','prepare.py']}},ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared: every visible x, N=3..61 odd, no point rejection. Source hashes verified.')
