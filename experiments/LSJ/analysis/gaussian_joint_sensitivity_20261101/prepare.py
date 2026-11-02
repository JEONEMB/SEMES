from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'gaussian_window_no_rejection_20261101'
BASE=HERE.parent/'reference_line_methods_108_20261030'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest=json.loads((BASE/'configs/inputs_manifest.json').read_text(encoding='utf-8'))
assert all(sha(r['path'])==r['sha256'] for r in manifest['references'])
old_hashes=json.loads((OLD/'output_hashes.json').read_text())
assert sha(OLD/'profiles.jsonl')==old_hashes['profiles.jsonl']
assert sha(OLD/'arrays.npz')==old_hashes['arrays.npz']
(HERE/'original_nelder_mead.hpp').write_bytes((OLD/'original_nelder_mead.hpp').read_bytes())
tracked=[BASE/'cpp/src/methods_points.cpp',BASE/'cpp/src/effective_settings.cpp',BASE/'configs/methods.json',
         BASE/'configs/targets.json',BASE/'configs/inputs_manifest.json',OLD/'profiles.jsonl',OLD/'arrays.npz',OLD/'report.html']
design={
 'scope':'Joint sensitivity checks of Gaussian sample count (2), initial sigma (3), numerical stopping (4). No Huber, line fitting, theta calibration, significance tests or production setting selection.',
 'inputs':'Exactly the previously verified 23056 visible-column profiles from four reference images, horizontal step 1 pixel. No new coarse-edge selection or image preprocessing.',
 'sample_counts':list(range(3,62,2)),
 'sample_count_basis':'Paper 2015 tests odd N=3..21; 23..61 is the already discussed extension. Neither range boundary is an acceptance rule.',
 'fixed_initialization_arms':{
   'sigma':[5,11,21],
   'basis':'Reuse the three previously compared initial widths as an explicit exploratory range, not a paper-prescribed optimum or exhaustive initialization test.',
   'A':'coarse_response, fixed across all N and sigma arms at a given location',
   'mu':'coarse_y, fixed across all N and sigma arms',
   'simplex_steps':['0.2*A',1,5.5],
   'simplex_basis':'Existing 11-sample implementation step sizes held fixed to avoid changing search size when changing N or initial sigma. These inherited numerical choices are not validated by the paper.',
   'limits':'Only initial sigma sensitivity is varied; this is not an exhaustive test of all A/mu/simplex initializations.'},
 'paper_reproduction_arm':{'A':'max of fitted neighborhood','mu':'coarse_y','sigma':'N','simplex_steps':['0.2*A',1,'0.5*N'],
   'role':'Reproduce previous unfiltered output and compare its numerical settings. Not pooled into fixed-start window/initial-sigma effects.'},
 'iteration_caps':[800,1600,3200,6400],
 'iteration_basis':'Doubling the inherited 800 cap is an exploratory numerical sensitivity sequence. 6400 is a finite experiment budget, not proof that every solve is sufficient.',
 'legacy_stopping':'Original objective-only: max f-min f <= 1e-10*(abs(best f)+1e-30), evaluated before the iteration cap.',
 'dual_stopping':{
   'tolerances':[1e-6,1e-8,1e-10],
   'cap':6400,
   'function_test':'max f-min f <= tol*(abs(best f)+1e-30)',
   'parameter_test':'max over vertices and A,mu,sigma of abs(q_vertex-q_best)/(1+abs(q_best-origin)) <= tol; origin=(0, coarse_y, 0)',
   'reason':'MathWorks fminsearch documents checking both function and parameter termination. This experiment adds that principle to the existing solver; the scaling and tolerance grid here are explicitly our experiment design, not an exact MATLAB implementation or Fabijanska prescription.',
   'tolerance_basis':'Logarithmically spaced probes ending at the inherited 1e-10, not px thresholds or task-accuracy acceptance limits.'},
 'objective':'Same unweighted SSE, A*exp(-(y-mu)^2/(2*(abs(sigma)+1e-12)^2)); no offset, no position/sigma/amplitude bounds.',
 'rejection_policy':'None. Keep weak, endpoint, nonconverged, nonpositive-amplitude, large-shift, out-of-image and nonfinite returns. Nonfinite summary statistics are marked unavailable, never silently removed.',
 'checkpoint_implementation':'One identical simplex trajectory per profile,N,initialization is continued to the strictest pending stopping condition or 6400 steps; all seven conditions retain their own first stop/cap return. Continuing a trajectory does not mutate an already recorded result.',
 'reporting':'Raw binary64 results for every combination, counts, medians and maxima over all locations. No good-case selection, no best-start merging, no acceptance cutoff. SSE is compared only for the same input sample set.',
 'sources':[
   'https://www.researchgate.net/publication/283556095_Subpixel_edge_detection_in_blurry_and_noisy_images',
   'https://www.annals-csis.org/Volume_2/pliks/136.pdf',
   'https://www.mathworks.com/help/matlab/ref/fminsearch.html',
   'https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd142.htm']}
(HERE/'provenance.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'design':design,'inputs':manifest,
 'unchanged_files':{str(p.resolve()):sha(p) for p in tracked},
 'experiment_sources':{p:sha(HERE/p) for p in ['experiment.cpp','CMakeLists.txt','prepare.py','original_nelder_mead.hpp']}},ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared; original image/cache hashes verified. Free disk bytes:',shutil.disk_usage(HERE).free)
