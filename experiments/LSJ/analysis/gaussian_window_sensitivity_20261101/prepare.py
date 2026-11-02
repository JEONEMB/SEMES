"""Freeze the existing implementation, inputs and experiment design; no image fits."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'reference_line_methods_108_20261030'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

source = BASE / 'cpp/src/methods_points.cpp'
text = source.read_text(encoding='utf-8')
start = text.index('template <class F>')
end = text.index('std::string excluded', start)
function = text[start:end]
(HERE / 'original_nelder_mead.hpp').write_text(
    '// Extracted verbatim from methods_points.cpp; see provenance.json.\n' + function,
    encoding='utf-8')
manifest = json.loads((BASE / 'configs/inputs_manifest.json').read_text(encoding='utf-8'))
for ref in manifest['references']:
    ref['observed_sha256'] = sha(ref['path'])
    if ref['sha256'] != ref['observed_sha256']:
        raise RuntimeError('Input hash mismatch: ' + ref['reference_id'])
tracked = [source, BASE / 'configs/targets.json', BASE / 'configs/methods.json',
           BASE / 'configs/inputs_manifest.json', BASE / 'results/line_measurements.json']
tracked += sorted((BASE / 'results/intermediates').glob('*/E01/*.json'))
provenance = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'inputs': manifest,
    'source_hashes': {str(p.resolve()): sha(p) for p in tracked},
    'extracted_nm_sha256': hashlib.sha256(function.encode()).hexdigest(),
    'design': {
        'scope': 'Gaussian window-size sensitivity only; no Huber, theta model, or hypothesis test',
        'target_ids': ['upper_structure', 'lower_structure'],
        'sample_counts': list(range(3, 22, 2)),
        'diagnostic_only_counts': [3],
        'main_sample_counts': list(range(5, 22, 2)),
        'sampling_step_px': 1,
        'coarse_selection': 'Original per-column positive Sobel maximum, calculated once and frozen',
        'coarse_gate': 'Original min_peak=8; reject peak at search-band endpoints',
        'primary_initial_sigma_px': 11,
        'auxiliary_fixed_initial_sigmas_px': [5, 21],
        'initial_simplex_sigma_step': '0.5 * initial_sigma, constant across windows in each start series',
        'initial_A': 'frozen coarse maximum',
        'initial_mu': 'frozen coarse y',
        'initial_simplex_A_step': '0.2 * coarse maximum',
        'initial_simplex_mu_step_px': 1,
        'solver': 'Original C++ Nelder-Mead verbatim; max_iter=800, relative objective spread tol=1e-10',
        'legacy_gate': 'converged, finite parameters/SSE, A>0, abs(mu-coarse_y)<=2px',
        'raw_results': 'All attempted fits retained; no Huber trimming or newly invented quality cutoff',
        'primary_comparison_cohort': 'Same columns accepted by legacy gate in ALL main windows, fixed initial sigma=11',
        'initialization_audit': 'Compare fixed starts 5,11,21 at same window; never select best start to mask sensitivity',
        'acceptance_threshold': None,
        'reason': 'No approved px tolerance for window-induced mu change. Descriptive sensitivity, not pass/fail accuracy.',
        'real_ground_truth': None,
        'new_selector_methods_applied': False,
    },
}
(HERE / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
print('Verified 4 original hashes; extracted exact Nelder-Mead; recorded experiment design.')
