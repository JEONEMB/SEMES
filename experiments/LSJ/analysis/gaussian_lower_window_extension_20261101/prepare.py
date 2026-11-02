"""Isolated lower-edge window extension, preserving the prior experiment."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'gaussian_window_sensitivity_20261101'
BASE=HERE.parent/'reference_line_methods_108_20261030'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

manifest=json.loads((BASE/'configs/inputs_manifest.json').read_text(encoding='utf-8'))
for ref in manifest['references']:
    assert sha(ref['path'])==ref['sha256'], ref['reference_id']
src=(OLD/'experiment.cpp').read_text(encoding='utf-8')
replacements={
    'if (t["kind"] != "external") continue;':'if (t["id"] != "lower_structure") continue;',
    'const int pad=14,':'const int pad=34,',
    'for (int k=-10;k<=10;++k)':'for (int k=-30;k<=30;++k)',
    'sobel_samples_m10_p10':'sobel_samples_m30_p30',
    'for (int n=3;n<=21;n+=2) {':'for (int n=11;n<=61;n+=2) {\n                        if (n>11 && n<21) continue;',
    'samples[k+10]':'samples[k+30]',
}
for old,new in replacements.items():
    assert src.count(old)==1,old
    src=src.replace(old,new)
(HERE/'experiment.cpp').write_text(src,encoding='utf-8')
(HERE/'original_nelder_mead.hpp').write_bytes((OLD/'original_nelder_mead.hpp').read_bytes())
cmake=(OLD/'CMakeLists.txt').read_text(encoding='utf-8').replace('gaussian_window_sensitivity','gaussian_lower_window_extension')
(HERE/'CMakeLists.txt').write_text(cmake,encoding='utf-8')
tracked=[OLD/p for p in ['experiment.cpp','original_nelder_mead.hpp','summary.json','paired_arrays.npz','profiles.json','provenance.json']]
tracked += [BASE/p for p in ['cpp/src/methods_points.cpp','configs/methods.json','configs/targets.json','configs/inputs_manifest.json']]
design={
    'scope':'Lower structure only. Gaussian window sensitivity; no Huber, lines, physical theta validation or production settings changes.',
    'sample_counts':[11]+list(range(21,62,2)),
    'overlap_counts_for_parity':[11,21],
    'new_counts':list(range(23,62,2)),
    'half_widths_px':list(range(11,31)),
    'grid_rationale':'Exploratory extension of the existing 1px-spaced odd window grid to +/-30px (61 samples); this upper bound is not an established optimal window or accuracy requirement.',
    'inputs':manifest,
    'coarse_selection':'Same original positive Sobel maximum within the same band, unchanged once per image and column.',
    'fixed_initial_sigmas_px':[11,5,21],
    'primary_initial_sigma_px':11,
    'solver':'Exact old Nelder-Mead, 800 iterations, objective-spread tolerance 1e-10.',
    'legacy_gate':'Same convergence, finite, A>0 and abs(mu-coarse_y)<=2px. An existing diagnostic filter, not a validated accuracy threshold.',
    'comparison':'Report all 6104 columns and failures. Preserve prior 2176-column common cohort as a named cohort; disclose new exclusions. Also report a fixed cohort accepted throughout the extended range.',
    'acceptance_threshold':None,
    'new_boundary_selectors_applied':False,
    'real_ground_truth':None,
    'source_replacements':replacements,
}
(HERE/'provenance.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
    'design':design,'unchanged_source_hashes':{str(p.resolve()):sha(p) for p in tracked},
    'generated_source_hashes':{p:sha(HERE/p) for p in ['experiment.cpp','original_nelder_mead.hpp','CMakeLists.txt']}},ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared lower-only extension: N=11,21,23..61; 4 source image hashes verified.')
