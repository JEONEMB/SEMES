"""Copy previously computed coefficients/ranges, without fitting or image analysis."""
import hashlib
import json
from pathlib import Path

stage = Path(__file__).resolve().parent
root = stage.parents[1]
payload = stage / 'payload'
paths = {
    'ranges': root / 'analysis/zero_equivalence/without_group4/equivalence_results.json',
    'slopes': root / 'analysis/first_y_slopes/y_slope_results.json',
    'measurements': root / 'analysis/initial_error/measurements.json',
}
data = {key: json.loads(path.read_text(encoding='utf-8')) for key, path in paths.items()}
groups = {g['capture_group_number']: g for g in data['slopes']['groups']}
config = {
    'schema_version': 1, 'width': 3840, 'height': 2160,
    'search_margin_px': 110, 'tick_mm': 0.05,
    'target_profile': 'without_group4_intersection_of_primary_and_sensitivity',
    'target_selected_by_user_on': '2026-10-29',
    'coordinate': 'zero-based tracked ROI pixel-center x = reference_roi_x + (width-1)/2 + affine_dx',
    'target_groups': [2, 3, 5, 6, 7], 'slope_groups': [2, 3, 6, 7],
    'additional_tolerance_px': 0, 'markers': [],
}
for name in ('L', 'C', 'R'):
    target = data['ranges']['summaries'][name]['intersection_of_primary_and_sensitivity']
    assert target['nonempty'] and target['low_px'] <= target['high_px']
    x0, y0, x1, y1 = data['measurements']['regions_xyxy'][name]
    config['markers'].append({
        'name': name, 'roi_xywh': [x0, y0, x1-x0, y1-y0],
        'low_px': target['low_px'], 'high_px': target['high_px'],
        'slopes_px_per_mm': [groups[g]['markers'][name]['slope_px_per_nominal_mm'] for g in config['slope_groups']],
    })
out = payload / 'configs/y_alignment.json'
out.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
provenance = {
    'sources': {k: {'path': str(p.relative_to(root)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()},
    'config_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'reference_sha256': next(r['sha256'] for r in data['measurements']['measurements'] if r['axis'] == 'center'),
    'user_choice': '4번 제외 OLS·HAC 공통 범위 사용',
    'refit_performed': False, 'new_equivalence_analysis': False, 'second_dataset_used': False,
    'range_is_not': ['a geometric QR bounding box', 'an extra +/-0.045mm band', 'a proven future-frame accuracy guarantee'],
    'decision_rule': 'Intersect movements satisfying all 3 marker ranges under each of 4 saved Y slopes; choose smallest absolute reachable 0.05mm tick.',
    'all_slopes_treated_as_possible': 'Implementation choice, not a confidence or probability statement.',
}
directory = payload / 'experiments/LSJ/y_controller'
directory.mkdir(parents=True, exist_ok=True)
(directory/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
reference = Path(data['measurements']['data_root']) / 'center.bmp'
(stage/'reference_path.txt').write_text(str(reference), encoding='utf-8')
print('Exported 3 unchanged target intervals, 12 unchanged Y slopes, provenance; no refit.')
