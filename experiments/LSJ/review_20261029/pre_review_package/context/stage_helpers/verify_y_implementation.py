"""Native implementation checks only; no new fitting or dataset accuracy evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess

stage = Path(__file__).resolve().parent
payload = stage / 'payload'
root = stage.parents[1]
validation = stage / 'validation'
validation.mkdir(exist_ok=True)
cmake_bin = Path('C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin')
executables = stage / 'build/Release'
checks = []

def run(name, args, cwd, expected=0):
    p = subprocess.run([str(a) for a in args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (validation/f'{name}.log').write_bytes(p.stdout)
    checks.append({'name': name, 'exit_code': p.returncode, 'expected_exit_code': expected,
                   'passed': p.returncode == expected, 'command': [str(a) for a in args]})
    assert p.returncode == expected, (name, p.returncode, p.stdout.decode('utf-8', errors='replace'))

run('ctest', [cmake_bin/'ctest.exe', '--test-dir', stage/'build', '-C', 'Release', '--output-on-failure'], root)
run('native_reference_and_adapter', [executables/'y_alignment_test.exe', 'configs/theta_alignment.json',
                                    'configs/y_alignment.json', '../reference_path.txt'], payload)
reference = Path((stage/'reference_path.txt').read_text(encoding='utf-8'))
inputs = [reference, reference.parent/'Theta/T+0.10.bmp', reference.parent/'Theta/T-0.10.bmp']
for path in inputs:
    assert path.is_file(), path
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
inputs.append(validation/'intentional_missing_image_for_error_contract.bmp')
assert not inputs[-1].exists()
(validation/'inputs.txt').write_text('\n'.join(str(p) for p in inputs)+'\n', encoding='utf-8')
run('combined_cli', [executables/'main.exe', '--reference', reference, '--config', 'configs/theta_alignment.json',
                    '--y-config', 'configs/y_alignment.json', '--input-list', validation/'inputs.txt',
                    '--output', validation/'combined_cli.json'], payload, expected=1)
def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f'Duplicate JSON key {key}'
        result[key] = value
    return result
output = json.loads((validation/'combined_cli.json').read_text(encoding='utf-8'), object_pairs_hook=unique_keys)
rows = output['results']
assert len(rows) == 4 and rows[0]['status'] == 'aligned' and rows[0]['command_y_mm'] == 0 and rows[0]['command_theta_deg'] == 0
for r in rows[1:3]:
    assert r['valid'] == 1 and r['status'].startswith('move_theta') and r['y_evaluated'] == 0 and r['command_y_mm'] == 0
assert rows[3]['valid'] == 0 and rows[3]['status'] == 'measurement_failed'
for p in inputs[:-1]:
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before[str(p)]

config = json.loads((payload/'configs/y_alignment.json').read_text(encoding='utf-8'))
ranges = json.loads((root/'analysis/zero_equivalence/without_group4/equivalence_results.json').read_text(encoding='utf-8'))
slopes = json.loads((root/'analysis/first_y_slopes/y_slope_results.json').read_text(encoding='utf-8'))
groups = {r['capture_group_number']: r for r in slopes['groups']}
for m in config['markers']:
    saved = ranges['summaries'][m['name']]['intersection_of_primary_and_sensitivity']
    assert m['low_px'] == saved['low_px'] and m['high_px'] == saved['high_px']
    assert m['slopes_px_per_mm'] == [groups[g]['markers'][m['name']]['slope_px_per_nominal_mm'] for g in [2,3,6,7]]
provenance = json.loads((payload/'experiments/LSJ/y_controller/provenance.json').read_text(encoding='utf-8'))
for info in provenance['sources'].values():
    assert hashlib.sha256((root/info['path']).read_bytes()).hexdigest() == info['sha256']

summary = {
    'date_kst': '2026-10-29', 'opencv_native_version': output['opencv_version'],
    'build': 'MSVC v141 / x64 Release / CMake Visual Studio 18 2026',
    'ctest_suites_passed': 2, 'reference_and_known_5px_translation_integration_passed': True,
    'skeleton_signature_preserved': 'Alignment_t alignment(cv::Mat &src)',
    'y_ranges_and_12_slopes_equal_saved_values': True, 'source_hashes_unchanged': True,
    'raw_image_hashes_unchanged': True,
    'cli_checks': [{'file': Path(r['file']).name, 'status': r['status'], 'valid': r['valid']} for r in rows],
    'execution_counter': 'Unit-tested: count acknowledged nonzero moves, block 21st move, allow final observation after 20th.',
    'not_tested': ['physical robot accuracy', 'whole-dataset correction performance', 'physical closed-loop convergence within 20 moves'],
    'no_new_regression_or_equivalence_fit': True,
    'source_file_sha256': {str(p.relative_to(payload)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest()
                           for folder in ['src','include','tests','configs'] for p in (payload/folder).glob('*') if p.is_file()},
}
(payload/'experiments/LSJ/y_controller/validation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(validation/'commands.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('PASS: two CTest suites, native reference/translation/adapter, CLI routing/error JSON, unchanged calibration/ranges/raw images.')
