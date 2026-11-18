"""Apply confirmed logical label corrections and keep unresolved cases excluded."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
inventory_path = HERE/'inventory.json'
raw = inventory_path.read_bytes()
inventory = json.loads(raw)
correction_path = HERE/'label_corrections.json'
correction_raw = correction_path.read_bytes()
corrections = json.loads(correction_raw)
confirmed = {r['source_file']: r for r in corrections['confirmed']}
all_paths = {r['file'] for r in inventory['images']}
assert set(confirmed).issubset(all_paths)
texts = {r['file']: r for r in inventory['text_records']}
excluded_paths = set(inventory['summary']['recommended_holdout_from_labeled_validation'])-set(confirmed)
included, excluded = [], []
for source in inventory['images']:
    row = dict(source)
    row['validation_y_mm'] = row['folder_y_mm']
    row['validation_theta_deg'] = row['folder_theta_deg']
    if row['file'] in confirmed:
        correction = confirmed[row['file']]
        row['validation_y_mm'] = correction['validation_y_mm']
        row['validation_theta_deg'] = correction['validation_theta_deg']
        row['label_correction'] = correction
    sidecar = (confirmed[row['file']]['sidecar_file'] if row['file'] in confirmed
        else str(Path(row['file']).with_suffix('.txt')))
    text_record = texts[sidecar]
    row['source_sidecar_file'] = sidecar
    row['effective_y_actual'] = text_record['Y_Actual']
    row['effective_theta_actual'] = text_record['T_Actual']
    row['effective_sidecar_path'] = str(Path(inventory['source_root'])/sidecar)
    if row['file'] in confirmed and 'corrected_theta_actual' in confirmed[row['file']]:
        row['effective_theta_actual'] = confirmed[row['file']]['corrected_theta_actual']
        corrected_path = HERE/'corrected_sidecars'/sidecar
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"Y_Actual\tT_Actual\r\n{row['effective_y_actual']:.3f}\t\t{row['effective_theta_actual']:.3f}"
        corrected_path.write_bytes(content.encode('utf-8'))
        row['effective_sidecar_path'] = str(corrected_path)
    (excluded if row['file'] in excluded_paths else included).append(row)
assert len(included)+len(excluded) == len(inventory['images'])
for row in excluded:
    row['exclusion_reason'] = ('Folder/filename theta disagreement; no exact filename TXT match.'
        if row['label_or_sidecar_anomaly'] else 'Theta 0.57 label conflicts with recorded T_Actual in all three TXT files.')
selection = {
    'decision_date': '2026-10-29',
    'authority': corrections['authority'],
    'scope': 'Selection for future validation; no angle extraction, regression or controller test executed.',
    'source_root': inventory['source_root'],
    'inventory_sha256': hashlib.sha256(raw).hexdigest(),
    'label_corrections_sha256': hashlib.sha256(correction_raw).hexdigest(),
    'included_image_count': len(included),
    'included_condition_count': len({(r['validation_y_mm'], r['validation_theta_deg']) for r in included}),
    'excluded_image_count': len(excluded),
    'confirmed_correction_count': len(confirmed),
    'pending_corrections': corrections['pending'],
    'input_contract': 'Use file with source_root to read original BMP; use validation_y_mm/validation_theta_deg and effective_sidecar_path for corrected analysis inputs. Raw folder labels and anomaly flags are provenance.',
    'included': included,
    'excluded': excluded,
    'raw_files_modified': False,
}
(HERE/'validation_selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"Selected {len(included)} images / {selection['included_condition_count']} conditions; {len(excluded)} images pending correction values.")
