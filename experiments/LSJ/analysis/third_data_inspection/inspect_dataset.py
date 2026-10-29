"""Inspect third-delivery labels, files and visibility; do not measure line angles."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import struct

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LABEL = re.compile(r'Y([+-]\d+(?:\.\d+)?) T([+-]\d+(?:\.\d+)?)')
REPEAT = re.compile(r'\((\d+)\)$')


def coordinate_label_anomalies(text_records):
    modal_zero = Counter(r['implied_label_zero_theta'] for r in text_records).most_common(1)[0][0]
    return [{**r, 'modal_implied_theta_zero': modal_zero,
        'theta_from_recorded_coordinate_and_modal_zero': float(Decimal(str(r['T_Actual']))-Decimal(str(modal_zero))),
        'handling': 'Flag label/TXT disagreement; do not choose a ground truth from image response.'}
        for r in text_records if r['implied_label_zero_theta'] != modal_zero]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    assert root.is_dir(), root
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    bmps = [p for p in paths if p.suffix.lower() == '.bmp']
    txts = [p for p in paths if p.suffix.lower() == '.txt']
    previous = json.loads((HERE.parent/'theta_line_hypothesis/line_measurements.json').read_text(encoding='utf-8'))
    old_hashes = defaultdict(list)
    for row in previous['measurements']:
        old_hashes[row['sha256']].append({'dataset': row['dataset'], 'file': row['file']})
    second = json.loads((HERE.parent/'second_data_inspection/inventory.json').read_text(encoding='utf-8'))
    second_reference = second['summary']['reference_recorded_coordinates']
    ref_y = Decimal(str(second_reference[0]['Y_Actual']))
    ref_t = Decimal(str(second_reference[0]['T_Actual']))

    text_records = []
    for path in txts:
        label = LABEL.fullmatch(path.parent.name)
        assert label, path
        text = path.read_text(encoding='utf-8-sig').strip()
        lines = text.splitlines()
        assert lines[0].split() == ['Y_Actual', 'T_Actual'], path
        actual_y, actual_t = [Decimal(v) for v in lines[1].split()]
        label_y, label_t = [Decimal(v) for v in label.groups()]
        text_records.append({'file': str(path.relative_to(root)), 'label_y_mm': float(label_y),
            'label_theta_deg': float(label_t), 'Y_Actual': float(actual_y), 'T_Actual': float(actual_t),
            'implied_label_zero_y': float(actual_y-label_y),
            'implied_label_zero_theta': float(actual_t-label_t),
            'delta_from_second_reference_minus_y_label': float(actual_y-ref_y-label_y),
            'delta_from_second_reference_minus_theta_label': float(actual_t-ref_t-label_t)})

    rows, anomalies, decode_errors, unchanged_checks = [], [], [], []
    hash_groups = defaultdict(list)
    preview_images = {}
    preview_conditions = {Decimal('0.04'), Decimal('0.44'), Decimal('0.82')}
    for i, path in enumerate(bmps):
        before = path.stat()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        header = raw[:54]
        signature = header[:2]
        declared_size = struct.unpack_from('<I', header, 2)[0]
        width, height = struct.unpack_from('<ii', header, 18)
        depth = struct.unpack_from('<H', header, 28)[0]
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (abs(height), width):
            decode_errors.append(str(path.relative_to(root)))
        assert signature == b'BM' and declared_size == len(raw), path
        folder_label = LABEL.fullmatch(path.parent.name)
        file_label = LABEL.match(path.stem)
        repeat = int(REPEAT.search(path.stem)[1])
        assert folder_label and file_label
        exact_sidecar = path.with_suffix('.txt')
        candidates = [p for p in path.parent.glob('*.txt') if REPEAT.search(p.stem) and int(REPEAT.search(p.stem)[1]) == repeat]
        anomaly = folder_label.groups() != file_label.groups() or not exact_sidecar.is_file()
        if anomaly:
            anomalies.append({'file': str(path.relative_to(root)), 'folder_labels': list(folder_label.groups()),
                'filename_labels': list(file_label.groups()), 'exact_sidecar_exists': exact_sidecar.is_file(),
                'same_folder_repeat_sidecar_candidates': [str(p.relative_to(root)) for p in candidates],
                'handling': 'Flag ambiguous label; do not rename or automatically relabel.'})
        row = {'file': str(path.relative_to(root)), 'sha256': digest, 'bytes': len(raw),
            'width': width, 'height': height, 'bits_per_pixel': depth,
            'folder_y_mm': float(folder_label[1]), 'folder_theta_deg': float(folder_label[2]), 'repeat': repeat,
            'mtime_local': datetime.fromtimestamp(before.st_mtime).isoformat(),
            'ctime_local': datetime.fromtimestamp(before.st_ctime).isoformat(),
            'label_or_sidecar_anomaly': anomaly, 'matching_previous_images': old_hashes.get(digest, [])}
        rows.append(row)
        hash_groups[digest].append(row['file'])
        after = path.stat()
        unchanged_checks.append(before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns)
        if Decimal(folder_label[2]) in preview_conditions and repeat == 1:
            preview_images[folder_label[2]] = image
        if (i+1) % 60 == 0 or i+1 == len(bmps):
            print(f'Inspected headers, sidecars, hashes and image decoding: {i+1}/{len(bmps)}', flush=True)

    conditions = sorted(set(r['folder_theta_deg'] for r in rows))
    cents = {int(round(v*100)) for v in conditions}
    condition_records = []
    for theta in conditions:
        group = [r for r in rows if r['folder_theta_deg'] == theta]
        text_group = [r for r in text_records if r['label_theta_deg'] == theta]
        condition_records.append({'Y_mm': group[0]['folder_y_mm'], 'theta_deg': theta,
            'image_count': len(group), 'txt_count': len(text_group), 'repeat_numbers': [r['repeat'] for r in group],
            'recorded_y_actual_values': sorted(set(r['Y_Actual'] for r in text_group)),
            'recorded_theta_actual_values': sorted(set(r['T_Actual'] for r in text_group))})
    summary = {'image_count': len(rows), 'txt_count': len(txts), 'condition_count': len(conditions),
        'y_label_values_mm': sorted(set(r['folder_y_mm'] for r in rows)),
        'theta_min_max_deg': [min(conditions), max(conditions)], 'nominal_grid_step_deg': .01,
        'missing_theta_inside_min_max_deg': [v/100 for v in range(min(cents), max(cents)+1) if v not in cents],
        'image_count_per_condition_distribution': dict(Counter(r['image_count'] for r in condition_records)),
        'header_shapes_and_depth': sorted(set((r['width'], r['height'], r['bits_per_pixel']) for r in rows)),
        'decode_errors': decode_errors, 'unique_image_hashes': len(hash_groups),
        'duplicate_image_groups': [v for v in hash_groups.values() if len(v)>1],
        'images_identical_to_prior_deliveries': [r for r in rows if r['matching_previous_images']],
        'Y_Actual_values': sorted(set(r['Y_Actual'] for r in text_records)),
        'implied_label_zero_y_values': sorted(set(r['implied_label_zero_y'] for r in text_records)),
        'implied_label_zero_theta_values': sorted(set(r['implied_label_zero_theta'] for r in text_records)),
        'max_abs_y_difference_vs_second_reference': max(abs(r['delta_from_second_reference_minus_y_label']) for r in text_records),
        'max_abs_theta_difference_vs_second_reference': max(abs(r['delta_from_second_reference_minus_theta_label']) for r in text_records),
        'mtime_local_range': [min(r['mtime_local'] for r in rows), max(r['mtime_local'] for r in rows)],
        'all_image_size_and_mtime_unchanged_during_read': all(unchanged_checks),
        'anomalies': anomalies,
        'coordinate_label_anomalies': coordinate_label_anomalies(text_records)}
    flagged = {r['file'] for r in rows if r['label_or_sidecar_anomaly']}
    for row in summary['coordinate_label_anomalies']:
        flagged.add(str(Path(row['file']).with_suffix('.bmp')))
    summary['recommended_holdout_from_labeled_validation'] = sorted(flagged)
    summary['remaining_images_if_flagged_records_are_withheld'] = len(rows)-len(flagged)
    summary['remaining_conditions_if_flagged_records_are_withheld'] = len({
        (r['folder_y_mm'], r['folder_theta_deg']) for r in rows if r['file'] not in flagged})
    output = {'source_root': str(root), 'scope': 'Data suitability inspection only; no angle extraction, regression, inference or algorithm changes.',
        'summary': summary, 'conditions': condition_records, 'images': rows, 'text_records': text_records,
        'second_reference_records_for_comparison': second_reference,
        'limits': ['Filesystem times are not independently verified capture timestamps.',
            'Actual fields are recorded coordinates; no independently measured physical pose is supplied.',
            'Three files per condition do not establish independent repositioning trials.',
            'Matching coordinate origin does not establish matching optical zero or lighting.',
            'No new theta-zero reference, negative theta or varying Y is present.',
            'Archive completeness/CRC was not checked; inspected existing extracted files.']}
    (HERE/'inventory.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    fig, axes = plt.subplots(3, 1, figsize=(12, 16), layout='constrained')
    for ax, (theta, image) in zip(axes, sorted(preview_images.items(), key=lambda item: float(item[0]))):
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Third delivery: Y=+0.05 mm, theta={theta} deg, repeat 1; visibility inspection only')
        ax.axis('off')
    fig.savefig(HERE/'sample_visibility.png', dpi=120)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
