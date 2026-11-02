"""Interpret recorded masks without inventing fit provenance for older results."""

def fit_point_view(intermediates):
    points = intermediates.get('fit_points', [])
    def mask(key):
        if key not in intermediates:
            return None
        flags = intermediates[key]
        if len(flags) != len(points) or any(x not in (0, 1) for x in flags):
            raise ValueError('invalid recorded mask: ' + key)
        return [bool(x) for x in flags]

    used = mask('fit_points_used_flag')
    final = mask('fit_points_final_inlier_flag')
    if final is None:
        final = mask('fit_points_inlier_flag')
    if used is not None:
        basis, label, excluded = 'actual_last_fit_input', '실제 마지막 피팅 사용점', '마지막 피팅 미사용점'
        display = used
    elif final is not None:
        basis, label, excluded = 'legacy_final_distance_inliers', '최종 거리 내 점(실제 사용점 미기록)', '최종 거리 밖 점'
        display = final
    else:
        basis, label, excluded = 'unrecorded', '점 분류 미기록', '점 분류 미기록'
        display = [False] * len(points)
    return dict(points=points, used=used, final_inlier=final, display_mask=display,
                basis=basis, label=label, excluded_label=excluded)

def convergence_label(record):
    if record.get('method_id') in ('Q11', 'Q12'):
        return 'IRLS 미사용'
    diag = record.get('diagnostics', {})
    runs = diag.get('irls_runs')
    if runs is None:
        return '수렴 상태 미기록(과거 결과)'
    if not runs:
        return 'IRLS 시작 전 실패: ' + str(diag.get('fit_failure_code', 'unknown'))
    names = {'converged': '수렴', 'max_iterations': '한도 도달·수렴 미확인',
             'invalid_input': '입력 오류', 'numerical_error': '계산 오류', 'degenerate_geometry': '퇴화 입력'}
    return ' / '.join(f"{r['stage']}: {names.get(r['termination'], r['termination'])} {r['iterations']}회" for r in runs)

def rms_label(record):
    if record.get('fit_rms_point_set') == 'actual_last_fit_input':
        return '실제 피팅 사용점 RMS'
    if record.get('fit_rms_px') is None:
        return 'RMS 없음'
    return '최종 거리 내 점 RMS(과거 결과)'
