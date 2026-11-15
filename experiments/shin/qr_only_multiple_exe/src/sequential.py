"""Anchor Y by stage position and close the theta loop using QR displacement."""
from dataclasses import dataclass
from math import isfinite

THETA_PX_PER_DEG = 85.0
QR_TOLERANCE_PX = .30


def theta_correction_mdeg(shift, maximum=300):
    # Positive theta moves the QR left.
    return max(-maximum, min(maximum, round(1000*shift/THETA_PX_PER_DEG)))


@dataclass(frozen=True)
class Settings:
    center_y_um: int
    center_theta_mdeg: int
    max_moves: int = 20
    tolerance_px: float = QR_TOLERANCE_PX


def converge(stage, observe, settings, record, estimate_mixed=None):
    """Send one absolute command per correction and count it against the budget."""
    s = settings
    if not 1 <= s.max_moves <= 20:
        raise ValueError('Move budget must be between 1 and 20')
    limits = stage.limits()
    y, theta = stage.status()['position']
    used = 0

    def result(reason):
        return {'result': reason, 'moves': used, 'y_um': y, 'theta_mdeg': theta,
                'status_y_within': abs(y-s.center_y_um) <= 45,
                'status_theta_within': abs(theta-s.center_theta_mdeg) <= 10}

    def move(sy, st):
        nonlocal y, theta, used
        if used >= s.max_moves:
            return False
        if not (limits[0] <= sy <= limits[1] and limits[2] <= st <= limits[3]):
            raise ValueError(f'Target exceeds stage limits: {(sy, st)}')
        used += 1
        try:
            stage.start(sy, st, relative=False, allow_t=True)
            done = stage.wait_done(timeout_ms=60000)
        except BaseException:
            try:
                stage.stop()
            except Exception:
                pass
            raise
        y, theta = done['position']
        record({'kind': 'move', 'move': used, 'target': [sy, st],
                'position': [y, theta], 'stage_result': done['result']})
        if done['result'] != 0:
            raise RuntimeError(f'Stage move failed: {done["result"]}')
        return True

    if estimate_mixed is not None:
        error_y, error_theta = estimate_mixed()
        if not (isfinite(error_y) and isfinite(error_theta)):
            raise ValueError('Non-finite mixed pose estimate')
        # Preserve the same per-command theta cap as the QR-only loop.
        correction = max(-300, min(300, round(1000 * error_theta)))
        target_y, target_theta = y - round(1000 * error_y), theta - correction
        if not (limits[0] <= target_y <= limits[1] and limits[2] <= target_theta <= limits[3]
                and abs(target_theta - s.center_theta_mdeg) <= 1050):
            return result('outside_calibrated_range')
        record({'kind': 'mixed_estimate', 'y_error_mm': float(error_y),
                'theta_error_deg': float(error_theta), 'position': [y, theta]})
        if (target_y, target_theta) != (y, theta):
            move(target_y, target_theta)

    start_theta = theta
    if not move(s.center_y_um, theta):
        return result('move_budget_or_resolution')
    if abs(theta-start_theta) > 1:
        return result('theta_changed_during_y_anchor')

    while True:
        if abs(y-s.center_y_um) > 5:
            return result('y_anchor_lost')
        shift = observe()
        if not isfinite(shift):
            raise ValueError('Non-finite QR displacement')
        record({'kind': 'observe', 'move': used, 'dx_px': shift, 'position': [y, theta]})
        if abs(shift) <= s.tolerance_px:
            confirmation = observe()
            if not isfinite(confirmation):
                raise ValueError('Non-finite QR confirmation')
            record({'kind': 'confirm', 'move': used, 'dx_px': confirmation,
                    'position': [y, theta]})
            if abs(confirmation) <= s.tolerance_px:
                return result('qr_aligned')
            shift = confirmation

        # Positive theta moves the QR left, with a nominal sensitivity of 85 px/deg.
        correction = theta_correction_mdeg(shift)
        target = theta + correction
        if abs(target-s.center_theta_mdeg) > 1050:
            return result('outside_calibrated_range')
        if not correction or not move(s.center_y_um, target):
            return result('move_budget_or_resolution')
