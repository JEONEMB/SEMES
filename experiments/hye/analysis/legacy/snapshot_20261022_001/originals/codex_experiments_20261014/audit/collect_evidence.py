"""Read-only audit of original reports and image metadata; outputs stay here."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
IMAGE_ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")

q = pd.read_csv(PROJECT / "14_데이터_QR마커위치.csv")
q["x"] = q[["dLx", "dCx", "dRx"]].mean(axis=1)
q["t"] = [
    (IMAGE_ROOT / n if n == "center.bmp" else IMAGE_ROOT / ("Theta" if n[0] == "T" else "Y") / n).stat().st_mtime
    for n in q.file
]
q.t = (q.t - q.t.min()) / 60
audit = {"image_root": str(IMAGE_ROOT), "n_images": len(q), "axes": {}}
for axis in ["Y", "T"]:
    g = q[q.file.str.startswith(axis)].sort_values("t").copy()
    v, t, x = (g[c].to_numpy() for c in ["value", "t", "x"])
    seg = np.cumsum(np.r_[0, (np.diff(t) > 3) | (np.sign(v[1:]) != np.sign(v[:-1]))])
    dr = np.empty(len(g))
    sessions = []
    for s in np.unique(seg):
        m = seg == s
        dr[m] = np.sign(np.polyfit(t[m], v[m], 1)[0])
        slope, intercept = np.polyfit(v[m], x[m], 1)
        sessions.append({
            "segment": int(s), "n": int(m.sum()), "direction": float(dr[m][0]),
            "value_time_correlation": float(np.corrcoef(v[m], t[m])[0, 1]),
            "slope_px_per_unit": float(slope), "zero_intercept_px": float(intercept),
            "intercept_in_command_unit": float(intercept / slope),
            "linear_fit_rms_px": float(np.sqrt(np.mean((x[m] - slope * v[m] - intercept) ** 2))),
        })
    A = np.c_[v, np.ones(len(g)), dr, t, 1 - np.exp(-t / 3)]
    coef = np.linalg.lstsq(A, x, rcond=None)[0]
    omitted = []
    for s in np.unique(seg):
        tr = seg != s
        omitted.append({"heldout_segment": int(s), "rank": int(np.linalg.matrix_rank(A[tr])), "n_columns": int(A.shape[1])})
    audit["axes"][axis] = {
        "sessions": sessions, "v4_design_condition_number": float(np.linalg.cond(A)),
        "v4_design_coefficients": coef.tolist(),
        "v4_fit_rms_px": float(np.sqrt(np.mean((x - A @ coef) ** 2))),
        "leave_session_out_calibration_ranks": omitted,
    }

table = pd.read_csv(PROJECT / "54_v7_최종평가.csv")
audit["original_v7_reported_scores"] = table.to_dict(orient="records")
pred = pd.read_csv(PROJECT / "55_v7_사진별예측.csv")
axis_ok = pred.axis == pred.pred_axis
sign_ok = np.sign(pred.cmd) == np.sign(pred.pred)
audit["original_v7_corrected_free_odd_even"] = {
    "n": len(pred), "axis_correct": int(axis_ok.sum()),
    "sign_correct_command_basis": int(sign_ok.sum()),
    "axis_and_sign_correct": int((axis_ok & sign_ok).sum()),
    "axis_errors": pred.loc[~axis_ok].to_dict(orient="records"),
}
audit["sources"] = [
    {"title": "OpenCV: Basic concepts of the homography explained with code", "url": "https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html", "evidence": "Planar image correspondence uses a homography; pose examples assume intrinsics and known object geometry. Relative homography contains translation divided by plane distance."},
    {"title": "OpenCV: Object Tracking", "url": "https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html", "evidence": "Pyramidal LK returns per-point tracking status/errors; ECC aligns intensities with 2/3/6/8 parameter models and requires adequate initialization for large displacement."},
    {"title": "OpenCV: Perspective-n-Point (PnP) pose computation", "url": "https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html", "evidence": "Pose is rotation/translation relating object coordinates to camera coordinates using corresponding object/image points and camera parameters."},
]
(HERE / "evidence.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"output": str(HERE / 'evidence.json'), "n_images": len(q), "design_condition_numbers": {a: audit['axes'][a]['v4_design_condition_number'] for a in audit['axes']}}, ensure_ascii=False))
