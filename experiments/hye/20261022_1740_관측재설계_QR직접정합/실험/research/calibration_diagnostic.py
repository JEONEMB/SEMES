"""Linear Zhang calibration feasibility diagnostic, never a physical K claim.

Real decoded module coordinates anchor each QR to the fixed center. Current
module->image homographies are composed from independently measured dense ECC
warps. Each outer-training subset is processed separately; no test observation
or Y/T command enters its calibration. Lens distortion is not modeled here.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import cv2
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REBOOT=HERE.parent
WORKSPACE=HERE.parents[2]
GENERAL=REBOOT.parent/'joint_yt_generalization_20261022'
sys.path.insert(0,str(GENERAL))
from data import load

# Image-coordinate normalization improves numerical scaling. It is NOT a prior
# that the true principal point equals the image center.
WIDTH,HEIGHT=3840.,2160.
N=np.array([[1/WIDTH,0,-.5],[0,1/WIDTH,-HEIGHT/(2*WIDTH)],[0,0,1.]])

def project(H,P):
    q=np.c_[P,np.ones(len(P))]@H.T
    return q[:,:2]/q[:,2,None]

def v(h,i,j):
    return np.array([h[0,i]*h[0,j],h[0,i]*h[1,j]+h[1,i]*h[0,j],h[1,i]*h[1,j],
                     h[2,i]*h[0,j]+h[0,i]*h[2,j],h[2,i]*h[1,j]+h[1,i]*h[2,j],h[2,i]*h[2,j]])

def constraints(homographies,normalize=True):
    rows=[]
    for H in homographies:
        h=N@H if normalize else H.copy()
        block=np.stack([v(h,0,1),v(h,0,0)-v(h,1,1)])
        # A common factor for each two-row block removes homography scale
        # while preserving the relative two constraints within that view.
        block/=max(np.linalg.norm(block),1e-30)
        rows.extend(block)
    return np.array(rows)

def solve_zhang(homographies):
    V=constraints(homographies)
    _,s,vt=np.linalg.svd(V,full_matrices=False)
    b=vt[-1]
    B=np.array([[b[0],b[1],b[3]],[b[1],b[2],b[4]],[b[3],b[4],b[5]]])
    if np.trace(B)<0:B=-B;b=-b
    eigen=np.linalg.eigvalsh(B)
    raw_s=np.linalg.svd(constraints(homographies,normalize=False),compute_uv=False)
    out=dict(homographies=len(homographies),constraint_rows=len(V),singular_values=s.tolist(),
             normalized_identifiable_condition_s0_s4=float(s[0]/max(s[-2],1e-30)),
             normalized_full_condition_s0_s5=float(s[0]/max(s[-1],1e-30)),
             nullspace_gap_s4_s5=float(s[-2]/max(s[-1],1e-30)),
             raw_pixel_identifiable_condition_s0_s4=float(raw_s[0]/max(raw_s[-2],1e-30)),
             constraint_rms=float(np.sqrt(np.mean((V@b)**2))),
             B_normalized_pixel_coordinates=B.tolist(),B_eigenvalues=eigen.tolist(),
             positive_definite=bool(eigen[0]>0),K_candidate=None,
             interpretation='Diagnostic candidate; never used as known camera calibration or physical command conversion')
    if eigen[0]>0:
        # B=K^-T K^-1, so upper Cholesky factor is K^-1 up to scale.
        L=np.linalg.cholesky(B)
        Kn=np.linalg.inv(L.T)
        Kn/=Kn[2,2]
        K=np.linalg.inv(N)@Kn
        K/=K[2,2]
        out.update(K_candidate=K.tolist(),fx=float(K[0,0]),fy=float(K[1,1]),skew=float(K[0,1]),
                   cx=float(K[0,2]),cy=float(K[1,2]),positive_focal_lengths=bool(K[0,0]>0 and K[1,1]>0),
                   principal_point_inside_image=bool(0<=K[0,2]<=WIDTH and 0<=K[1,2]<=HEIGHT),
                   B_condition=float(eigen[-1]/eigen[0]))
    return out

def pose_consistency(K,H,grid):
    q=np.linalg.solve(K,H)
    lam=2/(np.linalg.norm(q[:,0])+np.linalg.norm(q[:,1]))
    if lam*q[2,2]<0:lam=-lam
    r1,r2,t=lam*q[:,0],lam*q[:,1],lam*q[:,2]
    approx=np.c_[r1,r2,np.cross(r1,r2)]
    u,_,vt=np.linalg.svd(approx)
    R=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
    Hpose=K@np.c_[R[:,:2],t]
    error=project(Hpose,grid)-project(H,grid)
    return float(np.sqrt(np.mean(np.sum(error**2,axis=1))))

def sanity_check():
    K=np.array([[2800.,0,1920],[0,2820,1080],[0,0,1.]])
    rng=np.random.default_rng(8022)
    hh=[]
    for _ in range(20):
        R=cv2.Rodrigues(rng.uniform(-.4,.4,3))[0]
        t=np.r_[rng.uniform(-5,5,2),rng.uniform(35,60)]
        hh.append(K@np.c_[R[:,:2],t])
    got=solve_zhang(np.array(hh))
    error=float(np.max(abs(np.array(got['K_candidate'])-K)))
    assert got['positive_definite'] and error<1e-6
    return dict(type='Algebra-only synthetic known-K check, unrelated to actual camera calibration',
                passed=True,max_abs_K_recovery_error=error)

def main():
    m,old=load()
    a=np.load(REBOOT/'direct/features.npz')
    corner_path=WORKSPACE/'joint_center_630_20261022/assets/corners.json'
    corners=json.loads(corner_path.read_text())
    by_id={int(row['id']):row for row in corners}
    assert len(corners)==630 and np.array_equal(a['ids'],old['ids'])
    rows=[by_id[int(i)] for i in a['ids']]
    grid=np.array([[r['gx'],r['gy']] for r in rows],dtype=float)
    qr=np.array([r['qr']-1 for r in rows])
    assert np.array_equal(qr,a['regions'])
    center_json=np.array([[r['cx'],r['cy']] for r in rows])
    # corners.json stores the initial structural boundary centers; the immutable
    # reference.npz contains the subsequently refined center measurements.
    assert np.array_equal(a['points'],old['points'])
    initial_refined_delta=np.linalg.norm(center_json-a['points'],axis=1)
    hc=[];center_audit=[];grids=[]
    for k in range(3):
        mask=qr==k
        H,_=cv2.findHomography(grid[mask],a['points'][mask],method=0)
        H/=H[2,2];hc.append(H);grids.append(grid[mask])
        errors=np.linalg.norm(project(H,grid[mask])-a['points'][mask],axis=1)
        den=np.c_[grid[mask],np.ones(mask.sum())]@H[2]
        center_audit.append(dict(qr=int(k+1),points=int(mask.sum()),module_extent_min=grid[mask].min(0).tolist(),
                                 module_extent_max=grid[mask].max(0).tolist(),H_module_to_center=H.tolist(),
                                 grid_homography_fit_rms_px=float(np.sqrt(np.mean(errors**2))),
                                 grid_homography_fit_max_px=float(errors.max()),
                                 projective_denominator_relative_span=float(np.ptp(den)/abs(den.mean()))))
    hc=np.array(hc)
    # Explicit direction: observed H maps fixed-center pixels to current pixels.
    H=np.einsum('nqab,qbc->nqac',a['H_homography_g5'],hc)
    assert H.shape==(341,3,3,3) and np.isfinite(H).all()
    observations=[]
    for k in range(3):
        spans=[]
        for h in H[:,k]:
            den=np.c_[grids[k],np.ones(len(grids[k]))]@h[2]
            spans.append(float(np.ptp(den)/abs(den.mean())))
        observations.append(dict(qr=k+1,denominator_relative_span_min=float(min(spans)),
                                 denominator_relative_span_median=float(np.median(spans)),
                                 denominator_relative_span_max=float(max(spans))))
    records=[];training_splits=[]
    for fold in range(5):
        tr=np.flatnonzero(m.fold.to_numpy()!=fold) if fold<4 else np.arange(len(m))
        test=np.flatnonzero(m.fold.to_numpy()==fold) if fold<4 else np.array([],int)
        if fold<4:assert not set(m.iloc[tr].cv_group)&set(m.iloc[test].cv_group)
        label=f'outer_train{fold}' if fold<4 else 'all341_development_descriptive'
        training_splits.append(dict(scope=label,train_indices=m.iloc[tr]['index'].tolist(),
                                    excluded_test_indices=m.iloc[test]['index'].tolist()))
        for ks in [[0],[1],[2],[0,1,2]]:
            selected=H[tr][:,ks].reshape(-1,3,3)
            result=solve_zhang(selected)
            if result['positive_definite']:
                K=np.array(result['K_candidate'])
                residual=[pose_consistency(K,H[i,k],grids[k]) for i in tr for k in ks]
                result['train_homography_pose_consistency_rms_px']=float(np.sqrt(np.mean(np.square(residual))))
                result['train_homography_pose_consistency_max_view_rms_px']=float(max(residual))
            records.append(dict(scope=label,qr_subset=[int(k+1) for k in ks],**result))
    fold_records=[r for r in records if r['scope'].startswith('outer_')]
    stability=[]
    for ks in [[1],[2],[3],[1,2,3]]:
        same=[r for r in fold_records if r['qr_subset']==ks]
        valid=[r for r in same if r['positive_definite']]
        s=dict(qr_subset=ks,outer_training_fits=4,positive_definite_fits=len(valid),
               identifiable_condition_min=float(min(r['normalized_identifiable_condition_s0_s4'] for r in same)),
               identifiable_condition_max=float(max(r['normalized_identifiable_condition_s0_s4'] for r in same)),
               nullspace_gap_min=float(min(r['nullspace_gap_s4_s5'] for r in same)),
               nullspace_gap_max=float(max(r['nullspace_gap_s4_s5'] for r in same)))
        if valid:
            s['candidate_ranges']={key:dict(min=float(min(r[key] for r in valid)),max=float(max(r[key] for r in valid)),
                                           std=float(np.std([r[key] for r in valid]))) for key in ['fx','fy','skew','cx','cy']}
        stability.append(s)
    out=dict(status='diagnostic_completed_not_validated_calibration',
             source={'Zhang':'https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf',
                     'corners':str(corner_path),'dense_homographies':str(REBOOT/'direct/features.npz'),
                     'corners_sha256':hashlib.sha256(corner_path.read_bytes()).hexdigest(),
                     'dense_features_sha256':hashlib.sha256((REBOOT/'direct/features.npz').read_bytes()).hexdigest()},
             method='Actual decoded(gx,gy) -> fixed-center all-point homography; H_current=H_dense@H_center; Zhang Vb=0 SVD; B=K^-T K^-1; strict SPD then Cholesky K. No nearest-SPD repair, guessed K, distortion fit or command labels.',
             image_coordinate_normalization=N.tolist(),
             numerical_interpretation={'condition_s0_s4':'Condition of five identifiable directions after arbitrary common image normalization. Larger is worse; not a calibrated uncertainty bound.',
                                       'nullspace_gap_s4_s5':'Separation of the expected null direction; close to1 weakly identifies a unique solution.',
                                       'constraint_rms':'Algebraic train residual in normalized constraints, not pixel or Y/T error.',
                                       'pose_consistency':'Closest orthonormal pose projection compared with composed homography on training module points; not independent measured-image reprojection validation.'},
             unit_convention='Each decoded square module uses arbitrary length1. K can in principle be estimated independent of absolute scale; no translation in mm or physical T angle is recovered.',
             reference_geometry=center_audit,
             initial_corner_to_refined_reference_delta_px=dict(median=float(np.median(initial_refined_delta)),max=float(initial_refined_delta.max()),interpretation='gx/gy and IDs come from corners.json; module-to-center H uses immutable refined reference points, not the initial cx/cy values'),
             observed_projective_spread=observations,training_splits=training_splits,
             calibration_fits=records,fold_stability=stability,synthetic_algebra_check=sanity_check(),
             findings=[
                 'All four pooled-three-QR outer-training fits yield indefinite B. A common pinhole intrinsic matrix is not recovered by this linear method.',
                 'QR1 and QR2 each yield positive-definite solutions in all four training splits, but their candidate principal points differ materially. QR3 is positive-definite in only one split and that candidate is extreme.',
                 'Positive focal lengths and small homography pose-consistency residuals alone do not validate physical camera calibration.',
                 'Center square-grid homography residuals are about0.82-0.87px. Weak motion diversity, lens distortion, printed-grid departures and boundary localization bias remain possible contributors; this diagnostic does not isolate one cause.'],
             limitations=[
                 'Numeric candidates are diagnostics only. No K in this file is certified true or used by the inference pipelines.',
                 'The fixed center homographies use the allowed reference image only; no current held-out observations enter an outer-training calibration.',
                 'Pinhole lens, square printed modules and per-QR rigidity are assumptions. Lens distortion, printed-module/edge localization bias and weak orientation diversity may all affect results.',
                 'Small denominator variation is a projective-image diagnostic, not calibrated proof of a fronto-parallel physical camera pose.',
                 'Indefinite B or unstable K shows failure of this linear calibration under these observations and assumptions; it does not prove all self-calibration is impossible.',
                 'All341 are previously exposed development images. No independent calibration target capture or actual robot center-return validation exists.'])
    (HERE/'calibration_diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    flat=[]
    for r in records:
        flat.append({k:v for k,v in r.items() if k in ['scope','qr_subset','positive_definite','normalized_identifiable_condition_s0_s4','nullspace_gap_s4_s5','constraint_rms','fx','fy','skew','cx','cy','principal_point_inside_image','train_homography_pose_consistency_rms_px']})
    pd.DataFrame(flat).to_csv(HERE/'calibration_diagnostic.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({'center':center_audit,'stability':stability,'synthetic':out['synthetic_algebra_check']},indent=2),flush=True)

if __name__=='__main__':
    cv2.setNumThreads(1)
    with threadpool_limits(1):main()
