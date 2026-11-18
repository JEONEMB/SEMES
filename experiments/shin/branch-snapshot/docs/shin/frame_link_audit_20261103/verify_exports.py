"""Check export accounting, fold separation, and QR origin consistency."""
from refine_origin import *
import cv2
from pypdf import PdfReader

original,f,a=load();d=audited_data(original,True)
c=pd.read_csv(P/'coordinates_standard_v3.csv')
cv=pd.read_csv(P/'refinement_CV.csv')
j=json.loads((P/'standard_origin_v3.json').read_text())
assert len(c)==575 and c.path.nunique()==575 and c.row_id.nunique()==575
assert np.count_nonzero(~np.isclose(c.T_original,c['T']))==1
assert c.Actual_log_conflict.sum()==3
assert (c.training_image_weight<.999).sum()==4
assert c[c.group.isin([16,17])].cohort.eq('reference').all()
assert len(c[c.group==17])==1 and c[c.group==17].path.iloc[0].endswith('center3.bmp')
assert not cv.duplicated(['config','scheme','path']).any()
sizes=cv.groupby(['config','scheme']).size()
assert len(sizes)==12 and sizes.eq(555).all()
assert cv.QR_RMS_px.notna().all() and cv.right_RMS_px.notna().all()
cal=calibration_mask(d)
assert int(cal.sum())==20
assert set(cv.path)==set(d.loc[~cal,'path'])
assert cv.groupby(['config','scheme','pose']).fold.nunique().max()==1
detail=pd.read_csv(P/'refinement_feature_CV.csv')
assert not detail.duplicated(['config','scheme','fold','path','feature_id']).any()
points=pd.read_csv(P/'standard_origin_v3_QR630.csv')
assert len(points)==630 and points.groupby('qr_id').size().tolist()==[212,210,208]
fs=cv2.FileStorage(str(ROOT/'analysis/session_eda_20261102/remeasured/cache/reference.yml.gz'),cv2.FILE_STORAGE_READ)
ref=fs.getNode('points').mat();fs.release()
xy=points[['x_standard_px','y_standard_px']].to_numpy()-ref[:,:2]
means=np.concatenate([xy[points.qr_id==i].mean(0) for i in range(3)])
err=float(np.max(np.abs(means-np.array(j['origin_QR_features_6']))))
assert err<1e-7,(means,j['origin_QR_features_6'])
assert j['gradient_relative_error']<1e-5
pdf=ROOT/'output/pdf/T_axis_common_origin_v3_audit_20261103.pdf'
reader=PdfReader(pdf)
assert len(reader.pages)==18
assert all(len(p.extract_text())>100 for p in reader.pages)
assert '순환 검증' in ''.join(p.extract_text() for p in reader.pages)
checks={'images':575,'corrected_labels':1,'actual_conflicts_retained':3,'soft_weighted':4,
 'hard_removed':0,'configuration_split_pairs':12,'test_images_per_pair':555,
 'calibration_images':20,'duplicate_prediction_rows':0,'duplicate_feature_rows':0,
 'same_pose_cross_fold_leakage':False,'qr_counts':[212,210,208],
 'QR_origin_max_discrepancy_px':err,'gradient_relative_error':j['gradient_relative_error'],
 'pdf_pages':len(reader.pages),'optimizer_rounds':j['optimizer_rounds'],'raw_bound_hits':j['raw_bound_hits']}
save('export_checks.json',checks)
print(json.dumps(checks,ensure_ascii=False,indent=2))
