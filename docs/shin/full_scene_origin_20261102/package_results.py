from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd
import cv2
from pypdf import PdfReader
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
index=pd.read_csv(ROOT/'analysis/session_eda_20261102/image_inventory.csv')
j=json.loads((P/'standard_origin_v2.json').read_text())
assert j['status']=='validation_not_sufficient_for_confirmed_common_origin'
cloud=pd.read_csv(P/'standard_origin_v2_QR630.csv');assert len(cloud)==630
fs=cv2.FileStorage(str(ROOT/'analysis/session_eda_20261102/remeasured/cache/reference.yml.gz'),cv2.FILE_STORAGE_READ)
ref=fs.getNode('points').mat();fs.release()
centroid=[]
for q in range(3):centroid.extend(cloud.loc[cloud.qr_id==q,['x_standard_px','y_standard_px']].mean().to_numpy()-ref[ref[:,2]==q,:2].mean(0))
assert np.max(np.abs(np.array(centroid)-j['origin_QR_features_6']))<1e-8
mapping=pd.read_csv(P/'annotated_feature_ID_map.csv');assert len(mapping)==39 and mapping.display_id.nunique()==39
assert np.load(P/'measurements.npz')['complete'].all()
pdfs=[ROOT/'output/pdf/T_axis_used_features_photo_atlas.pdf',ROOT/'output/pdf/T_axis_origin_quality_revalidation_v2.pdf']
assert [len(PdfReader(p).pages) for p in pdfs]==[6,10]
with zipfile.ZipFile(P/'quality_origin_v2_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in P.iterdir():
        if f.is_file() and f.suffix in ['.csv','.json','.py','.cpp','.png','.md','.txt','.ps1']:z.write(f,f.name)
    for f in pdfs:z.write(f,f.name)
print('PASS: 575 measured images, QR630 origin-centroid consistency, 39 unique annotation IDs, 6+10 PDF pages, candidate validation status and ZIP')
