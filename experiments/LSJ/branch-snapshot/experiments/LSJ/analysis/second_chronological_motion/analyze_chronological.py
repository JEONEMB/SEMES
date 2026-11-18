"""Direct ZNCC -> affine ECC for consecutive second-data images in ZIP time order.

No subtraction of earlier center-relative measurements, no new axis separation.
The two unit equivalents apply separate single-axis assumptions to image dx.
"""

from collections import Counter
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MARKERS = ("L", "C", "R")
SEARCH_MARGIN = 230
MAX_ITERATIONS = 60
EPSILON = 1e-6
GAUSSIAN = 5
Y_SCALE = 14.36
THETA_SCALE = 80.4
KINDS = {"same_label": "동일 라벨", "Y_only": "Y만 변화",
         "Theta_only": "θ만 변화", "both_axes": "두 축 변화"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_image(path, expected_hash):
    data = np.fromfile(path, dtype=np.uint8)
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"Source hash mismatch: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2160, 3840):
        raise ValueError(f"Unexpected image: {path}")
    return image


def quadratic(a, b, c):
    denominator = a - 2*b + c
    return float(np.clip(.5*(a-c)/denominator, -1, 1)) if denominator < -1e-10 else 0.0


def zncc(reference, current, roi, margin):
    x0, y0, x1, y1 = map(int, roi)
    if not (0 <= x0-margin < x1+margin <= current.shape[1]
            and 0 <= y0-margin < y1+margin <= current.shape[0]):
        raise ValueError(f"Search region exceeds image: {roi}")
    template = reference[y0:y1, x0:x1]
    search = current[y0-margin:y1+margin, x0-margin:x1+margin]
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, maximum, _, (mx, my) = cv2.minMaxLoc(scores)
    edge_distance = min(mx, my, scores.shape[1]-1-mx, scores.shape[0]-1-my)
    if edge_distance == 0:
        raise ValueError("ZNCC peak reached search boundary")
    offset = np.array([mx-margin, my-margin], dtype=np.float64)
    subpixel = offset + [quadratic(scores[my,mx-1], scores[my,mx], scores[my,mx+1]),
                         quadratic(scores[my-1,mx], scores[my,mx], scores[my+1,mx])]
    return {"peak": float(maximum), "integer_delta_xy_px": offset.tolist(),
            "subpixel_delta_xy_px": subpixel.tolist(), "peak_distance_to_boundary_px": int(edge_distance)}


def locate(reference, current, base_rois):
    result = {}
    for marker, roi in base_rois.items():
        initial = zncc(reference, current, roi, SEARCH_MARGIN)
        dx, dy = map(int, initial["integer_delta_xy_px"])
        x0, y0, x1, y1 = roi
        result[marker] = {"roi_xyxy": [x0+dx,y0+dy,x1+dx,y1+dy], "zncc": initial}
    return result


def register_pair(reference, current, roi):
    initial = zncc(reference, current, roi, SEARCH_MARGIN)
    x0,y0,x1,y1 = roi
    dx,dy = map(int, initial["integer_delta_xy_px"])
    template = reference[y0:y1,x0:x1]
    target = current[y0+dy:y1+dy,x0+dx:x1+dx]
    warp = np.eye(2,3,dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, MAX_ITERATIONS, EPSILON)
    correlation,warp = cv2.findTransformECC(template, target, warp, cv2.MOTION_AFFINE, criteria, None, GAUSSIAN)
    center = np.array([(x1-x0-1)/2, (y1-y0-1)/2, 1.0])
    delta = np.array([dx,dy]) + warp @ center - center[:2]
    determinant = float(np.linalg.det(warp[:,:2]))
    if not np.all(np.isfinite(warp)) or not np.isfinite(correlation) or determinant <= 0:
        raise ValueError("Invalid affine fit")
    return {"reference_roi_xyxy": roi,
            "reference_roi_center_xy": [x0+center[0],y0+center[1]],
            "zncc": initial,
            "affine": {"correlation": float(correlation), "delta_xy_px": delta.tolist(),
                       "warp_local_2x3": warp.tolist(), "linear_determinant": determinant}}


def aggregate(regions):
    zncc_xy = np.array([regions[m]["zncc"]["subpixel_delta_xy_px"] for m in MARKERS])
    affine_xy = np.array([regions[m]["affine"]["delta_xy_px"] for m in MARKERS])
    z = zncc_xy.mean(axis=0)
    a = affine_xy.mean(axis=0)
    return {"zncc_mean_delta_xy_px": z.tolist(), "affine_mean_delta_xy_px": a.tolist(),
            "affine_mean_vector_length_px": float(np.linalg.norm(a)),
            "affine_marker_dx_range_px": float(np.ptp(affine_xy[:,0])),
            "min_zncc_peak": min(regions[m]["zncc"]["peak"] for m in MARKERS),
            "min_affine_correlation": min(regions[m]["affine"]["correlation"] for m in MARKERS),
            "zncc_Y_only_equivalent_mm": float(z[0]/Y_SCALE),
            "zncc_Theta_only_equivalent_degree": float(z[0]/THETA_SCALE),
            "affine_Y_only_equivalent_mm": float(a[0]/Y_SCALE),
            "affine_Theta_only_equivalent_degree": float(a[0]/THETA_SCALE)}


def capture_record(row, order):
    return {"order": order, "file": row["file"], "time": row["archive_modified_time_no_timezone"],
            "label_y_mm": row["label_y_mm"], "label_theta_degree": row["label_theta_degree"],
            "Y_Actual": row["Y_Actual"], "T_Actual": row["T_Actual"], "sha256": row["sha256"]}


def short_row(pair):
    a=pair["aggregate"]
    return [f"{pair['from']['order']:02d}→{pair['to']['order']:02d}", KINDS[pair["kind"]],
            f"{pair['label_delta_y_mm']:+.2f}", f"{pair['label_delta_theta_degree']:+.2f}",
            f"{a['zncc_mean_delta_xy_px'][0]:+.4f}", f"{a['zncc_mean_delta_xy_px'][1]:+.4f}",
            f"{a['affine_mean_delta_xy_px'][0]:+.4f}", f"{a['affine_mean_delta_xy_px'][1]:+.4f}",
            f"{a['affine_mean_vector_length_px']:.4f}",
            f"{a['affine_Y_only_equivalent_mm']:+.6f}",
            f"{a['affine_Theta_only_equivalent_degree']:+.6f}"]


def report(result):
    explanation = [
        "2차 데이터: 수정 시각순 인접 이미지의 ZNCC → 아핀 ECC 변위",
        "비교: 60장을 시간순으로 정렬하여 1→2, 2→3, ..., 59→60을 원본 이미지에서 직접 정합 (59쌍).",
        "기존 center1 대비 변위의 차분이 아니며, 사진 3장의 평균끼리 비교하지도 않았습니다.",
        "좌표 부호: 다음 사진 - 이전 사진. 영상 오른쪽 +Δx, 아래쪽 +Δy.",
        "각 QR를 별도로 측정한 뒤, QR 3개 ROI 중심의 변위 벡터를 산술 평균했습니다.",
        "ZNCC는 TM_CCOEFF_NORMED, 최대점 주변 이차보간값도 저장. 정수 최대점으로 아핀 ECC를 초기화했습니다.",
        "탐색 범위 ±230px / ECC 최대 60회, epsilon=1e-6, Gaussian 5. 기존 분석 설정이며 합격 기준이 아닙니다.",
        "시간은 원본 ZIP 내 BMP 수정 시각이며 시간대와 실제 노출 시각은 확인되지 않았습니다.",
        "라벨 ΔY(mm), Δθ(°)는 파일명 차이입니다. TXT Actual 차이는 JSON에 별도 저장했습니다.",
        "Y가정환산(mm) = 아핀 Δx / 14.36(px/mm). θ가정환산(°) = 같은 아핀 Δx / 80.4(px/°).",
        "위 배율은 기존 1차 단일축 결과에서 사용한 근삿값입니다. ZNCC 기준 환산값도 JSON에 저장했습니다.",
        "두 환산값은 서로 다른 단일축 가정의 참고값이며 동시 추정한 실제 Y·θ 이동량이나 기계 오차가 아닙니다.",
        "θ만 변한 구간도 1차 배율을 적용한 근사 환산입니다. 같은 라벨의 차이는 어느 축에서 생겼는지 알 수 없습니다.",
        "아핀 행렬의 영상 내 회전각을 로봇 PAN θ로 사용하지 않았습니다.",
        "표시 단위의 소수 자릿수는 계산 결과 표시용이며 실제 측정 정확도를 뜻하지 않습니다.",
    ]
    summary=result["summary"]
    explanation += ["", "구간 수: "+", ".join(f"{KINDS[k]} {v}쌍" for k,v in summary["kind_counts"].items()),
                    f"QR 정합: {summary['qr_pair_count']}개 / 실패: 0개 / 최소 ZNCC {summary['min_zncc_peak']:.6f} / 최소 ECC {summary['min_affine_correlation']:.6f}",
                    "위 상관 점수는 정합 상태 참고값이며 정확도 검증이나 합격 판정이 아닙니다."]
    headings=["순번", "라벨구분", "라벨ΔY(mm)", "라벨Δθ(°)", "ZNCCΔx(px)", "ZNCCΔy(px)",
              "아핀Δx(px)", "아핀Δy(px)", "아핀변위길이(px)", "Y가정환산(mm)", "θ가정환산(°)"]
    lines=explanation+["", " | ".join(headings)]
    lines += [" | ".join(short_row(p)) for p in result["pairs"]]
    lines += ["", "이미지 순번표: 순번 | 수정 시각 | 파일 | Y 라벨(mm) | θ 라벨(°)"]
    lines += [f"{r['order']:02d} | {r['time']} | {r['file']} | {r['label_y_mm']:.2f} | {r['label_theta_degree']:.2f}" for r in result["captures"]]
    (HERE/"chronological_results.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    table=[]
    for p in result["pairs"]:
        cells=short_row(p)
        detail=f"{Path(p['from']['file']).name} → {Path(p['to']['file']).name}\n{p['from']['time']} → {p['to']['time']}"
        table.append(f'<tr data-kind="{p["kind"]}" title="{html.escape(detail,quote=True)}">'+"".join(f"<td>{html.escape(c)}</td>" for c in cells)+"</tr>")
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2차 시간순 ZNCC·아핀 변위</title><style>
body{font-family:'Malgun Gothic',sans-serif;margin:30px;color:#182b43;background:#f5f7fa}h1{font-size:24px}.card{background:white;border:1px solid #d9e2ec;padding:20px;border-radius:10px;margin:18px 0}p{line-height:1.6;margin:6px 0;font-size:14px}.table{max-height:70vh;overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:10px;border-bottom:1px solid #e3e9ef;white-space:nowrap;text-align:right}th{position:sticky;top:0;background:#e7eff8}td:nth-child(1),td:nth-child(2){text-align:left}tr[data-kind='both_axes']{background:#fff4e8}tr[data-kind='Theta_only']{background:#eef5fc}select{font:inherit;padding:7px;margin:10px}small{color:#52657b}
</style><h1>2차 데이터 · 메타데이터 순서의 인접 사진 비교</h1><div class="card">'''
    document += "".join("<p>"+html.escape(line)+"</p>" for line in explanation[1:])
    document += '''</div><div class="card"><label>구간 표시 <select id="filter"><option value="all">전체 59쌍</option><option value="same_label">동일 라벨</option><option value="Theta_only">θ만 변화</option><option value="both_axes">두 축 변화</option></select></label><small>행에 마우스를 올리면 파일명과 시간을 볼 수 있습니다.</small><div class="table"><table><thead><tr>'''
    document += "".join("<th>"+html.escape(h)+"</th>" for h in headings)+"</tr></thead><tbody>"+"".join(table)+"</tbody></table></div></div>"
    document += '''<script>document.getElementById('filter').addEventListener('change',event=>{document.querySelectorAll('tbody tr').forEach(row=>{row.hidden=event.target.value!=='all'&&row.dataset.kind!==event.target.value})});</script></html>'''
    (HERE/"chronological_results.html").write_text(document,encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cv2.setNumThreads(2)
    started=time.perf_counter()
    inventory=read_json(BASE/"second_data_inspection/inventory.json")
    previous_measurements=read_json(BASE/"increment_consistency/second_measurements.json")
    rois={m:previous_measurements["regions_xyxy"][m] for m in MARKERS}
    root=Path(inventory["summary"]["extracted_root"])
    with ZipFile(inventory["summary"]["source_archive"]) as archive:
        entries={i.filename:i for i in archive.infolist() if not i.is_dir()}
        assert {n for n in entries if n.lower().endswith('.bmp')} == {r['file'] for r in inventory['images']}
        for row in inventory['images']:
            assert "%04d-%02d-%02d %02d:%02d:%02d" % entries[row['file']].date_time == row['archive_modified_time_no_timezone']
    rows=sorted(inventory["images"],key=lambda r:(r["archive_modified_time_no_timezone"],r["file"]))
    assert len(rows)==60 and len({r['archive_modified_time_no_timezone'] for r in rows})==60
    assert Path(rows[0]["file"]).name=="center1.bmp"
    captures=[capture_record(r,i) for i,r in enumerate(rows,1)]
    reference=read_image(root/rows[0]["file"],rows[0]["sha256"])
    previous=reference
    previous_locations=locate(reference,previous,rois)
    localizations=[{"order":1,"regions":previous_locations}]
    pairs=[]
    for i,row in enumerate(rows[1:],1):
        current=read_image(root/row["file"],row["sha256"])
        regions={m:register_pair(previous,current,previous_locations[m]["roi_xyxy"]) for m in MARKERS}
        before=rows[i-1]
        dy=round(row['label_y_mm']-before['label_y_mm'],8)
        dt=round(row['label_theta_degree']-before['label_theta_degree'],8)
        kind='same_label' if dy==dt==0 else 'Y_only' if dt==0 else 'Theta_only' if dy==0 else 'both_axes'
        pairs.append({"from":captures[i-1],"to":captures[i],"kind":kind,
                      "time_gap_seconds":(datetime.fromisoformat(row['archive_modified_time_no_timezone'])-datetime.fromisoformat(before['archive_modified_time_no_timezone'])).total_seconds(),
                      "label_delta_y_mm":dy,"label_delta_theta_degree":dt,
                      "sidecar_actual_delta_y":row['Y_Actual']-before['Y_Actual'],
                      "sidecar_actual_delta_theta":row['T_Actual']-before['T_Actual'],
                      "regions":regions,"aggregate":aggregate(regions)})
        previous=current
        previous_locations=locate(reference,current,rois)
        localizations.append({"order":i+1,"regions":previous_locations})
        if i%10==0 or i==59:
            print(f"Direct consecutive pairs: {i}/59",flush=True)
    summary={"image_count":60,"pair_count":len(pairs),"qr_pair_count":len(pairs)*3,
             "kind_counts":dict(Counter(p['kind'] for p in pairs)),
             "min_zncc_peak":min(p['aggregate']['min_zncc_peak'] for p in pairs),
             "min_affine_correlation":min(p['aggregate']['min_affine_correlation'] for p in pairs),
             "min_peak_distance_to_search_boundary_px":min(p['regions'][m]['zncc']['peak_distance_to_boundary_px'] for p in pairs for m in MARKERS),
             "elapsed_seconds":time.perf_counter()-started}
    for kind in ("same_label","Theta_only","both_axes"):
        selected=[p for p in pairs if p['kind']==kind]
        worst=max(selected,key=lambda p:p['aggregate']['affine_mean_vector_length_px'])
        summary[kind]={"max_affine_vector_pair":[worst['from']['order'],worst['to']['order']],
                       "max_affine_vector_length_px":worst['aggregate']['affine_mean_vector_length_px'],
                       "min_affine_dx_px":min(p['aggregate']['affine_mean_delta_xy_px'][0] for p in selected),
                       "max_affine_dx_px":max(p['aggregate']['affine_mean_delta_xy_px'][0] for p in selected)}
    result={"method":{"registration":"Direct consecutive source-image ZNCC initialization then affine ECC, not differences of center-relative registrations",
                      "roi":"Three established QR ROIs; each previous image ROI is localized from center1 by integer ZNCC; affine displacement evaluated at that previous ROI center",
                      "search_margin_px":SEARCH_MARGIN,"ecc_max_iterations":MAX_ITERATIONS,"ecc_epsilon":EPSILON,"gaussian_size":GAUSSIAN,
                      "units":"pixels in original resolution; target-minus-reference; x right and y down; arithmetic mean of three QR centers",
                      "parameters_note":"Reuse earlier implementation parameters; no new sensitivity study or accuracy/acceptance claim",
                      "opencv_version":cv2.__version__,"numpy_version":np.__version__},
            "sources":{"inventory":"../second_data_inspection/inventory.json","roi_source":"../increment_consistency/second_measurements.json (ROI coordinates only)",
                       "source_archive":inventory['summary']['source_archive'],"data_root":str(root),"all_image_sha256_verified":True,"all_zip_bmp_times_verified":True},
            "conversions":{"Y_scale_px_per_mm":Y_SCALE,"Theta_scale_px_per_degree":THETA_SCALE,
                           "scale_source":"Previously reported approximate first-dataset single-axis image sensitivities; not refitted here",
                           "formula":"Y_only_equivalent = dx / Y_scale; Theta_only_equivalent = same dx / Theta_scale",
                           "limitation":"Separate mutually exclusive single-axis equivalents, not joint Y/Theta estimates, verified physical moves, or motor errors"},
            "summary":summary,"captures":captures,"localizations":localizations,"pairs":pairs}
    assert len(pairs)==59 and len(localizations)==60
    (HERE/"chronological_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    report(result)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
