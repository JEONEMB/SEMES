"""Generate demo sessions for apps/web from the real modules.

Runs build/Release/main.exe (joint / control) on every photo a scenario reaches,
applies the agreed two-stage rules, locates the three QR boxes per photo with the
same NCC search joint uses, and writes preview JPEGs plus static/demo/data.js.
Temporary: replaced by the C++ service API once it exists.
"""
import json, math, os, subprocess, sys, time
import cv2
import numpy as np

REPO = sys.argv[1]
OUT = os.path.join(REPO, "apps", "web", "static", "demo")
DATA = os.path.join(REPO, "data")
MAIN = os.path.join(REPO, "build", "Release", "main.exe")

RULES = dict(tolerance=0.09, control_entry_t=0.09, max_iterations=20)
QR_BOXES = [(1552, 969, 184, 198), (1809, 971, 184, 198), (2067, 973, 185, 198)]  # joint observer.cpp
ROI = (1120, 620, 1600, 900)  # QR close-up, 16:9

os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
center = cv2.imread(os.path.join(DATA, "clean", "clean_20261014", "center.bmp"), cv2.IMREAD_GRAYSCALE)
templates = [center[y:y + h, x:x + w] for x, y, w, h in QR_BOXES]

def read_gray(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)

def locate_qr(img):
    boxes = []
    for (x, y, w, h), tpl in zip(QR_BOXES, templates):
        for pad in (130, 260, 512):
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            r = cv2.matchTemplate(img[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(r)
            if score >= 0.6:
                break
        boxes.append(dict(x=x0 + loc[0], y=y0 + loc[1], w=w, h=h, ncc=round(float(score), 4)))
    return boxes

image_cache = {}
def preview(rel):
    if rel in image_cache:
        return image_cache[rel]
    img = read_gray(os.path.join(DATA, rel))
    key = f"p{len(image_cache) + 1:02d}"
    full = cv2.resize(img, (1600, 900), interpolation=cv2.INTER_AREA)
    rx, ry, rw, rh = ROI
    roi = img[ry:ry + rh, rx:rx + rw]
    cv2.imwrite(os.path.join(OUT, "img", f"{key}_full.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 86])
    cv2.imwrite(os.path.join(OUT, "img", f"{key}_qr.jpg"), roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
    image_cache[rel] = dict(key=key, path=rel, full=f"demo/img/{key}_full.jpg", qr=f"demo/img/{key}_qr.jpg", qr_boxes=locate_qr(img))
    return image_cache[rel]

est_cache = {}
def estimate(mode, rel):
    k = (mode, rel)
    if k not in est_cache:
        t0 = time.perf_counter()
        p = subprocess.run([MAIN, mode, os.path.join(DATA, rel)], capture_output=True, text=True, cwd=REPO)
        ms = (time.perf_counter() - t0) * 1000
        if p.returncode != 0:
            est_cache[k] = dict(ok=False, error=p.stderr.strip())
        else:
            d = json.loads(p.stdout)
            est_cache[k] = dict(ok=True, y=d["y"], t=d["t"], measurement_valid=d["measurement_valid"],
                                outside_training_range=d["outside_training_range"], process_ms=round(ms))
    return est_cache[k]

def load_scenario(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    st = fs.getNode("steps")
    steps = [dict(label=st.at(i).getNode("label").string(), y=st.at(i).getNode("y").real(),
                  t=st.at(i).getNode("t").real(), image=st.at(i).getNode("image").string()) for i in range(st.size())]
    return dict(name=fs.getNode("name").string(), category=fs.getNode("category").string(),
                description=fs.getNode("description").string(), tol_y=fs.getNode("tolerance_y").real(),
                tol_t=fs.getNode("tolerance_t").real(), steps=steps)

def nearest(sc, y, t):
    best = min(sc["steps"], key=lambda s: abs(s["y"] - y) / sc["tol_y"] + abs(s["t"] - t) / sc["tol_t"])
    if abs(best["y"] - y) <= sc["tol_y"] and abs(best["t"] - t) <= sc["tol_t"]:
        return best
    return None

def r4(v):
    return round(v, 6)

def run(sc):
    pose = dict(y=sc["steps"][0]["y"], t=sc["steps"][0]["t"])
    stage, records, result = "joint", [], None
    for i in range(RULES["max_iterations"]):
        shot = nearest(sc, pose["y"], pose["t"])
        if shot is None:
            result = dict(status="failed", reason=f"시뮬레이션 자세 Y {pose['y']:+.4f}, T {pose['t']:+.4f} 근처에 준비된 사진이 없습니다")
            break
        img = preview(shot["image"])
        e = estimate(stage, shot["image"])
        rec = dict(index=i + 1, stage=stage, frame=dict(image=img["key"], label=shot["label"], pose=dict(y=shot["y"], t=shot["t"])),
                   pose_before=dict(y=r4(pose["y"]), t=r4(pose["t"])), estimate=e, command=dict(dy=0.0, dt=0.0), moved=False)
        if not e["ok"]:
            rec.update(decision="측정 실패", next="failed")
            records.append(rec)
            result = dict(status="failed", reason=e["error"])
            break
        tol = RULES["tolerance"]
        if stage == "joint":
            if abs(e["t"]) > RULES["control_entry_t"]:
                rec["command"] = dict(dy=0.0, dt=r4(-e["t"]))
                rec.update(moved=True, decision=f"T {abs(e['t']):.4f}가 기준 {RULES['control_entry_t']:.2f}보다 커서 T만 이동", next="joint")
            else:
                rec.update(decision=f"T {abs(e['t']):.4f}가 기준 {RULES['control_entry_t']:.2f} 이내라 2차로 전환", next="control")
        else:
            t_axis = e["y"] == 0 and e["t"] != 0
            if abs(e["y"]) <= tol and abs(e["t"]) <= tol:
                rec.update(decision="두 축 모두 기준 0.09 이내", next="done")
            elif t_axis:
                rec.update(decision="2차가 T 축을 선택해 1차로 복귀", next="joint")
            else:
                rec["command"] = dict(dy=r4(-e["y"]), dt=0.0)
                rec.update(moved=True, decision=f"Y {abs(e['y']):.4f}가 기준 0.09보다 커서 Y만 이동", next="control")
        pose = dict(y=pose["y"] + rec["command"]["dy"], t=pose["t"] + rec["command"]["dt"])
        rec["pose_after"] = dict(y=r4(pose["y"]), t=r4(pose["t"]))
        records.append(rec)
        if rec["next"] == "done":
            result = dict(status="done", reason="두 축 모두 기준 0.09 이내")
            break
        stage = rec["next"]
    if result is None:
        result = dict(status="failed", reason=f"최대 {RULES['max_iterations']}회를 넘었습니다")
    result.update(steps=len(records), moves=sum(r["moved"] for r in records),
                  final_pose=dict(y=r4(pose["y"]), t=r4(pose["t"])))
    return records, result

scen_dir = os.path.join(DATA, "scenarios")
scenarios = []
for name in sorted(os.listdir(scen_dir)):
    p = os.path.join(scen_dir, name, "scenario.yml")
    if not os.path.isfile(p):
        continue
    sc = load_scenario(p)
    records, result = run(sc)
    scenarios.append(dict(id=sc["name"], category=sc["category"], description=sc["description"],
                          start=dict(y=sc["steps"][0]["y"], t=sc["steps"][0]["t"]), steps=records, result=result))
    print(sc["name"], result["status"], result["steps"], "steps", result.get("reason", ""))

demo = dict(
    generated=time.strftime("%Y-%m-%d %H:%M"),
    source="build/Release/main.exe 실행 결과 (시나리오 재생, 실제 촬영 아님)",
    models=dict(joint="00000_final_Y_T_joint", control="00000_final_Y_T control", adopted=True),
    rules=RULES, roi=dict(zip(("x", "y", "w", "h"), ROI)), frame=dict(w=3840, h=2160),
    images={v["key"]: v for v in image_cache.values()}, scenarios=scenarios)
with open(os.path.join(OUT, "data.js"), "w", encoding="utf-8") as f:
    f.write("// Generated demo data. Real joint/control estimates on scenario photos.\n")
    f.write("window.SEMES_DEMO = " + json.dumps(demo, ensure_ascii=False, indent=1) + ";\n")
print("images", len(image_cache), "estimates", len(est_cache))
