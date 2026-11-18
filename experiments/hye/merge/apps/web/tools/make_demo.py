"""Generate demo sessions for apps/web from the real modules.

두 모델 중 하나를 골라 재생한다 (2026-11-13 사용자 지시).
  semes: SEMES 사진으로 학습한 채택 모듈(selected_modules.cmake 의 SEMES_JOINT_DIR). build/Release/main.exe joint 를 사진마다 실행한다.
  rig:   우리 장비 사진 1959장 전체로 학습한 값 7개 모델(experiments/hye/20261113_0950_.../실험/assets_rig, L2 라벨, 실험 자료).
         사진마다 공개 API(initialize_joint, joint)로 낸 예측 결과/predict_L2/predictions.tsv 를 읽는다.
Y와 T를 함께 움직여 두 축이 기준 안에 들 때까지 반복하고(2차 control 은 쓰지 않는다), 사진마다 QR 상자를 NCC 로 찾아
미리보기 JPEG 와 static/demo/data.js 를 쓴다.
Temporary: replaced by the C++ service API once it exists.
"""
import csv, hashlib, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

REPO = sys.argv[1]
OUT = os.path.join(REPO, "apps", "web", "static", "demo")
DATA = os.path.join(REPO, "data")
MAIN = os.path.join(REPO, "build", "Release", "main.exe")
RIG_EXP = os.path.join(REPO, "experiments", "hye", "20261113_0950_우리장비사진거부_장비기준자료와라벨보정")
RIG_ASSETS = os.path.join(RIG_EXP, "실험", "assets_rig")

# 2026-11-13 사용자 지시: 한 장 허용오차 |Y| <= 0.045, |T| <= 0.01. Y와 T를 동시에 보정하고 2차 control 은 쓰지 않는다.
# include/pipeline.hpp 의 서비스 흐름 기본값(두 축 0.09)과 다르다. 화면에 이 기준을 표시한다.
RULES = dict(tolerance_y=0.045, tolerance_t=0.01, max_iterations=20, sim_tolerance_t=0.02)

JOINT_DIR = dict(re.findall(r'set\(SEMES_(JOINT|CONTROL)_DIR\s+"([^"]+)"\)',
                 open(os.path.join(REPO, "experiments", "selected_modules.cmake"), encoding="utf-8-sig").read()))["JOINT"]
RIG_PROV = json.load(open(os.path.join(RIG_ASSETS, "provenance.json"), encoding="utf-8"))
RIG_SITE = json.load(open(os.path.join(RIG_ASSETS, "site.json"), encoding="utf-8"))

# 사진의 화면 자리. 장비 사진은 QR 이 SEMES 보다 약 62 px 아래에 있다(data/clean/rig_20261112/README_20261112.md).
SITES = dict(
    semes=dict(center=os.path.join(DATA, "clean", "clean_20261014", "center.bmp"),
               qr=[(1552, 969, 184, 198), (1809, 971, 184, 198), (2067, 973, 185, 198)],  # joint observer.cpp
               crop=(1340, 800, 1130, 530),  # control tracker full_crop
               points=os.path.join(REPO, "experiments", "hye", "20261028_1327_final_Y_T control", "assets", "points.json"),
               roi=(1120, 620, 1600, 900)),
    rig=dict(center=os.path.join(RIG_ASSETS, "center.bmp"),
             qr=[tuple(q) for q in RIG_SITE["qr"]], crop=tuple(RIG_SITE["full_crop"]),
             points=os.path.join(RIG_ASSETS, "points.json"),
             roi=(1125, 682, 1600, 900)),
)
MODELS = dict(
    semes=dict(id="semes", name=os.path.basename(JOINT_DIR), label="SEMES 사진으로 학습",
               train="SEMES 1~4차 702장", path=JOINT_DIR, t_unit="명령 단위", adopted=True,
               note="채택 모듈. 우리 장비 사진은 화면 자리가 달라 거부한다"),
    rig=dict(id="rig", name=RIG_PROV["model_id"], label="우리 장비 사진 전체로 학습",
             train=f"우리 장비 {RIG_PROV['training_images']}장 전체 (rig_1st 280, rig_grid 1679, 보정 라벨 L2)",
             path=os.path.relpath(RIG_ASSETS, REPO).replace("\\", "/"), t_unit="도", adopted=False,
             note="실험 자료, 전달 모듈 아님. 학습한 사진을 그대로 다시 넣은 재대입 결과. 부호는 우리 장비 스테이지 방향(SEMES 와 반대)"),
)
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)


def read_gray(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)


def load_site(k):
    S = dict(SITES[k])
    S["center_img"] = read_gray(S["center"])
    S["templates"] = [S["center_img"][y:y + h, x:x + w] for x, y, w, h in S["qr"]]
    pts = [q for q in json.load(open(S["points"], encoding="utf-8"))["points"] if q[3] == 1]
    S["ref_pts"] = np.array([[q[0], q[1]] for q in pts], np.float32)
    S["ref_qr"] = np.array([int(q[2]) for q in pts])
    cx, cy, cw, ch = S["crop"]
    S["center_blur"] = cv2.GaussianBlur(S["center_img"][cy:cy + ch, cx:cx + cw], (0, 0), 0.8)
    return S


SITE = {k: load_site(k) for k in SITES}


def locate_qr(img, S):
    boxes = []
    for (x, y, w, h), tpl in zip(S["qr"], S["templates"]):
        for pad in (130, 260, 512):
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            r = cv2.matchTemplate(img[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(r)
            if score >= 0.6:
                break
        boxes.append(dict(x=x0 + loc[0], y=y0 + loc[1], w=w, h=h, ncc=round(float(score), 4)))
    return boxes


# Feature-point motion: the 536 fixed QR points (points.json selected=1) against that site's center photo.
# Approximation for display: per-QR NCC shift + LK 15x15 on sigma 0.8 blur (the modules' GN refinement omitted).
def feature_motion(img, boxes, S):
    cx, cy, cw, ch = S["crop"]
    REF_PTS, REF_QR = S["ref_pts"], S["ref_qr"]
    cur = cv2.GaussianBlur(img[cy:cy + ch, cx:cx + cw], (0, 0), 0.8)
    p0 = (REF_PTS - [cx, cy]).astype(np.float32)
    shift = np.array([[b["x"] - q[0], b["y"] - q[1]] for b, q in zip(boxes, S["qr"])], np.float32)
    p1 = (p0 + shift[REF_QR]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-4)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(S["center_blur"], cur, p0.reshape(-1, 1, 2), p1.reshape(-1, 1, 2), winSize=(15, 15), maxLevel=0, criteria=crit, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, S["center_blur"], p1, p0.reshape(-1, 1, 2).copy(), winSize=(15, 15), maxLevel=0, criteria=crit, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    d = p1.reshape(-1, 2) - p0
    valid = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back.reshape(-1, 2) - p0, axis=1) < 0.5)
    # similarity fit on valid points: dx = tx - r*v + s*u, dy = ty + r*u + s*v (u, v from the point centroid)
    c = REF_PTS[valid].mean(axis=0)
    u, v = (REF_PTS[valid] - c).T
    A = np.zeros((2 * valid.sum(), 4)); b = np.zeros(2 * valid.sum())
    A[0::2] = np.c_[np.ones_like(u), np.zeros_like(u), -v, u]; b[0::2] = d[valid, 0]
    A[1::2] = np.c_[np.zeros_like(u), np.ones_like(u), u, v]; b[1::2] = d[valid, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = np.sqrt(np.mean((A @ sol - b) ** 2))
    return dict(
        points=len(REF_PTS), valid=int(valid.sum()), crop=dict(x=cx, y=cy, w=cw, h=ch),
        tx=round(float(sol[0]), 4), ty=round(float(sol[1]), 4), rotation_mrad=round(float(sol[2]) * 1000, 4),
        scale_ppm=round(float(sol[3]) * 1e6, 1), residual_px=round(float(residual), 4),
        field=[[round(float(x), 1), round(float(y), 1), round(float(a), 3), round(float(bb), 3), int(ok)] for (x, y), (a, bb), ok in zip(REF_PTS, d, valid)])


image_cache = {}
def preview(rel, site):
    if rel in image_cache:
        return image_cache[rel]
    S = SITE[site]
    img = read_gray(os.path.join(DATA, rel))
    key = f"p{len(image_cache) + 1:02d}"
    full = cv2.resize(img, (1600, 900), interpolation=cv2.INTER_AREA)
    rx, ry, rw, rh = S["roi"]
    roi = img[ry:ry + rh, rx:rx + rw]
    cv2.imwrite(os.path.join(OUT, "img", f"{key}_full.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 86])
    cv2.imwrite(os.path.join(OUT, "img", f"{key}_qr.jpg"), roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
    boxes = locate_qr(img, S)
    image_cache[rel] = dict(key=key, path=rel, full=f"demo/img/{key}_full.jpg", qr=f"demo/img/{key}_qr.jpg", qr_boxes=boxes,
                            roi=dict(zip(("x", "y", "w", "h"), S["roi"])), motion=feature_motion(img, boxes, S))
    return image_cache[rel]


# 장비 모델: final00_tool predict 가 공개 API 로 1959장을 한 장씩 예측한 결과 (센터 사진은 목록에 없다)
def tsv(*parts):
    return list(csv.DictReader(open(os.path.join(RIG_EXP, "결과", *parts), encoding="utf-8"), delimiter="\t"))
RIG_SHOTS = {r["id"]: r for r in tsv("촬영목록.tsv")}
RIG_L2 = {r["id"]: r for r in tsv("labels_L2.tsv")}
RIG_PRED = {"clean/rig_20261112/" + RIG_SHOTS[r["id"]]["file"]: r for r in tsv("predict_L2", "predictions.tsv")}

# SEMES 모델의 main.exe 결과는 build/demo_cache 에 모아 둔다(사진 719장 전체는 한 번에 약 4분, 8개씩 실행).
# 모델이 바뀌면 파일 이름이 달라져 새로 잰다.
CACHE_DIR = os.path.join(REPO, "build", "demo_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
SEMES_CACHE = os.path.join(CACHE_DIR, f"semes_{os.path.basename(JOINT_DIR)}.json")
est_cache = {("semes", k): v for k, v in (json.load(open(SEMES_CACHE, encoding="utf-8")) if os.path.isfile(SEMES_CACHE) else {}).items()}
def save_semes_cache():
    json.dump({k[1]: v for k, v in est_cache.items() if k[0] == "semes"}, open(SEMES_CACHE, "w", encoding="utf-8"), ensure_ascii=False)

def estimate(model, rel):
    k = (model, rel)
    if k in est_cache:
        return est_cache[k]
    if model == "rig":
        r = RIG_PRED.get(rel)
        if r is None:
            est_cache[k] = dict(ok=False, error="장비 모델 예측 결과에 없는 사진")
        else:
            est_cache[k] = dict(ok=r["valid"] == "1", y=float(r["Y"]), t=float(r["T"]), measurement_valid=r["valid"] == "1",
                                outside_training_range=r["outside_training_range"] == "1", process_ms=None,
                                error=r["message"] or None)
        return est_cache[k]
    t0 = time.perf_counter()
    p = subprocess.run([MAIN, "joint", os.path.join(DATA, rel)], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=REPO)
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


# Simulated camera after a move: the nearest photo of that model's data within tolerance,
# never the reference itself (it is the module's assets/center.bmp, so it would compare to itself).
# semes: data/clean 의 SEMES 사진, 자세 = 파일명 명령값.
# rig:   장비 사진 1959장, 자세 = 보정 라벨 L2(절대 영점, 영상으로 잰 실제 자세에 가깝다).
#        명령값은 영점이 T 약 −0.023 도 어긋나 있어 자세로 쓰지 않는다.
SIM_TOL_T = RULES["sim_tolerance_t"]
REFERENCE_SHA = "01a69df6771b840d2fcf9cf727760592b5fb199c12277bfc173af514e26fd685"
def load_pool(model):
    pool = []
    if model == "rig":
        for i, r in RIG_L2.items():
            pool.append(dict(image="clean/rig_20261112/" + RIG_SHOTS[i]["file"], y=float(r["Y_abs"]), t=float(r["T_abs"]),
                             cmd_y=float(r["Y_cmd"]), cmd_t=float(r["T_cmd"])))
        return sorted(pool, key=lambda p: p["image"])
    for d in sorted(os.listdir(os.path.join(DATA, "clean"))):
        m = os.path.join(DATA, "clean", d, "_metadata", "manifest.csv")
        if not os.path.isfile(m):
            continue
        for r in csv.DictReader(open(m, encoding="utf-8-sig")):
            if r["sha256"] == REFERENCE_SHA:
                continue
            num = lambda s: float(s) if s else 0.0
            y, t = num(r["filename_Y_command"]), num(r["filename_T_command"])
            pool.append(dict(image=f"clean/{d}/{r['relative_path']}", y=y, t=t, cmd_y=y, cmd_t=t))
    return sorted(pool, key=lambda p: p["image"])
POOL = {k: load_pool(k) for k in MODELS}


def shot_label(head, s, model):
    if model == "rig":
        return f"{head} (명령 Y {s['cmd_y']:+.2f}, T {s['cmd_t']:+.2f}, 보정 자세 Y {s['y']:+.3f}, T {s['t']:+.3f})"
    return f"{head} (명령 Y {s['cmd_y']:+.2f}, T {s['cmd_t']:+.2f})"


def nearest(sc, y, t):
    # 허용 범위 안의 사진 가운데 가장 가까운 것을 고른다
    tol_t = max(sc["tol_t"], SIM_TOL_T)
    ok = [s for s in POOL[sc["model"]] if abs(s["y"] - y) <= sc["tol_y"] and abs(s["t"] - t) <= tol_t]
    if ok:
        best = min(ok, key=lambda s: abs(s["y"] - y) / sc["tol_y"] + abs(s["t"] - t) / tol_t)
        return dict(best, label=shot_label("재촬영 대체 사진", best, sc["model"]))
    return None


def r4(v):
    return round(v, 6)


def run(sc):
    pose = dict(y=sc["steps"][0]["y"], t=sc["steps"][0]["t"])
    records, result = [], None
    tol_y, tol_t = RULES["tolerance_y"], RULES["tolerance_t"]
    for i in range(RULES["max_iterations"]):
        shot = dict(sc["steps"][0], label=sc["steps"][0].get("label") or "시나리오 시작 사진") if i == 0 else nearest(sc, pose["y"], pose["t"])
        if shot is None:
            result = dict(status="failed", reason=f"시뮬레이션 자세 Y {pose['y']:+.4f}, T {pose['t']:+.4f} 근처에 데이터 사진이 없습니다")
            break
        img = preview(shot["image"], sc["model"])
        e = estimate(sc["model"], shot["image"])
        rec = dict(index=i + 1, stage="joint", frame=dict(image=img["key"], label=shot["label"], pose=dict(y=shot["y"], t=shot["t"])),
                   pose_before=dict(y=r4(pose["y"]), t=r4(pose["t"])), estimate=e, command=dict(dy=0.0, dt=0.0), moved=False)
        if not e["ok"]:
            rec.update(decision="측정 실패", next="failed")
            records.append(rec)
            result = dict(status="failed", reason=e.get("error") or "측정 실패")
            break
        # 2026-11-13 사용자 지시: 2차 control 을 쓰지 않는다. 모델의 (Y, T) 로 두 축을 한 번에 움직이고 통과할 때까지 반복한다.
        if abs(e["y"]) <= tol_y and abs(e["t"]) <= tol_t:
            rec.update(decision=f"Y {abs(e['y']):.4f} (기준 {tol_y:.3f}), T {abs(e['t']):.4f} (기준 {tol_t:.2f}) 모두 이내", next="done")
        else:
            over = [n for n, v, k in (("Y", e["y"], tol_y), ("T", e["t"], tol_t)) if abs(v) > k]
            rec["command"] = dict(dy=r4(-e["y"]), dt=r4(-e["t"]))
            rec.update(moved=True, decision=f"{', '.join(over)}가 기준 밖이라 Y와 T를 함께 이동", next="joint")
        pose = dict(y=pose["y"] + rec["command"]["dy"], t=pose["t"] + rec["command"]["dt"])
        rec["pose_after"] = dict(y=r4(pose["y"]), t=r4(pose["t"]))
        records.append(rec)
        if rec["next"] == "done":
            result = dict(status="done", reason=f"Y {tol_y:.3f}, T {tol_t:.2f} 이내")
            break
    if result is None:
        result = dict(status="failed", reason=f"최대 {RULES['max_iterations']}회를 넘었습니다")
    result.update(steps=len(records), moves=sum(r["moved"] for r in records),
                  final_pose=dict(y=r4(pose["y"]), t=r4(pose["t"])))
    return records, result


scenarios = []
def add(sc):
    records, result = run(sc)
    scenarios.append(dict(id=sc["name"], model=sc["model"], category=sc["category"], description=sc["description"],
                          start=dict(y=sc["steps"][0]["y"], t=sc["steps"][0]["t"]), steps=records, result=result))
    print(sc["model"], sc["name"], result["status"], result["steps"], "steps", result.get("reason", ""))


scen_dir = os.path.join(DATA, "scenarios")
for name in sorted(os.listdir(scen_dir)):
    p = os.path.join(scen_dir, name, "scenario.yml")
    if os.path.isfile(p):
        sc = load_scenario(p)
        sc["steps"][0]["label"] = ""
        add(dict(sc, model="semes"))

# 장비 시나리오: SEMES 6개와 같은 구성의 시작 사진. 자세는 그 사진의 보정 라벨 L2.
# 이웃 사진 허용은 격자 간격(Y 0.1 mm, T 0.05 도)의 절반.
RIG_START = [("R01_Y_small", "Y 작게", "rig_1st/Y/Y+0.20.bmp"), ("R02_Y_large", "Y 크게", "rig_1st/Y/Y+1.50.bmp"),
             ("R03_T_small", "T 작게", "rig_1st/Theta/T+0.20.bmp"), ("R04_T_large", "T 크게", "rig_1st/Theta/T+1.00.bmp"),
             ("R05_YT_pp_small", "YT ++ 작게", "rig_grid/Grid/Y+0.10 T+0.10 (1).bmp"),
             ("R06_YT_pp_large", "YT ++ 크게", "rig_grid/Grid/Y+1.00 T+1.00 (1).bmp")]
by_image = {s["image"]: s for s in POOL["rig"]}
for name, cat, f in RIG_START:
    s0 = by_image["clean/rig_20261112/" + f]
    add(dict(name=name, model="rig", category=cat, tol_y=0.05, tol_t=0.025,
             description=f"우리 장비 {f.split('/')[0]} 명령 Y {s0['cmd_y']:+.2f}, T {s0['cmd_t']:+.2f}에서 시작",
             steps=[dict(image=s0["image"], y=s0["y"], t=s0["t"], label=shot_label("시나리오 시작 사진", s0, "rig"))]))

# ---------- 시뮬레이션 자료 (2026-11-13 사용자 지시) ----------
# 화면에서 아무 (Y, T)나 넣고 바로 돌린다. 이동 뒤 자세 근처(시나리오와 같은 허용)에 실제 사진이 있으면 그 사진의 실제 모델 추정값을 쓰고,
# 없으면 "참 자세 + 그 모델의 검증 오차(주 검증 rms, Y·T 독립 정규)"로 추정한다(오차 모형). 계산은 화면(app.js)에서 한다.
# 그래서 모델마다 사진 전체의 추정값과 QR 상자, QR 확대 미리보기(640×360)를 미리 만든다.
def eval_rms(path):
    line = next(l for l in open(path, encoding="utf-8") if l.startswith("main_cv_fails"))
    v = line.split()
    return float(v[v.index("rmsY") + 1]), float(v[v.index("rmsT") + 1])
ERR = dict(semes=(eval_rms(os.path.join(REPO, JOINT_DIR, "evidence", "cpp", "eval.txt")), f"{JOINT_DIR}/evidence/cpp/eval.txt 주 검증"),
           rig=(eval_rms(os.path.join(RIG_EXP, "결과", "eval_L2", "eval.txt")), "11-13 실험 결과/eval_L2/eval.txt 주 검증"))
SIM_TOL = dict(semes=(0.025, max(0.005, SIM_TOL_T)), rig=(0.05, 0.025))
THUMB = (640, 360)

semes_todo = [ph["image"] for ph in POOL["semes"] if ("semes", ph["image"]) not in est_cache]
print("semes estimates to run", len(semes_todo))
with ThreadPoolExecutor(8) as ex:
    list(ex.map(lambda rel: estimate("semes", rel), semes_todo))
save_semes_cache()

BOX_CACHE = os.path.join(CACHE_DIR, "sim_boxes.json")
box_cache = json.load(open(BOX_CACHE, encoding="utf-8")) if os.path.isfile(BOX_CACHE) else {}
def sim_photo(model, ph):
    rel = ph["image"]
    name = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12] + ".jpg"
    out = os.path.join(OUT, "sim", model, name)
    if rel not in box_cache or not os.path.isfile(out):
        S = SITE[model]
        img = read_gray(os.path.join(DATA, rel))
        box_cache[rel] = [[b["x"], b["y"], b["w"], b["h"], b["ncc"]] for b in locate_qr(img, S)]
        rx, ry, rw, rh = S["roi"]
        cv2.imwrite(out, cv2.resize(img[ry:ry + rh, rx:rx + rw], THUMB, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
    e = estimate(model, rel)
    return dict(p=rel, y=round(ph["y"], 5), t=round(ph["t"], 5), cy=ph["cmd_y"], ct=ph["cmd_t"], img=f"demo/sim/{model}/{name}",
                q=box_cache[rel], ok=int(e["ok"]), ey=round(e["y"], 6) if e["ok"] else None, et=round(e["t"], 6) if e["ok"] else None,
                err=None if e["ok"] else (e.get("error") or "측정 실패")[:200])
sim = dict(rules=RULES, models={})
for model in MODELS:
    os.makedirs(os.path.join(OUT, "sim", model), exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        photos = list(ex.map(lambda ph: sim_photo(model, ph), POOL[model]))
    (sy, st), src = ERR[model]
    sim["models"][model] = dict(tol_y=SIM_TOL[model][0], tol_t=SIM_TOL[model][1], roi=dict(zip(("x", "y", "w", "h"), SITES[model]["roi"])),
                                err=dict(sd_y=sy, sd_t=st, source=src), range=dict(y=[-2, 2], t=[-1, 1]), photos=photos)
    print("sim", model, len(photos), "photos,", sum(not ph["ok"] for ph in photos), "measurement failures")
json.dump(box_cache, open(BOX_CACHE, "w", encoding="utf-8"))
with open(os.path.join(OUT, "sim.js"), "w", encoding="utf-8") as f:
    f.write("// Generated by apps/web/tools/make_demo.py. Per-photo real estimates for the service tab simulation.\n")
    f.write("window.SEMES_SIM = " + json.dumps(sim, ensure_ascii=False, separators=(",", ":")) + ";\n")

demo = dict(
    generated=time.strftime("%Y-%m-%d %H:%M"),
    source="시나리오 재생, 실제 촬영 아님. Y와 T 동시 보정, 판정 기준 Y 0.045, T 0.01",
    models=MODELS, default_model="semes",
    rules=RULES, roi=dict(zip(("x", "y", "w", "h"), SITES["semes"]["roi"])), frame=dict(w=3840, h=2160),
    images={v["key"]: v for v in image_cache.values()}, scenarios=scenarios)
with open(os.path.join(OUT, "data.js"), "w", encoding="utf-8") as f:
    f.write("// Generated demo data (apps/web/tools/make_demo.py). Real Y/T estimates of two models on scenario photos.\n")
    f.write("window.SEMES_DEMO = " + json.dumps(demo, ensure_ascii=False, indent=1) + ";\n")
print("images", len(image_cache), "estimates", len(est_cache))
