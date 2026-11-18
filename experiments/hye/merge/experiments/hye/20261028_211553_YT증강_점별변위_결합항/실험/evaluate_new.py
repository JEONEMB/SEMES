"""새로 받은 사진으로 증강 생성기를 평가한다 (실험용 Python, 팀 전달 코드 아님).

    python -I evaluate_new.py <사진 폴더> <결과 이름>

1. 사진마다 고정 630점을 기존과 같은 흐린 경계 직접 맞춤으로 측정한다
   (joint_center_630_20261022/measurement.py 의 StructuralExtractor, 기준은 20261014 center.bmp).
2. 오늘 고정한 results/generator.npz 로 예측한 점과 실제 점을 비교한다. 생성기는 다시 맞추지 않는다.
   - 새 센터 사진: 기존 영점 15개 구름 안에 들어오는지 (주성분 2개 공간의 마할라노비스 거리, 가장 가까운 영점)
   - 나머지 사진: 영점 가정 3가지로 잔차를 잰다
       a. 새 센터 사진들의 평균을 영점으로 (사진 정보만으로 가능한 예측)
       a2. a + 접근 방향 백래시 (T가 있으면 T 부호, 없으면 Y 부호로 +/− 접근을 가정)
       b. 기존 영점 15개 중 가장 잘 맞는 것 (영점 후보가 충분한지)
       c. 새 사진들로 세션 영점 1개를 맞춤 (진단용, 새 자료를 씀)
     결합항 YT는 끈 것과 켠 것을 둘 다 계산한다.
3. 잔차 RMS를 반복 촬영 잡음 바닥(오늘 0.12px)과 비교한다.
결과: results/new_<이름>/ (덮어쓰지 않는다)
"""
import csv, hashlib, json, re, sys
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
EXT = HERE.parents[3] / "joint_center_630_20261022"     # 측정 코드 (저장소 밖, 비교용 참고)
CENTER = HERE.parents[2] / "data" / "clean" / "clean_20261014" / "center.bmp"
GEN = HERE / "results" / "generator.npz"


def label(path):
    stem = path.stem
    if re.fullmatch(r"center[0-9]*", stem, re.I): return 0.0, 0.0, True
    v = {"Y": 0.0, "T": 0.0}
    for a, s, x in re.findall(r"([YT])([+-])(\d+(?:\.\d*)?)", re.sub(r"\s*\(\d+\)$", "", stem)): v[a] = float(x) * (1 if s == "+" else -1)
    return v["Y"], v["T"], False


def measure(folder, out):
    sys.path.insert(0, str(EXT))
    from measurement import StructuralExtractor
    ex = StructuralExtractor(cv2.imdecode(np.fromfile(str(CENTER), np.uint8), 0))
    rows, Ds, Vs = [], [], []
    for p in sorted(Path(folder).rglob("*.bmp")):
        Y, T, c = label(p); raw = p.read_bytes()
        r = ex.extract(cv2.imdecode(np.frombuffer(raw, np.uint8), 0))
        rows.append(dict(file=str(p.relative_to(folder)), Y=Y, T=T, center=c, sha256=hashlib.sha256(raw).hexdigest(), valid=int(r["valid"].sum())))
        Ds.append(r["D"]); Vs.append(r["valid"]); print("measured", rows[-1]["file"], flush=True)
    np.savez_compressed(out / "measurements.npz", D=np.stack(Ds), valid=np.stack(Vs))
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    return rows, np.stack(Ds), np.stack(Vs)


def main(folder, name):
    out = HERE / "results" / f"new_{name}"
    out.mkdir(parents=True, exist_ok=False)
    g = np.load(GEN)
    sys.path.insert(0, str(HERE))
    from generate import basis
    fam, coef, sel, O, yt = str(g["family"]), g["coef"], g["sel"], g["origins"], g["yt_coupling"]
    names = [str(x) for x in g["origin_names"]]
    floor = float(np.load(HERE / "results" / "fit.npz")["noise_floor_rms"])
    rows, D, V = measure(folder, out)
    pred0 = lambda Y, T, cpl: np.tensordot(basis([Y], [T], fam)[0], coef, 1) + (Y * T * yt if cpl else 0)
    rms = lambda r, ok: float(np.sqrt(np.mean(r[ok & sel] ** 2)))

    # 영점 구름: 실측 11개의 주성분 2개 공간
    meas = [i for i, n in enumerate(names) if not n.startswith("PC")]
    X = O[meas][:, sel].reshape(len(meas), -1); mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False); sig = S[:2] / np.sqrt(len(meas) - 1)
    pc = lambda field: ((field[sel].reshape(-1) - mu) @ Vt[:2].T) / sig

    cen = [i for i, r in enumerate(rows) if r["center"]]
    o_new = D[cen].mean(0) if cen else None
    report = dict(generator_sha256=hashlib.sha256(GEN.read_bytes()).hexdigest(), photos=len(rows), centers=len(cen), noise_floor=floor, centers_detail=[], photos_detail=[])
    for i in cen:
        z = pc(D[i]); near = min(range(len(names)), key=lambda k: rms(D[i] - O[k], V[i]))
        report["centers_detail"].append(dict(file=rows[i]["file"], mdx=float(D[i][sel, 0].mean()), mdy=float(D[i][sel, 1].mean()),
                                             pc_sigma=[float(z[0]), float(z[1])], inside_2sigma=bool(np.hypot(*z) <= 2),
                                             nearest=names[near], nearest_rms=rms(D[i] - O[near], V[i])))
    others = [i for i in range(len(rows)) if not rows[i]["center"]]
    for cpl in (False, True):
        base = {i: D[i] - pred0(rows[i]["Y"], rows[i]["T"], cpl) for i in others}
        ses = np.mean([base[i] for i in others], 0) if others else None
        for i in others:
            r = dict(file=rows[i]["file"], Y=rows[i]["Y"], T=rows[i]["T"], coupling=cpl)
            if o_new is not None:
                sgn = np.sign(rows[i]["T"]) or np.sign(rows[i]["Y"])
                o2 = o_new + (g["backlash_plus"] if sgn > 0 else g["backlash_minus"])
                r.update(a_new_center_rms=rms(base[i] - o_new, V[i]), a_mdx=float((base[i] - o_new)[sel, 0].mean()),
                         a2_backlash_rms=rms(base[i] - o2, V[i]), a2_mdx=float((base[i] - o2)[sel, 0].mean()))
            best = min(range(len(names)), key=lambda k: rms(base[i] - O[k], V[i]))
            r.update(b_best_origin=names[best], b_rms=rms(base[i] - O[best], V[i]), c_session_rms=rms(base[i] - ses, V[i]),
                     outside_range=abs(rows[i]["Y"]) > 2 or abs(rows[i]["T"]) > 1)
            report["photos_detail"].append(r)
    agg = {}
    for cpl in (False, True):
        L = [r for r in report["photos_detail"] if r["coupling"] == cpl and not r["outside_range"]]
        for k in ("a_new_center_rms", "a2_backlash_rms", "b_rms", "c_session_rms"):
            v = [r[k] for r in L if k in r]
            if v: agg[f"{'YT' if cpl else 'none'}|{k}"] = dict(n=len(v), rms=float(np.sqrt(np.mean(np.square(v)))), max=float(max(v)), over_2x_floor=int(sum(x > 2 * floor for x in v)))
    report["summary"] = agg
    json.dump(report, open(out / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(centers=report["centers_detail"], summary=agg), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
