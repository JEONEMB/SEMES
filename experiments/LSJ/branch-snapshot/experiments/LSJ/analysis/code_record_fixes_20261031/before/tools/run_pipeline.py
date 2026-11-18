"""Build, test and run line_methods with full logging.

python tools/run_pipeline.py            # configure + build + selftest + full run + repeat check + single-method check
python tools/run_pipeline.py --no-build # reuse the existing build

Every external command is logged under logs/ with command line, stdout, stderr, exit code and wall time.
Writes only inside this analysis folder.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
CMAKE = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
VS_INSTANCE = "C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools"
OPENCV_DIR = "C:/Users/SSAFY/opencv/build"
OPENCV_DLL = Path(r"C:\Users\SSAFY\opencv\build\x64\vc15\bin\opencv_world3415.dll")
EXE = ROOT / "build" / "Release" / "line_methods.exe"
SELFTEST = ROOT / "build" / "Release" / "line_selftest.exe"
MARKER = ".created_by_line_methods_pipeline"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(step, cmd, cwd=ROOT):
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True)
    elapsed = time.time() - start
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    (LOGS / f"{step}.stdout.txt").write_text(out, encoding="utf-8")
    (LOGS / f"{step}.stderr.txt").write_text(err, encoding="utf-8")
    rec = {"step": step, "command": cmd, "cwd": str(cwd), "exit_code": proc.returncode, "wall_time_s": round(elapsed, 3),
           "started": datetime.fromtimestamp(start).isoformat(timespec="seconds")}
    (LOGS / f"{step}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{step}] exit={proc.returncode} {elapsed:.1f}s", flush=True)
    return proc.returncode, out


def prepare_results(path):
    """Clear a results folder only if this pipeline created it; never touch other files."""
    if path.exists():
        if not (path / MARKER).exists() and any(path.iterdir()):
            raise SystemExit(f"{path} has files not created by this pipeline; refusing to overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / MARKER).write_text("created by tools/run_pipeline.py\n", encoding="utf-8")


def numeric_fields(rec):
    keys = ["status", "slope_dy_dx", "intercept_y_px", "image_angle_deg", "fit_rms_px", "n_candidates", "n_selected"]
    return {k: rec.get(k) for k in keys}


def compare_runs(a_path, b_path):
    a = {r["measurement_id"]: numeric_fields(r) for r in json.loads(Path(a_path).read_text(encoding="utf-8"))["records"]}
    b = {r["measurement_id"]: numeric_fields(r) for r in json.loads(Path(b_path).read_text(encoding="utf-8"))["records"]}
    diffs = []
    for mid in sorted(set(a) & set(b)):
        for k, va in a[mid].items():
            vb = b[mid][k]
            if isinstance(va, float) and isinstance(vb, float):
                if va != vb:
                    diffs.append({"measurement_id": mid, "field": k, "a": va, "b": vb, "abs_diff": abs(va - vb)})
            elif va != vb:
                diffs.append({"measurement_id": mid, "field": k, "a": va, "b": vb})
    return {"compared_ids": len(set(a) & set(b)), "only_in_a": sorted(set(a) - set(b)), "only_in_b": sorted(set(b) - set(a)),
            "differences": diffs, "identical": not diffs}


def tree_hashes(base, patterns):
    out = {}
    for pat in patterns:
        for p in sorted(base.glob(pat)):
            if p.is_file():
                out[str(p.relative_to(ROOT)).replace("\\", "/")] = sha(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "configs/inputs_manifest.json").read_text(encoding="utf-8"))
    before = {r["reference_id"]: sha(r["path"]) for r in manifest["references"]}

    if not args.no_build:
        rc, _ = run("01_configure", [CMAKE, "-S", "cpp", "-B", "build", "-G", "Visual Studio 18 2026", "-A", "x64", "-T", "v141",
                                     f"-DCMAKE_GENERATOR_INSTANCE={VS_INSTANCE}", f"-DOpenCV_DIR={OPENCV_DIR}"])
        if rc: sys.exit(rc)
        rc, _ = run("02_build", [CMAKE, "--build", "build", "--config", "Release", "--", "-m"])
        if rc: sys.exit(rc)
    rc, _ = run("03_selftest", [str(SELFTEST)])
    if rc: sys.exit(rc)

    results = ROOT / "results"
    prepare_results(results)
    rc, _ = run("04_full_run", [str(EXE), "--root", ".", "--out", "results"])
    if rc: sys.exit(rc)

    repeat = LOGS / "repeat_run"
    prepare_results(repeat)
    rc, _ = run("05_repeat_full_run", [str(EXE), "--root", ".", "--out", "logs/repeat_run"])
    if rc: sys.exit(rc)
    rep = compare_runs(results / "line_measurements.json", repeat / "line_measurements.json")
    (LOGS / "repeatability.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("repeat identical:", rep["identical"], "differences:", len(rep["differences"]))
    # Keep only the comparison; the repeated intermediates duplicate results/.
    shutil.rmtree(repeat / "intermediates", ignore_errors=True)

    single = LOGS / "single_method_example"
    prepare_results(single)
    rc, _ = run("06_single_method_run", [str(EXE), "--root", ".", "--out", "logs/single_method_example", "--methods", "E04,Q12", "--refs", "R02"])
    if rc: sys.exit(rc)
    sm = compare_runs(results / "line_measurements.json", single / "line_measurements.json")
    (LOGS / "single_vs_full.json").write_text(json.dumps(sm, ensure_ascii=False, indent=2), encoding="utf-8")
    print("single-method run matches full run:", sm["identical"], "ids:", sm["compared_ids"])

    after = {r["reference_id"]: sha(r["path"]) for r in manifest["references"]}
    cl_dir = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC")
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "cmake": CMAKE, "generator": "Visual Studio 18 2026", "toolset": "v141",
        "msvc_toolsets_installed": sorted(p.name for p in cl_dir.iterdir()) if cl_dir.exists() else None,
        "opencv_dir": OPENCV_DIR,
        "opencv_dll": {"path": str(OPENCV_DLL), "size": OPENCV_DLL.stat().st_size, "sha256": sha(OPENCV_DLL)},
        "opencv_dll_copied_next_to_exe": {"path": str(EXE.parent / OPENCV_DLL.name), "sha256": sha(EXE.parent / OPENCV_DLL.name)},
        "exe_sha256": sha(EXE),
        "inputs_sha256_before": before, "inputs_sha256_after": after,
        "inputs_match_manifest": all(before[r["reference_id"]] == r["sha256"] == after[r["reference_id"]] for r in manifest["references"]),
        "system_changes": "none (no PATH change, no install, no OpenCV change)",
    }
    (LOGS / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    hashes = {
        "cpp": tree_hashes(ROOT, ["cpp/**/*.cpp", "cpp/**/*.hpp", "cpp/CMakeLists.txt"]),
        "configs": tree_hashes(ROOT, ["configs/*.json"]),
        "third_party_sources": tree_hashes(ROOT, ["third_party/lsd_1.6/*", "third_party/devernay_1.0/*", "third_party/ED_Lib/*",
                                                  "third_party/ELSED/*", "third_party/fld_port/*", "third_party/nlohmann_json/include/nlohmann/json.hpp"]),
        "third_party_downloads": tree_hashes(ROOT, ["third_party/_downloads/*"]),
        "tools": tree_hashes(ROOT, ["tools/*.py"]),
    }
    (LOGS / "hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    print("inputs unchanged and match manifest:", env["inputs_match_manifest"])


if __name__ == "__main__":
    main()
