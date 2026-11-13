"""장비 사진 목록을 만든다(자료 준비만 한다. 측정은 C++ 도구가 한다).

입력: data/clean/rig_20261112/{rig_1st,rig_grid} (data/raw/rig_1st.zip, rig_grid.zip 을 그대로 푼 것)
출력
  <출력 폴더>/photos.tsv   id<TAB>사진 절대 경로 (C++ 도구 입력, 머리줄 없음)
  결과/촬영목록.tsv        id, set, kind, seq, file, Y_cmd, T_cmd, status_y_um, status_t_mdeg, time, seconds
id 규칙: rig_1st 는 100000 + 촬영 순번(0 = 센터), rig_grid 는 200000 + 촬영 순번(0 = 센터).
         다시 찍은 센터(_check)는 그 세트의 90000 + 직전 순번(end 는 마지막 순번 + 1).
seconds 는 rig_1st 센터 촬영 시각(14:54:39)부터 잰 초다. 명령값은 파일 이름이 아니라 manifest.csv 의 command 열에서 읽는다(둘은 같다).

실행: 저장소 루트에서  python "<이 폴더>/prepare_rig_inputs.py" <출력 폴더(ASCII 경로)>
"""
import csv, datetime, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RIG = ROOT / "data/clean/rig_20261112"
T0 = datetime.datetime(2026, 11, 12, 14, 54, 39)


def sidecar(path):
    return dict(line.split(": ", 1) for line in path.read_text(encoding="utf-8").splitlines() if ": " in line)


def main():
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for base, name in ((100000, "rig_1st"), (200000, "rig_grid")):
        manifest = list(csv.DictReader((RIG / name / "manifest.csv").open(encoding="utf-8")))
        for seq, r in enumerate(manifest):
            assert r["move_result"] == "OK" and r["settled"] == "1"
            rows.append(dict(id=base + seq, set=name, kind="center" if seq == 0 else "shot", seq=seq, file=f"{name}/{r['file']}.bmp",
                             Y_cmd=float(r["command_y_mm"]), T_cmd=float(r["command_t_deg"]), status_y_um=r["status_y_um"], status_t_mdeg=r["status_t_mdeg"], time=r["time"]))
        last = len(manifest) - 1
        for p in sorted((RIG / name / "_check").glob("center_*.bmp")):
            tag = p.stem.split("_")[-1]
            seq = last + 1 if tag == "end" else int(tag)
            s = sidecar(p.with_suffix(".txt"))
            rows.append(dict(id=base + 90000 + seq, set=name, kind="recheck", seq=seq, file=f"{name}/_check/{p.name}", Y_cmd=0.0, T_cmd=0.0,
                             status_y_um=s["status_abs_y_um"], status_t_mdeg=s["status_abs_t_mdeg"], time=s["time"]))
    for r in rows:
        assert (RIG / r["file"]).exists(), r["file"]
        r["seconds"] = int((datetime.datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S") - T0).total_seconds())
    rows.sort(key=lambda r: (r["seconds"], r["id"]))
    with (out / "photos.tsv").open("w", encoding="utf-8", newline="") as f:
        for r in rows: f.write(f"{r['id']}\t{(RIG / r['file']).as_posix()}\n")
    keys = ["id", "set", "kind", "seq", "file", "Y_cmd", "T_cmd", "status_y_um", "status_t_mdeg", "time", "seconds"]
    with (HERE.parent / "결과" / "촬영목록.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, keys, delimiter="\t"); w.writeheader(); w.writerows(rows)
    print("photos", len(rows), {k: sum(1 for r in rows if r["kind"] == k) for k in ("center", "shot", "recheck")})


if __name__ == "__main__":
    main()
