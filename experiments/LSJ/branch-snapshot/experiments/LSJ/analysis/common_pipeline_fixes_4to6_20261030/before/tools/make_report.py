"""report.html / report.md from results/, logs/, configs/ and figures/ (no measurement is computed here)."""
import html
import json
from pathlib import Path
from fit_point_views import fit_point_view, convergence_label, rms_label

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


lm = jload(RES / "line_measurements.json")
combos = jload(RES / "combinations.json")
mstatus = jload(RES / "method_status.json")
targets = {t["id"]: t for t in jload(ROOT / "configs/targets.json")["targets"]}
tcfg = jload(ROOT / "configs/targets.json")
methods_cfg = jload(ROOT / "configs/methods.json")
sources = jload(ROOT / "configs/method_sources.json")
env = jload(ROOT / "logs/environment.json")
rep = jload(ROOT / "logs/repeatability.json")
single = jload(ROOT / "logs/single_vs_full.json")
crops = jload(ROOT / "figures/viewer/crops.json")
meta = lm["meta"]
recs = {r["measurement_id"]: r for r in lm["records"]}
REFS = ["R01", "R02", "R03", "R04"]
EXT_T = ["upper_structure", "lower_structure"]
QR_T = ["qr_L", "qr_C", "qr_R"]
E_IDS = [f"E{i:02d}" for i in range(1, 10)]
Q_IDS = [f"Q{i:02d}" for i in range(1, 13)]
CAT_KO = {"detection_failed": "검출 실패", "fit_failed": "피팅 실패(상세 사유 확인)", "corner_not_identified": "코너 식별 실패",
          "boundary_confusion": "경계 혼동", "not_implemented": "미구현", "input_mismatch": "입력 불일치", "exception": "실행 예외"}
T_KO = {"upper_structure": "상단 구조물", "lower_structure": "하단 구조물", "qr_L": "QR L", "qr_C": "QR C", "qr_R": "QR R"}
e = html.escape


def method_name(mid):
    for m in mstatus["methods"]:
        if m["method_id"] == mid: return m["name"]


def mcfg(mid):
    for k, v in methods_cfg["methods"].items():
        if mid in k.split("/"): return v


def fmt(v, d=5):
    return "—" if v is None else f"{v:+.{d}f}"


# ---------------------------------------------------------------- Q12 port check (old Python result, read-only)
q12_check = []
old_finder = ROOT.parent / "reference_qr_finder_20261029/finder_results.json"
if old_finder.exists():
    old = jload(old_finder)
    by_sha = {r["sha256"]: r for r in old["rows"]}
    for rid, rr in zip(REFS, lm["meta"]["inputs"]):
        o = by_sha.get(rr["sha256_before"])
        if not o: continue
        for q in "LCR":
            new = recs[f"{rid}_Q12_qr_{q}"]["image_angle_deg"]
            q12_check.append((rid, q, o["qrs"][q]["image_angle_deg"], new))

# ---------------------------------------------------------------- data embedded in the HTML
def pts_of(r):
    it = jload(RES / r["intermediate_json"])["intermediates"]
    view = fit_point_view(it)
    return ([[round(a, 2), round(b, 2), int(c)] for (a, b), c in zip(view['points'], view['display_mask'])], view['label'])


DATA = {"crops": crops, "records": {}, "combos": [], "methods": {}, "targets": {}}
for mid, r in recs.items():
    point_rows, point_label = pts_of(r) if r['status'] == 'ok' else ([], '사용점 없음')
    DATA["records"][mid] = {
        "ref": r["reference_id"], "m": r["method_id"], "t": r["target_id"], "st": r["status"],
        "cat": r["failure_category"], "why": r["failure_reason"], "ang": r["image_angle_deg"], "slope": r["slope_dy_dx"],
        "b": r["intercept_y_px"], "rms": r["fit_rms_px"], "rmsnull": r["fit_rms_null_reason"], "nc": r["n_candidates"], "ns": r["n_selected"],
        "sup": r["support_type"], "draw": r["draw_endpoints_at_x_range"], "fc": r["feature_class"], "pts": point_rows,
        "point_label": point_label, "rms_label": rms_label(r), "fit_trace": convergence_label(r), "n_fit_used": r.get('n_fit_used')}
for c in combos["combinations"]:
    DATA["combos"].append([c["combination_id"], c["status"], c["lines_ok"], c["failed_measurements"]])
for m in mstatus["methods"]:
    DATA["methods"][m["method_id"]] = m["name"]
for tid, t in targets.items():
    DATA["targets"][tid] = {"x": t["x_range"], "y": t["band_y"], "ex": t["exclude_x"], "roi": t.get("roi_xyxy")}

# ---------------------------------------------------------------- HTML pieces
def slot_matrix(ids, tids):
    head = "".join(f"<th>{rid}<br><small>{T_KO[t]}</small></th>" for rid in REFS for t in tids)
    rows = []
    for mid in ids:
        cells = []
        for rid in REFS:
            for t in tids:
                r = recs[f"{rid}_{mid}_{t}"]
                if r["status"] == "ok":
                    cells.append(f'<td class="ok" title="{e(r["measurement_id"])}: {r["image_angle_deg"]:+.5f}°">✓</td>')
                else:
                    cells.append(f'<td class="bad" title="{e(r["measurement_id"])}: {e(r["failure_reason"] or "")}">✗<br><small>{e(CAT_KO.get(r["failure_category"], r["failure_category"]))}</small></td>')
        rows.append(f'<tr><th class="l"><a href="#m-{mid}">{mid}</a> <small>{e(method_name(mid))}</small></th>{"".join(cells)}</tr>')
    return f'<table class="matrix"><tr><th>방법</th>{head}</tr>{"".join(rows)}</table>'


def angle_table(ids, tids):
    head = "".join(f"<th>{T_KO[t]}<br><small>{rid}</small></th>" for t in tids for rid in REFS)
    rows = []
    for mid in ids:
        cells = []
        for t in tids:
            for rid in REFS:
                r = recs[f"{rid}_{mid}_{t}"]
                if r["status"] == "ok":
                    cells.append(f'<td class="num">{r["image_angle_deg"]:+.4f}</td>')
                else:
                    cells.append(f'<td class="bad" title="{e(r["failure_reason"] or "")}">{e(CAT_KO.get(r["failure_category"], ""))}</td>')
        badge = ' <span class="badge">다른 특징: 파인더 중심선</span>' if mid == "Q12" else ""
        rows.append(f'<tr><th class="l"><a href="#m-{mid}">{mid}</a>{badge}</th>{"".join(cells)}</tr>')
    return f'<table class="angles"><tr><th>방법</th>{head}</tr>{"".join(rows)}</table>'


def method_status_table():
    rows = []
    for m in mstatus["methods"]:
        mid = m["method_id"]
        c = m["slot_status_counts"]
        n_ok, n_bad = c.get("ok", 0), c.get("failed", 0)
        src = sources["methods"][mid]
        fails = [r["measurement_id"] for r in lm["records"] if r["method_id"] == mid and r["status"] != "ok"]
        rows.append(f'<tr><th class="l"><a href="#m-{mid}">{mid}</a></th><td>{e(m["name"])}</td><td><code>{e(m["implementation"])}</code></td>'
                    f'<td>{e(src["third_party_code"] or "직접 구현")}</td><td>구현·빌드됨</td><td class="num">{n_ok}</td>'
                    f'<td class="num {"bad" if n_bad else ""}">{n_bad}</td><td><small>{e(", ".join(fails)) or "—"}</small></td></tr>')
    return ('<table><tr><th>ID</th><th>방법</th><th>코드 위치</th><th>외부 코드</th><th>구현/빌드</th><th>성공 슬롯</th><th>실패 슬롯</th><th>실패 위치</th></tr>'
            + "".join(rows) + "</table>")


METHOD_NOTES = {
    "01": "S01 III.B의 Gaussian 피팅 단계를 구현했습니다. 거친 경계는 논문의 ISODATA·골격화 대신 band 안 열별 극성 Sobel 최대값을 씁니다. 기존 Python ERF(흐린 계단) 방식은 σ=1 평활 밝기에 누적정규 계단을 맞추는 것으로, 이번 E01/Q01(평활 없는 Sobel 미분에 Gaussian)과 다른 모델입니다.",
    "02": "저자 C 코드 devernay()가 Canny 비최대 억제 화소마다 Devernay 2차 보간 보정을 적용한 서브픽셀 점을 냅니다. 그림 3열은 그 점과 가장 가까운 정수 화소(반올림)를 함께 보여 줍니다. 정수 맵을 그대로 쓰지 않았습니다.",
    "03": "Canny 화소 중 대상 극성·band·가림 밖 화소만 표준 허프에 투표시키고, 최다 득표이면서 band 안에 있는 선을 고른 뒤 그 선 주변 Canny 정수 화소로 재피팅합니다. 허프 θ는 법선 각도입니다. 투표 맵은 OpenCV가 내보내지 않아 저장하지 않았고, 반환된 후보선의 득표만 저장했습니다.",
    "04": "저자 LSD 1.6을 scale=1.0으로 실행합니다(재표본화 금지). 지지 영역(reg_img)을 저장해 선택 선분의 실제 지지 화소 잔차를 진단값으로 따로 기록합니다.",
    "05": "S05 식 31·41·42·25를 그대로 구현했습니다. 7×7 창, k_edge=1.80, 흐림 σ는 계측하지 않은 설정값(external 1.5px, QR 1.0px)입니다. 하단 구조물처럼 전이 폭이 창보다 넓은 경계에서는 모델 가정이 맞지 않을 수 있습니다.",
    "06": "PAE 2차 기본형을 3×9 창으로 직접 유도·구현했습니다. 원 논문 본문을 확인하지 못했고 평활·floating 변형을 넣지 않았으므로 원 논문 그대로의 재현이 아닙니다.",
    "07": "저자 ED_Lib의 ED+EDLines를 그대로 호출합니다. 선분별 지지 화소는 ED 체인 구간으로 재구성했습니다.",
    "08": "opencv_contrib 3.4 FastLineDetector 검출 코드를 ximgproc 없이 이식했습니다. 각 선분의 fitLine 지지 체인 화소를 함께 받습니다.",
    "09": "저자 ELSED C++ 코어를 그대로 호출합니다. 반환 선분별 픽셀을 보관하도록 5줄을 추가했습니다.",
    "10": "Otsu 이진화 → 외곽 윤곽 → 4꼭짓점 근사 → 위 두 꼭짓점 사이 윤곽점(양 끝 8% 제외)에 공통 피팅합니다. 윤곽점은 정수 화소입니다.",
    "11": "Shi-Tomasi 코너 후보 중 사분면 밝기 패턴(좌상: 오른쪽 아래만 밝음, 우상: 왼쪽 아래만 밝음)을 만족하는 가장 위쪽 코너를 바깥 테두리 코너로 식별하고 cornerSubPix로 정밀화한 뒤 연결합니다. 두 점 연결이라 fit RMS는 null입니다.",
    "12": "기존 measure_finders.py를 C++로 이식했습니다. 파인더 외곽 사각형 대각선 교점(중심) A·B를 연결한 방향선이며, 바깥 테두리 상단 경계와 다른 특징입니다.",
}


def method_sections():
    out = []
    for mid in E_IDS + Q_IDS:
        src = sources["methods"][mid]
        cfg = mcfg(mid)
        kind = "external" if mid[0] == "E" else "qr"
        tids = EXT_T if kind == "external" else QR_T
        rows = []
        for t in tids:
            cells = []
            for rid in REFS:
                r = recs[f"{rid}_{mid}_{t}"]
                if r["status"] == "ok":
                    rms = "null" if r["fit_rms_px"] is None else f'{r["fit_rms_px"]:.3f}'
                    cells.append(f'<td class="num">{r["image_angle_deg"]:+.5f}°<br><small>m={r["slope_dy_dx"]:+.7f}, b={r["intercept_y_px"]:.2f}, RMS {rms}, 지지 {r["n_selected"]}</small></td>')
                else:
                    cells.append(f'<td class="bad">{e(CAT_KO.get(r["failure_category"], ""))}<br><small>{e(r["failure_reason"] or "")}</small></td>')
            rows.append(f'<tr><th class="l">{T_KO[t]}</th>{"".join(cells)}</tr>')
        sup = next((recs[f"{rid}_{mid}_{t}"]["support_type"] for rid in REFS for t in tids if recs[f"{rid}_{mid}_{t}"]["support_type"]), "—")
        tabs = "".join(f'<button class="tab{" on" if rid == "R01" else ""}" onclick="setFig(\'{mid}\',\'{rid}\',this)">{rid}</button>' for rid in REFS)
        params = json.dumps(cfg[kind], ensure_ascii=False)
        srcs = ", ".join(f'<a href="#src-{s}">{s}</a>' for s in src["sources"])
        badge = ' <span class="badge">다른 특징: 파인더 중심선</span>' if mid == "Q12" else ""
        out.append(f'''<section class="method" id="m-{mid}"><h3>{mid} · {e(method_name(mid))}{badge}</h3>
<p>{e(METHOD_NOTES[mid[1:]])}</p>
<ul class="kv"><li><b>코드</b> <code>cpp/src/{e(mstatus["methods"][[m["method_id"] for m in mstatus["methods"]].index(mid)]["implementation"])}</code></li>
<li><b>근거 자료</b> {srcs} · <b>외부 코드</b> {e(src["third_party_code"] or "없음(직접 구현)")}</li>
<li><b>변경점</b> {e(src["change"])}</li>
<li><b>지지점 종류</b> {e(sup or "—")}</li>
<li><b>설정({kind})</b> <code class="wrap">{e(params)}</code></li></ul>
<table class="mres"><tr><th>대상</th>{"".join(f"<th>{rid}</th>" for rid in REFS)}</tr>{"".join(rows)}</table>
<div class="tabs">{tabs} <span class="hint">추출 과정 그림(실제 실행 산출물). 클릭하면 원본 크기로 열립니다.</span></div>
<a id="fa-{mid}" href="figures/methods/{mid}_R01.png" target="_blank"><img id="fig-{mid}" class="stage" loading="lazy" src="figures/methods/{mid}_R01.png" alt="{mid} R01 추출 과정"></a>
</section>''')
    return "\n".join(out)


def sources_appendix():
    rows = []
    for sid, s in sources["sources"].items():
        used = [m for m, v in sources["methods"].items() if sid in v["sources"]] or (["(공통 입출력)"] if sid == "X01" else [])
        links = " ".join(f'<a href="{e(l)}">{e(l)}</a>' if l.startswith("http") else f"<code>{e(l)}</code>" for l in s["links"])
        rows.append(f'''<div class="src" id="src-{sid}"><h4>{sid} · {e(s["title"])}</h4>
<ul class="kv"><li><b>저자·연도</b> {e(s["authors"] or "—")} · {s["year"] or "—"}</li>
<li><b>DOI</b> {f'<a href="https://doi.org/{e(s["doi"])}">{e(s["doi"])}</a>' if s["doi"] else "—"}</li>
<li><b>링크</b> {links}</li>
<li><b>코드 버전/commit</b> {e(s["code"] or "—")} · <b>라이선스</b> {e(s["license"] or "—")}</li>
<li><b>사용 방법 ID</b> {e(", ".join(used))}</li>
<li><b>뒷받침하는 내용</b> {e(s["supports"])}</li>
<li><b>직접 확인 범위</b> {e(s["checked_scope"])}</li></ul></div>''')
    maprows = []
    for mid in E_IDS + Q_IDS:
        v = sources["methods"][mid]
        maprows.append(f'<tr><th class="l">{mid}</th><td>{", ".join(v["sources"])}</td><td>{e("; ".join(sources["sources"][s]["supports"] for s in v["sources"]))}</td>'
                       f'<td><small>{e(" / ".join(sources["sources"][s]["checked_scope"] for s in v["sources"]))}</small></td><td>{e(v["change"])}</td></tr>')
    return ("".join(rows) + '<h4>방법 ID별 연결</h4><table class="srcmap"><tr><th>방법</th><th>자료</th><th>뒷받침하는 내용</th><th>직접 확인 범위</th><th>우리 구현에서의 변경</th></tr>'
            + "".join(maprows) + "</table>")


fails = [r for r in lm["records"] if r["status"] != "ok"]
n_ok = len(lm["records"]) - len(fails)
inputs_rows = "".join(
    f'<tr><td>{rr["reference_id"]}</td><td>{next(r["dataset"] for r in jload(ROOT / "configs/inputs_manifest.json")["references"] if r["reference_id"] == rr["reference_id"])}</td>'
    f'<td><code class="wrap">{e(rr["path"])}</code></td><td>{rr["bmp_header"]["width"]}×{rr["bmp_header"]["height"]}, {rr["bmp_header"]["bits_per_pixel"]}-bit</td>'
    f'<td><code>{rr["sha256_before"]}</code></td><td>{"일치" if rr["sha256_before"] == rr["expected_sha256"] else "불일치"} / {"변경 없음" if rr["unchanged_after_run"] else "변경됨"}</td></tr>'
    for rr in meta["inputs"])
q12_rows = "".join(f"<tr><td>{a}</td><td>QR {q}</td><td class='num'>{o:+.6f}</td><td class='num'>{n:+.6f}</td><td class='num'>{n - o:+.2e}</td></tr>" for a, q, o, n in q12_check)

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--mut:#5f6368;--line:#dadce0;--ok:#e6f4ea;--okfg:#137333;--bad:#fce8e6;--badfg:#a50e0e;--acc:#1a73e8;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 "Malgun Gothic","Apple SD Gothic Neo",sans-serif}
nav{position:fixed;top:0;left:0;bottom:0;width:210px;overflow:auto;border-right:1px solid var(--line);padding:16px 12px;background:#f3f3f1;font-size:13px}
nav a{display:block;color:var(--fg);text-decoration:none;padding:2px 4px}nav a:hover{background:#e8eaed}
main{margin-left:210px;padding:20px 28px;max-width:1500px}
h1{font-size:22px;margin:4px 0 8px}h2{font-size:18px;border-bottom:2px solid var(--fg);padding-bottom:4px;margin-top:40px}h3{font-size:16px;margin:28px 0 6px}h4{margin:14px 0 4px}
table{border-collapse:collapse;margin:8px 0;font-size:12.5px;background:var(--card)}th,td{border:1px solid var(--line);padding:3px 6px;vertical-align:top}th{background:#f1f3f4}
th.l{text-align:left;white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.ok{background:var(--ok);color:var(--okfg);text-align:center}td.bad,.bad{background:var(--bad);color:var(--badfg)}
.matrix td{text-align:center;min-width:46px}code{font-family:Consolas,monospace;font-size:12px}code.wrap{word-break:break-all}
.note{background:#fff8e1;border-left:4px solid #f9ab00;padding:8px 12px;margin:10px 0}.badge{background:#ede7f6;color:#4527a0;border-radius:3px;padding:0 5px;font-size:11px;font-weight:normal}
img.fig{max-width:100%;border:1px solid var(--line);background:#fff}img.stage{width:100%;border:1px solid var(--line);background:#fff}
.tabs{margin:6px 0}.tab{border:1px solid var(--line);background:#fff;padding:2px 10px;cursor:pointer}.tab.on{background:var(--fg);color:#fff}.hint{color:var(--mut);font-size:12px}
ul.kv{padding-left:18px;margin:4px 0}ul.kv li{margin:1px 0}
.viewer{background:var(--card);border:1px solid var(--line);padding:10px}.ctl{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:8px}
.ctl label{white-space:nowrap}canvas{border:1px solid var(--line);background:#222;max-width:100%}#zoomwrap{cursor:grab}
.grid4{display:grid;grid-template-columns:1fr 1fr;gap:8px}.grid4 canvas{width:100%}.small{font-size:12px;color:var(--mut)}
.src{border-top:1px solid var(--line);padding-top:4px}.srcmap td{font-size:12px}
.sw{display:inline-block;width:18px;height:4px;vertical-align:middle;margin-right:3px}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const COL = {upper_structure:'#ff6438', lower_structure:'#00aedb', qr_L:'#ff2bd6', qr_C:'#39ff14', qr_R:'#ffd60a'};
const TK = {upper_structure:'상단 구조물', lower_structure:'하단 구조물', qr_L:'QR L', qr_C:'QR C', qr_R:'QR R'};
const CAT = {detection_failed:'검출 실패', fit_failed:'피팅 실패', corner_not_identified:'코너 식별 실패', input_mismatch:'입력 불일치', exception:'실행 예외'};
const IMG = {}; const PS = 1280/3840;
function img(src){ if(!IMG[src]){ const i=new Image(); i.src=src; i.onload=()=>draw(); IMG[src]=i;} return IMG[src]; }
function setFig(mid, rid, btn){ document.getElementById('fig-'+mid).src='figures/methods/'+mid+'_'+rid+'.png';
  document.getElementById('fa-'+mid).href='figures/methods/'+mid+'_'+rid+'.png';
  btn.parentNode.querySelectorAll('.tab').forEach(b=>b.classList.remove('on')); btn.classList.add('on'); }
const S = {ref:'R01', e:'E01', q:'Q01', region:'upper', z:4, px:0, py:0, show:{upper_structure:1,lower_structure:1,qr_L:1,qr_C:1,qr_R:1}, roi:1, pts:1};
function ids(ref){ return {upper_structure:`${ref}_${S.e}_upper_structure`, lower_structure:`${ref}_${S.e}_lower_structure`,
  qr_L:`${ref}_${S.q}_qr_L`, qr_C:`${ref}_${S.q}_qr_C`, qr_R:`${ref}_${S.q}_qr_R`}; }
function segs(t, r){ // drawn pieces: external line split at the occlusion (dashed there)
  const d=r.draw; if(!d) return [];
  const m=(d[1][1]-d[0][1])/(d[1][0]-d[0][0]); const y=x=>d[0][1]+m*(x-d[0][0]);
  const ex=D.targets[t].ex; if(!ex.length) return [[d[0][0],d[0][1],d[1][0],d[1][1],0]];
  const a=ex[0][0], b=ex[0][1];
  return [[d[0][0],y(d[0][0]),a,y(a),0],[a,y(a),b,y(b),1],[b,y(b),d[1][0],y(d[1][0]),0]];
}
function drawLines(ctx, ref, tf, lw){
  const I=ids(ref);
  for(const t in I){ if(!S.show[t]) continue; const r=D.records[I[t]]; if(!r) continue;
    ctx.strokeStyle=COL[t]; ctx.fillStyle=COL[t];
    if(r.st!=='ok'){ const T=D.targets[t]; const [x,y]=tf(T.x[0], T.y[0]-8); ctx.font='bold 13px sans-serif'; ctx.fillText('✗ '+TK[t]+' 실패', x, y); continue; }
    for(const s of segs(t,r)){ ctx.lineWidth=lw; ctx.setLineDash(s[4]?[6,5]:[]); ctx.beginPath(); let p=tf(s[0],s[1]); ctx.moveTo(p[0],p[1]); p=tf(s[2],s[3]); ctx.lineTo(p[0],p[1]); ctx.stroke(); }
    ctx.setLineDash([]);
  }
}
function drawROIs(ctx, tf){
  if(!S.roi) return; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  for(const t in D.targets){ const T=D.targets[t]; ctx.strokeStyle=COL[t]; const a=tf(T.x[0]-.5,T.y[0]-.5), b=tf(T.x[1]+.5,T.y[1]+.5); ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]); }
  ctx.setLineDash([]);
}
function drawMain(cv, ref, scale){
  const ctx=cv.getContext('2d'); const im=img('figures/viewer/'+ref+'_preview.jpg');
  ctx.fillStyle='#222'; ctx.fillRect(0,0,cv.width,cv.height); if(im.complete) ctx.drawImage(im,0,0,cv.width,cv.height);
  const k=cv.width/3840; const tf=(x,y)=>[(x+.5)*k,(y+.5)*k];
  drawROIs(ctx, tf); drawLines(ctx, ref, tf, scale||2);
  ctx.fillStyle='#fff'; ctx.font='12px sans-serif'; ctx.fillText(`${ref}  ${S.e} × ${S.q}  (표시 배율 ${(k).toFixed(3)}, 원본 3840×2160, 종횡비 유지)`, 8, cv.height-8);
}
function drawZoom(){
  const cv=document.getElementById('zoom'); const ctx=cv.getContext('2d'); const C=D.crops[S.region];
  const im=img('figures/viewer/'+S.ref+'_'+S.region+'.png'); ctx.imageSmoothingEnabled=false;
  ctx.fillStyle='#222'; ctx.fillRect(0,0,cv.width,cv.height);
  const z=S.z, cw=C[2]-C[0]+1, ch=C[3]-C[1]+1;
  S.px=Math.max(0,Math.min(S.px, cw-cv.width/z)); S.py=Math.max(0,Math.min(S.py, Math.max(0,ch-cv.height/z)));
  if(im.complete) ctx.drawImage(im, S.px, S.py, cv.width/z, cv.height/z, 0, 0, cv.width, cv.height);
  const tf=(x,y)=>[(x-C[0]+.5-S.px)*z,(y-C[1]+.5-S.py)*z];
  drawROIs(ctx, tf);
  if(S.pts){ const I=ids(S.ref); for(const t in I){ if(!S.show[t]) continue; const r=D.records[I[t]]; if(!r||!r.pts) continue;
     for(const p of r.pts){ const q=tf(p[0],p[1]); if(q[0]<-2||q[0]>cv.width+2||q[1]<-2||q[1]>cv.height+2) continue;
       ctx.fillStyle=p[2]?'rgba(255,159,28,.9)':'rgba(155,89,182,.9)'; ctx.fillRect(q[0]-1.5,q[1]-1.5,3,3);} } }
  drawLines(ctx, S.ref, tf, 1.5);
  document.getElementById('zinfo').textContent=`원본 픽셀 ×${z} 확대 (가로·세로 같은 배율). 보이는 범위 x ${(C[0]+S.px).toFixed(0)}–${(C[0]+S.px+cv.width/z).toFixed(0)}, y ${(C[1]+S.py).toFixed(0)}–${(C[1]+S.py+cv.height/z).toFixed(0)}. 드래그로 이동.`;
}
function table(){
  const I=ids(S.ref); let h='<table><tr><th>대상</th><th>측정 ID</th><th>상태</th><th>m (dy/dx)</th><th>b (px)</th><th>영상 각도(°)</th><th>RMS(px)·계산 대상</th><th>점·수렴 기록</th><th>추출 과정</th></tr>';
  for(const t in I){ const r=D.records[I[t]]; if(!r) continue; const ok=r.st==='ok';
    h+=`<tr><th class="l"><span class="sw" style="background:${COL[t]}"></span>${TK[t]}${r.fc==='qr_finder_center_baseline'?' <span class="badge">파인더 중심선</span>':''}</th><td><code>${I[t]}</code></td>`+
      `<td class="${ok?'ok':'bad'}">${ok?'ok':(CAT[r.cat]||r.cat)+'<br><small>'+(r.why||'')+'</small>'}</td>`+
      `<td class="num">${ok?r.slope.toFixed(8):'—'}</td><td class="num">${ok?r.b.toFixed(3):'—'}</td><td class="num">${ok?r.ang.toFixed(5):'—'}</td>`+
      `<td class="num">${r.rms==null?('null'+(r.rmsnull&&ok?'<br><small>'+r.rmsnull.slice(0,60)+'</small>':'')):r.rms.toFixed(3)}<br><small>${r.rms_label}</small></td>`+
      `<td>기존 선택 수: ${r.ns}<br>실제 fit 입력: ${r.n_fit_used==null?'미기록/해당 없음':r.n_fit_used}<br><small>${r.point_label}<br>${r.fit_trace}</small></td>`+
      `<td><a href="#m-${r.m}" onclick="goFig('${r.m}','${S.ref}')">${r.m} ${S.ref}</a></td></tr>`; }
  document.getElementById('vtable').innerHTML=h+'</table>';
  const c=D.combos.find(x=>x[0]===`${S.ref}_${S.e}_${S.q}`);
  document.getElementById('cstat').innerHTML=c?`조합 <code>${c[0]}</code>: <b>${c[1]}</b> (선 ${c[2]}/5 성공)`+(c[3].length?` · 실패 선: ${c[3].join(', ')}`:''):'';
}
function goFig(mid, ref){ const sec=document.getElementById('m-'+mid); const b=[...sec.querySelectorAll('.tab')].find(x=>x.textContent===ref); if(b) setFig(mid,ref,b); }
function draw4(){
  REFS4.forEach(ref=>drawMain(document.getElementById('c4-'+ref), ref, 1.6));
  let h='<table><tr><th>대상</th>'+REFS4.map(r=>`<th>${r}</th>`).join('')+'</tr>';
  for(const t of ['upper_structure','lower_structure','qr_L','qr_C','qr_R']){ h+=`<tr><th class="l">${TK[t]}</th>`;
    for(const ref of REFS4){ const r=D.records[ids(ref)[t]]; h+=r.st==='ok'?`<td class="num">${r.ang.toFixed(5)}°</td>`:`<td class="bad">${CAT[r.cat]||r.cat}</td>`; } h+='</tr>'; }
  document.getElementById('t4').innerHTML=h+'</table>';
}
const REFS4=['R01','R02','R03','R04'];
function draw(){ drawMain(document.getElementById('main'), S.ref, 2); drawZoom(); table(); draw4(); }
function sel(k,v){ S[k]=v; if(k==='region'){S.px=S.region==='qr'?0:300; S.py=0;} draw(); }
function setCombo(id){ const [r,e,q]=id.split('_'); S.ref=r; S.e=e; S.q=q;
  document.getElementById('sref').value=r; document.getElementById('se').value=e; document.getElementById('sq').value=q; draw();
  document.getElementById('viewer').scrollIntoView(); }
function listCombos(){ const f=document.getElementById('cfilter').value; let h='<table><tr><th>조합 ID</th><th>상태</th><th>성공 선</th><th>실패 선</th></tr>';
  for(const c of D.combos){ if(f!=='all'&&c[1]!==f) continue;
    h+=`<tr><td><a href="javascript:void(0)" onclick="setCombo('${c[0]}')">${c[0]}</a></td><td class="${c[1]==='complete'?'ok':'bad'}">${c[1]}</td><td class="num">${c[2]}/5</td><td><small>${c[3].join(', ')}</small></td></tr>`; }
  document.getElementById('clist').innerHTML=h+'</table>'; }
(function init(){
  const z=document.getElementById('zoom'); let drag=null;
  z.addEventListener('mousedown',ev=>{drag=[ev.clientX,ev.clientY,S.px,S.py];});
  window.addEventListener('mouseup',()=>drag=null);
  window.addEventListener('mousemove',ev=>{ if(!drag) return; const r=z.getBoundingClientRect(), k=z.width/r.width;
    S.px=drag[2]-(ev.clientX-drag[0])*k/S.z; S.py=drag[3]-(ev.clientY-drag[1])*k/S.z; drawZoom(); });
  document.getElementById('main').addEventListener('click',ev=>{ const r=ev.target.getBoundingClientRect(); const x=(ev.clientX-r.left)/r.width*3840, y=(ev.clientY-r.top)/r.height*2160;
    const C=D.crops; let reg=y<1300&&y>900&&x>1500&&x<2300?'qr':(y<1300?'upper':'lower'); S.region=reg; document.getElementById('sreg').value=reg;
    const c=C[reg], zz=document.getElementById('zoom'); S.px=x-c[0]-zz.width/S.z/2; S.py=y-c[1]-zz.height/S.z/2; draw(); });
  listCombos(); draw();
})();
"""

opts = lambda ids_: "".join(f'<option value="{i}">{i} · {e(method_name(i))}</option>' for i in ids_)
html_doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>기준 4장 직선 추출 비교</title><style>{CSS}</style></head><body>
<nav><b>목차</b>
<a href="#s1">1. 전체 개요</a><a href="#s2">2. 측정 대상 확인</a><a href="#s3">3. 방법별 추출 과정</a>
{''.join(f'<a href="#m-{m}" style="padding-left:14px">{m}</a>' for m in E_IDS + Q_IDS)}
<a href="#viewer">4. 108개 조합 열람</a><a href="#s5">5. 측정값 표</a><a href="#s6">6. 해석 범위·관찰</a><a href="#s7">7. 재현·검증</a><a href="#s8">부록. 근거 자료 및 구현 출처</a></nav>
<main>
<h1>기준 이미지 4장: 외부 구조물·QR 직선 추출 (외부 9 × QR 12 = 108 조합)</h1>
<p class="small">생성: {e(env["timestamp"])} · 결과 폴더 <code>analysis/reference_line_methods_108_20261030</code></p>
<div class="note">모든 각도는 <b>영상 속 선의 각도</b>(원본 픽셀, +x 오른쪽, +y 아래, atan2(dy,dx), x 증가 방향, [-90°,90°))입니다. 로봇 θ, θ 보정치, mm로 환산하지 않았습니다. 소수 자릿수는 계산 출력이며 정확도 보장이 아닙니다. 이번 결과로 통계적 균일성 합격, Y·θ 분리 성공, ±0.005° 정렬 성공, 최적 방법 확정을 주장하지 않습니다.</div>

<h2 id="s1">1. 전체 개요</h2>
<h3>사용한 4장 (실행 전후 SHA-256 확인)</h3>
<table><tr><th>ID</th><th>데이터</th><th>경로</th><th>BMP 헤더</th><th>SHA-256(실행 전)</th><th>manifest 일치 / 실행 후</th></tr>{inputs_rows}</table>
<h3>실행 환경</h3>
<ul class="kv"><li>Windows x64 · {e(env["platform"])}</li>
<li>C++17, MSVC v141 (_MSC_FULL_VER {meta.get("msvc_full_version")}), CMake(Visual Studio 18 2026 생성기, 도구 집합 v141), OpenCV {e(meta["opencv_version"])} (opencv_world3415.dll SHA-256 <code>{env["opencv_dll"]["sha256"][:16]}…</code>), OpenCV 스레드 {meta["opencv_threads"]}</li>
<li>측정 수치: 전부 C++ 실행(<code>build/Release/line_methods.exe</code>)에서 얻음. Python(OpenCV 5.0.0)은 그림·보고서 작성에만 사용.</li>
<li>시스템 변경 없음: OpenCV 교체·PATH 변경·설치 없음.</li></ul>
<h3>21개 방법 구현·실행 현황</h3>
<p>21개 방법 모두 구현·빌드·실행했습니다. 미구현·컴파일 불가 방법은 없습니다. 실제 검출 실패는 아래 표와 행렬에 그대로 남겼습니다.</p>
{method_status_table()}
<h3>216개 주 선 슬롯 (성공 {n_ok}, 실패 {len(fails)})</h3>
<p class="small">✓ 성공, ✗ 실패(범주). 셀에 마우스를 올리면 측정 ID와 각도/사유가 보입니다.</p>
<h4>외부 구조물 72슬롯</h4>{slot_matrix(E_IDS, EXT_T)}
<h4>QR 144슬롯</h4>{slot_matrix(Q_IDS, QR_T)}
<h3>432개 이미지-조합 항목</h3>
<p>complete {combos["meta"]["complete"]} · partial {combos["meta"]["partial"]} · failed {combos["meta"]["failed"]} (합계 {combos["meta"]["count"]}). partial은 모두 E04 또는 E08의 하단 구조물 실패 때문입니다. 조합은 같은 단독 결과를 재사용해 함께 열람하는 항목이며, 108개의 독립 실험이나 새 직선이 아닙니다. 외부·QR 각도를 평균하거나 합치지 않았습니다.</p>

<h2 id="s2">2. 측정 대상 확인</h2>
<p>상단은 <b>upper_structure 하나</b>, 하단은 <b>lower_structure 하나</b>입니다. 중앙 가림 구간(빗금)은 점 추출에서 빼고, 양쪽 가시 구간의 점을 <b>하나의 직선</b>으로 맞췄습니다. 좌우 기울기를 따로 구해 평균하지 않았습니다. 범위는 양 끝 포함이며 코드 내부는 [a, b+1) 반열림으로 처리합니다. QR ROI는 기존 Python 규약 그대로 x1·y1 미포함입니다.</p>
<ul class="kv"><li><span class="sw" style="background:#ff6438"></span>상단 구조물: x 24~3816, 탐색 y 692~736, 가림 x 1370~2450. 주 선 = y 증가 방향 어두움→밝음 경계</li>
<li><span class="sw" style="background:#00aedb"></span>하단 구조물: x 24~3816, 탐색 y 1845~1905, 가림 x 1560~2300. 같은 극성의 경계</li>
<li><span class="sw" style="background:#ff2bd6"></span>QR L/C/R: ROI {targets["qr_L"]["roi_xyxy"]}, {targets["qr_C"]["roi_xyxy"]}, {targets["qr_R"]["roi_xyxy"]}. Q01~Q11 주 선 = 바깥 밝은 사각 테두리 상단 경계(탐색 y 958~978, x 범위 L {targets["qr_L"]["x_range"]}, C {targets["qr_C"]["x_range"]}, R {targets["qr_R"]["x_range"]}). Q12 주 선 = 위쪽 두 파인더 중심 연결선(다른 특징)</li></ul>
<p>기존 ROI는 R01에서 물리적 대상을 모두 담고 있어 바꾸지 않았습니다. QR 상단 탐색 y 범위와 x 범위는 이번에 새로 정한 값(R01 확대 영상·밝기 프로파일 기준)이며 네 장에 같게 적용했습니다. 네 장의 결과를 보고 조정하지 않았습니다.</p>
<img class="fig" src="figures/targets_R01_overview.png" alt="R01 대상 위치">
<img class="fig" src="figures/targets_R01_zoom_profiles.png" alt="R01 대상 확대와 밝기 프로파일">
<img class="fig" src="figures/targets_4refs_same_regions.png" alt="네 장 같은 범위">

<h2 id="s3">3. 방법별 추출 과정</h2>
<p>각 그림은 저장된 후보·점·선분·코너·중간 영상으로 그립니다. 새 공통 피팅 결과는 실제 마지막 피팅 입력점과 최종 거리 분류를 따로 저장합니다. 주황 점의 의미는 각 그림·조회 표에 표시합니다. 과거 결과에는 실제 입력 마스크와 수렴 상태가 없어 최종 거리 내 점으로만 표시합니다. 선분의 주황색은 기존 선택 선분을 뜻합니다. 영상 패널은 원본 좌표·가로:세로 1:1입니다.</p>
<p class="small">공통 직선 피팅의 Huber C=1.345px, 제외 거리 external 2.0px/QR 1.0px, 재피팅 2회는 유지했습니다. 새 결과는 모든 IRLS 호출의 종료 사유·횟수·마지막 변화량을 저장하며, 반복 한도 종료나 유효성 검사 실패를 성공으로 반환하지 않습니다. 새 fit RMS는 실제 마지막 피팅 입력점 기준이며 최종 거리 내 점 RMS는 diagnostics에 별도 저장합니다. 과거 RMS는 최종 거리 내 점 기준입니다. 어느 잔차도 측정 정확도 자체가 아닙니다.</p>
{method_sections()}

<h2 id="viewer">4. 108개 조합 열람</h2>
<div class="viewer">
<div class="ctl"><label>기준 이미지 <select id="sref" onchange="sel('ref',this.value)">{"".join(f'<option>{r}</option>' for r in REFS)}</select></label>
<label>외부 방법 <select id="se" onchange="sel('e',this.value)">{opts(E_IDS)}</select></label>
<label>QR 방법 <select id="sq" onchange="sel('q',this.value)">{opts(Q_IDS)}</select></label>
{"".join(f'<label><input type="checkbox" checked onchange="S.show.{t}=this.checked?1:0;draw()"><span class="sw" style="background:{c}"></span>{T_KO[t]}</label>' for t, c in [("upper_structure", "#ff6438"), ("lower_structure", "#00aedb"), ("qr_L", "#ff2bd6"), ("qr_C", "#39ff14"), ("qr_R", "#ffd60a")])}
<label><input type="checkbox" checked onchange="S.roi=this.checked?1:0;draw()">탐색 범위</label>
<label><input type="checkbox" checked onchange="S.pts=this.checked?1:0;draw()">지지점(확대 화면)</label></div>
<div id="cstat"></div>
<canvas id="main" width="1280" height="720" title="클릭하면 그 위치를 확대 화면에 엽니다"></canvas>
<p class="small">전체 화면은 원본을 1/3(1280×720)로 줄인 미리보기입니다(종횡비 유지). 외부 구조물 선의 점선 부분은 가림 구간(측정점 없음)입니다. 이미지를 클릭하면 아래 확대 화면이 그 위치로 이동합니다.</p>
<div class="ctl"><label>확대 영역 <select id="sreg" onchange="sel('region',this.value)"><option value="upper">상단 구조물 띠</option><option value="lower">하단 구조물 띠</option><option value="qr">QR 3개</option></select></label>
<label>배율 <select onchange="sel('z',+this.value)"><option>1</option><option>2</option><option selected>4</option><option>8</option><option>16</option></select></label>
<span id="zinfo" class="small"></span></div>
<div id="zoomwrap"><canvas id="zoom" width="1200" height="380"></canvas></div>
<div id="vtable"></div>
<h3>같은 조합, 네 기준 이미지 나란히</h3>
<div class="grid4">{"".join(f'<canvas id="c4-{r}" width="720" height="405"></canvas>' for r in REFS)}</div>
<div id="t4"></div>
<h3>432개 항목 목록</h3>
<label>상태 <select id="cfilter" onchange="listCombos()"><option value="all">전체</option><option value="partial">partial</option><option value="failed">failed</option><option value="complete">complete</option></select></label>
<div id="clist" style="max-height:360px;overflow:auto"></div>
</div>

<h2 id="s5">5. 측정값 표 (영상 각도, °)</h2>
<p class="small">셀 색: 빨강 = 실패(범주 표시, 마우스를 올리면 사유). Q12는 파인더 중심 연결선이며 Q01~Q11(바깥 테두리 상단 경계)과 특징 정의가 다릅니다. 경계 혼동은 자동 판정하지 않았습니다.</p>
<h4>외부 구조물</h4>{angle_table(E_IDS, EXT_T)}
<h4>QR</h4>{angle_table(Q_IDS, QR_T)}
<p>슬롯별 전체 수치(m, b, 일반형, 기준점·방향, 끝점, 후보·지지점 수, fit RMS, 실행 시간, 설정 해시)는 <code>results/line_measurements.json</code>, <code>results/line_measurements.csv</code>에 있습니다.</p>

<h2 id="s6">6. 해석 범위와 관찰</h2>
<ul>
<li>실패 7슬롯은 모두 하단 구조물입니다. E04(LSD)는 네 장 모두 가시 구간에서 조건을 만족하는 선분을 찾지 못했습니다(검출 선분은 가림 구간 안이거나 각도 조건 탈락). E08(FLD)은 R01~R03에서 짧은 단편 선분만 남아 지지점이 최소 개수(50)에 못 미쳤습니다. 다른 방법으로 대체하지 않았습니다.</li>
<li>하단 구조물 경계는 R01에서 밝기 전이가 약 15 px에 걸친 약한 경계입니다. E01 등의 잔차 패널에서 직선 대비 수 px 규모의 체계적인 휨이 보입니다. 원인(실제 형상, 렌즈 왜곡, 반사 등)은 미확인입니다.</li>
<li>QR 바깥 테두리 상단 경계는 지지 길이가 약 166 px로 짧습니다. 끝과 끝의 세로 차이 1 px가 약 0.35°에 해당하므로, 정수 화소를 지지점으로 쓰는 방법(Q03 Canny 화소, Q07~Q09 선분 지지 화소 진단값, Q10 윤곽점)은 계단 양자화 영향을 크게 받습니다. 예: R04 Q03 qr_C는 모든 지지 화소가 한 행에 있어 각도가 정확히 0입니다.</li>
<li>선분형 방법에서 선분 1개만 선택된 경우 표본 RMS가 정의상 0이 되므로 fit RMS를 null로 두고, 검출기 자체 지지 화소의 잔차를 diagnostics에 따로 기록했습니다.</li>
<li>두 점 연결 방법(Q11, Q12)의 잔차 0은 정확도 근거가 아니므로 fit RMS를 null로 두었습니다.</li>
<li>Q12 C++ 이식 확인: 기존 Python(OpenCV 5.0.0) 결과와의 차이 (참고, 동일 정의):
<table><tr><th>기준</th><th>QR</th><th>기존 Python(°)</th><th>이번 C++(°)</th><th>차이</th></tr>{q12_rows}</table></li>
<li>네 기준 이미지는 서로 다른 촬영이므로, 네 장 사이 각도 차이를 측정 오차로 해석할 수 없습니다. 이번 결과는 기준 4장에 대한 추출 결과이며 Y·θ 이동 반응을 검증한 것이 아닙니다.</li>
<li>일반 선 검출 논문은 우리 기준 이미지나 장비 θ 정확도를 입증하지 않습니다. 부록의 '직접 확인 범위'를 참고하세요.</li>
</ul>

<h2 id="s7">7. 재현·검증</h2>
<ul class="kv"><li>전체: <code>python tools/run_pipeline.py</code> (구성·빌드·자체시험·전체 실행·재실행 비교·단독 실행 비교·환경/해시 기록) → <code>python -I tools/make_figures.py</code> → <code>python tools/make_report.py</code></li>
<li>방법별 단독: <code>build\\Release\\line_methods.exe --root . --out &lt;새 폴더&gt; --methods E04,Q12 --refs R02</code></li>
<li>자체 시험(알려진 점 입력: 끝점 순서·180° 정규화·수직선 null·Huber 복원·SHA-256): <code>build\\Release\\line_selftest.exe</code> → 11/11 PASS (<code>logs/03_selftest.stdout.txt</code>)</li>
<li>재실행 비교: 같은 설정 2회 실행의 216슬롯 수치 {"완전히 동일" if rep["identical"] else f'차이 {len(rep["differences"])}건'} (<code>logs/repeatability.json</code>). OpenCV 스레드 1로 고정.</li>
<li>단독 실행(E04·Q12, R02) 결과와 전체 실행 결과: {"동일" if single["identical"] else "차이 있음"} ({single["compared_ids"]}슬롯, <code>logs/single_vs_full.json</code>)</li>
<li>원본 4장: 실행 전후 SHA-256이 manifest와 같음 ({"확인" if env["inputs_match_manifest"] else "불일치"}, <code>logs/environment.json</code>)</li>
<li>명령·stdout·stderr·종료 코드·실행 시간: <code>logs/0*_*.json|txt</code>. 코드·설정·외부 소스 해시: <code>logs/hashes.json</code>.</li></ul>

<h2 id="s8">부록. 근거 자료 및 구현 출처</h2>
<p class="small">'직접 확인 범위'는 이번 작업에서 실제로 읽은 범위입니다. 논문 본문을 읽지 않은 항목은 그렇게 적었습니다. 외부 코드 변경 diff: <code>third_party/*/CHANGES.diff</code>, 출처 요약: <code>third_party/PROVENANCE.md</code>. LSD·Devernay는 AGPL-3.0-or-later이므로 배포 전 라이선스 검토가 필요합니다.</p>
{sources_appendix()}
</main>
<script id="data" type="application/json">{json.dumps(DATA, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")}</script>
<script>{JS}</script>
</body></html>"""
(ROOT / "report.html").write_text(html_doc, encoding="utf-8")

# ---------------------------------------------------------------- report.md
def md_angles(ids, tids):
    head = "| 방법 | " + " | ".join(f"{T_KO[t]} {rid}" for t in tids for rid in REFS) + " |"
    sep = "|---|" + "---:|" * (len(tids) * 4)
    rows = []
    for mid in ids:
        cells = []
        for t in tids:
            for rid in REFS:
                r = recs[f"{rid}_{mid}_{t}"]
                cells.append(f"{r['image_angle_deg']:+.4f}" if r["status"] == "ok" else f"실패({CAT_KO.get(r['failure_category'], '')})")
        rows.append(f"| {mid}{' (파인더 중심선, 다른 특징)' if mid == 'Q12' else ''} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


md = [f"# 기준 이미지 4장: 외부 구조물·QR 직선 추출 (108 조합)", "",
      f"생성: {env['timestamp']}. 상호작용 열람은 `report.html`을 브라우저로 여세요.", "",
      "> 모든 각도는 영상 속 선의 각도(원본 픽셀, +x 오른쪽, +y 아래, atan2(dy,dx), [-90°,90°))입니다. 로봇 θ·보정치·mm로 환산하지 않았습니다. 소수 자릿수는 정확도 보장이 아닙니다. 통계적 균일성 합격, Y·θ 분리 성공, ±0.005° 정렬 성공, 최적 방법 확정을 주장하지 않습니다.", "",
      "## 1. 개요", "",
      "| ID | 경로 | SHA-256 | manifest 일치 / 실행 후 |", "|---|---|---|---|"]
for rr in meta["inputs"]:
    md.append(f"| {rr['reference_id']} | `{rr['path']}` | `{rr['sha256_before']}` | {'일치' if rr['sha256_before'] == rr['expected_sha256'] else '불일치'} / {'변경 없음' if rr['unchanged_after_run'] else '변경됨'} |")
md += ["", f"- 실행 환경: Windows x64, C++17, MSVC v141(_MSC_FULL_VER {meta.get('msvc_full_version')}), CMake(VS 18 2026 생성기, v141), OpenCV {meta['opencv_version']}. 측정 수치는 전부 C++ 실행 결과입니다. Python(OpenCV 5.0.0)은 그림·보고서에만 사용했습니다.",
       f"- 21개 방법 모두 구현·빌드·실행. 216개 주 선 슬롯: 성공 {n_ok}, 실패 {len(fails)}.",
       f"- 432개 조합: complete {combos['meta']['complete']}, partial {combos['meta']['partial']}, failed {combos['meta']['failed']}.", "",
       "### 실패 슬롯", "", "| 측정 ID | 범주 | 사유 |", "|---|---|---|"]
for r in fails:
    md.append(f"| {r['measurement_id']} | {CAT_KO.get(r['failure_category'], r['failure_category'])} | {r['failure_reason']} |")
md += ["", "## 2. 측정 대상", "",
       "- 상단 구조물 1개(upper_structure): x 24~3816, 탐색 y 692~736, 가림 x 1370~2450 제외. 하단 구조물 1개(lower_structure): x 24~3816, 탐색 y 1845~1905, 가림 x 1560~2300 제외. 주 선은 y 증가 방향 어두움→밝음 경계. 양쪽 가시 구간 점을 하나의 직선으로 맞춤(좌우 평균 없음).",
       "- QR L/C/R: 기존 ROI 유지. Q01~Q11 = 바깥 흰 테두리 상단 경계(탐색 y 958~978). Q12 = 위쪽 두 파인더 중심 연결선(다른 특징).", "",
       "![R01 대상 위치](figures/targets_R01_overview.png)", "", "![R01 확대·밝기 프로파일](figures/targets_R01_zoom_profiles.png)", "",
       "![네 장 같은 범위](figures/targets_4refs_same_regions.png)", "", "## 3. 방법별 추출 과정 (R01 그림, 나머지는 figures/methods/<ID>_<R0n>.png)", ""]
for mid in E_IDS + Q_IDS:
    src = sources["methods"][mid]
    md += [f"### {mid} · {method_name(mid)}", "", METHOD_NOTES[mid[1:]], "",
           f"- 근거: {', '.join(src['sources'])} · 외부 코드: {src['third_party_code'] or '없음(직접 구현)'}",
           f"- 변경점: {src['change']}", "", f"![{mid} R01](figures/methods/{mid}_R01.png)", ""]
md += ["## 4. 108개 조합", "", "`report.html` 4장에서 기준 이미지·외부 방법·QR 방법을 골라 선을 겹쳐 보고, 확대·선 숨기기·네 장 나란히 비교가 가능합니다. 조합-측정 ID 참조는 `results/combinations.json`에 있습니다.", "",
       "## 5. 영상 각도 표 (°)", "", "### 외부 구조물", "", md_angles(E_IDS, EXT_T), "", "### QR", "", md_angles(Q_IDS, QR_T), "",
       "## 6. 해석 범위와 관찰", "",
       "- 실패 7슬롯은 모두 하단 구조물(E04 4장, E08 3장)입니다. 다른 방법으로 대체하지 않았습니다.",
       "- 하단 경계는 약한 넓은 전이이며 잔차에 체계적 휨이 보입니다. 원인은 미확인입니다.",
       "- QR 상단 경계는 약 166 px로 짧아 1 px 세로 차이가 약 0.35°입니다. 정수 화소 지지 방법은 양자화 영향이 큽니다(예: R04 Q03 qr_C = 0°).",
       "- 두 점 연결(Q11, Q12)과 단일 선분 표본의 잔차 0은 정확도 근거가 아니므로 fit RMS를 null로 두었습니다.",
       "- 네 기준 이미지는 서로 다른 촬영이므로 장 사이 차이를 측정 오차로 해석할 수 없습니다. Y·θ 이동 반응 검증이 아닙니다.", "",
       "## 7. 재현·검증", "",
       f"- `python tools/run_pipeline.py` → `python -I tools/make_figures.py` → `python tools/make_report.py`",
       f"- 자체 시험 11/11 PASS, 재실행 수치 {'동일' if rep['identical'] else '차이 있음'}, 단독 실행=전체 실행 {'동일' if single['identical'] else '차이'}, 원본 해시 {'변경 없음' if env['inputs_match_manifest'] else '불일치'}.", "",
       "## 부록. 근거 자료 및 구현 출처", ""]
for sid, s in sources["sources"].items():
    used = [m for m, v in sources["methods"].items() if sid in v["sources"]]
    md += [f"### {sid} · {s['title']}", "", f"- 저자·연도: {s['authors']} · {s['year'] or '—'}", f"- DOI: {s['doi'] or '—'}", f"- 링크: {' '.join(s['links'])}",
           f"- 코드 버전/commit: {s['code'] or '—'} · 라이선스: {s['license'] or '—'}", f"- 사용 방법 ID: {', '.join(used) or '(공통 입출력)'}",
           f"- 뒷받침하는 내용: {s['supports']}", f"- 직접 확인 범위: {s['checked_scope']}", ""]
md += ["### 방법 ID별 연결", "", "| 방법 | 자료 | 우리 구현에서의 변경 |", "|---|---|---|"]
for mid in E_IDS + Q_IDS:
    v = sources["methods"][mid]
    md.append(f"| {mid} | {', '.join(v['sources'])} | {v['change']} |")
(ROOT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("report.html", round((ROOT / "report.html").stat().st_size / 1e6, 2), "MB; report.md written")
