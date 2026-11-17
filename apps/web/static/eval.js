// 평가 tab. Reads result folders written by tool/03_benchmark.cpp (summary.json + predictions.csv).
// Folders are opened with the browser folder picker, so no server is needed.
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { results: [], active: 0, module: "joint" };
  const MODULE_NAME = { joint: "1차 joint", control: "2차 control" };
  const CONDITION_NAME = { center: "기준 위치", Y: "Y만", T: "T만", mixed: "Y와 T 함께", unlabeled: "정답 없음" };
  const CONDITION_ORDER = ["center", "Y", "T", "mixed", "unlabeled"];

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v) ? "-" : v === 0 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v).toFixed(d));
  const num = (v, d = 4) => (v === null || v === undefined ? "-" : v.toFixed(d));
  const pct = (a, b) => (b ? `${((a / b) * 100).toFixed(1)}%` : "-");

  // ---------- CSV ----------
  function parseCsv(text) {
    text = text.replace(/^﻿/, "");
    const rows = [];
    let row = [], cur = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) {
        if (c === '"' && text[i + 1] === '"') { cur += '"'; i++; }
        else if (c === '"') q = false;
        else cur += c;
      } else if (c === '"') q = true;
      else if (c === ",") { row.push(cur); cur = ""; }
      else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(cur); rows.push(row); row = []; cur = "";
      } else cur += c;
    }
    if (cur || row.length) { row.push(cur); rows.push(row); }
    const head = rows.shift();
    return rows.filter((r) => r.length === head.length).map((r) => Object.fromEntries(head.map((h, i) => [h, r[i]])));
  }
  const toNum = (s) => (s === "" || s === undefined ? null : Number(s));
  function normalize(row) {
    return {
      path: row.path, condition: row.condition, module: row.module,
      labelY: toNum(row.label_y), labelT: toNum(row.label_t),
      estY: toNum(row.est_y), estT: toNum(row.est_t),
      measured: row.measured === "1", outside: row.outside_training_range === "1",
      errY: toNum(row.error_y), errT: toNum(row.error_t), pass: row.pass === "1",
      ms: toNum(row.ms), message: row.message,
    };
  }

  // ---------- loading ----------
  function addResult(name, summary, csvText) {
    const rows = parseCsv(csvText).map(normalize);
    const existing = state.results.findIndex((r) => r.name === name);
    const entry = { name, summary, rows };
    if (existing >= 0) state.results[existing] = entry; else state.results.push(entry);
    state.active = state.results.indexOf(entry);
    if (!summary.results[state.module]) state.module = Object.keys(summary.results)[0];
  }

  $("evalPicker").addEventListener("change", async (ev) => {
    const files = Array.from(ev.target.files);
    const folders = new Map();
    files.forEach((f) => {
      const parts = f.webkitRelativePath.split("/");
      const name = parts.pop();
      if (name !== "summary.json" && name !== "predictions.csv") return;
      const dir = parts.join("/");
      if (!folders.has(dir)) folders.set(dir, {});
      folders.get(dir)[name] = f;
    });
    const complete = [...folders.entries()].filter(([, v]) => v["summary.json"] && v["predictions.csv"]).sort(([a], [b]) => a.localeCompare(b));
    if (!complete.length) { alert("선택한 폴더에 summary.json과 predictions.csv가 없습니다"); return; }
    for (const [dir, v] of complete) {
      try {
        const summary = JSON.parse(await v["summary.json"].text());
        addResult(dir.split("/").pop(), summary, await v["predictions.csv"].text());
      } catch (e) { alert(`${dir}: 읽을 수 없습니다 (${e.message})`); }
    }
    ev.target.value = "";
    render();
  });

  // ---------- new run command ----------
  function renderCommand() {
    const path = $("evalPath").value.trim() || "data\\clean";
    const leaf = path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || "data";
    const d = new Date(), p2 = (n) => String(n).padStart(2, "0");
    const stamp = `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}_${p2(d.getHours())}${p2(d.getMinutes())}`;
    const which = $("evalWhich").value;
    $("evalCommand").textContent = `.\\build\\Release\\benchmark.exe "${path}" "evals\\${stamp}_${leaf}"${which === "both" ? "" : " " + which}`;
  }
  $("evalPath").addEventListener("input", renderCommand);
  $("evalWhich").addEventListener("change", renderCommand);
  $("evalCopy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("evalCommand").textContent); $("evalCopy").textContent = "복사됨"; }
    catch { $("evalCopy").textContent = "직접 선택해 복사하세요"; }
    setTimeout(() => { $("evalCopy").textContent = "명령 복사"; }, 1500);
  });

  // ---------- result list ----------
  function renderResultList() {
    const ul = $("evalResults");
    ul.replaceChildren();
    state.results.forEach((r, i) => {
      const li = el("li", i === state.active ? "is-on" : "");
      const pick = el("button", "result-pick");
      pick.append(el("span", "rf-name", r.name), el("span", "rf-meta", `${r.summary.photos}장  ${r.summary.created}${i === 0 ? "  기준" : ""}`));
      pick.addEventListener("click", () => { state.active = i; render(); });
      const close = el("button", "rf-close", "닫기");
      close.addEventListener("click", () => {
        state.results.splice(i, 1);
        state.active = Math.min(state.active, state.results.length - 1);
        render();
      });
      li.append(pick, close);
      ul.append(li);
    });
  }

  // ---------- header + KPIs ----------
  function moduleRows(r, m = state.module) { return r.rows.filter((x) => x.module === m); }

  function renderHeader(r) {
    $("evalTitle").textContent = r.name;
    const mods = Object.entries(r.summary.modules).map(([k, v]) => `${MODULE_NAME[k]} ${v.name}`).join("   ");
    $("evalMeta").textContent = `${r.summary.dataset}   ${r.summary.created}   ${mods}`;
    const dev = /clean_20261014|clean_20261020|data\/clean$/.test(r.summary.dataset);
    $("evalNotice").textContent = dev
      ? "개발에 이미 쓴 사진입니다. 아래 숫자는 재대입 성적이며 새 사진 성능이 아닙니다"
      : "같은 사진에 저장된 모델을 적용한 성적입니다. 이 사진이 학습에 쓰이지 않았는지 확인하세요";
    document.querySelectorAll("#evalModule button").forEach((b) => {
      b.disabled = !r.summary.results[b.dataset.module];
      b.classList.toggle("is-on", b.dataset.module === state.module);
    });
  }

  function renderKpis(r) {
    const s = r.summary.results[state.module];
    const box = $("evalKpis");
    box.replaceChildren();
    const kpi = (label, value, sub, cls) => {
      const d = el("div", "kpi");
      d.append(el("span", "kpi-label", label), el("span", "kpi-value" + (cls ? " " + cls : ""), value), el("span", "kpi-sub", sub));
      box.append(d);
    };
    const allPass = s.pass === s.labeled;
    kpi("두 축 0.09 이내", `${s.pass} / ${s.labeled}`, pct(s.pass, s.labeled), allPass ? "is-pass" : "is-fail");
    kpi("Y 오차 평균", num(s.error_y.mean), `최대 ${num(s.error_y.max)}   p95 ${num(s.error_y.p95)}`);
    kpi("T 오차 평균", num(s.error_t.mean), `최대 ${num(s.error_t.max)}   p95 ${num(s.error_t.p95)}`);
    kpi("처리 시간 중앙값", `${s.ms_median.toFixed(0)} ms`, `측정 성공 ${s.measured} / ${s.photos}`);
  }

  // ---------- comparison ----------
  function renderCompare() {
    const card = $("evalCompareCard");
    const avail = state.results.filter((r) => r.summary.results[state.module]);
    card.hidden = avail.length < 2;
    if (card.hidden) return;
    const base = avail[0].summary.results[state.module];
    const t = $("evalCompare");
    t.replaceChildren();
    const head = el("tr");
    ["결과", "데이터", "통과", "Y 평균", "Y 최대", "T 평균", "T 최대", "시간"].forEach((h) => head.append(el("th", null, h)));
    t.append(head);
    const diffCell = (v, b, d = 4, lowerBetter = true) => {
      const td = el("td");
      td.append(el("span", "cell-main", num(v, d)));
      if (v !== b) {
        const better = lowerBetter ? v < b : v > b;
        td.append(el("span", "cell-diff " + (better ? "is-pass" : "is-fail"), `${fmt(v - b, d)} ${better ? "개선" : "악화"}`));
      }
      return td;
    };
    avail.forEach((r, i) => {
      const s = r.summary.results[state.module];
      const tr = el("tr", i === 0 ? "is-base" : "");
      tr.append(el("td", "cell-strong", r.name + (i === 0 ? "  기준" : "")), el("td", null, r.summary.dataset.split("/").pop()));
      const passTd = el("td");
      passTd.append(el("span", "cell-main", `${s.pass} / ${s.labeled}`));
      if (i > 0 && s.labeled && base.labeled) {
        const dr = s.pass / s.labeled - base.pass / base.labeled;
        if (Math.abs(dr) > 1e-9) passTd.append(el("span", "cell-diff " + (dr > 0 ? "is-pass" : "is-fail"), `${dr > 0 ? "+" : "−"}${Math.abs(dr * 100).toFixed(1)}%p`));
      }
      tr.append(passTd);
      if (i === 0) {
        [s.error_y.mean, s.error_y.max, s.error_t.mean, s.error_t.max].forEach((v) => tr.append(el("td", null, num(v))));
        tr.append(el("td", null, `${s.ms_median.toFixed(0)} ms`));
      } else {
        tr.append(diffCell(s.error_y.mean, base.error_y.mean), diffCell(s.error_y.max, base.error_y.max), diffCell(s.error_t.mean, base.error_t.mean), diffCell(s.error_t.max, base.error_t.max));
        tr.append(el("td", null, `${s.ms_median.toFixed(0)} ms`));
      }
      t.append(tr);
    });
  }

  // ---------- conditions ----------
  function renderConditions(r) {
    const s = r.summary.results[state.module];
    const box = $("evalConditions");
    box.replaceChildren();
    CONDITION_ORDER.filter((c) => s.by_condition[c]).forEach((c) => {
      const v = s.by_condition[c];
      const row = el("div", "cond-row");
      const ratio = v.photos ? v.pass / v.photos : 0;
      row.append(el("span", "cond-name", CONDITION_NAME[c] || c));
      const bar = el("span", "cond-bar");
      const fill = el("span", "cond-fill" + (ratio === 1 ? " is-pass" : " is-fail"));
      fill.style.width = `${ratio * 100}%`;
      bar.append(fill);
      row.append(bar, el("span", "cond-value", `${v.pass} / ${v.photos}`));
      box.append(row);
    });
    if (state.module === "control" && s.by_condition.mixed && s.by_condition.mixed.pass < s.by_condition.mixed.photos) {
      box.append(el("p", "muted cond-note", "2차 control은 한 축만 추정하므로 Y와 T가 함께 틀어진 사진은 1차 joint로 T를 먼저 맞춘 뒤 쓰는 것이 전제입니다"));
    }
  }

  // ---------- scatter ----------
  function renderScatter(svgId, rows, axis) {
    const svg = $(svgId);
    svg.replaceChildren();
    const ns = "http://www.w3.org/2000/svg";
    const add = (tag, attrs, text) => {
      const n = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
      if (text !== undefined) n.textContent = text;
      svg.append(n);
      return n;
    };
    const pts = rows.filter((x) => x.measured && x.labelY !== null).map((x) => ({
      l: axis === "Y" ? x.labelY : x.labelT, e: axis === "Y" ? x.estY : x.estT, pass: x.pass, path: x.path,
    }));
    const W = 260, H = 220, L = 40, B = 26, T = 14, R = 10;
    const ext = Math.max(0.1, ...pts.map((p) => Math.max(Math.abs(p.l), Math.abs(p.e))));
    const lo = -ext * 1.05, hi = ext * 1.05;
    const X = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
    const Y = (v) => H - B - ((v - lo) / (hi - lo)) * (H - B - T);
    const tol = 0.09;
    add("polygon", { points: `${X(lo)},${Y(lo + tol)} ${X(hi - tol)},${Y(hi)} ${X(hi)},${Y(hi)} ${X(hi)},${Y(hi - tol)} ${X(lo + tol)},${Y(lo)} ${X(lo)},${Y(lo)}`, fill: "#e6f4ee" });
    add("line", { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), stroke: "#b7dfcd" });
    add("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, stroke: "#d5dbe5" });
    add("line", { x1: L, y1: T, x2: L, y2: H - B, stroke: "#d5dbe5" });
    const tick = Number((ext).toPrecision(1));
    [-tick, 0, tick].forEach((v) => {
      add("text", { x: X(v), y: H - B + 14, "text-anchor": "middle", "font-size": 9, fill: "#8a94a6" }, v === 0 ? "0" : fmt(v, 2));
      add("text", { x: L - 6, y: Y(v) + 3, "text-anchor": "end", "font-size": 9, fill: "#8a94a6" }, v === 0 ? "0" : fmt(v, 2));
    });
    add("text", { x: W - R, y: H - 2, "text-anchor": "end", "font-size": 9, fill: "#8a94a6" }, "명령값");
    add("text", { x: L + 6, y: T + 8, "font-size": 9, fill: "#8a94a6" }, "추정");
    pts.sort((a, b) => a.pass - b.pass).reverse().forEach((p) => {
      const c = add("circle", { cx: X(p.l), cy: Y(p.e), r: p.pass ? 2.4 : 3.2, fill: p.pass ? "#1d4ed8" : "#c62f3a", "fill-opacity": p.pass ? 0.55 : 0.9 });
      c.append(Object.assign(document.createElementNS(ns, "title"), { textContent: `${p.path}\n명령 ${fmt(p.l, 2)}  추정 ${fmt(p.e)}` }));
    });
  }

  // ---------- worst ----------
  function renderWorst(r) {
    const rows = moduleRows(r).filter((x) => x.measured && x.errY !== null)
      .map((x) => ({ ...x, worst: Math.max(x.errY / 0.09, x.errT / 0.09) }))
      .sort((a, b) => b.worst - a.worst).slice(0, 15);
    const failed = moduleRows(r).filter((x) => !x.measured);
    $("evalWorstNote").textContent = `상위 ${rows.length}장   기준 대비 큰 축 순${failed.length ? `   측정 실패 ${failed.length}장` : ""}`;
    const t = $("evalWorst");
    t.replaceChildren();
    const head = el("tr");
    ["사진", "조건", "명령 Y", "명령 T", "추정 Y", "추정 T", "오차 Y", "오차 T", "판정"].forEach((h) => head.append(el("th", null, h)));
    t.append(head);
    rows.forEach((x) => {
      const tr = el("tr");
      tr.append(el("td", "cell-path", x.path), el("td", null, CONDITION_NAME[x.condition] || x.condition));
      [fmt(x.labelY, 2), fmt(x.labelT, 2), fmt(x.estY), fmt(x.estT)].forEach((v) => tr.append(el("td", "cell-num", v)));
      tr.append(el("td", "cell-num" + (x.errY > 0.09 ? " is-fail" : ""), num(x.errY)), el("td", "cell-num" + (x.errT > 0.09 ? " is-fail" : ""), num(x.errT)));
      tr.append(el("td", "cell-verdict " + (x.pass ? "is-pass" : "is-fail"), x.pass ? "통과" : "미통과"));
      t.append(tr);
    });
  }

  function render() {
    renderResultList();
    const has = state.results.length > 0;
    $("evalEmpty").hidden = has;
    $("evalBody").hidden = !has;
    if (!has) return;
    const r = state.results[state.active];
    renderHeader(r);
    renderKpis(r);
    renderCompare();
    renderConditions(r);
    renderScatter("scatterY", moduleRows(r), "Y");
    renderScatter("scatterT", moduleRows(r), "T");
    renderWorst(r);
  }

  document.querySelectorAll("#evalModule button").forEach((b) => b.addEventListener("click", () => {
    if (b.disabled) return;
    state.module = b.dataset.module;
    render();
  }));

  // Review hook: a page may define window.SEMES_EVAL_PRELOAD = [{name, summary, csv}]
  (window.SEMES_EVAL_PRELOAD || []).forEach((p) => addResult(p.name, p.summary, p.csv));
  renderCommand();
  render();
})();
