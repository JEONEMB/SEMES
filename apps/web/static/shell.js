// Top tabs. Each [data-panel] element is shown only on its tab. #eval opens the 평가 tab, #dist or #dist/<id> the 데이터 분포 tab, #report or #report/<id> the 보고서 tab.
(() => {
  const tabs = document.querySelectorAll(".tab[data-tab]");
  // 보고서 탭의 보고서들. 경로는 static/index.html 기준
  const REPORTS = {
    augment: { title: "Y-T 평면 데이터 증강", src: "../../augment/index.html", badge: "가정 시뮬레이션, 실측 성적 아님" },
  };
  // 데이터 분포 탭의 자료들. 페이지마다 plotly.js 를 품은 큰 파일이라 블록을 누를 때 불러온다
  const DISTS = {
    semes: { title: "SEMES 1~4차", src: "../../distribution/index.html" },
    rig: { title: "우리 장비 2026-11-12", src: "../../distribution/rig/index.html" },
  };
  function show(name) {
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("is-active", on);
      if (on) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current");
    });
    document.querySelectorAll("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  }
  // 목록(list)과 보기(view)를 바꾸고, 고른 항목을 iframe 에 띄운다
  function openItem(item, ids) {
    const frame = document.getElementById(ids.frame);
    document.getElementById(ids.list).hidden = !!item;
    document.getElementById(ids.view).hidden = !item;
    if (!item) return;
    document.getElementById(ids.title).textContent = item.title;
    document.getElementById(ids.open).href = item.src;
    if (frame.getAttribute("src") !== item.src) frame.setAttribute("src", item.src);
  }
  function openReport(id) {
    const r = REPORTS[id];
    openItem(r, { frame: "reportFrame", list: "reportList", view: "reportView", title: "reportViewTitle", open: "reportOpen" });
    if (!r) return;
    const badge = document.getElementById("reportViewBadge");
    badge.textContent = r.badge || ""; badge.hidden = !r.badge;
  }
  function openDist(id) {
    openItem(DISTS[id], { frame: "distFrame", list: "distList", view: "distView", title: "distViewTitle", open: "distOpen" });
  }
  function route() {
    const h = location.hash.slice(1);
    if (h === "eval") show("eval");
    else if (h === "dist" || h.startsWith("dist/")) { show("dist"); openDist(h.split("/")[1]); }
    else if (h === "report" || h.startsWith("report/")) { show("report"); openReport(h.split("/")[1]); }
    else show("service");
  }
  const go = (hash) => { history.replaceState(null, "", hash); route(); };
  tabs.forEach((t) => t.addEventListener("click", () => {
    const tab = t.dataset.tab;
    go(tab === "service" ? location.pathname : "#" + tab);
  }));
  document.querySelectorAll(".report-block[data-report]").forEach((b) => b.addEventListener("click", () => go("#report/" + b.dataset.report)));
  document.querySelectorAll(".report-block[data-dist]").forEach((b) => b.addEventListener("click", () => go("#dist/" + b.dataset.dist)));
  document.getElementById("reportBack").addEventListener("click", () => go("#report"));
  document.getElementById("distBack").addEventListener("click", () => go("#dist"));
  route();
})();
