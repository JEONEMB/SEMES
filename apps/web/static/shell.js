// Top tabs. Each [data-panel] element is shown only on its tab. #eval opens the 평가 tab, #report or #report/<id> the 보고서 tab.
(() => {
  const tabs = document.querySelectorAll(".tab[data-tab]");
  // 보고서 탭의 보고서들. 경로는 static/index.html 기준
  const REPORTS = {
    augment: { title: "Y-T 평면 데이터 증강", src: "../../augment/index.html", badge: "가정 시뮬레이션, 실측 성적 아님" },
  };
  function show(name) {
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("is-active", on);
      if (on) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current");
    });
    document.querySelectorAll("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  }
  function openReport(id) {
    const r = REPORTS[id], frame = document.getElementById("reportFrame");
    document.getElementById("reportList").hidden = !!r;
    document.getElementById("reportView").hidden = !r;
    if (!r) return;
    document.getElementById("reportViewTitle").textContent = r.title;
    const badge = document.getElementById("reportViewBadge");
    badge.textContent = r.badge || ""; badge.hidden = !r.badge;
    document.getElementById("reportOpen").href = r.src;
    if (frame.getAttribute("src") !== r.src) frame.setAttribute("src", r.src);
  }
  function route() {
    const h = location.hash.slice(1);
    if (h === "eval") show("eval");
    else if (h === "report" || h.startsWith("report/")) { show("report"); openReport(h.split("/")[1]); }
    else show("service");
  }
  tabs.forEach((t) => t.addEventListener("click", () => {
    const tab = t.dataset.tab;
    history.replaceState(null, "", tab === "service" ? location.pathname : "#" + tab);
    route();
  }));
  document.querySelectorAll(".report-block[data-report]").forEach((b) => b.addEventListener("click", () => {
    history.replaceState(null, "", "#report/" + b.dataset.report); route();
  }));
  document.getElementById("reportBack").addEventListener("click", () => { history.replaceState(null, "", "#report"); route(); });
  route();
})();
