// Top tabs. Each [data-panel] element is shown only on its tab. #eval opens the 평가 tab.
(() => {
  const tabs = document.querySelectorAll(".tab[data-tab]");
  function show(name) {
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("is-active", on);
      if (on) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current");
    });
    document.querySelectorAll("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  }
  tabs.forEach((t) => t.addEventListener("click", () => {
    show(t.dataset.tab);
    history.replaceState(null, "", t.dataset.tab === "eval" ? "#eval" : location.pathname);
  }));
  show(location.hash === "#eval" ? "eval" : "service");
})();
