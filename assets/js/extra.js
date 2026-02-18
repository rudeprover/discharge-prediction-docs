(() => {
  let activeTimer = null;

  function ensureMathJax() {
    const hasMath =
      document.querySelector(".arithmatex") ||
      document.querySelector('script[type="math/tex"]') ||
      document.querySelector('script[type="math/tex; mode=display"]');
    if (!hasMath) return;

    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise();
      return;
    }

    if (document.querySelector('script[data-codex-mathjax="true"]')) return;

    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
    script.async = true;
    script.setAttribute("data-codex-mathjax", "true");
    document.head.appendChild(script);
  }

  function collapseIntegratedToc() {
    const primarySidebar = document.querySelector(".md-sidebar--primary");
    if (!primarySidebar) return;

    const tocToggle = document.getElementById("__toc");
    if (!(tocToggle instanceof HTMLInputElement)) return;

    if (activeTimer) {
      window.clearInterval(activeTimer);
      activeTimer = null;
    }

    let userInteracted = false;
    const markInteraction = () => {
      userInteracted = true;
    };

    primarySidebar.addEventListener("pointerdown", markInteraction, { capture: true, once: true });
    primarySidebar.addEventListener("keydown", markInteraction, { capture: true, once: true });

    // MkDocs Material may auto-expand the integrated ToC during init (or restore state).
    // Force it collapsed on load, but stop interfering as soon as the user interacts.
    const start = performance.now();
    activeTimer = window.setInterval(() => {
      if (userInteracted || performance.now() - start > 3000) {
        window.clearInterval(activeTimer);
        activeTimer = null;
        return;
      }
      if (tocToggle.checked) tocToggle.checked = false;
    }, 50);
  }

  document.addEventListener("DOMContentLoaded", collapseIntegratedToc, { once: true });
  document.addEventListener("DOMContentLoaded", ensureMathJax, { once: true });

  // Support "instant navigation" in Material: run after each page change.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(collapseIntegratedToc);
    window.document$.subscribe(ensureMathJax);
  }
})();
