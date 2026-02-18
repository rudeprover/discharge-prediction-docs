document.addEventListener("DOMContentLoaded", () => {
  const primarySidebar = document.querySelector(".md-sidebar--primary");
  if (!primarySidebar) return;

  const tocToggle = document.getElementById("__toc");
  if (!(tocToggle instanceof HTMLInputElement)) return;

  let userInteracted = false;
  const markInteraction = () => {
    userInteracted = true;
  };

  primarySidebar.addEventListener("pointerdown", markInteraction, { capture: true, once: true });
  primarySidebar.addEventListener("keydown", markInteraction, { capture: true, once: true });

  // MkDocs Material may auto-expand the integrated ToC during init (or restore state).
  // Force it collapsed on load, but stop interfering as soon as the user interacts.
  const start = performance.now();
  const timer = window.setInterval(() => {
    if (userInteracted || performance.now() - start > 3000) {
      window.clearInterval(timer);
      return;
    }
    if (tocToggle.checked) tocToggle.checked = false;
  }, 50);
});
