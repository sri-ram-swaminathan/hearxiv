// Adds an accessible "Listen with hearxiv" button to arXiv and alphaXiv paper pages.
// alphaXiv is a single-page app, so we re-check on DOM changes instead of running once.

const ID_RE = /(\d{4}\.\d{4,5})(v\d+)?/;

function paperId() {
  const m = location.pathname.match(ID_RE);
  return m ? m[1] + (m[2] || "") : null;
}

function ensureButton() {
  const id = paperId();
  if (!id || document.getElementById("hearxiv-listen")) return;
  const btn = document.createElement("button");
  btn.id = "hearxiv-listen";
  btn.type = "button";
  btn.textContent = "Listen with hearxiv";
  btn.setAttribute("aria-label", `Listen to paper ${id} with hearxiv. Opens the hearxiv side panel.`);
  btn.accessKey = "l"; // Alt+L (browser-dependent modifier)
  Object.assign(btn.style, {
    position: "fixed", right: "16px", bottom: "16px", zIndex: 2147483647,
    padding: "10px 16px", fontSize: "16px", borderRadius: "8px",
    border: "2px solid #111", background: "#ffd400", color: "#111", cursor: "pointer",
  });
  btn.addEventListener("click", () => chrome.runtime.sendMessage({ type: "hearxiv:open", arxivId: id }));
  document.body.appendChild(btn);
}

ensureButton();
new MutationObserver(ensureButton).observe(document.documentElement, { childList: true, subtree: true });
