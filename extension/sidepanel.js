// Placeholder wiring; the audio/WebSocket client lands in milestone M1/M2 (docs/roadmap.md).
chrome.storage.session.get("arxivId").then(({ arxivId }) => {
  if (arxivId) document.getElementById("paper").textContent = `Paper: arXiv ${arxivId}`;
});
