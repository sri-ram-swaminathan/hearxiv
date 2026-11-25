// "Listen with hearxiv" opens the local voice app for this paper (python -m hearxiv serve).
// A full in-extension build (Phonon WASM + in-browser STT) is milestone M2; see docs/roadmap.md.
const APP = "http://localhost:8765";

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type !== "hearxiv:open") return;
  chrome.storage.sync.get({ mode: "5min" }).then(({ mode }) => {
    chrome.tabs.create({ url: `${APP}/?paper=${encodeURIComponent(msg.arxivId)}&mode=${mode}` });
  });
});
