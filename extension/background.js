// Opens the side panel for the paper the user asked to hear.
chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg.type !== "hearxiv:open" || !sender.tab) return;
  chrome.storage.session.set({ arxivId: msg.arxivId });
  chrome.sidePanel.open({ tabId: sender.tab.id });
});
