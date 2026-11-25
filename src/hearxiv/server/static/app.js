// hearxiv voice-first player. Plain JS, no build step, so it is easy to read and fork.
"use strict";

const $ = (id) => document.getElementById(id);
const S = {
  sentences: [], sections: {}, idx: 0, playing: false, speed: 1, cache: new Map(),
  talking: false, wasPlaying: false, busy: false, lastStatus: "",
};
const audio = new Audio();
audio.preservesPitch = true;
const clipAudio = new Audio();
clipAudio.preservesPitch = true;

// ---------- small helpers ----------
function announce(msg) {
  if (!msg || msg === S.lastStatus) return;
  S.lastStatus = msg;
  $("status").textContent = msg;
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.blob();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// Earcons for the microphone (distinct from the narration's section/figure earcons)
let actx;
function beep(freq, ms, when = 0) {
  actx = actx || new AudioContext();
  const o = actx.createOscillator(), g = actx.createGain();
  o.frequency.value = freq; o.connect(g); g.connect(actx.destination);
  const t = actx.currentTime + when;
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.2, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + ms / 1000);
  o.start(t); o.stop(t + ms / 1000 + 0.02);
}
const listenOn = () => { beep(880, 90); beep(1320, 90, 0.1); };
const listenOff = () => { beep(1320, 90); beep(880, 90, 0.1); };

// ---------- loading a paper ----------
async function loadPaper(paper, mode) {
  try {
    await post("/api/load", { paper, mode });
  } catch (e) { announce(e.message); return; }
  $("progress-bar").style.width = "0";
  poll();
}

async function poll() {
  const st = await api("/api/status");
  announce(st.message);
  $("progress-bar").style.width = `${Math.round(st.progress * 100)}%`;
  if (st.status === "working") return setTimeout(poll, 900);
  if (st.status === "ready") return onReady(st);
}

async function onReady(st) {
  const sc = await api("/api/script");
  S.sentences = sc.sentences; S.sections = sc.sections; S.idx = 0; S.cache.clear();
  $("player").hidden = false;
  $("title").textContent = st.title;
  buildNav();
  $("keys-region").focus();
  loadLibrary();
  try {
    await playClip((await post("/api/speak", { text: st.message })).url, st.message);
    playFrom(0);
  } catch {
    announce("Ready. Press K or the Play button to start.");
  }
}

async function loadLibrary() {
  try {
    const lib = await api("/api/library");
    if (!lib.papers.length) return;
    $("library").hidden = false;
    $("library-list").replaceChildren(...lib.papers.map((p) => {
      const li = document.createElement("li"), b = document.createElement("button");
      b.textContent = p.title; b.onclick = () => { $("paper").value = p.id; loadPaper(p.id, mode()); };
      li.append(b); return li;
    }));
  } catch {}
}

function buildNav() {
  const wrap = $("nav-lists");
  const secs = S.sentences.filter((s) => s.kind === "heading" || s.kind === "abstract");
  const figs = S.sentences.filter((s, i) => (s.kind === "figure" || s.kind === "table") &&
                                            (i === 0 || S.sentences[i - 1].seg !== s.seg));
  const list = (items, label) => {
    const h = document.createElement("h4"); h.textContent = label;
    const ul = document.createElement("ul");
    for (const s of items) {
      const li = document.createElement("li"), b = document.createElement("button");
      b.textContent = s.text.slice(0, 90); b.onclick = () => playFrom(s.i);
      li.append(b); ul.append(li);
    }
    return items.length ? [h, ul] : [];
  };
  wrap.replaceChildren(...list(secs, "Sections"), ...list(figs, "Figures and tables"));
}

// ---------- playback ----------
async function audioFor(i) {
  if (!S.cache.has(i)) S.cache.set(i, api(`/api/audio/${i}`).then((b) => URL.createObjectURL(b)));
  return S.cache.get(i);
}

async function playFrom(i) {
  if (!S.sentences.length) return;
  i = Math.max(0, Math.min(i, S.sentences.length - 1));
  S.idx = i; S.playing = true;
  showSentence(i);
  const url = await audioFor(i);
  if (S.idx !== i || !S.playing) return; // user moved on while we were waiting
  audio.src = url; audio.playbackRate = S.speed;
  try { await audio.play(); } catch { S.playing = false; announce("Press K or Play to start."); }
  $("btn-play").textContent = "Pause";
  audioFor(i + 1); audioFor(i + 2); // prefetch
}

audio.addEventListener("ended", () => {
  if (!S.playing) return;
  if (S.idx + 1 < S.sentences.length) playFrom(S.idx + 1);
  else { S.playing = false; $("btn-play").textContent = "Play"; announce("End. Hold Space to ask a question."); }
});

function pause() {
  S.playing = false; audio.pause(); clipAudio.pause();
  $("btn-play").textContent = "Play";
}
function toggle() { S.playing ? (pause(), announce("Paused.")) : playFrom(S.idx); }

function showSentence(i) {
  const s = S.sentences[i];
  const cap = $("caption"); cap.classList.remove("answer"); cap.textContent = s.text;
  $("section").textContent = s.section && S.sections[s.section] ? `Section ${S.sections[s.section]}` : "";
  $("progress-bar").style.width = `${Math.round((100 * i) / Math.max(1, S.sentences.length - 1))}%`;
}

// Section boundaries: headings in full mode, changes of source section in summaries
function sectionStarts() {
  return S.sentences.filter((s, i) => s.kind === "heading" || s.kind === "abstract" ||
    (s.section && (i === 0 || S.sentences[i - 1].section !== s.section) && S.sentences[i - 1]?.seg !== s.seg)).map((s) => s.i);
}
function nextSection() {
  const n = sectionStarts().find((i) => i > S.idx);
  n === undefined ? announce("This is the last section.") : playFrom(n);
}
function prevSection() {
  const starts = sectionStarts().filter((i) => i <= S.idx);
  const cur = starts.at(-1) ?? 0;
  playFrom(S.idx - cur > 2 || starts.length < 2 ? cur : starts.at(-2));
}

function playClip(url, text) {
  return new Promise((resolve, reject) => {
    if (text) { const cap = $("caption"); cap.textContent = text; cap.classList.add("answer"); }
    clipAudio.src = url; clipAudio.playbackRate = S.speed;
    clipAudio.onended = resolve; clipAudio.onerror = reject;
    clipAudio.play().catch(reject);
  });
}

// ---------- talking to it ----------
async function handleReply(r, wasPlaying) {
  if (r.transcript) logLine("you", r.transcript);
  if (r.say) logLine("hearxiv", r.say);
  if (r.speed) { S.speed = r.speed; $("speed").textContent = `Speed ${S.speed}×`; audio.playbackRate = S.speed; }
  for (const c of r.clips || []) {
    try { await playClip(c.url, c.text); } catch { break; }
  }
  if (r.resume === null || r.resume === undefined) { pause(); return; }
  const wantsPlay = wasPlaying || !(r.clips || []).length ||
                    /\b(resume|continue|play|go on|go to|jump|skip|next|previous|back|repeat|again)\b/i.test(r.transcript || "");
  if (wantsPlay) playFrom(r.resume); else { S.idx = r.resume; showSentence(r.resume); announce("Paused. Press K to continue."); }
}

function logLine(who, text) {
  const li = document.createElement("li");
  const b = document.createElement("span"); b.className = who === "you" ? "you" : "";
  b.textContent = who === "you" ? "You: " : "hearxiv: ";
  li.append(b, document.createTextNode(text));
  $("log").prepend(li);
}

async function sendCommand(text) {
  if (!S.sentences.length) { announce("Load a paper first."); return; }
  const wasPlaying = S.playing; pause();
  announce("Thinking…");
  try { await handleReply(await post("/api/command", { text, sentence: S.idx }), wasPlaying); }
  catch (e) { announce(e.message); }
}

// Microphone: 16 kHz mono capture with the browser's echo cancellation (narration is played by
// this same page, which is what lets Chrome cancel it from the mic signal).
let mic = null;
async function startTalk() {
  if (S.talking || S.busy) return;
  S.talking = true; S.wasPlaying = S.playing; pause();
  $("btn-talk").classList.add("listening"); $("btn-talk").textContent = "🎙 Listening… release to send";
  listenOn(); announce("Listening.");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
    const ctx = new AudioContext({ sampleRate: 16000 });
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(proc); proc.connect(ctx.destination);
    mic = { stream, ctx, proc, chunks };
    if (!S.talking) stopTalk(); // released before the mic was ready
  } catch (e) {
    S.talking = false; resetTalkButton();
    announce("I can't use the microphone. Allow microphone access, or type your question instead.");
  }
}

function resetTalkButton() { $("btn-talk").classList.remove("listening"); $("btn-talk").textContent = "🎙 Hold to talk"; }

async function stopTalk(cancel = false) {
  if (!S.talking) return;
  S.talking = false; resetTalkButton();
  if (!mic) return;
  const { stream, ctx, proc, chunks } = mic; mic = null;
  proc.disconnect(); stream.getTracks().forEach((t) => t.stop()); await ctx.close();
  listenOff();
  if (cancel) { announce("Cancelled."); if (S.wasPlaying) playFrom(S.idx); return; }
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const pcm = new Float32Array(n); let o = 0; for (const c of chunks) { pcm.set(c, o); o += c.length; }
  S.busy = true; announce("Thinking…");
  try {
    const r = await api(`/api/listen?sentence=${S.idx}`, { method: "POST", body: pcm.buffer });
    if (!S.sentences.length) {
      announce(r.transcript ? `I heard “${r.transcript}”. Load a paper first, by pasting its link.` : "Load a paper first.");
    } else {
      await handleReply(r, S.wasPlaying);
    }
  } catch (e) { announce(e.message); }
  finally { S.busy = false; }
}

// ---------- wiring ----------
const mode = () => document.querySelector("input[name=mode]:checked").value;
$("load-form").addEventListener("submit", (e) => { e.preventDefault(); const p = $("paper").value.trim(); if (p) loadPaper(p, mode()); });
$("ask-form").addEventListener("submit", (e) => { e.preventDefault(); const t = $("ask").value.trim(); if (t) { $("ask").value = ""; sendCommand(t); } });
$("btn-play").onclick = toggle;
$("btn-back").onclick = () => playFrom(S.idx - 1);
$("btn-fwd").onclick = () => playFrom(S.idx + 1);
$("btn-prev-sec").onclick = prevSection;
$("btn-next-sec").onclick = nextSection;
$("btn-slower").onclick = () => sendCommand("slower");
$("btn-faster").onclick = () => sendCommand("faster");
$("btn-where").onclick = () => sendCommand("where am I");
$("btn-help").onclick = () => sendCommand("what can I say");
const talk = $("btn-talk");
talk.addEventListener("pointerdown", (e) => { e.preventDefault(); startTalk(); });
talk.addEventListener("pointerup", () => stopTalk());
talk.addEventListener("pointerleave", () => S.talking && stopTalk());
talk.addEventListener("keydown", (e) => { if ((e.key === " " || e.key === "Enter") && !e.repeat) { e.preventDefault(); startTalk(); } });
talk.addEventListener("keyup", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); stopTalk(); } });

function typing(e) { const t = e.target; return t.tagName === "INPUT" && t.type === "text" || t.tagName === "TEXTAREA"; }
document.addEventListener("keydown", (e) => {
  if (typing(e) || e.ctrlKey || e.metaKey || e.altKey || e.target === talk) return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); if (!e.repeat) startTalk(); return; }
  if (k === "escape") { if (S.talking) stopTalk(true); else { clipAudio.pause(); } return; }
  if (!S.sentences.length) return;
  const act = { k: toggle, j: () => playFrom(S.idx - 1), l: () => playFrom(S.idx + 1),
                "[": prevSection, "]": nextSection, "-": () => sendCommand("slower"), "=": () => sendCommand("faster"),
                w: () => sendCommand("where am I"), h: () => sendCommand("what can I say"), "?": () => sendCommand("what can I say"),
                t: () => $("ask").focus() }[k];
  if (act) { e.preventDefault(); act(); }
});
document.addEventListener("keyup", (e) => { if (e.key === " " && !typing(e) && e.target !== talk) { e.preventDefault(); stopTalk(); } });

// text size, remembered per browser
let scale = 1;
try { scale = parseFloat(localStorage.getItem("hearxiv-scale")) || 1; } catch {}
const setScale = (s) => { scale = Math.min(2, Math.max(0.8, s)); document.documentElement.style.setProperty("--scale", scale);
                          try { localStorage.setItem("hearxiv-scale", scale); } catch {} };
setScale(scale);
$("a-minus").onclick = () => setScale(scale - 0.1);
$("a-plus").onclick = () => setScale(scale + 0.1);

// Opened from the browser extension or a link: /?paper=1706.03762&mode=5min
const q = new URLSearchParams(location.search);
if (q.get("paper")) {
  $("paper").value = q.get("paper");
  if (q.get("mode")) document.querySelector(`input[name=mode][value="${q.get("mode")}"]`)?.click();
  loadPaper(q.get("paper"), mode());
} else {
  announce("Welcome to hearxiv. Paste an arXiv or alphaXiv link and press Enter.");
  $("paper").focus();
}
loadLibrary();
