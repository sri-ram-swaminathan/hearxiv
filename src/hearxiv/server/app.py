"""hearxiv local app: a voice-first web UI served on http://localhost:8765.

    uv run python -m hearxiv serve            (then open the URL; the extension links here too)

Everything runs on this machine except the optional cloud models (figure descriptions, summaries,
Q&A) chosen in .env. Endpoints are deliberately simple so any front end (this page, the browser
extension, a future mobile app) can drive them.

  POST /api/load        {"paper": "<id or url>", "mode": "full|5min|10min"}  -> starts preparing
  GET  /api/status      progress of the current job (spoken by the UI)
  GET  /api/script      the sentences to play, with section ids and earcons
  GET  /api/audio/{i}   WAV for sentence i (synthesised on demand, cached, next ones prefetched)
  POST /api/listen      raw 16 kHz float32 PCM of the listener's speech -> transcript + reply
  POST /api/command     {"text": ..., "sentence": i} typed or button commands -> reply
  GET  /api/clip/{key}  WAV for reply sentences
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hearxiv.ingest.fetch import parse_arxiv_id

STATIC = Path(__file__).parent / "static"
SR = 24_000


# Sentences --------------------------------------------------------------------------------------

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def split_sentences(text: str, max_chars: int = 260) -> list[str]:
    out = []
    for s in _SPLIT.split(text.strip()):
        while len(s) > max_chars:  # very long sentences: break at a comma or semicolon near the limit
            cut = max(s.rfind(", ", 0, max_chars), s.rfind("; ", 0, max_chars))
            cut = cut if cut > 60 else max_chars
            out.append(s[: cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out


def _wav_bytes(pcm: np.ndarray) -> bytes:
    from hearxiv.tts.phonon import write_wav

    buf = io.BytesIO()
    write_wav(buf, pcm)
    return buf.getvalue()


# App state (single local user) ------------------------------------------------------------------


@dataclass
class State:
    status: str = "idle"  # idle | working | ready | error
    message: str = "Paste an arXiv or alphaXiv link, or type a paper id, then press Load."
    progress: float = 0.0
    doc: object | None = None
    script: object | None = None
    session: object | None = None
    sentences: list[dict] = field(default_factory=list)
    audio: dict[int, bytes] = field(default_factory=dict)
    clips: dict[str, bytes] = field(default_factory=dict)
    clip_text: dict[str, str] = field(default_factory=dict)
    key: str = ""


S = State()
_tts = None
_stt = None
_tts_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)  # TTS is CPU-bound; one worker keeps latency predictable


def tts():
    global _tts
    if _tts is None:
        from hearxiv.tts.phonon import PhononTTS

        _tts = PhononTTS()
    return _tts


def stt():
    global _stt
    if _stt is None:
        from hearxiv.stt import STT

        _stt = STT()
    return _stt


def synth(text: str, earcon_kind: str | None = None) -> bytes:
    from hearxiv.tts.phonon import earcon

    with _tts_lock:
        pcm = tts().synthesize(text)
    parts = ([earcon(earcon_kind)] if earcon_kind else []) + [pcm, np.zeros(int(SR * 0.18), np.float32)]
    return _wav_bytes(np.concatenate(parts))


def default_describer() -> str:
    if os.environ.get("HEARXIV_DESCRIBER"):
        return os.environ["HEARXIV_DESCRIBER"]
    if os.environ.get("OPENAI_API_KEY"):
        return "openai:gpt-6.1-terra"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini:gemini-3.5-flash-lite"
    return "none"  # captions only, but still fully usable


def _prepare(paper: str, mode: str):
    from hearxiv.agent.session import Session
    from hearxiv.cli import load_or_ingest, paper_dir
    from hearxiv.narrate.script import Script, build_script

    try:
        S.status, S.progress = "working", 0.05
        S.message = "Fetching the paper from arXiv."
        aid, ver = parse_arxiv_id(paper)
        doc = load_or_ingest(paper, log=lambda m: None)
        S.message = f"Found {doc.title}. {len(doc.figures)} figures, {len(doc.tables)} tables."
        S.progress = 0.2
        need = [f for f in doc.figures if f.image_urls and not f.description]
        describer = default_describer()
        if need and describer != "none":
            from hearxiv.ingest.enrich import describe_figures

            for k, fig in enumerate(need, 1):
                S.message = f"Describing figure {k} of {len(need)}. This happens once per paper."
                S.progress = 0.2 + 0.5 * k / len(need)
                fig_only = doc.model_copy(update={"figures": [fig]})
                describe_figures(fig_only, describer, log=lambda m: None)
            (paper_dir(aid, ver) / "paper.json").write_text(doc.model_dump_json(indent=1), encoding="utf-8")
        S.progress = 0.75
        cached = paper_dir(aid, ver) / f"script_{mode}.json"
        if cached.exists():
            script = Script.model_validate_json(cached.read_text(encoding="utf-8"))
        else:
            S.message = "Writing the script." if mode == "full" else f"Writing your {mode.replace('min', ' minute')} summary."
            script = build_script(doc, mode)
            cached.write_text(script.model_dump_json(indent=1), encoding="utf-8")
        sents = []
        for si, seg in enumerate(script.segments):
            for j, t in enumerate(split_sentences(seg.text)):
                sents.append({"i": len(sents), "seg": si, "kind": seg.kind, "text": t,
                              "earcon": seg.earcon if j == 0 else None, "section": seg.section_id})
        S.doc, S.script, S.sentences = doc, script, sents
        S.session = Session(doc, script)
        S.audio, S.key = {}, f"{aid}{ver or ''}:{mode}"
        S.progress, S.status = 1.0, "ready"
        mins = script.minutes()
        S.message = (f"Ready: {doc.title}. About {round(mins)} minutes. Press Play, or hold Space to ask "
                     f"a question at any time.")
        _prefetch(0, 3)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        S.status, S.message = "error", f"Sorry, I couldn't prepare that paper: {e}"


def _prefetch(start: int, n: int = 2):
    for i in range(start, min(start + n, len(S.sentences))):
        if i not in S.audio:
            _pool.submit(_ensure_audio, i)


def _ensure_audio(i: int) -> bytes:
    if i not in S.audio:
        s = S.sentences[i]
        S.audio[i] = synth(s["text"], s["earcon"])
    return S.audio[i]


def _first_sentence_of(seg_idx: int) -> int:
    return next((s["i"] for s in S.sentences if s["seg"] >= seg_idx), len(S.sentences) - 1)


def _reply(text: str, sentence: int) -> dict:
    """Run one utterance through the session and return what the UI should do."""
    sess = S.session
    sess.cursor = S.sentences[sentence]["seg"] if S.sentences else 0
    r = sess.handle(text)
    clips = []
    if r.say:
        # return immediately; each clip is synthesised when requested (first ones prefetched now)
        for t in split_sentences(r.say):
            key = hashlib.sha1(t.encode()).hexdigest()[:16]
            S.clip_text[key] = t
            clips.append({"url": f"/api/clip/{key}", "text": t})
        for c in clips[:2]:
            _pool.submit(_ensure_clip, c["url"].rsplit("/", 1)[1])
    resume = None if r.resume_at is None else _first_sentence_of(r.resume_at)
    if resume is not None:
        _prefetch(resume, 2)
    return {"say": r.say, "clips": clips, "resume": resume, "speed": r.speed}


# HTTP -------------------------------------------------------------------------------------------

app = FastAPI(title="hearxiv")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


class LoadReq(BaseModel):
    paper: str
    mode: str = "5min"


@app.post("/api/load")
def load(req: LoadReq):
    try:
        parse_arxiv_id(req.paper)
    except ValueError:
        raise HTTPException(400, "That doesn't look like an arXiv link or id.")
    if S.status == "working":
        raise HTTPException(409, "Still preparing the previous paper.")
    S.status, S.message, S.progress = "working", "Starting.", 0.0
    threading.Thread(target=_prepare, args=(req.paper, req.mode), daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def status():
    title = getattr(S.doc, "title", None) if S.status == "ready" else None
    return {"status": S.status, "message": S.message, "progress": S.progress, "title": title,
            "sentences": len(S.sentences), "key": S.key}


@app.get("/api/library")
def library():
    from hearxiv.ingest.fetch import CACHE_DIR

    papers = []
    for pj in sorted(CACHE_DIR.glob("*/paper.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        try:
            title = json.loads(pj.read_text(encoding="utf-8"))["title"]
        except Exception:  # noqa: BLE001
            continue
        papers.append({"id": pj.parent.name, "title": title})
    return {"papers": papers}


@app.get("/api/script")
def script():
    sections = {}
    if S.doc is not None:
        sections = {s.id: (f"{s.number} " if s.number else "") + s.title for s in S.doc.iter_sections()}
    return {"sentences": S.sentences, "sections": sections, "key": S.key}


@app.get("/api/audio/{i}")
def audio(i: int):
    if not 0 <= i < len(S.sentences):
        raise HTTPException(404)
    data = _pool.submit(_ensure_audio, i).result() if i not in S.audio else S.audio[i]
    _prefetch(i + 1, 2)
    return Response(data, media_type="audio/wav")


def _ensure_clip(key: str) -> bytes:
    if key not in S.clips:
        S.clips[key] = synth(S.clip_text[key])
    return S.clips[key]


@app.get("/api/clip/{key}")
def clip(key: str):
    if key not in S.clips and key not in S.clip_text:
        raise HTTPException(404)
    data = S.clips.get(key) or _pool.submit(_ensure_clip, key).result()
    # prefetch the next two clips of the same reply
    keys = list(S.clip_text)
    if key in keys:
        for k in keys[keys.index(key) + 1: keys.index(key) + 3]:
            if k not in S.clips:
                _pool.submit(_ensure_clip, k)
    return Response(data, media_type="audio/wav")


class CommandReq(BaseModel):
    text: str
    sentence: int = 0


@app.post("/api/command")
def command(req: CommandReq):
    if S.session is None:
        raise HTTPException(409, "Load a paper first.")
    return {"transcript": req.text, **_reply(req.text, req.sentence)}


@app.post("/api/listen")
async def listen(request: Request, sentence: int = 0):
    raw = await request.body()
    pcm = np.frombuffer(raw, dtype=np.float32)
    text = stt().transcribe(pcm)
    if not text:
        return {"transcript": "", "say": "Sorry, I didn't catch that.",
                "clips": [{"url": f"/api/clip/{_static_clip('Sorry, I did not catch that.')}",
                           "text": "Sorry, I didn't catch that."}], "resume": sentence}
    if S.session is None:  # before a paper is loaded, the only useful thing to say is a paper
        return {"transcript": text, "say": None, "clips": [], "resume": None}
    return {"transcript": text, **_reply(text, sentence)}


@app.post("/api/speak")
def speak(req: CommandReq):
    """Speak arbitrary UI text (welcome, status) with the same voice."""
    return {"url": f"/api/clip/{_static_clip(req.text)}"}


def _static_clip(text: str) -> str:
    key = hashlib.sha1(text.encode()).hexdigest()[:16]
    if key not in S.clips:
        S.clips[key] = synth(text)
    return key


@app.on_event("startup")
def warm():
    # load models in the background so the first interaction is fast
    def _warm():
        tts()
        _static_clip("Welcome to hearxiv.")
        stt()
    threading.Thread(target=_warm, daemon=True).start()


def run(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn

    print(f"hearxiv is running at http://localhost:{port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
