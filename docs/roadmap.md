# Roadmap and development plan

Development is notebook-driven. Each open question gets an experiment notebook with a measurable
outcome. Then the winning approach moves into `src/hearxiv/` with tests.

## Milestones

### M0: foundations (this commit)
- [x] Repo layout, uv project, CI-able tests
- [x] arXiv HTML fetch and parse into `PaperDoc`, with a first-mention index for figures and tables
- [x] Spoken-text normaliser and CS/ML lexicon
- [x] Full-mode script (deterministic) and N-minute script agent
- [x] Q&A agent and session command grammar
- [x] Phonon TTS wrapper and earcons
- [x] **Notebook 01:** figure-description model benchmark on standard seaborn plots

### M1: end-to-end local demo
- [x] Local server (FastAPI): sentence-level Phonon audio with prefetch, push-to-talk audio in
- [x] Web app player: play/pause, speed, push-to-talk, earcons, keyboard shortcuts, large captions,
  live-region announcements, typed commands (screen-reader friendly: every control is a real button)
- [x] On-device STT (Moonshine-base via transformers)
- [x] Extension "Listen" button opens the app for that paper
- [ ] Hands-free barge-in with VAD (Silero / `@ricky0123/vad-web`) as an option next to push-to-talk
- [ ] Streaming Q&A answers (speak the first sentence while the LLM is still writing)
- [ ] Testing with screen readers (NVDA, JAWS, VoiceOver) and with blind users
- [ ] Notebook 02: pronunciation ASR round-trip over the lexicon
- [ ] Notebook 03: parse coverage on 200 random recent cs.* papers (sections found, figures with
  images, `.ltx_ERROR` rate)

### M2: browser extension (fits into alphaXiv)
- [ ] MV3 content script on `arxiv.org/abs|html|pdf/*` and `alphaxiv.org/abs/*` adds a "Listen"
  button (see `extension/`)
- [ ] Side panel UI talking to the local server, then a fully in-browser build (`phonon-tts` WASM)
- [ ] Shared ingest cache: a static JSON per paper on a CDN / GitHub Pages, so most users never run
  a vision model

### M3: quality
- [ ] Number-verification pass for summaries
- [ ] Per-paper glossary at ingest
- [ ] Table header merging; table navigation by row and column
- [ ] Diagram graph-walk mode
- [ ] PDF fallback via Docling for papers without HTML
- [ ] User study with blind and low-vision researchers (recruit via NFB / RNIB / university
  disability offices)

## Notebooks

| # | Question | Status |
|---|---|---|
| 01 | Which vision model should describe figures (quality vs. cost vs. local)? | done (v1) |
| 02 | Does Phonon pronounce lexicon respellings correctly? (ASR round-trip) | planned |
| 03 | How often does arXiv HTML parsing fail, and how? | planned |
| 04 | Summary faithfulness: hallucinated numbers per minute of summary | planned |
| 05 | Q&A grounding: answer accuracy on a hand-written question set for 10 papers | planned |
| 06 | Real paper figures (not synthetic): diagrams, heatmaps, multi-panel | planned |

## Model strategy (the zero-cost goal)

- **Default for everyone:** the shared ingest cache. Descriptions are computed once (by any
  contributor or by CI), and listening needs only Phonon plus an LLM for Q&A.
- **Fully local:** Qwen3.5 (0.8B / 2B / 4B) or Granite Vision via llama.cpp GGUF for figures.
  The same small model, or Gemma, via llama.cpp / Ollama for Q&A, using `HEARXIV_LLM=local:...`.
- **Free cloud:** Gemini Flash-Lite free tier (about 500 requests per day) is enough for about 50
  papers a day of figure descriptions.
