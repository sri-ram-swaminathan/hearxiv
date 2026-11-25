# Research notes: prior art and lessons (Nov 2026)

These notes are compiled from a literature and product survey done before writing code. Sources
are linked so the claims can be checked.

## Existing tools and their gaps

| Tool | What it does | Gap for hearxiv |
|---|---|---|
| NotebookLM Audio Overviews (+ interactive mode) | two AI hosts; press "Join" to ask a question | button-press interrupt, cloud-only, no section navigation, no figure handling |
| Google Illuminate | arXiv CS papers turned into a two-voice discussion | about 5 per day, not interactive, experimental |
| alphaXiv audio (Mar 2025) | spoken overviews and talking with the paper | closed, summary-style; Pro plan for heavy use |
| arXiv × ScienceCast pilot | 60-second summaries (astro-ph.HE) | fixed length, no navigation |
| Paper2Audio | skips clutter, summarises math and figures, screen-reader friendly | closed, freemium; developer says figures are still a work in progress |
| ElevenReader / Speechify | general PDF readers | ElevenReader's own docs say equations "may not be read as intended" and tables lose structure |
| Podcastfy, PDF2Audio, papercast, paper_to_podcast, NotebookLlama | open source, PDF → two-host podcast | cloud APIs, PDF text only, no figures, no interruption, no navigation |

**The gap we fill:** free and open source, TTS on the device, structure-aware, real figure
descriptions, interruptible by voice, navigable by section.

## Accessibility evidence

- **arXiv accessibility report** ([2212.07286](https://arxiv.org/abs/2212.07286)): only 30% of
  assistive-tech users find the research they need accessible without help. 55% prefer
  well-formatted HTML. One interviewee: "Figures are a crapshoot."
- **SciA11y** ([2105.00076](https://arxiv.org/abs/2105.00076)): broken headings are the worst
  error. Put figures at first mention. Link citations both ways. Admit extraction failures.
- **Lundgard & Satyanarayan** ([2110.04406](https://arxiv.org/abs/2110.04406)): the four semantic
  levels of chart descriptions. Blind readers value statistics and trends most, and editorialising
  least.
- **Olli / VoxLens / VizAbility:** a navigable tree from overview to detail, voice queries ("what's
  the max?"), and the finding that users need guidance on what they can ask. That is why hearxiv
  has a "what can I say" command.
- **DIAGRAM / NCAM guidelines:** brief, data-focused, consistent wording, short plus long
  descriptions.
- **SciCap+ / Chart-to-Text:** ground descriptions in the caption and the mentioning paragraphs,
  and verify numbers, because models make factual errors.

## Voice-UI evidence

- About 70% of screen-reader users navigate long pages by headings (WebAIM survey #10).
- Blind listeners comfortably follow faster speech than sighted listeners (Bragg et al.). Support
  up to 3–4× speed, adjusted as speech rate rather than by resampling.
- Barge-in is essential. The available commands are invisible, so the system must tell users what
  they can say.
- Earcons mark structure (section, figure, table) without extra words.

## Technology findings

- **arXiv HTML:** about 97% of new submissions get *some* HTML ([2605.16562](https://arxiv.org/abs/2605.16562)).
  Selectors are verified in `src/hearxiv/ingest/arxiv_html.py`. Vector figures arrive as
  `<object type="image/svg+xml">`.
- **Etiquette:** 1 request per 3 s, a descriptive User-Agent, cache everything, never bulk-crawl,
  and `/e-print` is disallowed for crawlers.
- **alphaXiv:** no public REST API. There is an MCP server (`api.alphaxiv.org/mcp/v1`, OAuth or API
  key, server-side only) and the id-swap URL trick (`arxiv` → `alphaxiv`). The integration path is a
  content script that adds a "Listen" button.
- **Phonon:** `gradium/phonon-31m-en`, Apache-2.0, ungated. Grapheme input, 24 kHz, streaming
  (exiting the stream stops generation), Python `ptts` and browser `phonon-tts` (WASM).
  Verified on this project's dev laptop (Ryzen 5 3500U, CPU only).
- **STT / VAD for barge-in:** Silero VAD (about 2 MB), Moonshine or Whisper-tiny. Play TTS from the
  same page that holds the microphone so browser echo cancellation works. Recommend headphones and
  offer push-to-talk.
- **Windows dev note:** Smart App Control can block DLLs in uv-managed Python builds and unsigned
  `.exe` launchers. Use a python.org interpreter (`python-preference = "system"`) and `python -m hearxiv`.
