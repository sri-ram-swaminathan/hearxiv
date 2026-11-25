# hearxiv

**Listen to arXiv papers, and talk back to them.**
hearxiv is a free, open-source, voice-first reader for research papers. It is built first for
blind and low-vision researchers in computer science.

- 🎧 **Three ways to listen:** a 5-minute summary, a 10-minute summary, or a full walkthrough of the paper.
- 📊 **Figures are actually described**, at the moment the text first discusses them, by a vision
  model that runs once per paper. Tables and equations are spoken properly too.
- 🗣️ **Interrupt at any time.** Ask "why do they scale by root d k?", "describe figure 3",
  "go to the results" or "where am I?". It answers from the paper, then picks up where you left off.
- 🔈 **On-device speech** with [Gradium's Phonon](https://huggingface.co/gradium/phonon-31m-en)
  (31M parameters, Apache-2.0). It runs at about 3× real time on a 2019 laptop CPU.
- 🆓 **Free forever.** Every component has a zero-cost path: local open models, Gemini's free
  tier, or a shared cache of pre-processed papers.

> Status: **early research prototype (M0).** The pipeline works end to end from the command line
> (parse → describe figures → script → speech → Q&A). The browser extension and live voice loop
> are next. See [docs/roadmap.md](docs/roadmap.md).

## Why

- arXiv's own accessibility study found that only 30% of assistive-technology users can access
  the research they need without help, and one participant said "figures are a crapshoot".
- Existing audio tools are either **static** (paper-to-podcast, which you can't question),
  **paid**, or **ignore figures**.

hearxiv combines a structure-aware parser, real figure descriptions, a faithful narrator and a
conversational loop. The novelty is the **presentation logic** that ties these together: what to
say, when, at what depth, and how to recover after an interruption. See [docs/agents.md](docs/agents.md).

## How it works

```mermaid
flowchart LR
    A["arXiv / alphaXiv URL"] --> P["parse once<br/>LaTeXML HTML → PaperDoc"]
    P --> V["describe figures once<br/>vision model (cloud or local)"]
    V --> J[("paper.json")]
    J --> S["script<br/>full · 5 min · 10 min"]
    S --> N["spoken-text pass<br/>math, numbers, jargon lexicon"]
    N --> T["Phonon TTS<br/>(on device)"]
    T --> L(("listener"))
    L -- "interrupts (voice)" --> Q["session: commands + grounded Q&A"]
    Q --> T
```

The detailed diagrams (per-paper ingest, runtime loop, barge-in sequence) are in
[docs/architecture.md](docs/architecture.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/sri-ram-swaminathan/hearxiv && cd hearxiv
uv sync --extra tts --extra svg          # add --extra local for on-device vision models
cp .env.example .env                     # optional: OPENAI_API_KEY and/or GEMINI_API_KEY

# 1. parse + describe figures (once per paper)
uv run python -m hearxiv ingest 1706.03762 --describer openai:gpt-6.1-terra --equations
#    zero-cost alternatives:  --describer gemini:gemini-3.5-flash-lite   (free tier)
#                             --describer hf:Qwen/Qwen3.5-2B             (local CPU, slow)

# 2. read the script (full mode needs no LLM at all)
uv run python -m hearxiv script 1706.03762 --mode full
uv run python -m hearxiv script 1706.03762 --mode 5min

# 3. hear it
uv run python -m hearxiv speak 1706.03762 --mode 5min -o attention.wav

# 4. ask it something
uv run python -m hearxiv ask 1706.03762 "Why do they scale the dot products?"
```

The LLM used for summaries and Q&A is set with `HEARXIV_LLM`: `openai:gpt-6.1-terra` (default),
`gemini:gemini-3.5-flash-lite`, or `local:<model>` for any OpenAI-compatible local server
(llama.cpp, Ollama, LM Studio).

## Which model describes figures best?

See [`notebooks/01_figure_description_benchmark.ipynb`](notebooks/01_figure_description_benchmark.ipynb).
It covers 18 standard seaborn/matplotlib figures with ground truth taken from the plotting code,
11 models (cloud and local CPU), automatic fact checks and an LLM judge. All descriptions are
side by side in [`benchmarks/figure_ocr/DESCRIPTIONS.md`](benchmarks/figure_ocr/DESCRIPTIONS.md).

<!-- BENCHMARK_SUMMARY -->

## Repository layout

```
hearxiv/
├── src/hearxiv/
│   ├── schema.py            # PaperDoc: the one-time, cacheable result of ingesting a paper
│   ├── ingest/              # fetch (polite, cached), LaTeXML parser, figure/equation enrichment
│   ├── vision/              # FigureDescriber: openai: / gemini: / hf: / local: backends + prompt
│   ├── narrate/             # script builder (full / N-min), spoken-text pass, jargon lexicon
│   ├── agent/               # Q&A agent + listening session (commands, cursor, resume)
│   ├── tts/                 # Phonon wrapper, earcons
│   ├── llm.py               # one OpenAI-compatible client for every provider
│   └── cli.py               # python -m hearxiv ...
├── benchmarks/figure_ocr/   # seaborn figure suite, runner, scorer, judge, results
├── notebooks/               # experiments (01 = figure-description benchmark)
├── extension/               # Chrome MV3 skeleton: "Listen" button on arXiv / alphaXiv
├── docs/                    # architecture, agents, pronunciation, research notes, roadmap
└── tests/
```

## Contributing

The most valuable contributions right now:
1. **Pronunciations.** Add terms to [`src/hearxiv/narrate/lexicon/cs_ml.json`](src/hearxiv/narrate/lexicon/cs_ml.json)
   (see [docs/pronunciation.md](docs/pronunciation.md)).
2. **Hard figures.** Real paper figures that stump the describers, with a correct description.
3. **Feedback from blind and low-vision researchers** on what to say, when, and how much.

Notes for Windows developers: if Smart App Control is on, use a python.org Python (the project sets
`python-preference = "system"`) and run `python -m hearxiv` rather than the `hearxiv.exe` launcher.

## License

MIT. Phonon weights are Apache-2.0 (Gradium). Paper content belongs to its authors: hearxiv fetches
papers on demand for the person listening and does not redistribute them.
