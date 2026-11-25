# How the "agents" reason about presentation

hearxiv's novelty is not any single model. It is the **presentation logic**: what to say, when,
at what depth, and how to recover when the listener interrupts. This page records those decisions
and the evidence behind them, so contributors can challenge them with data.

## The four roles

| Role | Input | Output | Model | Deterministic? |
|---|---|---|---|---|
| **Figure describer** | figure image + caption | long structured description, short 1–2 sentence form | vision model (see the benchmark) | no, but done once at ingest and cached |
| **Script builder (full)** | `PaperDoc` | `Script` of segments | none | yes |
| **Script agent (5 / 10 min)** | condensed `PaperDoc` + word budget | `Script` with each segment tagged by section | LLM | no, so it is constrained and source-tagged |
| **Conversation** | utterance + cursor + `PaperDoc` | reply, where to resume | regex grammar first, then LLM Q&A | commands yes, answers no |

## Presentation rules (and why)

1. **Describe a figure where it is first discussed, not where LaTeX floated it.**
   Every `a.ltx_ref` pointing at a figure is a mention. The first mention gets the short
   description inline, right after that paragraph. Later mentions get nothing extra, because the
   listener can say "describe figure 3". Figures that are never referenced are described at the
   end of their section. Sub-figure references (`S3.F2.sf1`) count as mentions of the parent.
   *Evidence:* SciA11y moved figures next to their first mention after readers lost their place
   when figures were grouped separately.

2. **Figure descriptions go from overview to detail on demand.**
   The order is chart type and subject, then axes and encodings, then series, then key numbers,
   then the main pattern. The short form keeps the overview and the main-pattern sentence.
   *Evidence:* Lundgard and Satyanarayan's four-level model. Blind readers rated statistics (L2)
   and trends (L3) most useful and editorialising (L4) least useful. Olli and VoxLens show the
   value of drilling down from an overview.

3. **Separate what the figure shows from what the authors claim.**
   The vision model describes only what is visible. Captions are given as naming context only.
   The authors' interpretation is said as "the authors argue…".

4. **Equations have three depths.**
   (a) A meaning-level sentence (LLM at ingest, cached). (b) A direct verbalisation for short
   formulas ("Attention of Q, K, V equals softmax of QK transpose over square root of d sub k,
   times V"). (c) "Read the equation" gives a symbol-by-symbol reading on demand.
   *Evidence:* the MathSpeak / ClearSpeak verbosity levels in the Speech Rule Engine.

5. **Tables: summary first, then a walk that always keeps the headers.**
   "Table 2, 11 rows by 4 columns…", then "read table 2" reads each value with its column header.
   *Evidence:* screen-reader table studies, where losing header context is the main failure.

6. **Citations are silent. Footnotes and the bibliography are on demand.**
   Bracketed citations are dropped from speech. Footnotes are kept in the `PaperDoc` but not read
   inline.

7. **Be honest about failures.**
   Parse warnings (LaTeX conversion errors, missing images, vector figures we could not render)
   are spoken at the start. A figure with no description says so and reads the caption.
   *Evidence:* SciA11y found that trust depends on admitting extraction failures.

8. **Single neutral narrator.**
   A two-host "podcast" format is a style choice that adds words and invented banter. Nothing in
   the accessibility literature asks for it.

## The summary (script) agent

- **Input:** a condensed `PaperDoc` (abstract, each section's first ~1,800 characters, figure
  short descriptions, table headers and first rows), plus a word budget of 150 wpm × minutes.
- **Constraints in the system prompt:** use only the facts given, no invented numbers, signpost
  sections, mention figures by number, label the authors' claims.
- **Output:** JSON `{"segments": [{"section_id", "text"}]}`. Unknown section ids are dropped, and
  the text goes through the same `to_spoken` normaliser.
- **Why source-tagging matters:** "tell me more" during a summary jumps into full mode at that
  section.
- **Planned:** a verification pass that checks every number in the summary against the
  `PaperDoc` text, and flags or removes any number it can't find.

## The conversation loop

1. Voice activity stops TTS immediately (barge-in).
2. The utterance is matched against a small command grammar: pause, resume, repeat, go back,
   next/previous section, go to <name or number>, where am I, what can I say, faster/slower,
   describe figure N, read table N. These are instant, free and predictable.
3. Anything else is a **question**. The Q&A agent receives:
   - the passage being heard when interrupted, so "what does *that* mean?" works,
   - the top-k retrieved excerpts (paragraphs, figure descriptions, tables, equations), with any
     figure or table named in the question always included,
   - the last three Q&A turns.
4. The answer is short, spoken-style, cites where it comes from ("In Section 4…"), and says "the
   paper doesn't say" rather than guessing.
5. Then: "Resuming Section 3.2", restarting at the beginning of the interrupted segment.

## Edge cases we know about

| Case | Current handling | Next step |
|---|---|---|
| Heatmaps without numbers | describer reports structure (diagonal, masks, hot spots) and colour-scale range | ask the model for approximate values only for the cells the text discusses |
| Architecture / flow diagrams | caption + vision description of boxes and arrows | a graph-walk mode: "what feeds into the decoder?" |
| Multi-panel figures | prompt asks for panel-by-panel; multi-image figures described per image | panel navigation ("panel b") |
| Vector figures (TikZ/SVG) | SVG rendered to PNG via PyMuPDF | — |
| Figure referenced before it appears / never | first-mention index / end of section | — |
| Multi-row table headers | first header row only | merge header rows into column paths |
| Very long equations | pointer + on-demand reading | LLM paraphrase at ingest (`--equations`) |
| Papers without HTML | error message | Docling PDF pipeline |
| Mispronounced new names | lexicon only | per-paper glossary built at ingest (see pronunciation.md) |
