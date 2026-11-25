# Figure-description benchmark

The question: **which vision model should describe a paper's figures for someone who can't see them?**

- **Test set:** 18 standard seaborn/matplotlib figures, from easy to hard: lines, bars, heatmaps with and
  without annotations, multi-panel figures, facet grids, pair plots, a clustermap and more.
- **Ground truth:** taken from the plotting code (`ground_truth.json`), so it is exact.
- **Prompt:** every model gets the same prompt (`src/hearxiv/vision/prompts.py`).

## Scoring

1. **Automatic checks** (`scoring.py`):
   - did the model name the chart type;
   - did it recall the axis, series and panel labels (fuzzy match);
   - did it get each numeric or qualitative fact right. Spoken numbers count too, for example
     "eighty-four point six".
2. **LLM judge** (`judge_results.py`, `gpt-6.2-sol`):
   - the judge sees the image plus the answer key;
   - it scores accuracy, completeness and listenability from 0 to 5, and lists hallucinated claims.
3. **Quality** = mean of fact accuracy, label recall, accuracy ÷ 5 and completeness ÷ 5, on a 0–1 scale.

Full analysis: [`notebooks/01_figure_description_benchmark.ipynb`](../../notebooks/01_figure_description_benchmark.ipynb).
Every description side by side: [`DESCRIPTIONS.md`](DESCRIPTIONS.md).

## Results (v1)

| Model | Runs | Quality | Hallucinations / figure | Median s / figure | Cost / 1,000 figures |
|---|---|---|---|---|---|
| `gpt-6.1-terra` | OpenAI | **0.93** | 0.22 | 8.4 | $3.09 |
| `gpt-6.2-astra` | OpenAI | 0.91 | **0.11** | 6.9 | $18.09 |
| `gemini-3.8-flash` *(5/18 only)* | Google free tier | 0.89 | 1.40 | — † | $0 |
| `gpt-6-luna` | OpenAI | 0.89 | 1.00 | 5.3 | $0.36 |
| `gemma-4-31b-it` *(17/18)* | Google API, open weights | 0.86 | 1.71 | — † | $0 |
| `gpt-5.4-mini` | OpenAI | 0.86 | 1.78 | 3.4 | $2.22 |
| `gemini-3.5-flash-lite` | Google free tier | 0.85 | 2.17 | **2.0** | $0 |
| `Qwen3.5-0.8B` | **local CPU** | 0.68 | 5.72 | 103 | $0 |
| `granite-docling-258M` (chart→table) *(3/18)* | local CPU | 0.33 | 4.67 | ~3,800 ‡ | $0 |
| `SmolVLM-256M-Instruct` | local CPU | 0.33 | 1.61 | 61 | $0 |

† Gemini and Gemma latencies are dominated by free-tier rate-limit and 503 waits, so they are not
comparable.
‡ This run was under heavy memory pressure.

Timings were measured on an AMD Ryzen 5 3500U with 14 GB RAM and no GPU. Partial runs average over a subset,
so treat them as indicative only.

## What to use

- **Default: `gpt-6.1-terra`.** It has the best overall quality and very few hallucinations. A typical paper
  has 5–10 figures, so describing one costs about 2–3 cents. That cost is paid **once per paper**, and with the
  shared ingest cache every later listener gets the result for free.
- **Not worth it: `gpt-6.2-astra`.** It has the best judged accuracy and the fewest invented claims, but at
  6× the price for no gain in composite quality.
- **Near-free: `gpt-6-luna`.** Quality is 0.89 at about $0.004 per paper. This is the best value.
- **Zero-cost: `gemini-3.5-flash-lite` (free tier).** It is fast and complete, but it invents about two
  claims per figure, usually a wrong number. For a listener who can't check the figure, a confident wrong
  number is the worst possible failure. Use it when nothing else is available, and treat its numbers as
  unverified.
- **Open weights: `gemma-4-31b-it`** reaches 0.86. It is the best open model here, but at 31B it is not
  laptop-sized.
- **Tiny local models are not good enough yet.**
  - Qwen3.5-0.8B reads values but contradicts itself and invents almost 6 claims per figure.
  - SmolVLM-256M (Docling's default describer) produces a title-length sentence.
  - Granite-Docling's chart-to-table mode gets most values wrong.
  - A local path probably needs a quantised 2–4B model through llama.cpp: Qwen3.5-2B/4B, Granite Vision 4.1
    or Gemma 4 E4B. None of these were tested here (see the caveats).

## What every model finds hard

The facts that models miss most often are not values. They are the *patterns*:

- "diminishing returns";
- "validation diverges after epoch 12";
- "shaded bands are seed variability";
- "the upper triangle is masked" (causal attention);
- facts inside one panel of a multi-panel figure.

These are exactly what a sighted reader takes in at a glance. This is why the prompt asks for the pattern
explicitly, and why hearxiv offers "describe figure N" for the full multi-panel version.

## Caveats

- **Synthetic figures.** Real paper figures (architecture diagrams, dense multi-panel plots) are harder. They
  are planned for notebook 06.
- **The judge is an OpenAI model**, which may favour OpenAI-style answers. The automatic fact checks do not
  depend on the judge. They rank the models in broadly the same order (fact accuracy: terra 0.97,
  astra and luna 0.93, Gemma 0.91, flash-lite 0.86, Qwen 0.8B 0.80).
- **Several runs did not finish:**
  - `Qwen3.5-2B` was never reached;
  - `granite-docling` and `gemini-3.8-flash` are partial;
  - Granite Vision 4.1 and Gemma 4 E2B/E4B need more than 14 GB in float32 on this CPU. llama.cpp binaries,
    which would run them quantised, are blocked by Windows Smart App Control on the test machine.

  To resume, rerun `run_benchmark.py`: it skips figures it has already done.

## Reproduce

```bash
uv run python benchmarks/figure_ocr/make_figures.py
uv run python benchmarks/figure_ocr/run_benchmark.py openai:gpt-6.1-terra gemini:gemini-3.5-flash-lite
uv run python benchmarks/figure_ocr/run_benchmark.py hf:Qwen/Qwen3.5-2B --max-side 768 --max-new-tokens 400
uv run python benchmarks/figure_ocr/judge_results.py
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace 01_figure_description_benchmark.ipynb
```
