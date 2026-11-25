"""Enrichment: the one-time, model-heavy pass over a parsed paper.

  * download each figure image (cached), render SVG to PNG
  * describe it with a vision model (long description), derive a one-sentence short form
  * optionally verbalise long equations at the meaning level with the LLM

The result is written back into the PaperDoc JSON so nothing here runs again for that paper.
"""

from __future__ import annotations

import re
from pathlib import Path

from hearxiv.ingest.fetch import CACHE_DIR, fetch_image
from hearxiv.schema import FigureDescription, PaperDoc
from hearxiv.vision import build_prompt, make_describer


def _svg_to_png(svg: Path) -> Path:
    import pymupdf  # optional dependency: `uv sync --extra svg`

    png = svg.with_suffix(".png")
    if not png.exists():
        with pymupdf.open(svg) as doc:
            doc[0].get_pixmap(dpi=200).save(png)
    return png


def _short_form(long: str) -> str:
    """First sentence (overview) + the sentence that states the main pattern, if we can find one."""
    sents = re.split(r"(?<=[.!?])\s+", long.strip())
    if not sents:
        return long
    short = sents[0]
    pattern = next((s for s in sents[1:] if re.search(
        r"\b(highest|lowest|best|worst|increas|decreas|outperform|trend|overall|peaks?|largest|smallest)\w*",
        s, re.I)), None)
    return f"{short} {pattern}" if pattern else short


def describe_figures(doc: PaperDoc, describer_spec: str, cache_dir: Path = CACHE_DIR,
                     overwrite: bool = False, log=print) -> PaperDoc:
    describer = make_describer(describer_spec)
    img_dir = cache_dir / f"{doc.arxiv_id}{doc.version or ''}".replace("/", "_") / "figures"
    for fig in doc.figures:
        if fig.description and not overwrite:
            continue
        if not fig.image_urls:
            continue
        parts, raw = [], []
        for i, url in enumerate(fig.image_urls):
            ext = Path(url).suffix.lower() or ".png"
            path = fetch_image(url, img_dir / f"{fig.id}_{i}{ext}")
            if ext == ".svg":
                path = _svg_to_png(path)
            prompt = build_prompt(caption=f"{fig.label}: {fig.caption}")
            if len(fig.image_urls) > 1:
                prompt += f"\nThis is panel {i + 1} of {len(fig.image_urls)} of the figure."
            out = describer.describe(path, prompt)
            log(f"  {fig.label} panel {i + 1}: {out.seconds:.1f}s")
            raw.append(out.text)
            parts.append(out.text if len(fig.image_urls) == 1 else f"Panel {i + 1}. {out.text}")
        n = len(raw)
        short = _short_form(raw[0]) if n == 1 else (
            f"It has {n} panels. Panel 1: {_short_form(raw[0])} Say 'describe {fig.label}' for all panels.")
        fig.description = FigureDescription(model=describer.name, short=short, long=" ".join(parts))
    return doc


EQUATION_SYSTEM = """\
You explain equations from research papers to a blind listener in ONE short spoken sentence that \
says what the equation computes and names its parts, e.g. "Attention is a softmax over the scaled \
dot products of queries and keys, used to weight the values." Use only the equation and context \
given. No LaTeX, no symbols. Return JSON: {"spoken": "..."}"""


def verbalize_equations(doc: PaperDoc, llm, min_chars: int = 60) -> PaperDoc:
    context = {p.text.split("[eq:")[0][-400:]: p for _, p in doc.iter_paragraphs()}
    for eq in doc.equations:
        if eq.spoken or len(eq.latex) < min_chars:
            continue
        ctx = next((k for k, p in context.items() if f"[eq:{eq.id}]" in p.text), "")
        out = llm.chat_json(EQUATION_SYSTEM, f"Context: {ctx}\nEquation (LaTeX): {eq.latex}", max_tokens=300)
        eq.spoken = out.get("spoken")
    return doc
