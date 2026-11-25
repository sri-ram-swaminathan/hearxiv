"""Build the spoken script for a paper.

A Script is a flat list of Segments. The flat list is the player's "cursor": "where am I",
"go back", "next section", and "resume after my question" are all just index arithmetic, and
every segment points back to its source (section / paragraph / figure id) so the Q&A agent and
"tell me more" can jump into the full text.

Modes
  full     Deterministic walk through the paper. Every sentence comes from the paper; figures are
           described right after the paragraph that FIRST mentions them (not where LaTeX happened
           to float them), later mentions get nothing extra (the listener can ask "describe figure
           3 again"). Figures never mentioned are described at the end of their section.
  5min /   An LLM "script agent" writes a summary with a word budget (~150 words per minute at
  10min    1x). It may only use facts from the condensed paper it is given and must tag every
           segment with the section it came from, so "dig deeper" can switch to full mode there.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from hearxiv.llm import LLM
from hearxiv.narrate.spoken import latex_to_words, to_spoken
from hearxiv.schema import Figure, PaperDoc, Section, Table

WORDS_PER_MINUTE = 150

SegmentKind = Literal["title", "abstract", "heading", "paragraph", "figure", "table", "summary", "note"]


class Segment(BaseModel):
    kind: SegmentKind
    text: str  # what is spoken (already normalised for TTS)
    source_id: str | None = None  # section / paragraph / figure / table id in the PaperDoc
    section_id: str | None = None
    earcon: str | None = None  # "section", "figure", "table", "footnote"


class Script(BaseModel):
    arxiv_id: str
    mode: str
    segments: list[Segment] = Field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)

    def minutes(self, speed: float = 1.0) -> float:
        return self.words / (WORDS_PER_MINUTE * speed)


# Shared helpers -------------------------------------------------------------------------------


def figure_utterance(fig: Figure, long: bool = False) -> str:
    """What is said for a figure. Short form inline; long form on request."""
    head = f"{fig.label}."
    if fig.description:
        body = fig.description.long if long else fig.description.short
        return f"{head} {body}"
    if fig.alt_text:
        return f"{head} The authors describe it as: {fig.alt_text}"
    # Be honest when we have no description (SciA11y: trust depends on admitting failures)
    return f"{head} I don't have a visual description of this figure. Its caption reads: {fig.caption}"


def table_utterance(tab: Table) -> str:
    if tab.summary:
        return f"{tab.label}. {tab.summary}"
    n_rows, n_cols = len(tab.rows), max((len(r) for r in tab.header + tab.rows), default=0)
    head = ", ".join(c for c in (tab.header[0] if tab.header else []) if c)
    shape = f"It has {n_rows} rows and {n_cols} columns" + (f", with columns {head}." if head else ".")
    return f"{tab.label}: {tab.caption} {shape} Say 'read the table' to go through it row by row."


_EQ_MARK = re.compile(r"\[eq:([^\]]*)\]")
MAX_VERBALIZED_EQ_CHARS = 90


def speak_equations(text: str, doc: PaperDoc) -> str:
    """Replace display-equation markers. Three depths (MathSpeak/ClearSpeak research): meaning-level
    sentence if the ingest step wrote one, a direct verbalisation for short formulas, and otherwise
    a pointer so the listener can ask for a symbol-by-symbol reading."""
    eqs = {e.id: e for e in doc.equations}

    def repl(m: re.Match) -> str:
        eq = eqs.get(m.group(1))
        if not eq:
            return " "
        label = eq.label or "an equation"
        if eq.spoken:
            return f" {label}: {eq.spoken} "
        if len(eq.latex) <= MAX_VERBALIZED_EQ_CHARS:
            return f" {label}: {latex_to_words(eq.latex)}. "
        return f" {label}, a longer formula. Say 'read the equation' to hear it symbol by symbol. "

    return _EQ_MARK.sub(repl, text)


def _section_heading(sec: Section) -> str:
    kind = {1: "Section", 2: "Subsection", 3: "Subsection", 4: ""}.get(sec.level, "")
    if sec.is_appendix:
        kind = "Appendix"
    num = f" {sec.number}" if sec.number and not sec.is_appendix else ""
    return re.sub(r"\s+", " ", f"{kind}{num}: {sec.title}." if kind else f"{sec.title}.").strip()


# Full mode ------------------------------------------------------------------------------------


def build_full_script(doc: PaperDoc, include_appendix: bool = False) -> Script:
    g = doc.glossary
    segs: list[Segment] = [
        Segment(kind="title", text=to_spoken(f"{doc.title}. By {_authors(doc)}.", g)),
    ]
    if doc.warnings:
        segs.append(Segment(kind="note", text="A note before we start: " + " ".join(doc.warnings)))
    if doc.abstract:
        segs.append(Segment(kind="abstract", text="Abstract. " + to_spoken(doc.abstract, g), earcon="section"))

    objects = {o.id: o for o in [*doc.figures, *doc.tables]}
    placed: set[str] = set()

    def place(obj_id: str, sec_id: str):
        obj = objects[obj_id]
        placed.add(obj_id)
        if isinstance(obj, Figure):
            segs.append(Segment(kind="figure", text=to_spoken(figure_utterance(obj), g), source_id=obj.id,
                                section_id=sec_id, earcon="figure"))
        else:
            segs.append(Segment(kind="table", text=to_spoken(table_utterance(obj), g), source_id=obj.id,
                                section_id=sec_id, earcon="table"))

    def walk(sec: Section):
        if sec.is_appendix and not include_appendix:
            return
        segs.append(Segment(kind="heading", text=to_spoken(_section_heading(sec), g), source_id=sec.id,
                            section_id=sec.id, earcon="section"))
        for p in sec.paragraphs:
            is_fn = p.text.startswith("[footnote]")
            if is_fn:
                continue  # footnotes are available on demand, not read inline
            text = to_spoken(speak_equations(p.text, doc), g)
            segs.append(Segment(kind="paragraph", text=text, source_id=p.id, section_id=sec.id))
            for m in p.mentions:  # describe each figure/table right after its FIRST mention
                if m.target_id in objects and m.target_id not in placed:
                    place(m.target_id, sec.id)
        for child in sec.children:
            walk(child)
        # figures/tables floated into this section but never referenced in the text
        for obj in objects.values():
            if obj.id not in placed and obj.section_id == sec.id and obj.first_mention_paragraph_id is None:
                place(obj.id, sec.id)

    for sec in doc.sections:
        walk(sec)
    segs.append(Segment(kind="note", text="End of paper. You can ask a question, or say 'go to' and a section name."))
    return Script(arxiv_id=doc.arxiv_id, mode="full", segments=segs)


def _authors(doc: PaperDoc) -> str:
    a = doc.authors
    if not a:
        return "unknown authors"
    if len(a) == 1:
        return a[0]
    if len(a) <= 3:
        return ", ".join(a[:-1]) + " and " + a[-1]
    return f"{a[0]}, {a[1]} and {len(a) - 2} others"


# Summary modes (LLM script agent) -------------------------------------------------------------

SUMMARY_SYSTEM = """\
You are the narrator of hearxiv, an audio reader of research papers for blind and low-vision \
researchers. You write scripts that are read aloud by a text-to-speech engine.

Rules:
- Single, neutral narrator. Plain spoken English. No markdown, lists, symbols or LaTeX; say \
"alpha", "d sub k", "O of n squared".
- Faithfulness first: use ONLY facts and numbers present in the material you are given. If a \
number is not given, do not invent one. Keep the authors' interpretation separate from fact \
("the authors argue...").
- Structure: say what the paper is about and why it matters, then the method, then the key \
results with their most important numbers, then limitations. Signpost each part ("Now, the method.").
- Figures: when a result depends on a figure, mention it by number with its one-sentence description \
(e.g. "Figure 3, a line plot of loss against steps, shows..."). The listener can ask for more.
- Respect the word budget closely.

Return JSON: {"segments": [{"section_id": "<id of the section this comes from, or null>", \
"text": "<spoken text>"}]}
"""


def condense(doc: PaperDoc, max_chars_per_section: int = 1800) -> str:
    """A compact view of the paper for the script agent: headings, leading text, figure/table info."""
    parts = [f"TITLE: {doc.title}", f"AUTHORS: {_authors(doc)}", f"ABSTRACT: {doc.abstract}"]
    for sec in doc.iter_sections():
        if sec.is_appendix:
            continue
        text = " ".join(p.text for p in sec.paragraphs if not p.text.startswith("[footnote]"))
        text = re.sub(r"\[(?:cite|eq):[^\]]*\]", "", text)
        parts.append(f"\n[section_id={sec.id}] {sec.number or ''} {sec.title}\n{text[:max_chars_per_section]}")
    for f in doc.figures:
        d = f.description.short if f.description else "(no visual description)"
        parts.append(f"\n[{f.label}, in section {f.section_id}] caption: {f.caption[:400]} | description: {d}")
    for t in doc.tables:
        parts.append(f"\n[{t.label}, in section {t.section_id}] caption: {t.caption[:300]} | "
                     f"header: {t.header[:1]} | first rows: {t.rows[:4]}")
    return "\n".join(parts)


def build_summary_script(doc: PaperDoc, minutes: int, llm: LLM | None = None) -> Script:
    llm = llm or LLM.from_env()
    budget = minutes * WORDS_PER_MINUTE
    user = (f"Write a {minutes}-minute spoken summary (about {budget} words, at most {int(budget * 1.1)}) "
            f"of this paper.\n\n{condense(doc)}")
    out = llm.chat_json(SUMMARY_SYSTEM, user, max_tokens=max(2000, budget * 3))
    segs = [Segment(kind="title", text=to_spoken(f"{doc.title}. A {minutes}-minute summary.", doc.glossary))]
    valid = {s.id for s in doc.iter_sections()}
    for s in out.get("segments", []):
        sid = s.get("section_id") if s.get("section_id") in valid else None
        segs.append(Segment(kind="summary", text=to_spoken(s["text"], doc.glossary), section_id=sid, source_id=sid))
    segs.append(Segment(kind="note", text="That's the summary. Ask me anything, or say 'tell me more' about any part."))
    return Script(arxiv_id=doc.arxiv_id, mode=f"{minutes}min", segments=segs)


def build_script(doc: PaperDoc, mode: str, llm: LLM | None = None) -> Script:
    if mode == "full":
        return build_full_script(doc)
    m = re.fullmatch(r"(\d+)\s*min", mode)
    if not m:
        raise ValueError("mode must be 'full', '5min' or '10min'")
    return build_summary_script(doc, int(m.group(1)), llm)


def dump(script: Script) -> str:
    return json.dumps(script.model_dump(), indent=1, ensure_ascii=False)
