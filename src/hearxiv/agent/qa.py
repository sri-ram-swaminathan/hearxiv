"""Answer a listener's question about the paper, grounded in the PaperDoc.

Retrieval is deliberately simple (keyword overlap with light IDF weighting, no embeddings):
it runs anywhere, needs no extra model, and is easy to inspect. Context given to the LLM:
  * the passage the listener was hearing when they interrupted (conversational anchor:
    "what does THAT mean?")
  * the top-k paragraphs / figure descriptions / tables matching the question
  * any figure or table explicitly named in the question ("figure 3")
The answer is spoken-style, short, says where in the paper it comes from, and says
"the paper doesn't say" instead of guessing.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from hearxiv.llm import LLM
from hearxiv.narrate.spoken import to_spoken
from hearxiv.schema import PaperDoc

_STOP = set("""a an the of to in on for and or is are was were be been with by as at from that this these those
it its we our they their which what how why when where who does do did can could would should into than then
there here about over under not no yes so such also more most very paper authors""".split())


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Chunk:
    ref: str  # human-readable location, e.g. "Section 3.2" or "Figure 2"
    id: str
    text: str


def chunks(doc: PaperDoc) -> list[Chunk]:
    out = [Chunk("the abstract", "abstract", doc.abstract)] if doc.abstract else []
    for sec, p in doc.iter_paragraphs():
        ref = f"Section {sec.number}" if sec.number else sec.title
        out.append(Chunk(f"{ref}, {sec.title}", p.id, re.sub(r"\[(?:cite|eq):[^\]]*\]", "", p.text)))
    for f in doc.figures:
        desc = f.description.long if f.description else ""
        out.append(Chunk(f.label, f.id, f"{f.label}. Caption: {f.caption} Visual description: {desc}"))
    for t in doc.tables:
        rows = "; ".join(" | ".join(r) for r in (t.header + t.rows)[:25])
        out.append(Chunk(t.label, t.id, f"{t.label}. Caption: {t.caption} Contents: {rows}"))
    for e in doc.equations:
        out.append(Chunk(e.label or "an equation", e.id, f"{e.label or 'Equation'}: {e.latex} {e.spoken or ''}"))
    return out


def retrieve(doc: PaperDoc, question: str, k: int = 6) -> list[Chunk]:
    cs = chunks(doc)
    df = Counter(t for c in cs for t in set(_tokens(c.text)))
    n = len(cs)
    q = _tokens(question)
    named = {m.group(0).lower() for m in re.finditer(r"\b(figure|fig\.?|table|equation|eq\.?)\s*\d+", question, re.I)}

    def score(c: Chunk) -> float:
        tf = Counter(_tokens(c.text))
        s = sum((1 + math.log(tf[t])) * math.log(1 + n / (1 + df[t])) for t in q if tf[t])
        if any(c.ref.lower().replace("fig.", "figure").startswith(nm.replace("fig.", "figure").replace("eq.", "equation"))
               for nm in named):
            s += 100  # explicitly named object always included
        return s / (1 + 0.002 * len(c.text))

    return sorted(cs, key=score, reverse=True)[:k]


QA_SYSTEM = """\
You are hearxiv, answering a blind researcher's spoken question about a paper they are listening to.
- Answer ONLY from the provided excerpts. If they do not contain the answer, say "The paper doesn't \
say" and, if helpful, what it does say that is related.
- Be brief: two to five spoken sentences unless asked for detail. Plain spoken English, no markdown, \
no LaTeX (say "d sub k", "alpha").
- Say where the answer comes from, naturally: "In Section 4..." or "Table 2 shows...".
- If the question is about what they just heard, use the CURRENT PASSAGE.
Return JSON: {"answer": "...", "sources": ["<ids of excerpts used>"]}
"""


@dataclass
class Answer:
    text: str
    sources: list[str]


def answer(doc: PaperDoc, question: str, current_passage: str | None = None, llm: LLM | None = None,
           history: list[tuple[str, str]] | None = None) -> Answer:
    llm = llm or LLM.from_env()
    ctx = "\n\n".join(f"[id={c.id}] ({c.ref}) {c.text[:1500]}" for c in retrieve(doc, question))
    hist = "\n".join(f"Q: {q}\nA: {a}" for q, a in (history or [])[-3:])
    user = (f"PAPER: {doc.title}\n\nCURRENT PASSAGE: {current_passage or '(none)'}\n\n"
            f"EXCERPTS:\n{ctx}\n\n{'EARLIER QUESTIONS:\n' + hist if hist else ''}\n\nQUESTION: {question}")
    out = llm.chat_json(QA_SYSTEM, user, max_tokens=800)
    return Answer(text=to_spoken(out.get("answer", ""), doc.glossary), sources=out.get("sources", []))
