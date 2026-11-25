"""The listening session: a cursor over the script plus a small command grammar.

This is transport-agnostic (no audio here). A front end (CLI, local web page, browser extension)
feeds it utterances from speech-to-text and plays whatever `Reply` comes back.

Design, from screen-reader and voice-UI research:
  * barge-in: the front end stops TTS the moment voice activity is detected, then calls handle()
  * after a question, resume at the START of the interrupted segment, and say where we are
  * "where am I", "what can I say", next/previous section, go to <section>, faster/slower are
    deterministic commands, matched before anything is sent to an LLM (fast, predictable, free)
  * everything else is a question for the Q&A agent, anchored on the current passage
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hearxiv.narrate.script import Script, figure_utterance, table_utterance
from hearxiv.narrate.spoken import to_spoken
from hearxiv.schema import PaperDoc

HELP = ("You can say: pause, resume, next section, previous section, go back, repeat, "
        "go to a section by name, where am I, describe figure and a number, read table and a number, "
        "faster, slower, or just ask a question about the paper.")


@dataclass
class Reply:
    say: str | None = None  # spoken immediately (answer, confirmation)
    resume_at: int | None = None  # then continue the script from this segment (None = stay paused)
    speed: float | None = None


@dataclass
class Session:
    doc: PaperDoc
    script: Script
    cursor: int = 0
    speed: float = 1.0
    history: list[tuple[str, str]] = field(default_factory=list)

    # navigation helpers ------------------------------------------------------------------------
    def _heading_indices(self) -> list[int]:
        """Section boundaries: explicit headings (full mode) or changes of source section (summaries)."""
        out, prev = [], None
        for i, s in enumerate(self.script.segments):
            if s.kind in {"heading", "abstract"} or (s.section_id and s.section_id != prev):
                out.append(i)
            prev = s.section_id or prev
        return sorted(set(out))

    def _section_index(self, sec_id: str) -> int | None:
        """First segment that comes from this section or one of its subsections."""
        return next((i for i, s in enumerate(self.script.segments)
                     if s.section_id and (s.section_id == sec_id or s.section_id.startswith(sec_id + "."))), None)

    def where_am_i(self) -> str:
        sec_id = next((s.section_id for s in reversed(self.script.segments[: self.cursor + 1]) if s.section_id), None)
        sec = next((s for s in self.doc.iter_sections() if s.id == sec_id), None)
        done = sum(len(s.text.split()) for s in self.script.segments[: self.cursor])
        total = max(1, self.script.words)
        left = (total - done) / (150 * self.speed)
        where = f"Section {sec.number}, {sec.title}" if sec and sec.number else (sec.title if sec else "the beginning")
        return f"You are in {where}, {round(100 * done / total)} percent through, about {round(left)} minutes left."

    def _find_object(self, kind: str, num: str):
        pool = self.doc.figures if kind.startswith("fig") else self.doc.tables
        return next((o for o in pool if re.search(rf"\b{num}\b", o.label)), None)

    # command handling --------------------------------------------------------------------------
    def handle(self, utterance: str) -> Reply:
        u = utterance.strip().lower().rstrip(".?!")
        here = self.cursor
        if u in {"pause", "stop", "wait", "hold on"}:
            return Reply()
        if u in {"resume", "continue", "go on", "play"}:
            return Reply(resume_at=here)
        if u in {"repeat", "say that again", "again"}:
            return Reply(resume_at=here)
        if u in {"go back", "back", "rewind"}:
            return Reply(resume_at=max(0, here - 1))
        if u in {"what can i say", "help", "commands"}:
            return Reply(say=HELP, resume_at=here)
        if u in {"where am i", "where are we"}:
            return Reply(say=self.where_am_i(), resume_at=here)
        if u in {"faster", "speed up"}:
            self.speed = min(4.0, self.speed + 0.25)
            return Reply(say=f"Speed {self.speed:g}.", resume_at=here, speed=self.speed)
        if u in {"slower", "slow down"}:
            self.speed = max(0.5, self.speed - 0.25)
            return Reply(say=f"Speed {self.speed:g}.", resume_at=here, speed=self.speed)
        if u in {"next section", "skip", "next"}:
            nxt = next((i for i in self._heading_indices() if i > here), None)
            return Reply(resume_at=nxt) if nxt is not None else Reply(say="This is the last section.", resume_at=here)
        if u in {"previous section", "last section"}:
            prev = [i for i in self._heading_indices() if i < here]
            target = prev[-2] if len(prev) >= 2 else 0
            return Reply(resume_at=target)
        if m := re.match(r"(?:go to|jump to|skip to)\s+(?:the\s+)?(?:section\s+)?(.+)", u):
            target = re.sub(r"^(?:the|section)\s+|\s+(?:section|part)$", "", m.group(1).strip()).strip()
            if target in {"abstract", "beginning", "start"}:
                i = next((i for i, s in enumerate(self.script.segments) if s.kind == "abstract"), 0)
                return Reply(resume_at=i)
            secs = list(self.doc.iter_sections())
            # exact number ("go to section 3.2"), then exact title, then title containing the words
            for test in (lambda s: s.number == target,
                         lambda s: s.title.lower() == target,
                         lambda s: target in s.title.lower(),
                         lambda s: all(w in s.title.lower() for w in target.split())):
                sec = next((s for s in secs if test(s)), None)
                if sec is None:
                    continue
                hit = self._section_index(sec.id)
                if hit is not None:
                    return Reply(resume_at=hit)
                return Reply(say=f"This summary doesn't cover {sec.title}. Load the whole paper to hear it.",
                             resume_at=here)
            return Reply(say=f"I couldn't find a section called {target}.", resume_at=here)
        if m := re.match(r"(?:describe|tell me about|explain)\s+(fig(?:ure)?|table)\.?\s*(\d+)", u):
            obj = self._find_object(m.group(1), m.group(2))
            if obj is None:
                return Reply(say=f"There is no {m.group(1)} {m.group(2)}.", resume_at=here)
            say = figure_utterance(obj, long=True) if hasattr(obj, "image_urls") and not hasattr(obj, "rows") \
                else table_utterance(obj)
            return Reply(say=say + f" Resuming.", resume_at=here)
        if m := re.match(r"read (?:the )?table\s*(\d+)?", u):
            tab = self._find_object("table", m.group(1)) if m.group(1) else None
            if tab is None:
                return Reply(say="Which table? Say read table and a number.", resume_at=here)
            rows = [f"Row {i + 1}: " + ", ".join(f"{h}: {c}" for h, c in zip(tab.header[0] if tab.header else [], r) if c)
                    for i, r in enumerate(tab.rows)]
            # always pair each value with its column header: losing header context is the main
            # failure mode of screen-reader table navigation
            return Reply(say=to_spoken(f"{tab.label}. " + " ".join(rows) + " End of table.", self.doc.glossary),
                         resume_at=here)

        # anything else: a question, anchored on what they were just hearing
        from hearxiv.agent.qa import answer

        current = self.script.segments[here].text if here < len(self.script.segments) else None
        ans = answer(self.doc, utterance, current_passage=current, history=self.history)
        self.history.append((utterance, ans.text))
        heading = self.where_am_i().split(",")[0].replace("You are in ", "")
        return Reply(say=f"{ans.text} Resuming {heading}.", resume_at=here)
