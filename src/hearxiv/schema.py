"""The PaperDoc: everything hearxiv knows about one paper, built ONCE and cached as JSON.

Ingestion (parse + figure description) is the expensive step; narration and Q&A only read
this file. Keeping it a plain, versioned JSON document means it can be shared, cached on a CDN,
diffed, corrected by hand, and consumed by any front end (extension, web app, CLI).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1"


class Mention(BaseModel):
    """An in-text reference to a figure/table/equation, e.g. 'as shown in Figure 3'."""

    target_id: str  # id of the Figure/Table/Equation referenced
    paragraph_id: str
    char_offset: int  # position of the reference in the paragraph text


class Paragraph(BaseModel):
    id: str
    text: str  # cleaned text; math replaced by its LaTeX in $...$, citations kept as [cite:key]
    mentions: list[Mention] = Field(default_factory=list)


class Section(BaseModel):
    id: str
    number: str | None = None  # "3.1"
    title: str
    level: int  # 1 = section, 2 = subsection, 3 = subsubsection, 4 = paragraph heading
    paragraphs: list[Paragraph] = Field(default_factory=list)
    children: list[Section] = Field(default_factory=list)
    is_appendix: bool = False


class FigureDescription(BaseModel):
    model: str  # e.g. "openai:gpt-6.1-terra" or "hf:Qwen/Qwen3.5-2B"
    short: str  # one or two sentences, read inline at first mention
    long: str  # full structured description, read on demand ("describe figure 3")
    data_table: str | None = None  # optional chart-to-table extraction (CSV)


class Figure(BaseModel):
    id: str  # DOM id, e.g. "S3.F2"
    label: str  # "Figure 2"
    caption: str
    image_urls: list[str]  # several for sub-figures
    alt_text: str | None = None  # author-supplied alt text, if any
    kind: Literal["figure", "table"] = "figure"
    section_id: str | None = None  # where it is placed in the document
    first_mention_paragraph_id: str | None = None
    description: FigureDescription | None = None


class Table(BaseModel):
    id: str
    label: str  # "Table 1"
    caption: str
    header: list[list[str]]  # header rows
    rows: list[list[str]]
    section_id: str | None = None
    first_mention_paragraph_id: str | None = None
    image_urls: list[str] = Field(default_factory=list)  # tables included as images
    summary: str | None = None  # spoken overview ("5 methods by 4 benchmarks; ours best on 3")


class Equation(BaseModel):
    id: str
    label: str | None = None  # "Equation 3"
    latex: str
    spoken: str | None = None  # meaning-level verbalisation, filled by the script agent


class Reference(BaseModel):
    key: str  # "bib.bib13"
    text: str


class PaperDoc(BaseModel):
    schema_version: str = SCHEMA_VERSION
    arxiv_id: str
    version: str | None = None
    source: Literal["arxiv-html", "ar5iv", "latex", "pdf"] = "arxiv-html"
    source_url: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    sections: list[Section] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)  # paper-specific pronunciations
    warnings: list[str] = Field(default_factory=list)  # parse problems, told to the user honestly

    def iter_sections(self):
        stack = list(reversed(self.sections))
        while stack:
            s = stack.pop()
            yield s
            stack.extend(reversed(s.children))

    def iter_paragraphs(self):
        for s in self.iter_sections():
            for p in s.paragraphs:
                yield s, p

    def get_figure(self, fig_id: str) -> Figure | None:
        return next((f for f in self.figures if f.id == fig_id), None)
