"""Parse arXiv's LaTeXML HTML (arxiv.org/html/<id>, ar5iv) into a PaperDoc.

Why HTML and not PDF: the arXiv accessibility study (arXiv:2212.07286) found blind readers
strongly prefer well-structured HTML, and SciA11y (arXiv:2105.00076) found that broken heading
structure is the single most damaging extraction error. LaTeXML keeps the author's real section
tree, the LaTeX of every equation (in `alttext`), and explicit links from in-text references
to figures, so "where is Figure 3 first discussed?" is a lookup, not a guess.

Selectors (verified on arxiv.org/html/1706.03762v7, Nov 2026):
  sections   section.ltx_section / ltx_subsection / ltx_subsubsection / ltx_paragraph / ltx_appendix
  headings   .ltx_title (with span.ltx_tag holding the number)
  paragraphs div.ltx_para > p.ltx_p
  figures    figure.ltx_figure  (img.ltx_graphics, figcaption)
  tables     figure.ltx_table   (table.ltx_tabular, figcaption)
  math       math[alttext]; display equations in table.ltx_equation(group)
  citations  cite.ltx_cite ; bibliography li.ltx_bibitem
  cross-refs a.ltx_ref[href="#S3.F1"]
  footnotes  .ltx_note.ltx_role_footnote
  errors     .ltx_ERROR (LaTeX that failed to convert)
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

from hearxiv.schema import Equation, Figure, Mention, PaperDoc, Paragraph, Reference, Section, Table

SECTION_CLASSES = {
    "ltx_section": 1, "ltx_appendix": 1, "ltx_subsection": 2,
    "ltx_subsubsection": 3, "ltx_paragraph": 4,
}
SKIP_SECTION_TITLES = re.compile(r"^(acknowledg|references|bibliography)", re.I)


_WS = re.compile(r"\s+")  # Unicode \s also matches non-breaking spaces (LaTeX ~)


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _classes(t: Tag) -> list[str]:
    return t.get("class") or []


class _TextBuilder:
    """Flatten a DOM subtree to narration-ready text while recording cross-references."""

    def __init__(self, ref_targets: set[str]):
        self.parts: list[str] = []
        self.length = 0
        self.mentions: list[tuple[str, int]] = []
        self.ref_targets = ref_targets
        self.footnotes: list[str] = []

    def add(self, s: str):
        self.parts.append(s)
        self.length += len(s)

    def walk(self, node):
        if isinstance(node, NavigableString):
            self.add(str(node))
            return
        if not isinstance(node, Tag):
            return
        cls = _classes(node)
        if node.name == "math":
            tex = node.get("alttext", "")
            self.add(f" ${tex}$ " if tex else "")
            return
        if node.name == "cite" or "ltx_cite" in cls:
            keys = [a["href"].lstrip("#") for a in node.find_all("a", href=True) if a["href"].startswith("#bib")]
            self.add(f"[cite:{','.join(keys)}]" if keys else "")
            return
        if "ltx_note" in cls:  # footnote: keep it out of the main flow, offer it on demand
            content = node.find(class_="ltx_note_content")
            if content:
                self.footnotes.append(_clean(content.get_text(" ")))
            return
        if node.name == "table" and ("ltx_equation" in cls or "ltx_equationgroup" in cls):
            # keep a marker where the display equation sits; the narrator decides how to say it
            self.add(f" [eq:{node.get('id', '')}] ")
            return
        if node.name == "figure" and ("ltx_figure" in cls or "ltx_table" in cls):
            return  # handled as separate objects
        if node.name == "a" and "ltx_ref" in cls:
            target = node.get("href", "").lstrip("#")
            base = _figure_root(target)
            if base in self.ref_targets:
                self.mentions.append((base, self.length))
        for child in node.children:
            self.walk(child)

    def text(self) -> str:
        return _clean("".join(self.parts))


def _figure_root(target: str) -> str:
    """Map sub-figure / panel ids (S3.F2.sf1, S3.F2.g1) to the parent figure id (S3.F2)."""
    m = re.match(r"^(.*?\.(?:F|T)\d+)", target)
    return m.group(1) if m else target


def _paragraph_text(el: Tag, ref_targets: set[str]) -> tuple[str, list[tuple[str, int]], list[str]]:
    b = _TextBuilder(ref_targets)
    b.walk(el)
    raw = "".join(b.parts)
    # offsets were recorded on the raw text; re-map them after whitespace normalisation
    text = b.text()
    mentions = []
    for tid, off in b.mentions:
        prefix = _WS.sub(" ", raw[:off]).lstrip()
        mentions.append((tid, min(len(prefix), len(text))))
    return text, mentions, b.footnotes


def _heading(sec: Tag) -> tuple[str | None, str]:
    h = sec.find(re.compile(r"^h[1-6]$"), class_="ltx_title")
    if not h:
        return None, ""
    tag = h.find(class_="ltx_tag")
    number = _clean(tag.get_text()).rstrip(".") if tag else None
    if tag:
        tag.extract()
    title = _clean(h.get_text(" "))
    if tag:
        h.insert(0, tag)  # restore the DOM
    return number, title


def _caption(fig: Tag) -> tuple[str, str]:
    cap = fig.find("figcaption")
    if not cap:
        return "", ""
    tag = cap.find(class_=re.compile(r"ltx_tag_(figure|table)"))
    label = _clean(tag.get_text()).rstrip(":").strip() if tag else ""
    text = _clean(cap.get_text(" "))
    if label and text.startswith(label):
        text = text[len(label):].lstrip(": ").strip()
    return label, text


def _table_cells(tab: Tag) -> tuple[list[list[str]], list[list[str]]]:
    header, rows = [], []
    for tr in tab.find_all("tr"):
        cells = [_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        if not any(cells):
            continue
        is_head = tr.find_parent("thead") is not None or all(td.name == "th" for td in tr.find_all(["td", "th"]))
        (header if is_head and not rows else rows).append(cells)
    if not header and rows:
        header, rows = [rows[0]], rows[1:]
    return header, rows


def parse_arxiv_html(html: str, arxiv_id: str, source_url: str, source: str = "arxiv-html",
                     version: str | None = None) -> PaperDoc:
    soup = BeautifulSoup(html, "html.parser")
    warnings: list[str] = []
    # MathML carries both a presentation tree and the LaTeX source; plain get_text() would read
    # both ("O ( n 2 ) O(n^{2})"). Keep only the LaTeX, delimited so the narrator can verbalise it.
    for m in soup.find_all("math"):
        tex = m.get("alttext", "")
        if m.find_parent("table", class_=re.compile(r"ltx_equation")):
            m.replace_with(tex)  # display equations: the marker/Equation object handles delimiting
        else:
            m.replace_with(f" ${tex}$ " if tex else "")

    title_el = soup.find(class_="ltx_title_document")
    title = _clean(title_el.get_text(" ")) if title_el else arxiv_id
    authors = []
    for a in soup.select(".ltx_authors .ltx_personname"):
        for junk in a.select(".ltx_note, sup"):  # footnote marks, affiliations-as-footnotes
            junk.extract()
        # a personname block often holds "Name<br>Affiliation<br>email": keep the first line
        first = re.split(r"<br\s*/?>", a.decode_contents())[0]
        name = _clean(BeautifulSoup(first, "html.parser").get_text(" "))
        if name:
            authors.append(name)
    abstract_el = soup.find(class_="ltx_abstract")
    abstract = ""
    if abstract_el:
        abstract = " ".join(_clean(p.get_text(" ")) for p in abstract_el.find_all("p"))

    n_err = len(soup.select(".ltx_ERROR"))
    if n_err:
        warnings.append(f"{n_err} pieces of LaTeX failed to convert to HTML; some text or math may be missing.")

    # Figures and tables ---------------------------------------------------------------------
    figures: list[Figure] = []
    tables: list[Table] = []
    for el in soup.find_all("figure"):
        cls = _classes(el)
        fid = el.get("id")
        if not fid or el.find_parent("figure") is not None:  # sub-figures belong to their parent
            continue
        label, caption = _caption(el)
        sec = el.find_parent("section")
        sec_id = sec.get("id") if sec else None
        # raster <img> plus vector graphics embedded as <object type="image/svg+xml" data=...>
        imgs = [urljoin(source_url, img["src"]) for img in el.find_all("img") if img.get("src")]
        imgs += [urljoin(source_url, o["data"]) for o in el.find_all("object") if o.get("data")]
        alts = [img.get("alt", "") for img in el.find_all("img")]
        alt = next((a for a in alts if a and a.lower() != "refer to caption"), None)
        if "ltx_table" in cls:
            tab = el.find("table", class_="ltx_tabular")
            header, rows = _table_cells(tab) if tab else ([], [])
            tables.append(Table(id=fid, label=label or "Table", caption=caption, header=header, rows=rows,
                                section_id=sec_id, image_urls=imgs if not tab else []))
        elif "ltx_figure" in cls:
            if not imgs:
                svg = el.find("svg")
                if svg is not None:
                    warnings.append(f"{label or fid} is a vector drawing (TikZ/SVG); only its caption is available.")
                else:
                    warnings.append(f"{label or fid} has no image in the HTML.")
            figures.append(Figure(id=fid, label=label or "Figure", caption=caption, image_urls=imgs,
                                  alt_text=alt, section_id=sec_id))
    ref_targets = {f.id for f in figures} | {t.id for t in tables}

    # Equations ------------------------------------------------------------------------------
    equations: list[Equation] = []
    for eq in soup.select("table.ltx_equation, table.ltx_equationgroup"):
        if eq.find_parent("table", class_=re.compile("ltx_equationgroup")) is not None:
            continue
        cells = eq.select("td.ltx_eqn_cell:not(.ltx_eqn_eqno)") or [eq]
        tex = _clean(" ".join(c.get_text(" ") for c in cells if not c.find(class_="ltx_tag_equation")))
        tag = eq.find(class_="ltx_tag_equation")
        equations.append(Equation(id=eq.get("id") or f"eq{len(equations)+1}",
                                  label=f"Equation {_clean(tag.get_text()).strip('()')}" if tag else None,
                                  latex=tex))

    # Sections and paragraphs ----------------------------------------------------------------
    para_counter = 0

    def build_section(sec: Tag) -> Section | None:
        nonlocal para_counter
        level = next((SECTION_CLASSES[c] for c in _classes(sec) if c in SECTION_CLASSES), 1)
        number, stitle = _heading(sec)
        s = Section(id=sec.get("id") or f"sec{para_counter}", number=number, title=stitle or "Untitled",
                    level=level, is_appendix="ltx_appendix" in _classes(sec))
        for child in sec.children:
            if not isinstance(child, Tag):
                continue
            ccls = _classes(child)
            if child.name == "section" and any(c in SECTION_CLASSES for c in ccls):
                sub = build_section(child)
                if sub:
                    s.children.append(sub)
            elif "ltx_para" in ccls or child.name == "p":
                text, mentions, foots = _paragraph_text(child, ref_targets)
                if not text:
                    continue
                para_counter += 1
                pid = child.get("id") or f"p{para_counter}"
                s.paragraphs.append(Paragraph(
                    id=pid, text=text,
                    mentions=[Mention(target_id=t, paragraph_id=pid, char_offset=o) for t, o in mentions]))
                for ft in foots:
                    para_counter += 1
                    s.paragraphs.append(Paragraph(id=f"{pid}.fn{para_counter}", text=f"[footnote] {ft}"))
        return s

    article = soup.find("article") or soup
    sections = []
    for sec in article.find_all("section", recursive=True):
        if sec.find_parent("section") is not None:
            continue
        if not any(c in SECTION_CLASSES for c in _classes(sec)):
            continue
        built = build_section(sec)
        if built and not SKIP_SECTION_TITLES.match(built.title):
            sections.append(built)
    if not sections:
        warnings.append("No section structure was found; navigation by heading will not work.")

    # First mentions: narrate a figure where it is first discussed, not where it is placed ----
    doc = PaperDoc(arxiv_id=arxiv_id, version=version, source=source, source_url=source_url, title=title,
                   authors=authors, abstract=abstract, sections=sections, figures=figures, tables=tables,
                   equations=equations, warnings=warnings)
    first: dict[str, str] = {}
    for _, p in doc.iter_paragraphs():
        for m in p.mentions:
            first.setdefault(m.target_id, p.id)
    for obj in [*doc.figures, *doc.tables]:
        obj.first_mention_paragraph_id = first.get(obj.id)

    # References -------------------------------------------------------------------------------
    for li in soup.select("li.ltx_bibitem"):
        doc.references.append(Reference(key=li.get("id", ""), text=_clean(li.get_text(" "))))
    return doc
