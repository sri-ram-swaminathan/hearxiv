from hearxiv.ingest.arxiv_html import parse_arxiv_html
from hearxiv.narrate.script import build_full_script

HTML = """
<html><body><article class="ltx_document">
<h1 class="ltx_title ltx_title_document">A Tiny Paper</h1>
<div class="ltx_authors"><span class="ltx_personname">Ada Lovelace<span class="ltx_note"><sup>1</sup></span><br>Somewhere Univ.</span></div>
<div class="ltx_abstract"><p>We study tiny things.</p></div>
<section id="S1" class="ltx_section"><h2 class="ltx_title"><span class="ltx_tag">1 </span>Introduction</h2>
  <div id="S1.p1" class="ltx_para"><p class="ltx_p">Prior work <cite class="ltx_cite">[<a href="#bib.bib1">1</a>]</cite> is slow.
  Our loss is <math alttext="O(n^{2})"><mi>O</mi></math> as seen in
  <a class="ltx_ref" href="#S2.F1.sf2">Figure&nbsp;1b</a>.</p></div>
  <figure id="S1.F2" class="ltx_figure"><img class="ltx_graphics" src="2401.00001v1/x2.png" alt="Refer to caption">
    <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">Figure 2: </span>Never mentioned.</figcaption></figure>
</section>
<section id="S2" class="ltx_section"><h2 class="ltx_title"><span class="ltx_tag">2 </span>Method</h2>
  <div id="S2.p1" class="ltx_para"><p class="ltx_p">We compute</p>
    <table id="S2.E1" class="ltx_equation"><tr><td class="ltx_eqn_cell"><math alttext="y=x^{2}"><mi>y</mi></math></td>
    <td class="ltx_eqn_cell ltx_eqn_eqno"><span class="ltx_tag ltx_tag_equation">(1)</span></td></tr></table></div>
  <div id="S2.p2" class="ltx_para"><p class="ltx_p">See <a class="ltx_ref" href="#S2.F1">Figure 1</a> again.</p></div>
  <figure id="S2.F1" class="ltx_figure">
    <figure id="S2.F1.sf1" class="ltx_figure"><img class="ltx_graphics" src="2401.00001v1/a.png"></figure>
    <figure id="S2.F1.sf2" class="ltx_figure"><object type="image/svg+xml" data="2401.00001v1/b.svg"></object></figure>
    <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">Figure 1: </span>Two panels.</figcaption></figure>
</section>
</article></body></html>
"""


def doc():
    return parse_arxiv_html(HTML, "2401.00001", "https://arxiv.org/html/2401.00001v1", version="v1")


def test_metadata_and_structure():
    d = doc()
    assert d.title == "A Tiny Paper"
    assert d.authors == ["Ada Lovelace"]
    assert [s.number for s in d.sections] == ["1", "2"]


def test_figures_images_and_first_mention():
    d = doc()
    f1 = d.get_figure("S2.F1")
    assert f1.label == "Figure 1"
    assert f1.image_urls == ["https://arxiv.org/html/2401.00001v1/a.png", "https://arxiv.org/html/2401.00001v1/b.svg"]
    # the sub-figure reference "Figure 1b" in Section 1 counts as the first mention of Figure 1
    assert f1.first_mention_paragraph_id == "S1.p1"
    assert d.get_figure("S1.F2").first_mention_paragraph_id is None


def test_math_citations_and_equations():
    d = doc()
    p = d.sections[0].paragraphs[0]
    assert "$O(n^{2})$" in p.text and "[cite:bib.bib1]" in p.text
    assert d.equations[0].label == "Equation 1" and d.equations[0].latex == "y=x^{2}"


def test_full_script_places_figures():
    s = build_full_script(doc())
    kinds = [(seg.kind, seg.source_id) for seg in s.segments]
    # Figure 1 right after its first mention (S1.p1), Figure 2 (never mentioned) at the end of Section 1
    i = kinds.index(("paragraph", "S1.p1"))
    assert kinds[i + 1] == ("figure", "S2.F1")
    assert kinds[i + 2] == ("figure", "S1.F2")
    text = " ".join(seg.text for seg in s.segments)
    assert "O of n squared" in text and "Equation 1: y equals x squared" in text
    assert "[cite" not in text
