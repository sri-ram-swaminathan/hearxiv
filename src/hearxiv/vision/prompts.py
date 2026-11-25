"""Prompts for describing figures to a blind or low-vision listener.

The structure follows Lundgard & Satyanarayan's four-level model of chart semantics
(arXiv:2110.04406): blind readers rated statistics (L2) and trends (L3) most useful and
editorialising (L4) least useful. So the description is built to:
  1. orient in one sentence (chart type + what it shows)              -> L1, short
  2. give the encodings needed to follow along (axes, units, legend)  -> L1
  3. give the key numbers and the main trend / comparison             -> L2 + L3
  4. note anything unusual (outliers, truncated or log axes, panels)
and never guess beyond what is visible.
"""

FIGURE_DESCRIPTION_PROMPT = """\
You are describing a figure from a computer-science research paper to a blind researcher who \
will HEAR your description read aloud by a text-to-speech engine.

Write plain spoken English (no markdown, no bullet points, no LaTeX). Follow this order:
1. One sentence: what kind of chart this is and what it shows overall.
2. The axes: labels, units, ranges, and whether an axis is logarithmic or does not start at zero. \
If there are several panels, say how many and describe each one in turn.
3. The legend or groups: name every series, category or panel.
4. The key numbers: read off the most important values (maxima, minima, values of each bar or \
group, start and end points of lines). Be precise; if you have to estimate, say "about".
5. The main pattern: trends, comparisons, rankings, clusters, correlations, and any outliers.

Only state what is visible in the image. Do not speculate about the paper. \
Keep it under 250 words.
"""

CAPTION_CONTEXT_TEMPLATE = """\

The paper's caption for this figure is: "{caption}"
Use the caption to name things correctly, but describe what the image actually shows.
"""


def build_prompt(caption: str | None = None) -> str:
    """The prompt sent to every vision model. Captions are optional context."""
    if caption:
        return FIGURE_DESCRIPTION_PROMPT + CAPTION_CONTEXT_TEMPLATE.format(caption=caption.strip())
    return FIGURE_DESCRIPTION_PROMPT
