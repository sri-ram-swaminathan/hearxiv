"""Turn paper text into text that SOUNDS right when read by a grapheme-based TTS (Phonon).

Pipeline (each step is a pure function, easy to test and to override):
  1. strip_citations   "[cite:bib.bib13]" -> "" (the bibliography is offered on demand instead)
  2. verbalize_math    "$d_{k}$" -> "d sub k", "$O(n^2)$" -> "O of n squared"; long formulas -> a
                       placeholder the script agent replaces with a meaning-level sentence
  3. verbalize_numbers "1e-4" -> "1 times ten to the minus 4", "7B" -> "7 billion", "3x3" -> "3 by 3"
  4. apply_lexicon     community CS/ML pronunciation lexicon + per-paper glossary
Phonon's own normaliser then handles plain numbers, dates, URLs and currency.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path(__file__).parent / "lexicon" / "cs_ml.json"

# 1. Citations ---------------------------------------------------------------------------------

_CITE = re.compile(r"\s*\[cite:[^\]]*\]")


def strip_citations(text: str) -> str:
    text = _CITE.sub("", text)
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)  # empty parens left behind, e.g. "( )"
    return re.sub(r"\s+([,.;:])", r"\1", text)


# 2. Math --------------------------------------------------------------------------------------

_GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi",
    "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
    "Phi", "Psi", "Omega",
}
_MATH_WORDS = {
    r"\in": " in ", r"\times": " times ", r"\cdot": " times ", r"\le": " at most ", r"\leq": " at most ",
    r"\ge": " at least ", r"\geq": " at least ", r"\neq": " not equal to ", r"\approx": " approximately ",
    r"\sim": " distributed as ", r"\to": " to ", r"\rightarrow": " to ", r"\infty": " infinity ",
    r"\log": " log ", r"\exp": " exp ", r"\max": " max ", r"\min": " min ", r"\sum": " sum of ",
    r"\prod": " product of ", r"\nabla": " gradient of ", r"\partial": " partial ", r"\top": " transpose ",
    r"\mathbb{R}": " R ", r"\mathbb{E}": " expected value of ", r"\ell": " ell ", r"\pm": " plus or minus ",
    r"\ldots": " dot dot dot ", r"\dots": " dot dot dot ", r"\cdots": " dot dot dot ", r"\|": " norm ",
}
MAX_INLINE_MATH_CHARS = 40  # longer formulas get a meaning-level paraphrase from the script agent


def latex_to_words(tex: str) -> str:
    r"""Verbalise a (short) LaTeX formula: d_{k} -> "d sub k", \frac{a}{b} -> "a over b"."""
    s = tex.strip()
    s = re.sub(r"\\(?:displaystyle|textstyle|left|right|big|Big|quad|qquad)\b|\\[,;!]|&", " ", s)
    s = s.replace(r"\_", "_")
    # [^{}]* only matches innermost groups, so apply the rules until nothing changes:
    # \frac{QK^{T}}{\sqrt{d_{k}}} -> \frac{QK transpose}{\sqrt{d sub k}} -> ... -> "QK transpose over ..."
    rules = [
        (r"\\(?:mathrm|mathbf|mathit|text|textrm|operatorname|mathcal|boldsymbol|bm)\{([^{}]*)\}", r"\1"),
        (r"\^\{2\}|\^2(?!\d)", " squared "),
        (r"\^\{3\}|\^3(?!\d)", " cubed "),
        (r"\^\{\\top\}|\^\\top|\^\{T\}|\^T(?![a-z])", " transpose "),
        (r"\^\{([^{}]*)\}", r" to the \1 "),
        (r"\^(\w)", r" to the \1 "),
        (r"_\{([^{}]*)\}", r" sub \1 "),
        (r"_(\w)", r" sub \1 "),
        (r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r" \1 over \2 "),
        (r"\\sqrt\{([^{}]*)\}", r" square root of \1 "),
    ]
    for _ in range(6):
        before = s
        for pat, rep in rules:
            s = re.sub(pat, rep, s)
        if s == before:
            break
    for k in sorted(_MATH_WORDS, key=len, reverse=True):
        s = s.replace(k, _MATH_WORDS[k])
    s = re.sub(r"\\(" + "|".join(sorted(_GREEK, key=len, reverse=True)) + r")\b", r" \1 ", s)
    s = re.sub(r"\b([A-Za-z]+)\s*\(", r"\1 of (", s)  # O(n) -> O of n, Attention(Q,K,V) -> Attention of Q,K,V
    s = re.sub(r"\s*,\s*", ", ", s)
    s = s.replace("=", " equals ").replace("+", " plus ").replace("<", " less than ").replace(">", " greater than ")
    s = re.sub(r"(?<=\s)-(?=\s|\w)", " minus ", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)  # drop any remaining commands
    s = re.sub(r"[{}()\\]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_INLINE_MATH = re.compile(r"\$([^$]+)\$")


def verbalize_math(text: str, placeholder: str = "a formula") -> str:
    def repl(m: re.Match) -> str:
        tex = m.group(1)
        if len(tex) > MAX_INLINE_MATH_CHARS:
            return f" {placeholder} "
        return f" {latex_to_words(tex)} "

    return re.sub(r"\s+", " ", _INLINE_MATH.sub(repl, text)).strip()


# 3. Numbers -----------------------------------------------------------------------------------

_SCALE = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}


def verbalize_numbers(text: str) -> str:
    # 1e-4, 3e5, 1.5E-3
    text = re.sub(r"\b(\d+(?:\.\d+)?)[eE]-(\d+)\b", r"\1 times ten to the minus \2", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)[eE]\+?(\d+)\b", r"\1 times ten to the \2", text)
    # model / dataset sizes: 7B, 1.5T, 300M, 8K (not when followed by more letters, e.g. "8Kb")
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s?([KMBT])\b", lambda m: f"{m.group(1)} {_SCALE[m.group(2)]}", text)
    # 3x3, 224×224 -> "3 by 3"
    text = re.sub(r"\b(\d+)\s?[x×]\s?(\d+)\b", r"\1 by \2", text)
    # 2x faster / 10× -> "2 times"
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s?[x×](?=\s|$|[,.;])", r"\1 times", text)
    return text


# 4. Lexicon -----------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_lexicon(path: Path = LEXICON_PATH) -> tuple[dict[str, str], dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    words = {k: v["say"] for k, v in data["case_sensitive"].items()}
    return words, data.get("symbols", {})


def _compile(words: dict[str, str]) -> re.Pattern:
    alts = sorted(words, key=len, reverse=True)
    # a following hyphen is allowed ("CIFAR-10", "LoRA-based"), a preceding one is not ("Swin-T")
    return re.compile(r"(?<![\w-])(" + "|".join(re.escape(a) for a in alts) + r")(?!\w)")


def apply_lexicon(text: str, glossary: dict[str, str] | None = None) -> str:
    words, symbols = load_lexicon()
    if glossary:
        words = {**words, **glossary}  # paper-specific terms win
    pat = _compile(words)
    text = pat.sub(lambda m: words[m.group(1)], text)
    for sym, say in symbols.items():
        text = text.replace(sym, f" {say} " if not say.startswith(" ") else say)
    return re.sub(r"\s+", " ", text).strip()


# Full pass ------------------------------------------------------------------------------------


def to_spoken(text: str, glossary: dict[str, str] | None = None) -> str:
    text = strip_citations(text)
    text = verbalize_math(text)
    text = verbalize_numbers(text)
    text = apply_lexicon(text, glossary)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text
