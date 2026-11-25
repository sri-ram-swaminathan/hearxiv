"""Score figure descriptions against code-derived ground truth.

Two complementary layers:

1. Automatic, deterministic checks (cheap, reproducible, no model bias)
   - type_hit      : the chart type is named (any keyword)
   - label_recall  : fraction of title / axis labels / series names mentioned (fuzzy match)
   - fact_acc      : fraction of checkable facts satisfied (numbers within tolerance, key phrases)

2. LLM-as-judge (catches hallucinations and judges usefulness for a listener)
   - accuracy, completeness, listenability on 0-5; hallucinations as a list of false claims
   The judge sees the ground truth AND the image, so it can verify claims the answer key
   does not cover. Caveat: an OpenAI judge may favour OpenAI-written text; compare with the
   automatic metrics, which have no such bias.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from rapidfuzz import fuzz

HERE = Path(__file__).parent

# Numbers -------------------------------------------------------------------------------------

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_NUM_RE = re.compile(
    r"(?<![A-Za-z])"  # avoid matching inside identifiers like 'ViT-B16'
    r"(?:10\s*\^\s*\{?(-?\d+)\}?"  # 10^-3, 10^{-3}
    r"|10(-\d+)"  # 10-3 after superscript normalisation (10⁻³ -> 10-3)
    r"|(-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?))"
)


def extract_numbers(text: str) -> list[float]:
    t = text.translate(_SUPERSCRIPT)
    out = []
    for m in _NUM_RE.finditer(t):
        exp1, exp2, plain = m.groups()
        try:
            if exp1 is not None:
                out.append(10 ** int(exp1))
            elif exp2 is not None:
                out.append(10 ** int(exp2))
                out.append(10.0)  # it could also simply be "10-3" meaning 10 and 3
            else:
                s = plain.replace(",", "").rstrip(".")
                if s in {"", "-"}:
                    continue
                out.append(float(s))
        except (ValueError, OverflowError):
            continue
    out += spoken_numbers(t)
    return out


# Spoken numbers ("four point five", "eighty-eight", "ten thousand", "zero point nine four") ----
# Models asked to write for text-to-speech often spell numbers out; the scorer must read them too.

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUMWORD = re.compile(r"[a-z]+(?:-[a-z]+)?|\.|,")


def spoken_numbers(text: str) -> list[float]:
    toks = _NUMWORD.findall(text.lower())
    out, i = [], 0
    while i < len(toks):
        parts = toks[i].split("-")
        if not all(p in _UNITS or p in _TENS for p in parts) and toks[i] not in ("a",):
            i += 1
            continue
        if toks[i] == "a" and (i + 1 >= len(toks) or toks[i + 1] not in _SCALES):
            i += 1
            continue
        total, current, j = 0, 0, i
        while j < len(toks):
            tok = toks[j]
            ps = tok.split("-")
            if tok == "a":
                current += 1
            elif all(p in _UNITS or p in _TENS for p in ps):
                current += sum(_UNITS.get(p, _TENS.get(p, 0)) for p in ps)
            elif tok == "hundred":
                current = max(current, 1) * 100
            elif tok in ("thousand", "million", "billion"):
                total += max(current, 1) * _SCALES[tok]
                current = 0
            elif tok == "and" and j + 1 < len(toks) and (toks[j + 1] in _UNITS or toks[j + 1].split("-")[0] in _TENS):
                pass
            else:
                break
            j += 1
        value = float(total + current)
        if j < len(toks) and toks[j] == "point":  # decimals are read digit by digit
            digits, k = "", j + 1
            while k < len(toks) and toks[k] in _UNITS and _UNITS[toks[k]] < 10:
                digits += str(_UNITS[toks[k]])
                k += 1
            if digits:
                value = float(f"{int(value)}.{digits}")
                j = k
        out.append(value)
        i = max(j, i + 1)
    return out


def number_hit(text: str, value: float, tol: float) -> bool:
    nums = extract_numbers(text)
    candidates = [value, value * 100, value / 100]  # 0.89 vs "89%", and vice versa
    tols = [tol, tol * 100, tol / 100]
    return any(abs(n - c) <= t for n in nums for c, t in zip(candidates, tols))


# Labels --------------------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.%/ -]+", " ", s.lower()).strip()


def label_present(label: str, text: str, threshold: int = 85) -> bool:
    lab = _norm(label)
    if not lab or lab in {"same", "count"}:
        return True if lab == "same" else ("count" in text.lower())
    return fuzz.partial_ratio(lab, _norm(text)) >= threshold


def expected_labels(gt: dict) -> list[str]:
    labels = []
    for key in ("title", "x_label", "y_label"):
        val = gt.get(key) or ""
        for part in re.split(r"\s+[/|]\s+", val):
            part = re.sub(r"^\([a-z]\)\s*", "", part)  # drop panel markers "(a) "
            part = re.sub(r"\s*\(r = [-\d.]+\)", "", part)  # numeric part scored as a fact instead
            if part and part.lower() != "same":
                labels.append(part)
    labels += gt.get("series", [])
    return labels


# Automatic score -----------------------------------------------------------------------------


def auto_score(gt: dict, text: str) -> dict:
    low = text.lower()
    type_hit = any(k.lower() in low for k in gt["type_keywords"])
    labels = expected_labels(gt)
    found = [lab for lab in labels if label_present(lab, text)]
    fact_results = []
    for f in gt["facts"]:
        ok = number_hit(text, f["value"], f["tol"]) if f["kind"] == "number" else any(a in low for a in f["any"])
        fact_results.append({"desc": f["desc"], "ok": ok})
    return {
        "type_hit": float(type_hit),
        "label_recall": len(found) / len(labels) if labels else 1.0,
        "fact_acc": sum(r["ok"] for r in fact_results) / len(fact_results) if fact_results else 1.0,
        "missed_labels": [lab for lab in labels if lab not in found],
        "missed_facts": [r["desc"] for r in fact_results if not r["ok"]],
        "words": len(text.split()),
    }


# LLM judge -----------------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are grading a description of a scientific figure that will be read aloud to a blind researcher.
You are given the figure image, a ground-truth answer key generated from the plotting code, and the \
candidate description. Use the image and the answer key together to verify every claim.

Answer key (JSON):
{gt}

Candidate description:
\"\"\"{desc}\"\"\"

Grade on these scales:
- accuracy (0-5): are the stated facts and numbers correct? 5 = no errors; deduct for each wrong value, \
wrong label, wrong ranking or wrong trend.
- completeness (0-5): does it convey chart type, axes, all series/panels, key values and the main \
pattern? 5 = a blind reader could answer most questions a sighted reader could.
- listenability (0-5): is it clear and well-ordered when HEARD (overview first, no markdown or \
symbol soup, not padded, not too terse)?
- hallucinations: list every specific claim that is false or not supported by the image (empty list if none).

Return only JSON: {{"accuracy": int, "completeness": int, "listenability": int, \
"hallucinations": [str], "rationale": "one or two sentences"}}
"""


def _gt_for_judge(gt: dict) -> dict:
    keep = ("chart_type", "title", "x_label", "y_label", "series")
    return {k: gt[k] for k in keep} | {"facts": [f["desc"] for f in gt["facts"]]}


def judge(gt: dict, desc: str, client, model: str = "gpt-6.2-sol") -> dict:
    b64 = base64.b64encode((HERE / gt["file"]).read_bytes()).decode()
    r = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": JUDGE_PROMPT.format(gt=json.dumps(_gt_for_judge(gt), indent=1), desc=desc)},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}", "detail": "high"},
        ]}],
        text={"format": {"type": "json_object"}},
    )
    out = json.loads(r.output_text)
    out["n_hallucinations"] = len(out.get("hallucinations", []))
    return out
