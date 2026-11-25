"""LLM-as-judge over all benchmark results, cached in results/judgments.jsonl.

    uv run python benchmarks/figure_ocr/judge_results.py [--judge gpt-6.2-sol]

A judgment is keyed by (model, figure, sha1(description)), so re-running only judges new outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from scoring import judge

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CACHE = RESULTS / "judgments.jsonl"


def key(model: str, fig: str, text: str) -> str:
    return f"{model}|{fig}|{hashlib.sha1(text.encode()).hexdigest()[:12]}"


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    return {json.loads(line)["key"]: json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines()}


def latest_results() -> list[dict]:
    """Last successful record per (model, figure) across all result files."""
    out: dict[tuple[str, str], dict] = {}
    for f in sorted(RESULTS.glob("*.jsonl")):
        if f.name == CACHE.name:
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("text") is not None and not r.get("error"):
                out[(r["model"], r["figure_id"])] = r
    return list(out.values())


def main(argv=None):
    load_dotenv(HERE.parents[1] / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--judge", default="gpt-6.2-sol")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(argv)
    gt = {g["id"]: g for g in json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))}
    cache = load_cache()
    todo = [r for r in latest_results() if key(r["model"], r["figure_id"], r["text"]) not in cache]
    print(f"{len(todo)} descriptions to judge with {a.judge}")
    client = OpenAI()

    def work(r):
        k = key(r["model"], r["figure_id"], r["text"])
        try:
            j = judge(gt[r["figure_id"]], r["text"], client, model=a.judge) if r["text"].strip() else {
                "accuracy": 0, "completeness": 0, "listenability": 0, "hallucinations": [],
                "rationale": "empty output", "n_hallucinations": 0}
        except Exception as e:  # noqa: BLE001
            print("  judge failed:", k, e)
            return None
        return {"key": k, "model": r["model"], "figure_id": r["figure_id"], "judge": a.judge, **j}

    with ThreadPoolExecutor(a.workers) as ex, CACHE.open("a", encoding="utf-8") as f:
        for rec in ex.map(work, todo):
            if rec:
                f.write(json.dumps(rec) + "\n")
                f.flush()
    print("done")


if __name__ == "__main__":
    main()
