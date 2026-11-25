"""Run figure describers over the benchmark suite. Resumable: finished (model, figure) pairs are skipped.

    uv run python benchmarks/figure_ocr/run_benchmark.py openai:gpt-6.1-terra gemini:gemini-3.5-flash-lite
    uv run python benchmarks/figure_ocr/run_benchmark.py hf:Qwen/Qwen3.5-2B --max-side 768

Results are appended to benchmarks/figure_ocr/results/<model-slug>.jsonl (one line per figure).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from hearxiv.vision import build_prompt, make_describer

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", spec)


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if not rec.get("error"):
            out.add(rec["figure_id"])
    return out


def run(spec: str, figure_ids: list[str] | None, describer_kw: dict, tag: str = "") -> None:
    gt = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))
    if figure_ids:
        gt = [g for g in gt if g["id"] in figure_ids]
    out_path = RESULTS / f"{slug(spec + tag)}.jsonl"
    RESULTS.mkdir(exist_ok=True)
    skip = done_ids(out_path)
    todo = [g for g in gt if g["id"] not in skip]
    if not todo:
        print(f"[{spec}{tag}] nothing to do")
        return
    print(f"[{spec}{tag}] {len(todo)} figures to describe", flush=True)
    d = make_describer(spec, **describer_kw)
    prompt = build_prompt()
    for g in todo:
        rec = {"model": spec + tag, "figure_id": g["id"]}
        try:
            o = d.describe(HERE / g["file"], prompt)
            rec |= {"text": o.text, "seconds": round(o.seconds, 2),
                    "input_tokens": o.input_tokens, "output_tokens": o.output_tokens}
            print(f"  {g['id']}: {o.seconds:.1f}s, {o.output_tokens} tok", flush=True)
        except Exception as e:  # keep going; failures are recorded and retried next run
            rec |= {"error": f"{type(e).__name__}: {e}"}
            print(f"  {g['id']}: ERROR {e}", flush=True)
            traceback.print_exc()
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def main(argv=None):
    load_dotenv(HERE.parents[1] / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="provider:model specs")
    p.add_argument("--figures", nargs="*", help="subset of figure ids")
    p.add_argument("--max-side", type=int, help="downscale images to this many pixels on the long side")
    p.add_argument("--max-new-tokens", type=int, help="local models only")
    p.add_argument("--tag", default="", help="suffix for the results file, e.g. '@768px'")
    p.add_argument("--prompt-override", help="local models only: replace the description prompt "
                   "(e.g. 'Convert chart to table.' for Granite-Docling)")
    a = p.parse_args(argv)
    kw = {}
    if a.max_side:
        kw["max_side"] = a.max_side
    if a.max_new_tokens:
        kw["max_new_tokens"] = a.max_new_tokens
    if a.prompt_override:
        kw["prompt_override"] = a.prompt_override
    for spec in a.models:
        try:
            run(spec, a.figures, dict(kw), a.tag)
        except Exception:
            traceback.print_exc()
            print(f"[{spec}] FAILED to initialise, skipping", file=sys.stderr)


if __name__ == "__main__":
    main()
