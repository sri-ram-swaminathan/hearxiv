"""hearxiv command line.

  hearxiv ingest 1706.03762 [--describer openai:gpt-6.1-terra | gemini:... | hf:Qwen/Qwen3.5-2B | none]
  hearxiv script 1706.03762 --mode full|5min|10min
  hearxiv speak  1706.03762 --mode 5min -o out.wav      (needs the `tts` extra)
  hearxiv ask    1706.03762 "What dataset did they use?"

Paper artefacts are cached under data/papers/<id>/ (source.html, paper.json, figures/, scripts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from hearxiv.ingest.fetch import CACHE_DIR, fetch_html, parse_arxiv_id
from hearxiv.schema import PaperDoc


def paper_dir(arxiv_id: str, version: str | None) -> Path:
    return CACHE_DIR / f"{arxiv_id}{version or ''}".replace("/", "_")


def load_or_ingest(ref: str, describer: str = "none", equations: bool = False, log=print) -> PaperDoc:
    from hearxiv.ingest.arxiv_html import parse_arxiv_html

    arxiv_id, version = parse_arxiv_id(ref)
    pj = paper_dir(arxiv_id, version) / "paper.json"
    if pj.exists():
        doc = PaperDoc.model_validate_json(pj.read_text(encoding="utf-8"))
    else:
        html, url, source = fetch_html(arxiv_id, version)
        doc = parse_arxiv_html(html, arxiv_id, url, source, version)
        log(f"parsed '{doc.title}': {len(list(doc.iter_sections()))} sections, {len(doc.figures)} figures, "
            f"{len(doc.tables)} tables, {len(doc.equations)} equations")
    if describer != "none":
        from hearxiv.ingest.enrich import describe_figures

        doc = describe_figures(doc, describer, log=log)
    if equations:
        from hearxiv.ingest.enrich import verbalize_equations
        from hearxiv.llm import LLM

        doc = verbalize_equations(doc, LLM.from_env())
    pj.parent.mkdir(parents=True, exist_ok=True)
    pj.write_text(doc.model_dump_json(indent=1), encoding="utf-8")
    return doc


def main(argv=None):
    load_dotenv()
    p = argparse.ArgumentParser(prog="hearxiv", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest", help="parse a paper and describe its figures (once)")
    pi.add_argument("paper")
    pi.add_argument("--describer", default="none")
    pi.add_argument("--equations", action="store_true", help="LLM meaning-level verbalisation of long equations")
    ps = sub.add_parser("script", help="build a spoken script")
    ps.add_argument("paper")
    ps.add_argument("--mode", default="full")
    pk = sub.add_parser("speak", help="render a script to a WAV file with Phonon")
    pk.add_argument("paper")
    pk.add_argument("--mode", default="5min")
    pk.add_argument("-o", "--out", default=None)
    pk.add_argument("--max-segments", type=int, default=None)
    pk.add_argument("--regenerate", action="store_true", help="rebuild the script instead of reusing the cached one")
    pa = sub.add_parser("ask", help="ask a question about a paper")
    pa.add_argument("paper")
    pa.add_argument("question")
    a = p.parse_args(argv)

    if a.cmd == "ingest":
        load_or_ingest(a.paper, a.describer, a.equations)
        return
    doc = load_or_ingest(a.paper)
    if a.cmd in {"script", "speak"}:
        from hearxiv.narrate.script import Script, build_script

        d = paper_dir(doc.arxiv_id, doc.version)
        cached = d / f"script_{a.mode}.json"
        if a.cmd == "speak" and cached.exists() and not a.regenerate:
            script = Script.model_validate_json(cached.read_text(encoding="utf-8"))
        else:
            script = build_script(doc, a.mode)
            cached.write_text(json.dumps(script.model_dump(), indent=1, ensure_ascii=False), encoding="utf-8")
        if a.cmd == "script":
            for s in script.segments:
                print(f"[{s.kind}] {s.text}\n")
            print(f"-- {len(script.segments)} segments, {script.words} words, ~{script.minutes():.1f} min at 1x",
                  file=sys.stderr)
            return
        import numpy as np

        from hearxiv.tts.phonon import PhononTTS, earcon, write_wav

        tts = PhononTTS()
        pcm = []
        for s in script.segments[: a.max_segments]:
            if s.earcon:
                pcm.append(earcon(s.earcon))
            pcm.append(tts.synthesize(s.text))
            pcm.append(np.zeros(int(24000 * 0.35), np.float32))
        out = Path(a.out or d / f"{a.mode}.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out, np.concatenate(pcm))
        print(f"wrote {out}")
    elif a.cmd == "ask":
        from hearxiv.agent.qa import answer

        print(answer(doc, a.question).text)


if __name__ == "__main__":
    main()
