"""Polite, cached fetching of arXiv content.

arXiv etiquette (https://info.arxiv.org/help/api/tou.html, robots.txt):
  * one request at a time with a delay between requests, never bulk-crawl
  * identify the client with a descriptive User-Agent
  * cache everything by id + version so a paper is downloaded once
hearxiv only fetches what a user explicitly asked to listen to.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import httpx

from hearxiv import __version__

USER_AGENT = f"hearxiv/{__version__} (open-source accessible paper reader; +https://github.com/sri-ram-swaminathan/hearxiv)"
CACHE_DIR = Path("data/papers")
MIN_DELAY_S = 3.0
_last_request = 0.0

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?|([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?")


def parse_arxiv_id(s: str) -> tuple[str, str | None]:
    """Accept '2401.01234', '2401.01234v2', or any arxiv.org / alphaxiv.org URL."""
    m = ARXIV_ID_RE.search(s)
    if not m:
        raise ValueError(f"no arXiv id found in {s!r}")
    if m.group(1):
        return m.group(1), m.group(2)
    return m.group(3), m.group(4)


def _get(url: str, client: httpx.Client) -> httpx.Response:
    global _last_request
    wait = MIN_DELAY_S - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    r = client.get(url, follow_redirects=True, timeout=60)
    _last_request = time.time()
    return r


def fetch_html(arxiv_id: str, version: str | None = None, cache_dir: Path = CACHE_DIR) -> tuple[str, str, str]:
    """Return (html, final_url, source) trying arxiv.org/html first, then ar5iv as a fallback."""
    key = f"{arxiv_id}{version or ''}".replace("/", "_")
    cache = cache_dir / key / "source.html"
    meta = cache_dir / key / "source.url"
    if cache.exists() and meta.exists():
        url, source = meta.read_text(encoding="utf-8").split("\n", 1)
        return cache.read_text(encoding="utf-8"), url, source
    candidates = [
        (f"https://arxiv.org/html/{arxiv_id}{version or ''}", "arxiv-html"),
        (f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}", "ar5iv"),
    ]
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for url, source in candidates:
            r = _get(url, client)
            if r.status_code == 200 and "ltx_document" in r.text:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(r.text, encoding="utf-8")
                # No trailing slash: LaTeXML image srcs look like "2401.01234v2/x1.png", so they must
                # resolve against https://arxiv.org/html/ (urljoin drops the last path segment).
                final = str(r.url).rstrip("/")
                meta.write_text(f"{final}\n{source}", encoding="utf-8")
                return r.text, final, source
    raise LookupError(
        f"No HTML version of arXiv:{arxiv_id} is available. "
        "PDF fallback (Docling) is planned; see docs/roadmap.md."
    )


def fetch_image(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        r = _get(url, client)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest
