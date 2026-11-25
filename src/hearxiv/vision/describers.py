"""Figure describers: one interface, many backends (cloud APIs and local open models).

    d = make_describer("openai:gpt-6.1-terra")
    d = make_describer("gemini:gemini-3.5-flash-lite")
    d = make_describer("hf:Qwen/Qwen3.5-2B")
    out = d.describe("fig.png", prompt)

Each call returns a `Description` with the text, wall-clock seconds and token counts so that
benchmarks can compare quality, speed and cost on equal terms.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image


@dataclass
class Description:
    text: str
    seconds: float
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)


class FigureDescriber(Protocol):
    name: str

    def describe(self, image_path: str | Path, prompt: str) -> Description: ...


def _load_image(path: str | Path, max_side: int | None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _retry(fn, tries: int = 5, base_delay: float = 10.0):
    """Retry on rate limits / transient errors with exponential backoff."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - SDKs raise many different error types
            msg = str(e).lower()
            transient = any(k in msg for k in ("429", "rate", "quota", "503", "500", "timeout", "unavailable", "overloaded"))
            if not transient or attempt == tries - 1:
                raise
            time.sleep(base_delay * (2**attempt))


# Cloud backends -------------------------------------------------------------------------------


class OpenAIDescriber:
    """OpenAI Responses API. Also works for any OpenAI-compatible server via base_url."""

    def __init__(self, model: str, detail: str = "high", max_side: int | None = 2048,
                 base_url: str | None = None, api_key: str | None = None, max_output_tokens: int = 2000):
        from openai import OpenAI

        self.model, self.detail, self.max_side = model, detail, max_side
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.name = f"openai:{model}"

    def describe(self, image_path, prompt):
        b64 = base64.b64encode(_png_bytes(_load_image(image_path, self.max_side))).decode()
        t0 = time.perf_counter()
        r = _retry(lambda: self.client.responses.create(
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}", "detail": self.detail},
            ]}],
        ))
        u = getattr(r, "usage", None)
        return Description(text=r.output_text.strip(), seconds=time.perf_counter() - t0, model=self.name,
                           input_tokens=getattr(u, "input_tokens", None),
                           output_tokens=getattr(u, "output_tokens", None))


class GeminiDescriber:
    """Google Gemini via google-genai. Free tier is rate limited, so we pace requests."""

    def __init__(self, model: str, max_side: int | None = 2048, min_interval_s: float = 13.0,
                 api_key: str | None = None):
        from google import genai

        self.model, self.max_side, self.min_interval_s = model, max_side, min_interval_s
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.name = f"gemini:{model}"
        self._last = 0.0

    def describe(self, image_path, prompt):
        from google.genai import types

        wait = self.min_interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        data = _png_bytes(_load_image(image_path, self.max_side))
        t0 = time.perf_counter()
        r = _retry(lambda: self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=data, mime_type="image/png"), prompt],
        ), base_delay=30.0)
        self._last = time.time()
        u = getattr(r, "usage_metadata", None)
        return Description(text=(r.text or "").strip(), seconds=time.perf_counter() - t0, model=self.name,
                           input_tokens=getattr(u, "prompt_token_count", None),
                           output_tokens=getattr(u, "candidates_token_count", None))


# Local backend --------------------------------------------------------------------------------


class HFLocalDescriber:
    """Any Hugging Face image-text-to-text model that ships a chat template (Qwen-VL, Gemma,
    SmolVLM, Granite Vision, Granite-Docling, ...). Runs fully offline after the first download.

    `max_side` downscales the image before encoding: on a laptop CPU the number of vision tokens
    dominates latency, so 768-1024 px is a good trade-off for charts.
    """

    def __init__(self, repo_id: str, max_new_tokens: int = 400, max_side: int | None = 1024,
                 dtype: str | None = None, device: str | None = None, chat_kwargs: dict | None = None,
                 prompt_override: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.repo_id, self.max_new_tokens, self.max_side = repo_id, max_new_tokens, max_side
        self.chat_kwargs = chat_kwargs or {}
        self.prompt_override = prompt_override
        self.name = f"hf:{repo_id}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bfloat16 is emulated (very slow) on CPUs without native bf16 support (e.g. Zen 1/2,
        # most laptops). float32 doubles memory but is far faster there.
        dtype = dtype or ("bfloat16" if self.device == "cuda" else "float32")
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            repo_id, dtype=getattr(torch, dtype)).to(self.device).eval()

    def describe(self, image_path, prompt):
        import torch

        img = _load_image(image_path, self.max_side)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": self.prompt_override or prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            **self.chat_kwargs,
        ).to(self.device)
        n_in = inputs["input_ids"].shape[-1]
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        dt = time.perf_counter() - t0
        new = out[0, n_in:]
        text = self.processor.decode(new, skip_special_tokens=True).strip()
        return Description(text=text, seconds=dt, model=self.name, input_tokens=n_in, output_tokens=len(new))


def make_describer(spec: str, **kw) -> FigureDescriber:
    """Build a describer from 'provider:model', e.g. 'openai:gpt-6.1-terra', 'hf:Qwen/Qwen3.5-2B'."""
    provider, _, model = spec.partition(":")
    if provider == "openai":
        return OpenAIDescriber(model, **kw)
    if provider == "gemini":
        return GeminiDescriber(model, **kw)
    if provider == "hf":
        return HFLocalDescriber(model, **kw)
    if provider == "local":  # any OpenAI-compatible local server (llama.cpp, Ollama, LM Studio, vLLM)
        return OpenAIDescriber(model, base_url=kw.pop("base_url", "http://localhost:8080/v1"),
                               api_key=kw.pop("api_key", "none"), **kw)
    raise ValueError(f"unknown provider in {spec!r}")
