"""One tiny LLM interface for every provider.

Everything speaks the OpenAI Chat Completions protocol, so swapping providers is a config change:

    HEARXIV_LLM=openai:gpt-6.1-terra            (OpenAI)
    HEARXIV_LLM=gemini:gemini-3.5-flash-lite    (Google, free tier, via its OpenAI-compatible endpoint)
    HEARXIV_LLM=local:qwen3.5-4b                (llama.cpp / Ollama / LM Studio on localhost)

No agent framework on purpose: the pipeline is a fixed sequence of steps, and plain SDK calls
keep the project easy to read, debug and fork.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import OpenAI

ENDPOINTS = {
    "openai": (None, "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "local": (os.environ.get("HEARXIV_LOCAL_URL", "http://localhost:8080/v1"), None),
    "ollama": ("http://localhost:11434/v1", None),
}
DEFAULT_SPEC = "openai:gpt-6.1-terra"


@dataclass
class LLM:
    spec: str = DEFAULT_SPEC

    def __post_init__(self):
        provider, _, self.model = self.spec.partition(":")
        if provider not in ENDPOINTS:
            raise ValueError(f"unknown LLM provider {provider!r}; use one of {list(ENDPOINTS)}")
        base_url, key_env = ENDPOINTS[provider]
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(key_env, "none") if key_env else "none")

    @classmethod
    def from_env(cls) -> LLM:
        return cls(os.environ.get("HEARXIV_LLM", DEFAULT_SPEC))

    def chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 4000) -> str:
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=max_tokens,
            **kw,
        )
        return r.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, max_tokens: int = 4000) -> dict:
        txt = self.chat(system, user, json_mode=True, max_tokens=max_tokens)
        txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        return json.loads(txt)
