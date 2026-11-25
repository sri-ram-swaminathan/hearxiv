"""On-device text-to-speech with Gradium's Phonon (phonon-31m-en, 31M params, Apache-2.0).

  pip install ptts        (or: uv sync --extra tts)

Phonon takes graphemes (plain text), not phonemes, so pronunciation is controlled upstream by
`hearxiv.narrate.spoken.to_spoken` (lexicon respellings). Its built-in normaliser (lang="en")
then handles numbers, dates and URLs. Streaming generation stops as soon as the `with` block
exits, which is what makes barge-in (interrupting the narrator) cheap.

Browser equivalent: `npm install phonon-tts` (WASM), same model, for a future in-extension build.
"""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "hf://gradium/phonon-31m-en"
SAMPLE_RATE = 24_000


class PhononTTS:
    def __init__(self, model: str = DEFAULT_MODEL, voice: str | None = None, quant: str = "q8"):
        import ptts

        self.tts = ptts.TTS(model, lang="en", quant=quant)
        self.voice = voice

    def _kw(self):
        return {"voice": self.voice} if self.voice else {}

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 PCM chunks at 24 kHz. Breaking out of the loop stops synthesis."""
        with self.tts.stream(text, **self._kw()) as audio:
            yield from audio

    def synthesize(self, text: str) -> np.ndarray:
        return np.concatenate(list(self.stream(text))) if text.strip() else np.zeros(0, np.float32)

    def save(self, text: str, path: str | Path) -> Path:
        self.tts.save(str(path), text, **self._kw())
        return Path(path)


def write_wav(path, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Write 16-bit mono WAV to a path or a binary file-like object."""
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    target = str(path) if isinstance(path, (str, Path)) else path
    with wave.open(target, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return path


def earcon(kind: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Short non-speech cues so structure is audible: rising chime = new section, two soft
    blips = figure, low tone = table. Distinct and brief per screen-reader UX guidance."""
    def tone(freq, dur, vol=0.25):
        t = np.linspace(0, dur, int(sample_rate * dur), endpoint=False)
        env = np.minimum(1, np.minimum(t, dur - t) * 40)
        return (vol * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    gap = np.zeros(int(sample_rate * 0.05), np.float32)
    if kind == "section":
        return np.concatenate([tone(660, 0.08), tone(880, 0.12), gap])
    if kind == "figure":
        return np.concatenate([tone(1200, 0.05), gap, tone(1200, 0.05), gap])
    if kind == "table":
        return np.concatenate([tone(440, 0.15), gap])
    return np.zeros(0, np.float32)
