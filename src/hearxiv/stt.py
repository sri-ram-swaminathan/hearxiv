"""On-device speech-to-text with Moonshine (UsefulSensors, MIT) via transformers.

Moonshine is built for short, live utterances (commands, questions): its compute scales with the
length of the audio, unlike Whisper's fixed 30-second window. Base is about 60M params; tiny is
about 27M. The browser sends 16 kHz mono float32 PCM, so no audio decoding library is needed.
"""

from __future__ import annotations

import threading

import numpy as np

DEFAULT_STT = "UsefulSensors/moonshine-base"


class STT:
    def __init__(self, model: str = DEFAULT_STT):
        from transformers import pipeline

        self.asr = pipeline("automatic-speech-recognition", model=model, device="cpu")
        self.lock = threading.Lock()

    def transcribe(self, pcm16k: np.ndarray) -> str:
        if pcm16k.size < 1600:  # < 0.1 s: nothing said
            return ""
        peak = float(np.abs(pcm16k).max())
        if peak > 0:
            pcm16k = pcm16k / peak * 0.9  # normalise quiet microphones
        with self.lock:
            out = self.asr({"raw": pcm16k.astype(np.float32), "sampling_rate": 16000})
        return out["text"].strip()
