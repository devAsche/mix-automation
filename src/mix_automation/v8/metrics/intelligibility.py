"""Lyric intelligibility of the vocal inside the mix (Phase 2b, for "聞こえない" issues).

Two measures, both against the time-aligned pre-processing vocal as the clean reference:
- ESTOI (Jensen & Taal 2016, IEEE/ACM TASLP 24(11)) via pystoi (MIT): reference = pre vocal,
  degraded = master, per phrase.
- ASR character error rate: the pre vocal and the master are transcribed with Whisper large-v3
  (OpenAI, MIT; run with faster-whisper / CTranslate2, MIT); CER of the master transcript against
  the pre-vocal transcript, per chunk. Same idea as the lyric-intelligibility task of the Cadenza
  challenge (ICASSP 2026 CLIP / LIWhiz baseline): an ASR stands in for the listener.
Japanese text is compared as hiragana (pykakasi) so kanji/kana spelling differences do not count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mix_automation.v8.metrics._core import Segment, to_mono

ESTOI_MIN_S = 0.4  # pystoi needs >= 30 frames of 384 ms analysis at 10 kHz -> skip shorter phrases


def estoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    from pystoi import stoi

    r, d = to_mono(ref), to_mono(deg)
    n = min(len(r), len(d))
    if n < int(ESTOI_MIN_S * sr):
        return float("nan")
    try:
        return float(stoi(r[:n], d[:n], sr, extended=True))
    except Exception:  # noqa: BLE001 - pystoi raises on near-silent input; report NaN
        return float("nan")


def estoi_segments(ref: np.ndarray, deg: np.ndarray, sr: int, segs: list[Segment]) -> list[tuple[float, float, float]]:
    r, d = to_mono(ref), to_mono(deg)
    out = []
    for s in segs:
        i0, i1 = int(s.start * sr), int(s.end * sr)
        out.append((s.start, s.end, estoi(r[i0:i1], d[i0:i1], sr)))
    return out


_KKS = None


def to_hiragana(text: str) -> str:
    global _KKS
    import pykakasi

    if _KKS is None:
        _KKS = pykakasi.kakasi()
    s = "".join(item["hira"] for item in _KKS.convert(text))
    return "".join(ch for ch in s if ch.isalnum())


def cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein distance / len(ref)) on hiragana-normalised text."""
    a, b = to_hiragana(ref), to_hiragana(hyp)
    if not a:
        return float("nan")
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


@dataclass
class Word:
    start: float
    end: float
    text: str


class Transcriber:
    """Whisper large-v3 through faster-whisper. Lazily loads the model (GPU if available)."""

    def __init__(self, model: str = "large-v3", device: str | None = None):
        from faster_whisper import WhisperModel

        try:
            import torch

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            dev = device or "cpu"
        self.model = WhisperModel(model, device=dev, compute_type="float16" if dev == "cuda" else "int8")

    def transcribe(self, x: np.ndarray, sr: int, words: bool = False, prompt: str | None = None) -> tuple[str, list[Word]]:
        import librosa

        y = librosa.resample(to_mono(x).astype(np.float32), orig_sr=sr, target_sr=16000)
        segs, _ = self.model.transcribe(
            y, language="ja", beam_size=5, condition_on_previous_text=False, vad_filter=False,
            word_timestamps=words, initial_prompt=prompt, temperature=0.0,
        )
        text, ws = [], []
        for s in segs:
            text.append(s.text)
            if words and s.words:
                ws += [Word(float(w.start), float(w.end), w.word) for w in s.words]
        return "".join(text), ws


def chunk_segments(segs: list[Segment], max_s: float = 12.0, max_gap_s: float = 1.5) -> list[Segment]:
    """Group phrases into ASR chunks of at most max_s, breaking at gaps longer than max_gap_s."""
    out: list[Segment] = []
    cur: Segment | None = None
    for s in segs:
        if cur is None:
            cur = s
        elif s.start - cur.end <= max_gap_s and s.end - cur.start <= max_s:
            cur = Segment(cur.start, s.end)
        else:
            out.append(cur)
            cur = s
    if cur is not None:
        out.append(cur)
    return out
