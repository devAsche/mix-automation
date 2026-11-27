"""Which cover does each Amo_* practice main take belong to? (R1 step 0)

Key-invariant chroma cross-correlation of each song's practice main (post-correction if any)
against every cover. Prints the best-matching cover per practice folder and the score margin.

    .venv\\Scripts\\python.exe scripts\\knowledge\\match_practice_cover.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "utattemita_practicesite_reference"
SR = 11025
HOP = 2048  # ~0.19 s


def load(p: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def chroma(y: np.ndarray) -> np.ndarray:
    c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    c = c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)
    return c


def score(a: np.ndarray, b: np.ndarray) -> float:
    """max over 12 key shifts and time lag of mean cosine (frames where `a` is voiced)."""
    voiced = a.sum(0) > 0.5
    best = 0.0
    n = a.shape[1]
    for k in range(12):
        ar = np.roll(a, k, axis=0)
        # correlate per chroma row, sum
        corr = sum(np.correlate(b[i], ar[i] * voiced, mode="full") for i in range(12))
        best = max(best, float(corr.max()) / max(voiced.sum(), 1))
    return best if n else 0.0


def main_take(song: Path) -> Path | None:
    wavs = sorted((song / "practice").rglob("*.wav"))
    mains = [w for w in wavs if re.search(r"メイン|main|^m\d", w.stem, re.I)]
    post = [w for w in mains if re.search(r"補正(後|語|有|あり|済)|修正(後|有|あり)", str(w))]
    pick = post or mains or wavs
    return pick[0] if pick else None


def main() -> None:
    songs = sorted(ROOT.glob("Amo_*"))
    names = [s.name.removeprefix("Amo_") for s in songs]
    covers = [chroma(load(next(s.glob("cover_*")))) for s in songs]
    rows, keep = [], []
    for s in songs:
        m = main_take(s)
        if m is None:
            continue
        c = chroma(load(m))
        rows.append([score(c, cv) for cv in covers])
        keep.append(s.name.removeprefix("Amo_"))
    mat = np.array(rows)
    # remove "this cover matches everything" and "this take matches everything" biases
    z = mat - mat.mean(0, keepdims=True)
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-9)
    for i, n in enumerate(keep):
        order = np.argsort(-z[i])
        b, b2 = order[0], order[1]
        own = z[i, names.index(n)]
        flag = "" if names[b] == n else "  <-- MISMATCH"
        print(f"{n:<16} best {names[b]:<16} z {z[i, b]:+.2f}  2nd {names[b2]:<16} z {z[i, b2]:+.2f}  own z {own:+.2f}{flag}")


if __name__ == "__main__":
    main()
