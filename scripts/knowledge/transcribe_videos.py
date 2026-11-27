"""Timestamped transcripts of the MIX tutorial videos (R1 step 1).

faster-whisper large-v3 on GPU. Writes per video (cache, never committed):
  data/processed/mix_knowledge/transcripts/<ID>.json  [{"t0", "t1", "text"}]
  data/processed/mix_knowledge/transcripts/<ID>.tsv   "mm:ss<TAB>text"

    .venv\\Scripts\\python.exe scripts\\knowledge\\transcribe_videos.py [IDs...]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "vocalmix_instructions"
OUT = REPO / "data" / "processed" / "mix_knowledge" / "transcripts"

PROMPT = (
    "ボーカルミックスの解説です。EQ、コンプレッサー、ディエッサー、リバーブ、ディレイ、"
    "スレッショルド、レシオ、アタック、リリース、プリディレイ、センド、バス、サチュレーション、"
    "ダブラー、ピッチ補正、オケ、ハモリ、dB、Hz、kHz、ms。"
)


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def load16k(p: Path):
    """Decode with the ffmpeg CLI (the installed PyAV is incompatible with faster-whisper's decoder)."""
    import numpy as np

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def main() -> None:
    import torch  # noqa: F401  (loads cublas/cudnn DLLs shipped with torch)
    from faster_whisper import WhisperModel

    OUT.mkdir(parents=True, exist_ok=True)
    ids = sys.argv[1:] or sorted(p.stem for p in SRC.glob("*.mp4"))
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for vid in ids:
        dst = OUT / f"{vid}.json"
        if dst.exists():
            print("skip", vid, flush=True)
            continue
        segs, info = model.transcribe(
            load16k(SRC / f"{vid}.mp4"), language="ja", vad_filter=True,
            initial_prompt=PROMPT, condition_on_previous_text=False,
        )
        rows = [{"t0": round(s.start, 2), "t1": round(s.end, 2), "text": s.text.strip()} for s in segs if s.text.strip()]
        dst.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
        (OUT / f"{vid}.tsv").write_text("".join(f"{mmss(r['t0'])}\t{r['text']}\n" for r in rows), encoding="utf-8")
        print(f"done {vid} segs={len(rows)} dur={info.duration:.0f}s", flush=True)


if __name__ == "__main__":
    main()
