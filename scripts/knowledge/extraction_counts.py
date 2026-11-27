"""Per-video extraction counts for R1 (transcript segments, frames, OCR setting boxes, note rows).

    .venv\\Scripts\\python.exe scripts\\knowledge\\extraction_counts.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
C = REPO / "data" / "processed" / "mix_knowledge"


def main() -> None:
    print("| ID | 長さ | 発話 | フレーム | OCR 設定値の枠 | ノート処理行 | 数値付き行 |")
    print("|---|---|---|---|---|---|---|")
    for mp4 in sorted((REPO / "vocalmix_instructions").glob("*.mp4")):
        v = mp4.stem
        tr = C / "transcripts" / f"{v}.json"
        segs = json.loads(tr.read_text(encoding="utf-8")) if tr.exists() else []
        dur = segs[-1]["t1"] if segs else 0
        ocr = C / "ocr" / f"{v}.json"
        o = json.loads(ocr.read_text(encoding="utf-8")) if ocr.exists() else []
        note = C / "notes" / f"{v}.md"
        rows = num = 0
        if note.exists():
            for ln in note.read_text(encoding="utf-8").splitlines():
                if re.match(r"\|\s*\d+[a-z]?\s*\|", ln):
                    rows += 1
                    if re.search(r"\d+(\.\d+)?\s*(dB|Hz|kHz|k\b|ms|%|:1|s\b|LUFS)", ln.split("|")[5] if ln.count("|") > 6 else ln):
                        num += 1
        print(f"| {v} | {int(dur // 60)}分 | {len(segs)} | {len(o)} | {sum(len(x['settings']) for x in o)} | {rows} | {num} |")


if __name__ == "__main__":
    main()
