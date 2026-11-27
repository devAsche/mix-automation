"""Frames + OCR from the MIX tutorial videos (R1 step 2). Cache only, never committed.

    python frames_ocr.py scenes [IDs]   # ffmpeg scene-change times (keyframes)  -> scenes/<ID>.json
    python frames_ocr.py pick   [IDs]   # keyword-in-transcript + scene times, <=30 frames -> frames/<ID>/*.jpg
    python frames_ocr.py ocr    [IDs]   # EasyOCR ja+en (needs PYTHONPATH=C:\\MIX_automation_ext\\ocr\\pylibs) -> ocr/<ID>.json

`ocr` keeps only text boxes that look like settings (numbers with units, parameter or plugin words)
and prints a per-video count.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "vocalmix_instructions"
CACHE = REPO / "data" / "processed" / "mix_knowledge"
MAX_FRAMES = 30
KW_FRAMES = 20
MIN_FRAMES = 15

KEYWORDS = re.compile(
    r"\d+(\.\d+)?\s*(dB|デシベル|Hz|ヘルツ|kHz|キロ|ms|ミリ|%|パーセント|対)|"
    r"レシオ|ratio|アタック|リリース|スレッショ|スレショ|センド|プリディレイ|ディケイ|"
    r"ゲイン|ニー|ルックアヘッド|ドライ|ウェット|ミックス量|"
    r"Pro-?Q|Pro-?C|Pro-?R|Pro-?DS|FabFilter|ファブフィルター|Waves|ウェーブス|Nectar|ネクター|Ozone|オゾン|"
    r"Neutron|ニュートロン|Soothe|スーズ|Valhalla|バルハラ|CLA|R-?Vox|Renaissance|ルネッサンス|"
    r"1176|LA-?2A|Saturn|サターン|Decapitator|EchoBoy|エコーボーイ|H-?Delay|Doubler|ダブラー|"
    r"Melodyne|メロダイン|Auto-?Tune|オートチューン|RX|Trash|Kotelnikov|OTT|Vocal ?Rider|ボーカルライダー",
    re.I,
)
OCR_KEEP = re.compile(
    r"-?\d+(\.\d+)?\s*(dB|Hz|kHz|ms|s|%|:1)\b|"
    r"ratio|attack|release|thresh|knee|gain|mix|dry|wet|decay|pre-?delay|size|width|feedback|drive|"
    r"lookahead|range|freq|output|input|send|"
    r"pro-?[qcrdsl]|fabfilter|waves|nectar|ozone|neutron|soothe|valhalla|izotope|rvox|r-vox|renaissance|"
    r"1176|la-?2a|saturn|decapitator|echoboy|delay|reverb|doubler|comp|eq|de-?ess|limiter|"
    r"レシオ|アタック|リリース|スレッショ|ゲイン|センド|リバーブ|ディレイ|コンプ|ディエッサー",
    re.I,
)


def ids_from(argv: list[str]) -> list[str]:
    return argv or sorted(p.stem for p in SRC.glob("*.mp4"))


def scenes(vid: str) -> list[float]:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-skip_frame", "nokey", "-i", str(SRC / f"{vid}.mp4"),
         "-vf", "scale=320:-2,select='gt(scene,0.3)',showinfo", "-an", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return [round(float(m), 2) for m in re.findall(r"pts_time:([\d.]+)", p.stderr)]


def dedupe(ts: list[float], gap: float) -> list[float]:
    out: list[float] = []
    for t in sorted(ts):
        if not out or t - out[-1] >= gap:
            out.append(t)
    return out


def thin(ts: list[float], n: int) -> list[float]:
    if len(ts) <= n:
        return ts
    step = len(ts) / n
    return [ts[int(i * step)] for i in range(n)]


def pick(vid: str) -> list[dict]:
    tr = CACHE / "transcripts" / f"{vid}.json"
    kw: list[float] = []
    if tr.exists():
        for r in json.loads(tr.read_text(encoding="utf-8")):
            if KEYWORDS.search(r["text"]):
                kw.append(r["t0"] + 1.5)  # the screen usually shows the value just after it is said
    kw = thin(dedupe(kw, 8.0), KW_FRAMES)
    sc = json.loads((CACHE / "scenes" / f"{vid}.json").read_text())
    sc = [t + 1.0 for t in sc if all(abs(t - k) > 5 for k in kw)]
    sc = thin(dedupe(sc, 5.0), MAX_FRAMES - len(kw))
    chosen = [{"t": round(t, 1), "why": "kw"} for t in kw] + [{"t": round(t, 1), "why": "scene"} for t in sc]
    if len(chosen) < MIN_FRAMES:  # static screen recordings: top up with evenly spaced frames
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(SRC / f"{vid}.mp4")], capture_output=True, text=True).stdout)
        have = [c["t"] for c in chosen]
        n = MIN_FRAMES - len(chosen)
        for i in range(n * 2):
            t = dur * (i + 0.5) / (n * 2)
            if len(chosen) < MIN_FRAMES and all(abs(t - h) > dur / (MIN_FRAMES * 2) for h in have):
                chosen.append({"t": round(t, 1), "why": "even"})
                have.append(t)
    chosen.sort(key=lambda d: d["t"])
    d = CACHE / "frames" / vid
    d.mkdir(parents=True, exist_ok=True)
    for c in chosen:
        f = d / f"{int(c['t']):05d}_{c['why']}.jpg"
        c["file"] = f.name
        if not f.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-ss", str(c["t"]), "-i", str(SRC / f"{vid}.mp4"),
                            "-frames:v", "1", "-vf", "scale=1600:-2", "-q:v", "3", str(f)], check=False)
    (d / "frames.json").write_text(json.dumps(chosen, indent=0), encoding="utf-8")
    return chosen


def ocr(vid: str, reader) -> dict:
    d = CACHE / "frames" / vid
    frames = json.loads((d / "frames.json").read_text(encoding="utf-8"))
    out = []
    for fr in frames:
        res = reader.readtext(str(d / fr["file"]), detail=1, paragraph=False)
        texts = [t for _, t, c in res if c > 0.35]
        keep = [t for t in texts if OCR_KEEP.search(t)]
        out.append({**fr, "n_text": len(texts), "settings": keep})
    (CACHE / "ocr").mkdir(exist_ok=True)
    (CACHE / "ocr" / f"{vid}.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    return {"frames": len(out), "with_settings": sum(1 for o in out if o["settings"]),
            "setting_boxes": sum(len(o["settings"]) for o in out)}


def main() -> None:
    cmd, ids = sys.argv[1], ids_from(sys.argv[2:])
    if cmd == "scenes":
        (CACHE / "scenes").mkdir(parents=True, exist_ok=True)
        for v in ids:
            f = CACHE / "scenes" / f"{v}.json"
            if not f.exists():
                f.write_text(json.dumps(scenes(v)))
            print(v, "scenes", len(json.loads(f.read_text())), flush=True)
    elif cmd == "pick":
        for v in ids:
            ch = pick(v)
            print(v, "frames", len(ch), "kw", sum(c["why"] == "kw" for c in ch), flush=True)
    elif cmd == "ocr":
        import easyocr
        import torch  # noqa: F401

        reader = easyocr.Reader(["ja", "en"], gpu=True, verbose=False,
                                model_storage_directory=r"C:\MIX_automation_ext\ocr\models")
        for v in ids:
            print(v, ocr(v, reader), flush=True)


if __name__ == "__main__":
    main()
