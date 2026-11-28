"""R5b: automatic correction EQ (v8.auto_eq) vs Ryo's Persistent Q settings, cross-check of thresholds.

    python scripts/analysis/auto_eq_check.py

For エバ and L&D: the auto EQ from the R5 lead_dry (the voice Ryo set his EQ on), with the broad high band
from the gap of the pre-tone lead vs the raw backing. Prints the bands, the dB RMS of the response
difference over 200 Hz-16 kHz (static RBJ, Ryo's muted bands left out, both curves with their mean over
200 Hz-16 kHz removed and also as is), and which of Ryo's cuts (Q >= 2, static) have an auto cut
within 1/6 oct. For Fall in Love only the gap and the broad band (no R5 render).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SR, print_table  # noqa: E402
from pultec_ceq_response import static_db  # noqa: E402

SONGS = {"エバ": ("エバ_v8", "ryo_エバ_20261128.json"), "ラブ&デストロイ": ("ラブandデストロイ_v8", "ryo_LD_20261128.json"),
         "Fall in Love": (None, None)}
MATCH_OCT = 1 / 6


def pre_tone(song: str) -> tuple[np.ndarray, np.ndarray]:
    from mix_automation.io.audio_loader import consolidate_vocal_stem_folder, load_audio
    from mix_automation.pipeline import discover_song_assets

    a = discover_song_assets(ROOT / "data" / "raw" / song)
    st = consolidate_vocal_stem_folder(list(a["stems"]), target_sr=SR)
    inst, _ = load_audio(Path(a["inst"]), target_sr=SR, mono=True)
    return np.asarray(st.lead_mono, np.float32), np.asarray(inst, np.float32)


def main() -> int:
    import soundfile as sf

    from mix_automation.v8 import auto_eq
    from mix_automation.v8 import correction_eq as ceq

    f = np.geomspace(200, 16000, 400)
    rows, bands_rows = [], []
    for song, (render, ryo_file) in SONGS.items():
        lead, inst = pre_tone(song)
        gap = auto_eq.high_gap_db(lead, inst)
        hb = auto_eq.broad_high_band(gap)
        if render is None:
            rows.append([song, f"{gap:+.2f}", f"{hb['net_target_db']:+.2f}", f"{hb['gain_db']:+.2f}", "–", "–", "–", "–"])
            continue
        x, sr = sf.read(str(ROOT / "data/outputs_v8r5/render" / render / "stems/lead_dry.wav"), dtype="float32")
        cfg = auto_eq.auto_profile(x, sr, gap_db=gap)
        ryo = ceq.load(ceq.PROFILE_DIR / ryo_file)
        ra, rr = static_db(cfg, f), static_db(ryo, f)
        rms = float(np.sqrt(np.mean((ra - rr) ** 2)))
        rms_c = float(np.sqrt(np.mean(((ra - ra.mean()) - (rr - rr.mean())) ** 2)))
        auto_hz = [b["frequency_hz"] for b in cfg["bands"]]
        ryo_cuts = [b["frequency_hz"] for b in ryo["bands"] if b["enabled"] and b["gain_db"] < 0 and b["q"] >= 2
                    and not (b.get("extra") or {}).get("mute") and not (b.get("extra") or {}).get("spectral")
                    and not (b.get("extra") or {}).get("dynamic")]
        hit = [h for h in ryo_cuts if any(abs(np.log2(h / a)) <= MATCH_OCT for a in auto_hz)]
        extra = [a for a in auto_hz if a != auto_eq.BROAD_HIGH_HZ and not any(abs(np.log2(h / a)) <= MATCH_OCT for h in ryo_cuts)]
        rows.append([song, f"{gap:+.2f}", f"{hb['net_target_db']:+.2f}", f"{hb['gain_db']:+.2f}", f"{rms:.2f}",
                     f"{rms_c:.2f}", f"{len(hit)}/{len(ryo_cuts)}", " ".join(f"{a:.0f}" for a in extra) or "–"])
        for p in cfg["source"]["peaks"]:
            bands_rows.append([song, str(p["range"]), f"{p['hz']:.0f}", "–" if p["prom_db"] is None else f"{p['prom_db']:.2f}",
                               f"{p['gain_db']:+.2f}", f"{p['q']}"])
        bands_rows.append([song, "ryo cuts (Q>=2)", " ".join(f"{h:.0f}" for h in sorted(ryo_cuts)), "", "", ""])
    print_table(["song", "range", "Hz", "prom dB", "gain", "Q"], bands_rows)
    print_table(["song", "gap 6-16k", "net target", "broad 8.3k", "dB RMS", "dB RMS (mean off)", "ryo cuts hit",
                 "auto cuts not in ryo"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
