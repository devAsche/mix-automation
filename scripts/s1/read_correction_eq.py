"""R5 B1: read the Persistent Q on a Studio One track into profiles/correction_eq/<voice>.json.

    .venv\\Scripts\\python.exe scripts\\s1\\read_correction_eq.py <song> [--track <name>] [--voice ryo] [--out <json>]

--track defaults to 'Vocal CorrEQ' (the v8 export track), else 'mainvocal'. The plugin state is loaded
into Persistent Q through pedalboard and read with correction_eq.bands_from_plugin; the values are
cross-checked against the plugin's own <PersistantQState> XML in the preset (stops on a mismatch).
Never overwrites: when the target exists the JSON goes to <voice>_<YYYYMMDD-HHMMSS>.json instead
(switching the file the engine uses is a human decision). The written path is printed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mix_automation.v8 import correction_eq as ceq  # noqa: E402
from mix_automation.v8.s1_preset import correction_eq_from_song  # noqa: E402


def free_path(path: Path, now: datetime | None = None) -> Path:
    if not path.exists():
        return path
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    alt = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    if alt.exists():
        raise FileExistsError(f"{alt} exists too")
    return alt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("song", type=Path)
    ap.add_argument("--track", default=None)
    ap.add_argument("--voice", default=ceq.DEFAULT_VOICE)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = correction_eq_from_song(a.song, a.track, a.voice)
    target = free_path(a.out or ceq.PROFILE_DIR / f"{a.voice}.json")
    ceq.save(cfg, target)
    print(f"track: {cfg['source']['track']} ({cfg['source']['slot']})  bands: {len(cfg['bands'])}  "
          f"active: {len(ceq.active_bands(cfg))}  global: {cfg['global'] or '{}'}  flat: {ceq.is_flat(cfg)}")
    for b in cfg["bands"]:
        slope = f" {b['slope_db_oct']:.0f} dB/oct" if b["slope_db_oct"] else ""
        print(f"  band {b['band']:>2} {b['type']:<9} {b['frequency_hz']:>8.1f} Hz {b['gain_db']:+6.2f} dB "
              f"Q {b['q']:.3f}{slope} {b['placement']}{' extra=' + str(b['extra']) if b['extra'] else ''}")
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
