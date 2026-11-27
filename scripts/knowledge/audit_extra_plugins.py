"""Load test for installed VST3s that scripts/audit_vst3_load.py does not cover (R1 step 4).

Same isolated-subprocess probe as audit_vst3_load.py, but focused on space / colour / width
candidates (delay, reverb, saturator, doubler, imager, clipper). Writes to the knowledge cache,
not to data/outputs/vst3_audit_report.json.

    .venv\\Scripts\\python.exe scripts\\knowledge\\audit_extra_plugins.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V = Path(r"C:\Program Files\Common Files\VST3")
K, M = V / "Kilohearts", V / "MeldaProduction"
OUT = REPO / "data" / "processed" / "mix_knowledge" / "vst3_extra_audit.json"

TARGETS = [
    ("kHs Delay", K / "kHs Delay.vst3"),
    ("kHs Reverb", K / "kHs Reverb.vst3"),
    ("kHs Haas", K / "kHs Haas.vst3"),
    ("kHs Stereo", K / "kHs Stereo.vst3"),
    ("kHs Chorus", K / "kHs Chorus.vst3"),
    ("kHs Ensemble", K / "kHs Ensemble.vst3"),
    ("kHs Pitch Shifter", K / "kHs Pitch Shifter.vst3"),
    ("kHs Distortion", K / "kHs Distortion.vst3"),
    ("kHs Clipper", K / "kHs Clipper.vst3"),
    ("kHs Limiter", K / "kHs Limiter.vst3"),
    ("kHs Transient Shaper", K / "kHs Transient Shaper.vst3"),
    ("kHs Dynamics", K / "kHs Dynamics.vst3"),
    ("kHs Filter", K / "kHs Filter.vst3"),
    ("kHs 3-Band EQ", K / "kHs 3-Band EQ.vst3"),
    ("Melda MDelay", M / "Delay" / "MDelay.vst3"),
    ("Melda MCharmVerb", M / "Reverb" / "MCharmVerb.vst3"),
    ("Melda MConvolutionEZ", M / "Reverb" / "MConvolutionEZ.vst3"),
    ("Melda MStereoExpander", M / "Stereo" / "MStereoExpander.vst3"),
    ("Melda MWaveShaper", M / "Distortion" / "MWaveShaper.vst3"),
    ("Melda MCompressor", M / "Dynamics" / "MCompressor.vst3"),
    ("VPRE-73", V / "VPRE-73.vst3"),
    ("TALLBOY", V / "TALLBOY.vst3"),
    ("Dagon", V / "Dagon.vst3"),
    ("Flash", V / "Flash.vst3"),
    ("Hybrid Filter", V / "Hybrid Filter.vst3"),
    ("LANDR FX Voice", V / "LANDR" / "LANDR FX Voice.vst3"),
    ("iZotope Ozone 12 Elements", V / "iZotope" / "Ozone 12 Elements.vst3"),
    ("iZotope VocalSynth 2", V / "iZotope" / "VocalSynth 2.vst3"),
    ("iZotope Nectar 3 (60s)", V / "iZotope" / "Nectar 3.vst3"),
]
LONG = {"iZotope Nectar 3 (60s)": 60}


def _audit_module():
    spec = importlib.util.spec_from_file_location("audit", REPO / "scripts" / "audit_vst3_load.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker_code() -> str:
    return _audit_module().WORKER_CODE


def main() -> None:
    audit = _audit_module()
    code = audit.WORKER_CODE
    res = {}
    for label, path in TARGETS:
        if not path.exists():
            res[label] = {"status": "missing"}
        else:
            try:
                # Same rule as PluginSpec.loadable_path (inner x86_64-win binary of a bundle when present).
                p = subprocess.run([sys.executable, "-c", code, str(audit.loadable_path(path))], capture_output=True,
                                   text=True, timeout=LONG.get(label, 15))
                lines = [ln for ln in p.stdout.splitlines() if ln.strip().startswith("{")]
                res[label] = json.loads(lines[-1]) if lines else {"status": "crash_or_no_json", "rc": p.returncode}
            except subprocess.TimeoutExpired:
                res[label] = {"status": "timeout_dialog_likely"}
        r = res[label]
        print(f"{label:<28} {r.get('status')} params={r.get('param_count', '-')} out_rms={r.get('out_rms', '-')}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
