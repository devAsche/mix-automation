"""Comprehensive VST3 loadability, sub-plugin discovery (WaveShell), and audio pass test.

Runs each VST3 target in an isolated subprocess with a short timeout so that license
dialogs or hung plugins never block the full audit.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

VST3_ROOT = Path(r"C:\Program Files\Common Files\VST3")

TARGETS: list[tuple[str, Path]] = [
    ("Persistent Q", VST3_ROOT / "Persistent Q.vst3"),
    ("ZL Spectrum Equalizer (root)", VST3_ROOT / "ZL Spectrum Equalizer.vst3"),
    (
        "ZL Spectrum Equalizer (inner)",
        VST3_ROOT / "ZL Spectrum Equalizer.vst3" / "Contents" / "x86_64-win" / "ZL Spectrum Equalizer.vst3",
    ),
    ("Initial Dynamic EQ", VST3_ROOT / "Initial Dynamic EQ.vst3"),
    ("TDR VOS SlickEQ", VST3_ROOT / "TDR VOS SlickEQ.vst3"),
    ("TDR Prism", VST3_ROOT / "TDR Prism.vst3"),
    ("Softube Saturation Knob", VST3_ROOT / "Saturation Knob.vst3"),
    ("Softube VCA Compressor", VST3_ROOT / "VCA Compressor.vst3"),
    ("Exponential Audio R2", VST3_ROOT / "R2.vst3"),
    ("Exponential Audio PhoenixVerb", VST3_ROOT / "PhoenixVerb.vst3"),
    ("Arturia Rev PLATE-140", VST3_ROOT / "Rev PLATE-140.vst3"),
    ("Oeksound Soothe2", VST3_ROOT / "soothe2_x64.vst3"),
    ("UAD LA-2A", VST3_ROOT / "uaudio_teletronix_la-2a_tc.vst3"),
    ("UAD Pultec EQP-1A", VST3_ROOT / "uaudio_pultec_eqp-1a.vst3"),
    ("UAD Pultec MEQ-5", VST3_ROOT / "uaudio_pultec_meq-5.vst3"),
    ("UAD Pultec HLF-3C", VST3_ROOT / "uaudio_pultec_hlf-3c.vst3"),
    ("WaveBreaker", VST3_ROOT / "WaveBreaker.vst3"),
    ("iZotope Nectar 3", VST3_ROOT / "iZotope" / "Nectar 3.vst3"),
    ("iZotope Ozone 12 Elements", VST3_ROOT / "iZotope" / "Ozone 12 Elements.vst3"),
    ("iZotope Ozone 9 Elements", VST3_ROOT / "iZotope" / "Ozone 9 Elements.vst3"),
    ("iZotope Trash 2", VST3_ROOT / "iZotope" / "iZotope Trash 2.vst3"),
    ("iZotope VEA", VST3_ROOT / "iZotope" / "VEA.vst3"),
    ("iZotope Vocal Doubler", VST3_ROOT / "iZotope" / "Vocal Doubler.vst3"),
    ("iZotope Relay", VST3_ROOT / "iZotope" / "Relay.vst3"),
    ("RX 10 Breath Control", VST3_ROOT / "iZotope" / "RX 10 Breath Control.vst3"),
    ("RX 10 Mouth De-click", VST3_ROOT / "iZotope" / "RX 10 Mouth De-click.vst3"),
    ("RX 10 De-click", VST3_ROOT / "iZotope" / "RX 10 De-click.vst3"),
    ("RX 10 De-plosive", VST3_ROOT / "iZotope" / "RX 10 De-plosive.vst3"),
    ("RX 10 De-ess", VST3_ROOT / "iZotope" / "RX 10 De-ess.vst3"),
    ("RX 10 Voice De-noise", VST3_ROOT / "iZotope" / "RX 10 Voice De-noise.vst3"),
    ("RX 10 Spectral De-noise", VST3_ROOT / "iZotope" / "RX 10 Spectral De-noise.vst3"),
    ("RX 10 De-reverb", VST3_ROOT / "iZotope" / "RX 10 De-reverb.vst3"),
    ("RX 10 De-clip", VST3_ROOT / "iZotope" / "RX 10 De-clip.vst3"),
    ("Melda MAGC", VST3_ROOT / "MeldaProduction" / "Dynamics" / "MAGC.vst3"),
    ("Melda MCompressor", VST3_ROOT / "MeldaProduction" / "Dynamics" / "MCompressor.vst3"),
    ("Melda MEqualizer", VST3_ROOT / "MeldaProduction" / "EQ" / "MEqualizer.vst3"),
    ("Melda MAutoPitch", VST3_ROOT / "MeldaProduction" / "Pitch Shift" / "MAutoPitch.vst3"),
    ("Melda MSaturator", VST3_ROOT / "MeldaProduction" / "Distortion" / "MSaturator.vst3"),
    ("kHs Dual Delay", VST3_ROOT / "Kilohearts" / "kHs Dual Delay.vst3"),
    ("kHs Compressor", VST3_ROOT / "Kilohearts" / "kHs Compressor.vst3"),
    ("WaveShell 14.12", VST3_ROOT / "WaveShell1-VST3 14.12_x64.vst3"),
    ("WaveShell 14.17", VST3_ROOT / "WaveShell1-VST3 14.17_x64.vst3"),
    ("WaveShell 14.19", VST3_ROOT / "WaveShell1-VST3 14.19_x64.vst3"),
    ("WaveShell 14.21", VST3_ROOT / "WaveShell1-VST3 14.21_x64.vst3"),
]

WORKER_CODE = """
import json, sys, numpy as np
from pedalboard import VST3Plugin, load_plugin

path = sys.argv[1]
try:
    names = VST3Plugin.get_plugin_names_for_file(path)
except Exception as e:
    print(json.dumps({"status": "scan_error", "error": str(e)}))
    sys.exit(0)

if not names:
    print(json.dumps({"status": "empty", "names": []}))
    sys.exit(0)

if len(names) > 1:
    # Multi-plugin container (e.g., WaveShell): test loading the first few or list all
    loaded_ok = []
    failed = []
    for n in names:
        try:
            p = load_plugin(path, plugin_name=n)
            x = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4800, dtype=np.float32)).reshape(2, -1)
            y = p(x, 48000)
            rms = float(np.sqrt(np.mean(y ** 2)))
            loaded_ok.append({"name": n, "params": len(p.parameters), "out_rms": round(rms, 4)})
        except Exception as sub_e:
            failed.append({"name": n, "error": str(sub_e)[:120]})
    print(json.dumps({"status": "multi", "count": len(names), "loaded": loaded_ok, "failed": failed}))
else:
    try:
        p = load_plugin(path)
        x = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4800, dtype=np.float32)).reshape(2, -1)
        y = p(x, 48000)
        rms = float(np.sqrt(np.mean(y ** 2)))
        print(json.dumps({
            "status": "ok",
            "plugin_name": names[0],
            "param_count": len(p.parameters),
            "sample_params": list(p.parameters.keys())[:8],
            "state_bytes": len(p.raw_state),
            "out_rms": round(rms, 4),
        }))
    except Exception as e:
        print(json.dumps({"status": "load_error", "names": names, "error": str(e)}))
"""


def main() -> None:
    results = {}
    for label, path in TARGETS:
        if not path.exists():
            results[label] = {"status": "missing", "path": str(path)}
            continue
        try:
            proc = subprocess.run(
                [sys.executable, "-c", WORKER_CODE, str(path)],
                capture_output=True,
                text=True,
                timeout=12,
            )
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
            if lines:
                results[label] = json.loads(lines[-1])
            else:
                results[label] = {
                    "status": "crash_or_no_json",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr[-300:],
                }
        except subprocess.TimeoutExpired:
            results[label] = {"status": "timeout_dialog_likely"}

    out_file = Path(r"C:\MIX_automation\data\outputs\vst3_audit_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for label, info in results.items():
        st = info.get("status")
        if st == "ok":
            print(
                f"[OK]      {label:30s} | params={info['param_count']:3d} | rms={info['out_rms']} | {info['sample_params'][:4]}"
            )
        elif st == "multi":
            ok_names = [x["name"] for x in info["loaded"]]
            print(f"[MULTI]   {label:30s} | total={info['count']} | loaded_ok={len(ok_names)}: {ok_names}")
        else:
            err = info.get("error", info.get("stderr", ""))[:90].replace("\n", " ")
            print(f"[{st.upper():8s}] {label:30s} | {err}")


if __name__ == "__main__":
    main()
