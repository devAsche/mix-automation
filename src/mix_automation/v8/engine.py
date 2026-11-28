"""v8 engine entry point (R3 chain + R4 knobs on real VST3 plugins, lead vocal only).

Plan: docs/V8_PLAN.md. Chain and constants: v8/chain.py; knobs for R4/R5: v8/knobs.py.
Only the main vocal (consolidate_vocal_stem_folder().lead_mono) is used; doubles, harmonies and
choirs are dropped on purpose. Any plugin load/parameter failure raises.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from mix_automation.io.audio_loader import consolidate_vocal_stem_folder, load_audio
from mix_automation.provenance import render_provenance
from mix_automation.v8 import chain, dsp
from mix_automation.v8 import correction_eq as ceq
from mix_automation.v8.knobs import load_knobs

ENGINE_NAME = "v8"
SR = 44100


def song_slug(song_name: str) -> str:
    """Same convention as v7 (pipeline.run_full_automix_for_song): '&' -> 'and', spaces -> '_'."""
    return song_name.replace("&", "and").replace(" ", "_")


def fresh_output_dir(base: Path, now: datetime | None = None) -> Path:
    """Never write into a folder that already holds files: use `<name>_YYYYMMDD-HHMMSS` instead
    (naming as pipeline.resolve_writable_output_dir)."""
    if not base.exists() or not any(base.iterdir()):
        return base
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    alt = base.with_name(f"{base.name}_{stamp}")
    print(f"[v8] '{base}' already has files; writing to '{alt}' (no overwrite).", flush=True)
    return alt


def estimate_tempo_bpm(inst: np.ndarray, sr: int) -> tuple[float, int]:
    """Tempo from Beat This! beats (median inter-beat interval). Raises when no beat grid is found."""
    from mix_automation.v8.metrics.timing import beat_grid

    beats, _ = beat_grid(inst, sr)
    if len(beats) < 4:
        raise RuntimeError(f"v8: Beat This! found only {len(beats)} beats; cannot set the delay grid")
    ioi = float(np.median(np.diff(beats)))
    return 60.0 / ioi, len(beats)


APPLIED_CEQ_NAME = "correction_eq_applied.json"  # read by scripts/s1/export_song.py


def select_correction_eq(source: str, cfg: dict | None, lead_dry: np.ndarray, sr: int, auto_enabled: bool,
                         raw_lead: np.ndarray | None = None, raw_inst: np.ndarray | None = None,
                         voice: str | None = ceq.DEFAULT_VOICE) -> tuple[str, dict | None, dict]:
    """Final (source, cfg, auto_info): "auto" becomes v8.auto_eq's profile of lead_dry (broad high band from
    high_gap_db of the unprocessed lead vs the raw inst, when given), or "none" (flat) when auto is
    disabled; other sources pass through."""
    if source != "auto":
        return source, cfg, {}
    if not auto_enabled:
        return "none", None, {}
    gap = None
    if raw_lead is not None and raw_inst is not None:
        ri = np.asarray(raw_inst)
        gap = ceq.auto_high_gap_db(raw_lead, ri.mean(axis=0) if ri.ndim == 2 else ri)
    cfg = ceq.auto_config(lead_dry, sr, voice=voice, gap_db=gap)
    return "auto", cfg, {"high_gap_db": None if gap is None else round(gap, 2),
                         "broad_high": (cfg.get("source") or {}).get("broad_high")}


def _write(path: Path, x: np.ndarray, sr: int, subtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(x, dtype=np.float32)
    sf.write(str(path), a.T if a.ndim == 2 else a, sr, subtype=subtype)


def run_v8_automix_for_song(
    song_folder: Path,
    output_root: Path,
    knobs_json: Path | None = None,
    knob_overrides: dict | None = None,
    voice: str = ceq.DEFAULT_VOICE,
    correction_eq_json: Path | None = None,
    auto_correction_eq: bool = True,
    **_options: object,
) -> dict:
    """R5: the correction EQ is applied to the lead after the phrase ride. Source, first found:
    `correction_eq_json` > profiles/correction_eq/index.json entry for the song folder > <voice>.json >
    auto (v8.auto_eq on lead_dry; `auto_correction_eq=False` -> flat). Flat -> skipped (no plugin loaded)."""
    from mix_automation.pipeline import discover_song_assets

    t0 = time.time()
    song_folder = Path(song_folder)
    knobs = load_knobs(knobs_json, knob_overrides)
    # validated before the long render; "auto" is computed once lead_dry exists
    ceq_source, ceq_path, ceq_cfg = ceq.resolve_source(song_folder.name, voice, correction_eq_json)
    assets = discover_song_assets(song_folder)
    stems = consolidate_vocal_stem_folder(list(assets["stems"]), target_sr=SR)
    lead = np.asarray(stems.lead_mono, dtype=np.float32)
    inst, _ = load_audio(Path(assets["inst"]), target_sr=SR, mono=False)
    inst = np.asarray(inst, dtype=np.float32)
    if inst.ndim == 1:
        inst = np.stack([inst, inst])
    n = inst.shape[1]
    lead = chain.fit_len(lead, n)

    slug = song_slug(song_folder.name)
    out_dir = fresh_output_dir(Path(output_root) / f"{slug}_{ENGINE_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = chain.StageLog()

    tempo_bpm, n_beats = estimate_tempo_bpm(inst, SR)
    inst_work = chain.prepare_inst(inst, SR, log)
    lead_dry, ride_stats = chain.process_lead(lead, inst_work, SR, knobs, log)
    sends = chain.build_sends(lead_dry, SR, knobs, tempo_bpm, log)
    inst_ducked, duck_stats = chain.duck_inst(inst_work, lead_dry, SR, knobs, log)
    ceq_source, ceq_cfg, ceq_auto = select_correction_eq(ceq_source, ceq_cfg, lead_dry, SR, auto_correction_eq,
                                                         raw_lead=lead, raw_inst=inst, voice=voice)
    # R5: sends and the duck key stay on the uncorrected lead; the VocalBus gets the corrected lead.
    lead_corrected = chain.correction_eq(lead_dry, SR, ceq_cfg, log)
    ceq_flat = lead_corrected is lead_dry

    vocal_bus = chain.vocal_bus(lead_corrected, sends, n)
    mix = inst_ducked.astype(np.float64) + vocal_bus
    master, mix_stats = chain.master_bus(mix.astype(np.float32), SR, knobs, log)

    # VocalBus file: the vocal part of the 2mix input times one uniform gain = master LUFS - pre-bus LUFS.
    # Approximation: the bus saturation / M-S comp / parallel glue / limiter are non-linear and not replayed per stem.
    bus_gain_db = float(mix_stats["master_lufs"] - dsp.integrated_lufs(mix, SR))
    vocal_bus_out = (vocal_bus * 10 ** (bus_gain_db / 20)).astype(np.float32)

    name = f"{slug}_{ENGINE_NAME}"
    master_path = out_dir / f"{name}_Master.wav"
    vb_path = out_dir / f"{name}_VocalBus.wav"
    _write(master_path, master, SR, "PCM_24")
    _write(vb_path, vocal_bus_out, SR, "FLOAT")
    _write(out_dir / "stems" / "lead_dry.wav", lead_dry, SR, "FLOAT")
    applied_path = out_dir / APPLIED_CEQ_NAME
    if not ceq.is_flat(ceq_cfg):
        ceq.save(ceq_cfg, applied_path)
    if not ceq_flat:
        _write(out_dir / "stems" / "lead_corrected.wav", lead_corrected, SR, "FLOAT")
    for k, w in sends.items():
        _write(out_dir / "stems" / f"wet_{k}.wav", w, SR, "FLOAT")
    _write(out_dir / "stems" / "inst_ducked.wav", inst_ducked, SR, "FLOAT")

    metrics = dsp.mix_metrics(vocal_bus, inst_ducked, master, SR)
    report = {
        **render_provenance(ENGINE_NAME, None),
        "stage_name": "R4",
        "song": song_folder.name,
        "song_folder": str(song_folder),
        "inst_file": str(assets["inst"]),
        "vocal_sources": {
            "used": "lead_mono (consolidate_vocal_stem_folder)",
            "dropped": ["double", "harmonies", "choir", "octave layers"],
            "stem_roles_summary": stems.stem_roles_summary,
            "stems_superseded": [str(x) for x in assets.get("stems_superseded") or []],
        },
        "sr": SR,
        "knobs": knobs.to_dict(),
        "tempo_bpm": round(tempo_bpm, 2),
        "n_beats": n_beats,
        "stages": log.stages,
        "ride": {k: v for k, v in ride_stats.items() if k != "phrases"},
        "ride_phrases": ride_stats["phrases"],
        "inst_duck": duck_stats,
        "correction_eq": {"source": ceq_source, "voice": voice, **ceq_auto,
                          **ceq.report_entry(ceq_path if ceq_path is not None else applied_path, ceq_cfg),
                          "applied_json": str(applied_path) if applied_path.exists() else None,
                          "lead_corrected_wav": None if ceq_flat else str(out_dir / "stems" / "lead_corrected.wav")},
        "mix_bus": mix_stats,
        "vocal_bus_note": (
            "VocalBus = (corrected lead + all sends) x uniform gain (master LUFS - pre-bus LUFS = "
            f"{bus_gain_db:.2f} dB); approximation, bus non-linearities not replayed. Stems are at pre-bus level."
        ),
        "metrics": metrics,
        "outputs": {
            "master_wav": str(master_path),
            "vocal_bus_wav": str(vb_path),
            "stems_dir": str(out_dir / "stems"),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "mix_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    report["output_dir"] = str(out_dir)
    return report


def _json_default(o: object) -> object:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
