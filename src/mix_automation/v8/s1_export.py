"""v8 R5 part B: a Studio One .song from a v8 engine output folder (for setting the correction EQ by ear).

Input: <v8 out>/mix_report.json, stems/{lead_dry,wet_doubler,wet_room,wet_plate,wet_delay,inst_ducked}.wav
and <slug>_v8_Master.wav. Template (read only): data/templates/test20261124/test20261124.song; the
template's own tracks, buses and FX channels are dropped, its input / main output channels are kept.

Tracks (fader 0 dB, pan centre, float WAVs in <out>/s1_media/, every file times the pipeline's
mix_work_gain so the master inserts see the level master_bus saw):
  Vocal CorrEQ      lead_dry (after the ride, before the correction EQ) on both channels, insert:
                    Persistent Q (flat = all bands off, or the correction EQ passed in)
  Wet doubler/room/plate/delay   the sends as rendered (no plugins)
  Inst              inst_ducked
  Ref v8 (pre-EQ)   muted; lead_dry L/R + all sends + inst_ducked = the sum of the tracks above with a flat
                    correction EQ (solo it for A/B; the master inserts then give ~the v8 master)
Master (template 'メイン') inserts, in master_bus order: Saturation Knob (KEEP LOW 0.8) -> [M/S compressor:
scipy DSP, not reproducible with a plugin, omitted] -> VCA Compressor parallel glue (calibrated
compression_db and dry/wet from mix_report) -> Wave Breaker (threshold -0.8 dB; its drive_db carries the
pre-limiter gain: measured 2026-11-28 to act as an input gain, 0.03 dB steps).

Verification (no GUI): reproduce_master() sums the written WAVs and runs them through plugins loaded
from the written .vstpresets, and compares with the v8 master (LUFS, null residual); check_song()
checks the zip, XML, media URLs, presets and IDs.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
import soundfile as sf

from mix_automation.io.song_builder import (
    _render_fx_slot_xml,
    _replace_channel_inserts,
    load_s1_plugin_specs,
)
from mix_automation.v8 import correction_eq as ceq
from mix_automation.v8 import s1_preset as sp

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_SONG = ROOT / "data" / "templates" / "test20261124" / "test20261124.song"
EXPORT_TAG = "v8r5"

MASTER_LABEL = "メイン"  # template main output channel
MASTER_UID = "{636828DD-7358-4765-B2A2-106A2780EE51}"
TRACK_CORR = "Vocal CorrEQ"
TRACK_INST = "Inst"
TRACK_REF = "Ref v8 (pre-EQ)"
SEND_KEYS = ("doubler", "room", "plate", "delay")
# plugin registry key -> Studio One insert name (load_s1_plugin_specs names)
MASTER_CHAIN = (("saturation_knob", "Saturation Knob"), ("vca_compressor", "VCA Compressor"),
                ("wave_breaker", "Wave Breaker"))
# Envelope files of a fresh track (copied from song_builder, which Studio One opened)
ENV_VOLUME = b"{i\x06events[]}"
ENV_PAN = b"{i\x07bipolari\x01i\x06events[]}"


class S1ExportError(RuntimeError):
    pass


# ------------------------------------------------------------------ the v8 render


@dataclass
class V8Render:
    out_dir: Path
    report: dict
    song: str
    slug: str
    sr: int
    tempo_bpm: float
    work_gain_db: float
    glue_compression_db: float
    glue_dry_wet_pct: float
    pre_gain_db: float
    ms_thr_db: float | None
    ms_ratio: float
    master_wav: Path
    stems: dict[str, Path] = field(default_factory=dict)


def _stage(report: dict, name: str) -> dict:
    hits = [s for s in report.get("stages", []) if s.get("stage") == name]
    if len(hits) != 1:
        raise S1ExportError(f"mix_report has {len(hits)} '{name}' stages (need 1)")
    return hits[0]


def read_v8_render(out_dir: Path | str) -> V8Render:
    from mix_automation.v8.engine import ENGINE_NAME, song_slug

    d = Path(out_dir)
    rp = d / "mix_report.json"
    if not rp.exists():
        raise S1ExportError(f"{rp} not found")
    rep = json.loads(rp.read_text(encoding="utf-8"))
    slug = song_slug(rep["song"])
    master = d / f"{slug}_{ENGINE_NAME}_Master.wav"
    stems = {"lead_dry": d / "stems" / "lead_dry.wav", "inst_ducked": d / "stems" / "inst_ducked.wav"}
    stems |= {f"wet_{k}": d / "stems" / f"wet_{k}.wav" for k in SEND_KEYS}
    missing = [str(p) for p in [master, *stems.values()] if not p.exists()]
    if missing:
        raise S1ExportError(f"missing file(s): {missing}")
    glue = _stage(rep, "mix_parallel_glue")
    lim = _stage(rep, "mix_limiter")
    ms = _stage(rep, "mix_ms_compress")
    return V8Render(
        out_dir=d, report=rep, song=rep["song"], slug=slug, sr=int(rep["sr"]),
        tempo_bpm=float(rep.get("tempo_bpm") or 120.0),
        work_gain_db=float(_stage(rep, "mix_work_gain")["gain_db"]),
        glue_compression_db=float(glue["calibration"]["value"]),
        glue_dry_wet_pct=float(rep["knobs"]["glue_wet"]) * 100.0,
        # the last try is the pre-gain of the written master (chain.master_bus loop)
        pre_gain_db=float(lim["measured"]["pre_gain_tries"][-1][0]),
        ms_thr_db=float(ms["calibration"]["value"]) if ms.get("calibration") else None,
        ms_ratio=float(ms.get("ratio", 2.0)),
        master_wav=master, stems=stems,
    )


def _read(path: Path) -> np.ndarray:
    a, _ = sf.read(str(path), dtype="float32", always_2d=True)
    a = a.T
    return np.vstack([a, a]) if a.shape[0] == 1 else a


def track_audio(r: V8Render) -> dict[str, np.ndarray]:
    """Stereo (2, n) float32 per track, times mix_work_gain."""
    g = np.float32(10.0 ** (r.work_gain_db / 20.0))
    lead = _read(r.stems["lead_dry"])
    n = lead.shape[1]

    def fit(a: np.ndarray) -> np.ndarray:
        return a[:, :n] if a.shape[1] >= n else np.pad(a, ((0, 0), (0, n - a.shape[1])))

    tr = {TRACK_CORR: lead}
    for k in SEND_KEYS:
        tr[f"Wet {k}"] = fit(_read(r.stems[f"wet_{k}"]))
    tr[TRACK_INST] = fit(_read(r.stems["inst_ducked"]))
    ref = np.zeros((2, n), dtype=np.float64)
    for a in tr.values():
        ref += a
    tr[TRACK_REF] = ref.astype(np.float32)
    return {k: (v * g).astype(np.float32) for k, v in tr.items()}


# ------------------------------------------------------------------ plugin states


def _load(key: str) -> Any:
    from mix_automation.config.plugin_registry import get_plugin_map
    from mix_automation.v8.chain import load_stage_plugin

    plug, _ = load_stage_plugin(get_plugin_map()[key])
    return plug


def correction_state(cfg: dict | None) -> bytes:
    """Persistent Q .vstpreset: `cfg` (correction EQ dict) or flat (all 32 bands off)."""
    plug = _load("persistent_q")
    flat = ceq.validate({"format": ceq.FORMAT, "bands": []})
    ceq.write_to_plugin(plug, ceq.validate(cfg) if cfg is not None else flat)
    return sp.vstpreset_from_plugin(plug, load_s1_plugin_specs()["persistent_q"].class_id)


def master_states(r: V8Render, pre_gain_db: float) -> dict[str, bytes]:
    """{registry key: .vstpreset} for the master inserts, set as chain.master_bus sets them."""
    from mix_automation.dsp.vst3_adapter import (
        configure_softube_saturation,
        configure_softube_vca,
        set_vst3_param_strict,
    )
    from mix_automation.v8 import chain

    specs = load_s1_plugin_specs()
    out = {}
    p = _load("saturation_knob")
    configure_softube_saturation(p, amount=chain.MASTER_SAT_AMOUNT, sat_type=chain.MASTER_SAT_TYPE)
    out["saturation_knob"] = sp.vstpreset_from_plugin(p, specs["saturation_knob"].class_id)
    p = _load("vca_compressor")
    configure_softube_vca(p, threshold_db=10.0, dry_wet=100.0, **chain.MASTER_GLUE_KW)
    set_vst3_param_strict(p, "compression_db", r.glue_compression_db)
    set_vst3_param_strict(p, "dry_wet_wet", r.glue_dry_wet_pct)
    out["vca_compressor"] = sp.vstpreset_from_plugin(p, specs["vca_compressor"].class_id)
    out["wave_breaker"] = master_states_wb(pre_gain_db)
    return out


def plugin_from_state(key: str, data: bytes) -> Any:
    p = _load(key)
    sp.check_class(data, load_s1_plugin_specs()[key].class_id)
    sp.load_vstpreset_into(p, data)
    return p


# ------------------------------------------------------------------ master reproduction


def lufs(x: np.ndarray, sr: int) -> float:
    from mix_automation.v8 import dsp

    return float(dsp.integrated_lufs(x, sr))


def null_db(ref: np.ndarray, y: np.ndarray) -> float:
    """Residual energy of y - ref relative to ref (dB)."""
    n = min(ref.shape[-1], y.shape[-1])
    a, b = ref[..., :n].astype(np.float64), y[..., :n].astype(np.float64)
    return float(10 * np.log10(np.sum((b - a) ** 2) / max(np.sum(a ** 2), 1e-30) + 1e-30))


def mix_tracks(tracks: dict[str, np.ndarray], corr_state: bytes, sr: int) -> np.ndarray:
    """What Studio One sums into the master: the Vocal CorrEQ track through its Persistent Q, plus the
    other unmuted tracks (Ref is muted)."""
    from mix_automation.v8.chain import run_plugin

    pq = plugin_from_state("persistent_q", corr_state)
    mix = run_plugin(pq, tracks[TRACK_CORR], sr).astype(np.float64)
    for k, a in tracks.items():
        if k not in (TRACK_CORR, TRACK_REF):
            mix += a
    return mix.astype(np.float32)


def master_pre_limiter(mix: np.ndarray, sr: int, states: dict[str, bytes], ms_thr_db: float | None = None,
                       ms_ratio: float = 2.0) -> np.ndarray:
    """Saturation -> (M/S compressor only when ms_thr_db is given) -> parallel glue."""
    from mix_automation.v8 import dsp
    from mix_automation.v8.chain import run_plugin

    x = run_plugin(plugin_from_state("saturation_knob", states["saturation_knob"]), mix, sr)
    if ms_thr_db is not None:
        x = dsp.ms_compress(x, sr, ms_thr_db, ratio=ms_ratio)[0].astype(np.float32)
    return run_plugin(plugin_from_state("vca_compressor", states["vca_compressor"]), x, sr)


def limit(glued: np.ndarray, sr: int, wb_state: bytes) -> np.ndarray:
    from mix_automation.v8.chain import run_plugin

    return run_plugin(plugin_from_state("wave_breaker", wb_state), glued, sr)


def reproduce_master(r: V8Render, tracks: dict[str, np.ndarray], corr_state: bytes, states: dict[str, bytes],
                     lufs_tol: float = 0.1, max_adjust: int = 3) -> tuple[dict[str, bytes], dict]:
    """Run the written tracks through the written states and compare with the v8 master. When the .song
    chain (no M/S compressor) misses the master's LUFS by more than `lufs_tol`, re-set the Wave Breaker
    drive (pre-limiter gain) to match it. Returns (final states, report)."""
    ref = _read(r.master_wav)
    sr = r.sr
    ref_l = lufs(ref, sr)
    mix = mix_tracks(tracks, corr_state, sr)
    full = limit(master_pre_limiter(mix, sr, states, r.ms_thr_db, r.ms_ratio), sr, states["wave_breaker"])
    glued = master_pre_limiter(mix, sr, states)
    drive = r.pre_gain_db
    tries = []
    out = limit(glued, sr, states["wave_breaker"])
    for _ in range(max_adjust + 1):
        out_l = lufs(out, sr)
        tries.append([round(drive, 3), round(out_l, 3)])
        if abs(out_l - ref_l) <= lufs_tol or len(tries) > max_adjust:
            break
        drive += ref_l - out_l
        states = {**states, "wave_breaker": master_states_wb(drive)}
        out = limit(glued, sr, states["wave_breaker"])
    rep = {
        "reference": str(r.master_wav),
        "reference_lufs": round(ref_l, 3),
        "with_ms_compressor": {"note": "plugins from the written presets + v8.dsp.ms_compress at the report "
                                       "threshold: checks the plugin states (should null deeply)",
                               "lufs": round(lufs(full, sr), 3), "lufs_diff": round(lufs(full, sr) - ref_l, 3),
                               "null_db": round(null_db(ref, full), 2)},
        "song_chain": {"note": "what the .song plays (M/S compressor omitted)",
                       "lufs": round(lufs(out, sr), 3), "lufs_diff": round(lufs(out, sr) - ref_l, 3),
                       "null_db": round(null_db(ref, out), 2),
                       "sample_peak_dbfs": round(float(20 * np.log10(np.max(np.abs(out)) + 1e-12)), 2)},
        "ms_omission": {"report_mean_gr_db": (_stage(r.report, "mix_ms_compress").get("mean_gr_db")),
                        "null_db_song_vs_full": round(null_db(full, out), 2)},
        "pre_gain_db_report": r.pre_gain_db,
        "drive_db_tries": tries,
        "drive_db_used": round(drive, 3),
        "drive_db_readjusted": len(tries) > 1,
    }
    return states, rep


def master_states_wb(drive_db: float) -> bytes:
    from mix_automation.dsp.vst3_adapter import set_vst3_param_strict
    from mix_automation.v8 import chain

    p = _load("wave_breaker")
    set_vst3_param_strict(p, "bypass", False)
    set_vst3_param_strict(p, "threshold_db", chain.MASTER_LIMITER_THRESHOLD_DB)
    set_vst3_param_strict(p, "drive_db", float(drive_db))  # 0.03 dB steps (parameter range -10..20)
    return sp.vstpreset_from_plugin(p, load_s1_plugin_specs()["wave_breaker"].class_id)


# ------------------------------------------------------------------ .song writing


def _uid() -> str:
    return "{" + str(uuid.uuid4()).upper() + "}"


def _attr(v: str) -> str:
    return _xml_escape(v, {'"': "&quot;"})


def _env_dir(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


@dataclass
class TrackSpec:
    name: str
    wav: Path
    color: str
    mute: bool = False
    inserts: tuple[tuple[str, str], ...] = ()  # (registry key, preset path)


def _clip_xml(media_id: str, wav: Path, tempo_spb: float, now: str) -> str:
    info = sf.info(str(wav))
    a, _ = sf.read(str(wav), dtype="float32", always_2d=True)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    bits = 32 if info.subtype == "FLOAT" else 24
    return (
        f'\t\t\t<AudioClip mediaID="{media_id}" useCount="1" peak="{peak:.10f}">\n'
        f'\t\t\t\t<Url x:id="path" type="1" url="{_attr(wav.resolve().as_uri())}"/>\n'
        f'\t\t\t\t<DateTime x:id="recordTime" time="{now}"/>\n'
        f'\t\t\t\t<AudioTempoMap x:id="tempoMap" tempoApproved="1">\n'
        f'\t\t\t\t\t<AudioTempoMapSegment offset="0" tempo="{tempo_spb:.16f}"/>\n'
        f'\t\t\t\t</AudioTempoMap>\n'
        f'\t\t\t\t<Attributes x:id="format" frameCount="{info.frames}" sampleRate="{info.samplerate}" '
        f'numChannels="{info.channels}" bitDepth="{bits}"/>\n'
        f'\t\t\t\t<List x:id="modifications">\n\t\t\t\t\t<AudioModification id="0">\n'
        f'\t\t\t\t\t\t<List x:id="propertySets">\n'
        f'\t\t\t\t\t\t\t<AudioBendProperties threshold="0.2" visible="0"/>\n'
        f'\t\t\t\t\t\t\t<AudioChordProperties tuningFrequency="440"/>\n'
        f'\t\t\t\t\t\t</List>\n\t\t\t\t\t</AudioModification>\n\t\t\t\t</List>\n'
        f'\t\t\t</AudioClip>'
    )


def _media_track_xml(t: TrackSpec, track_id: str, chan_id: str, clip_id: str, len_beats: float) -> str:
    nm, env = _attr(t.name), _env_dir(t.name)
    return (
        f'\t\t\t<MediaTrack mediaType="Audio" tempoFollow="0" materialMode="0" layersFollow="1" version="1" '
        f'trackID="{track_id}" timeFormat="2" name="{nm}" color="{t.color}">\n'
        f'\t\t\t\t<SpeakerSetup x:id="trackFormat" type="Stereo"/>\n'
        f'\t\t\t\t<UID x:id="channelID" uid="{chan_id}"/>\n'
        f'\t\t\t\t<Attributes x:id="AutomationRegionList" visible="0">\n'
        f'\t\t\t\t\t<AutomationRegion title="ボリューム" identity="param:///AudioMixer/{chan_id}/volume" '
        f'metaIdentity="AudioTrackChannel/volume" color="FFFFD829" mute="0">\n'
        f'\t\t\t\t\t\t<Url x:id="points" type="1" url="media:///Envelopes/{env}/ボリューム.envelopex"/>\n'
        f'\t\t\t\t\t</AutomationRegion>\n'
        f'\t\t\t\t\t<AutomationRegion title="パン" identity="param:///AudioMixer/{chan_id}/pan" '
        f'metaIdentity="AudioTrackChannel/pan" color="FFC7FF3A" mute="0">\n'
        f'\t\t\t\t\t\t<Url x:id="points" type="1" url="media:///Envelopes/{env}/パン.envelopex"/>\n'
        f'\t\t\t\t\t</AutomationRegion>\n'
        f'\t\t\t\t</Attributes>\n'
        f'\t\t\t\t<List x:id="Events">\n'
        f'\t\t\t\t\t<AudioEvent clipID="{clip_id}" timeFormat="2" length="{len_beats:.8f}" name="{nm}" speed="1" '
        f'transpose="0" tune="0" normalize="0" modification="0">\n'
        f'\t\t\t\t\t\t<Attributes x:id="volumeCurve" level="1"/>\n'
        f'\t\t\t\t\t</AudioEvent>\n'
        f'\t\t\t\t</List>\n'
        f'\t\t\t</MediaTrack>'
    )


def _channel_xml(t: TrackSpec, idx: int, chan_id: str, specs: dict) -> str:
    slots = [_render_fx_slot_xml(i + 1, specs[k], path, stereo_channel=True) for i, (k, path) in enumerate(t.inserts)]
    ins = "\n".join([*slots, '\t\t\t\t\t<Attributes x:id="Presets" pname="デフォルト" dirty="0"/>'])
    return (
        f'\t\t\t<AudioTrackChannel gain="1" pan="0.5" name="Channel{idx:02d}" label="{_attr(t.name)}" '
        f'mute="{int(t.mute)}" solo="0" soloSafe="0" color="{t.color}">\n'
        f'\t\t\t\t<SpeakerSetup x:id="speakerType" type="Stereo"/>\n'
        f'\t\t\t\t<UID x:id="uniqueID" uid="{chan_id}"/>\n'
        f'\t\t\t\t<Connection x:id="destination" objectID="{MASTER_UID}/Input" friendlyName="{MASTER_LABEL}"/>\n'
        f'\t\t\t\t<Attributes x:id="Automation" mode="0"/>\n'
        f'\t\t\t\t<Attributes x:id="Inserts">\n{ins}\n\t\t\t\t</Attributes>\n'
        f'\t\t\t\t<Attributes x:id="CueMix">\n\t\t\t\t\t<Attributes x:id="slots" version="1"/>\n\t\t\t\t</Attributes>\n'
        f'\t\t\t\t<Attributes x:id="InputFX" gain="0" invertPhaseL="0" invertPhaseR="0"/>\n'
        f'\t\t\t\t<Attributes x:id="Sends">\n\t\t\t\t\t<Attributes x:id="Presets" pname="デフォルト" dirty="0"/>\n'
        f'\t\t\t\t</Attributes>\n'
        f'\t\t\t</AudioTrackChannel>'
    )


def _replace_group(mixer: str, group: str, body: str) -> str:
    m = re.search(rf'(\t\t<ChannelGroup name="{group}" flags="1">\n)(.*?)(\t\t</ChannelGroup>)', mixer, re.DOTALL)
    if m is None:
        raise S1ExportError(f"template audiomixer.xml has no ChannelGroup {group}")
    return mixer[:m.start(2)] + body + mixer[m.end(2):]


def _sub1(pattern: str, repl: str, text: str, what: str) -> str:
    out, n = re.subn(pattern, lambda _m: repl, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise S1ExportError(f"template: could not set {what}")
    return out


def _set_attr(text: str, prefix: str, value: str, what: str) -> str:
    """Replace the attribute value after the regex `prefix` (one group ending at the opening quote)."""
    out, n = re.subn(prefix + r'[^"]*', lambda m: m.group(1) + value, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise S1ExportError(f"template: could not set {what}")
    return out


def write_song(song_path: Path, tracks: list[TrackSpec], presets: dict[str, bytes], sr: int, tempo_bpm: float,
               title: str, master_inserts: list[tuple[str, str]], template: Path = TEMPLATE_SONG) -> Path:
    """Write the .song (zip). `presets` maps preset path -> .vstpreset bytes; `master_inserts` is
    [(registry key, preset path)] for the main output."""
    specs = load_s1_plugin_specs()
    members = sp.read_song_members(template)
    # the template's own tracks / buses / FX go away with their presets and envelopes
    members = {k: v for k, v in members.items() if not k.startswith(("Presets/Channels/", "Envelopes/"))}
    dec = {k: members[k].decode("utf-8").replace("\r\n", "\n") for k in
           ("Song/song.xml", "Song/mediapool.xml", "Devices/audiomixer.xml", "Devices/mixerconsole.xml",
            "notepad.xml", "metainfo.xml")}
    spb = 60.0 / tempo_bpm
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S.000")
    dur = max(sf.info(str(t.wav)).frames for t in tracks) / sr
    len_sec = dur + 12.0
    ids = [(_uid(), _uid(), _uid()) for _ in tracks]  # track, channel, clip

    clips = "\n".join(_clip_xml(c, t.wav, spb, now) for t, (_tr, _ch, c) in zip(tracks, ids, strict=True))
    pool = dec["Song/mediapool.xml"].replace(
        '<Attributes x:id="rootFolder"/>',
        f'<Attributes x:id="rootFolder">\n\t\t<MediaFolder name="Audio">\n{clips}\n\t\t</MediaFolder>\n\t</Attributes>')
    pool = _sub1(r'<Attributes x:id="documentPath" type="1" url="[^"]*"/>',
                 f'<Attributes x:id="documentPath" type="1" url="{_attr(song_path.resolve().as_uri())}"/>', pool,
                 "documentPath")

    song = re.sub(r"\t\t\t<MediaTrack .*?</MediaTrack>\n", "", dec["Song/song.xml"], flags=re.DOTALL)
    mtracks = "\n".join(_media_track_xml(t, tr, ch, c, dur / spb) for t, (tr, ch, c) in zip(tracks, ids, strict=True))
    song = _sub1(r"\t\t</List>\n\t</Attributes>\n</Song>", f"{mtracks}\n\t\t</List>\n\t</Attributes>\n</Song>", song,
                 "track list")
    song = _sub1(r'sampleRate="\d+"', f'sampleRate="{sr}"', song, "sampleRate")
    song = _set_attr(song, r'(<Attributes x:id="Root"[^>]*? length=")', f"{len_sec:.2f}", "length")
    # tempo: envelope value = (bpm - 10) / 390 and segment tempo = seconds per beat (template: 146 bpm ->
    # 0.348718 / 0.410959; same mapping as song_builder)
    song = _set_attr(song, r'(<Attributes x:id="envelope">\s*<EnvelopePoint value=")',
                     f"{(tempo_bpm - 10.0) / 390.0:.16f}", "tempo envelope")
    song = _set_attr(song, r'(<TempoMapSegment [^>]*?tempo=")', f"{spb:.16f}", "tempo")
    song = _set_attr(song, r'(<MarkerEvent markerType="3" start=")', f"{len_sec / spb:.6f}", "end marker")

    mixer = dec["Devices/audiomixer.xml"]
    chans = "\n".join(_channel_xml(t, i + 1, ch, specs) for i, (t, (_tr, ch, _c)) in enumerate(zip(tracks, ids, strict=True)))
    mixer = _replace_group(mixer, "AudioTrack", chans + "\n")
    mixer = _replace_group(mixer, "AudioGroup", "")
    mixer = _replace_group(mixer, "AudioEffect", "")
    mslots = [_render_fx_slot_xml(i + 1, specs[k], path, stereo_channel=True) for i, (k, path) in enumerate(master_inserts)]
    mixer2 = _replace_channel_inserts(mixer, MASTER_LABEL, mslots, "v8 master")
    if mslots and mixer2 == mixer:
        raise S1ExportError("template: master inserts not replaced")
    mixer = mixer2

    console = dec["Devices/mixerconsole.xml"]
    keep = re.findall(r'\t\t<Section path="([0-9A-F]{32})">\n\t\t\t<Attributes [^>]*order="(\d+)"/>\n\t\t</Section>\n', console)
    in_ids = {u.strip("{}").replace("-", "") for u in re.findall(r'<AudioInputChannel [^>]*>\s*<SpeakerSetup[^>]*>\s*<UID x:id="uniqueID" uid="([^"]+)"', mixer)}
    inputs = sorted((int(o), p) for p, o in keep if p in in_ids)
    order = [p for _o, p in inputs] + [ch.strip("{}").replace("-", "") for _tr, ch, _c in ids] + \
        [MASTER_UID.strip("{}").replace("-", "")]
    secs = "".join(f'\t\t<Section path="{p}">\n\t\t\t<Attributes visible="1" expanded="{int(i >= len(inputs))}" order="{i}"/>\n'
                   f'\t\t</Section>\n' for i, p in enumerate(order))
    console = _sub1(r'(<Attributes x:id="channelSettings" version="1">\n).*?(\t</Attributes>)',
                    f'<Attributes x:id="channelSettings" version="1">\n{secs}\t</Attributes>', console, "channelSettings")
    vis = "".join(f'\t\t\t\t<UID uid="{ch}"/>\n' for _tr, ch, _c in ids)
    console = re.sub(r'(<List x:id="visible">\n).*?(\t\t\t</List>)', lambda m: m.group(1) + vis + m.group(2), console,
                     flags=re.DOTALL)

    note_items = re.findall(r'\t<NotepadItem id="([^"]+)" title="([^"]*)" text=""/>\n', dec["notepad.xml"])
    keep_ids = {"{" + "-".join([p[:8], p[8:12], p[12:16], p[16:20], p[20:]]) + "}" for p in [x for _o, x in inputs]}
    keep_ids.add(MASTER_UID)
    items = [f'\t<NotepadItem id="{i}" title="{t_}" text=""/>\n' for i, t_ in note_items if i in keep_ids]
    items += [f'\t<NotepadItem id="{tr}" title="{_attr(t.name)}" text=""/>\n' for t, (tr, _ch, _c) in zip(tracks, ids, strict=True)]
    notepad = _sub1(r"(<NotepadData>\n).*?(</NotepadData>)", "<NotepadData>\n" + "".join(items) + "</NotepadData>",
                    dec["notepad.xml"], "notepad")

    meta = dec["metainfo.xml"]
    for key, val in (("Document:Title", title), ("Media:Tempo", f"{tempo_bpm:.6f}"), ("Media:TrackCount", str(len(tracks))),
                     ("Media:Length", f"{len_sec:.2f}"), ("Media:SampleRate", str(sr))):
        meta = re.sub(rf'(<Attribute id="{key}" value=")[^"]*', lambda m, v=val: m.group(1) + _attr(v), meta, count=1)

    for k, v in (("Song/song.xml", song), ("Song/mediapool.xml", pool), ("Devices/audiomixer.xml", mixer),
                 ("Devices/mixerconsole.xml", console), ("notepad.xml", notepad), ("metainfo.xml", meta)):
        members[k] = ("﻿" + v.lstrip("﻿")).replace("\n", "\r\n").encode("utf-8")
    for t in tracks:
        members[f"Envelopes/{_env_dir(t.name)}/ボリューム.envelopex"] = ENV_VOLUME
        members[f"Envelopes/{_env_dir(t.name)}/パン.envelopex"] = ENV_PAN
    members.update(presets)
    song_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(song_path, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return song_path


# ------------------------------------------------------------------ checks (no GUI)


def check_song(song_path: Path | str, template: Path = TEMPLATE_SONG, load_presets: bool = True) -> list[str]:
    """Problems that would stop Studio One from opening the song as intended. Empty = OK."""
    issues: list[str] = []
    song_path = Path(song_path)
    try:
        with zipfile.ZipFile(song_path) as z:
            bad = z.testzip()
            if bad:
                issues.append(f"zip CRC error in {bad}")
    except zipfile.BadZipFile as e:
        return [f"not a zip: {e}"]
    m = sp.read_song_members(song_path)
    tm = sp.read_song_members(template)
    roots = {}
    for k, v in m.items():
        if k.endswith(".xml"):
            try:
                roots[k] = sp.parse_s1_xml(v)
            except Exception as e:  # noqa: BLE001 - any parse failure is a finding
                # Studio One's own Workspace/perspective.xml is not strict XML ('FrameItem::ViewState'):
                # a file copied unchanged from the template is not a finding
                if tm.get(k) != v:
                    issues.append(f"{k}: XML error {e}")
    tmpl = {k for k in tm if not k.startswith(("Presets/Channels/", "Envelopes/"))}
    issues += [f"missing template member {k}" for k in sorted(tmpl - set(m))]
    if issues:
        return issues
    pool, songx, mixer = roots["Song/mediapool.xml"], roots["Song/song.xml"], roots["Devices/audiomixer.xml"]
    clip_ids = []
    for c in pool.iter("AudioClip"):
        clip_ids.append(c.get("mediaID"))
        url = next((u.get("url") for u in c if u.get(sp.XID) == "path"), None)
        from urllib.parse import unquote, urlparse

        p = Path(unquote(urlparse(url or "").path).lstrip("/")) if url else None
        if p is None or not p.exists():
            issues.append(f"media URL does not exist: {url}")
    chan_ids, uniq = [], []
    for ch in mixer.iter():
        if ch.tag.endswith("Channel"):
            u = next((e.get("uid") for e in ch if e.get(sp.XID) == "uniqueID"), None)
            chan_ids.append(u)
    for e in mixer.iter("UID"):
        if e.get(sp.XID) == "uniqueID":
            uniq.append(e.get("uid"))
    track_ids = [t.get("trackID") for t in songx.iter("MediaTrack")]
    for what, lst in (("clip", clip_ids), ("mixer uniqueID", uniq), ("track", track_ids)):
        dup = sorted({x for x in lst if lst.count(x) > 1})
        if dup:
            issues.append(f"duplicate {what} IDs: {dup}")
    if set(track_ids) & set(chan_ids):
        issues.append("a track ID equals a channel ID")
    for t in songx.iter("MediaTrack"):
        ch = next((e.get("uid") for e in t if e.get(sp.XID) == "channelID"), None)
        if ch not in chan_ids:
            issues.append(f"track {t.get('name')}: channel {ch} not in audiomixer.xml")
        for ev in t.iter("AudioEvent"):
            if ev.get("clipID") not in clip_ids:
                issues.append(f"track {t.get('name')}: clip {ev.get('clipID')} not in mediapool.xml")
        for u in t.iter("Url"):
            ref = (u.get("url") or "").removeprefix("media:///")
            if ref and ref not in m:
                issues.append(f"track {t.get('name')}: {ref} not in the .song")
    specs = load_s1_plugin_specs()
    by_class = {s.class_id.upper(): k for k, s in specs.items()}
    for label, slots in sp.channel_inserts(mixer).items():
        for s in slots:
            data = m.get(s["preset_path"] or "")
            if data is None:
                issues.append(f"{label} {s['slot']}: preset {s['preset_path']} missing")
                continue
            try:
                sp.check_class(data, s["class_id"] or "")
                key = by_class.get((s["class_id"] or "").upper())
                if load_presets:
                    if key is None:
                        raise sp.S1PresetError(f"unknown class {s['class_id']}")
                    plugin_from_state(key, data)
            except Exception as e:  # noqa: BLE001 - reported, not raised
                issues.append(f"{label} {s['slot']} ({s['plugin']}): {e}")
    return issues


# ------------------------------------------------------------------ top level


def fresh_dir(base: Path) -> Path:
    if not base.exists() or not any(base.iterdir()):
        return base
    return base.with_name(f"{base.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")


def export_v8_song(v8_dir: Path | str, out_dir: Path | str | None = None, correction_cfg: dict | None = None,
                   verify: bool = True, template: Path = TEMPLATE_SONG) -> dict:
    """Build <out>/<slug>_v8r5.song (+ _orig copy), media in <out>/s1_media/, export_report.json.
    `out_dir` defaults to <v8 dir>/s1_export; an existing non-empty folder is never written into
    (a timestamped sibling is used). `correction_cfg` = correction EQ dict for the Vocal CorrEQ insert
    (None = flat)."""
    r = read_v8_render(v8_dir)
    out = fresh_dir(Path(out_dir) if out_dir is not None else r.out_dir / "s1_export")
    media = out / "s1_media"
    media.mkdir(parents=True, exist_ok=True)
    audio = track_audio(r)
    files = {}
    for name, a in audio.items():
        p = media / f"{_env_dir(name)}.wav"
        sf.write(str(p), a.T, r.sr, subtype="FLOAT")
        files[name] = p
    tracks_back = {k: _read(p) for k, p in files.items()}  # what Studio One will read
    corr = correction_state(correction_cfg)
    states = master_states(r, r.pre_gain_db)
    verify_rep = None
    if verify:
        states, verify_rep = reproduce_master(r, tracks_back, corr, states)
    corr_path = f"Presets/Channels/{TRACK_CORR}/1 - Persistent Q.vstpreset"
    presets = {corr_path: corr}
    master_inserts = []
    for i, (key, nm) in enumerate(MASTER_CHAIN):
        path = f"Presets/Channels/{MASTER_LABEL}/{i + 1} - {nm}.vstpreset"
        presets[path] = states[key]
        master_inserts.append((key, path))
    colors = {TRACK_CORR: "FFDD6105", TRACK_INST: "FF2D7DD2", TRACK_REF: "FF7F8C8D"}
    specs = [TrackSpec(TRACK_CORR, files[TRACK_CORR], colors[TRACK_CORR], inserts=(("persistent_q", corr_path),))]
    specs += [TrackSpec(f"Wet {k}", files[f"Wet {k}"], "FF16A085") for k in SEND_KEYS]
    specs += [TrackSpec(TRACK_INST, files[TRACK_INST], colors[TRACK_INST]),
              TrackSpec(TRACK_REF, files[TRACK_REF], colors[TRACK_REF], mute=True)]
    name = f"{r.slug}_{EXPORT_TAG}"
    song_path = write_song(out / f"{name}.song", specs, presets, r.sr, r.tempo_bpm, name, master_inserts, template)
    orig = out / f"{name}_orig.song"
    shutil.copyfile(song_path, orig)
    issues = check_song(song_path, template)
    rep = {
        "format": "v8_s1_export/1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "v8_dir": str(r.out_dir),
        "song": str(song_path), "song_orig": str(orig),
        "template": str(template),
        "mix_work_gain_db": r.work_gain_db,
        "tracks": {k: str(p) for k, p in files.items()},
        "vocal_corr_eq": {"flat": ceq.is_flat(correction_cfg), "bands": ceq.active_bands(correction_cfg)},
        "master_inserts": [nm for _k, nm in MASTER_CHAIN],
        "omitted": ["mix_ms_compress (v8.dsp.ms_compress, scipy; no plugin equivalent)"],
        "glue": {"compression_db": r.glue_compression_db, "dry_wet_pct": r.glue_dry_wet_pct},
        "verify": verify_rep,
        "check_issues": issues,
    }
    (out / "export_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    rep["out_dir"] = str(out)
    return rep


__all__ = [
    "S1ExportError",
    "TrackSpec",
    "V8Render",
    "check_song",
    "correction_state",
    "export_v8_song",
    "master_states",
    "null_db",
    "read_v8_render",
    "reproduce_master",
    "track_audio",
    "write_song",
]
