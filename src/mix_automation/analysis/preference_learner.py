"""Human-in-the-Loop Preference Learning & Advanced Studio One 5 Anchor Propagation.

Extracts quantitative user DAW tweaks between an auto-generated `.song` and a user-adjusted
`.song` (`mainvocal` / `harmvocal` faders, send levels, per-event `volumeCurve` clip gains,
and `Persistent Q` band gains), updates bounded EMA perceptual target offsets, applies learned
preferences during target resolution, and propagates anchor tweaks across sections with
proportional harmony scaling so harmonies never detach from the lead.
"""

from __future__ import annotations

import json
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from mix_automation.analysis.relative_profile import (
    VOCAL_MIX_BANDS,
    SectionRelativeProfile,
    compute_anchor_propagation,
    compute_section_relative_profile,
)
from mix_automation.io.audio_loader import load_audio
from mix_automation.io.song_builder import extract_steinberg_comp_and_cont

_SEND_ALIAS_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("doubler", ("doubler",)),
    ("delay", ("h-delay", "delay")),
    ("plate_reverb", ("r2", "plate")),
    ("saturation_parallel", ("saturation",)),
    ("early_room", ("phoenix", "earlyroom", "early room")),
)

_EMPTY_MIX_BANDS: dict[str, float] = {b: 0.0 for b in VOCAL_MIX_BANDS}


def _safe_parse_xml(raw_bytes: bytes) -> ET.Element:
    txt = raw_bytes.decode("utf-8", errors="ignore").lstrip("\ufeff")
    if "xmlns:x=" not in txt:
        txt = re.sub(
            r"<(?!\?)([A-Za-z0-9_]+)(\s|>)",
            r'<\1 xmlns:x="http://www.presonus.com/xml"\2',
            txt,
            count=1,
        )
    return ET.fromstring(txt)


def _lin_to_db(val: float, floor_db: float = -48.0) -> float:
    return float(max(floor_db, 20.0 * np.log10(max(float(val), 1e-6))))


def _map_freq_to_mix_band(freq_hz: float) -> str | None:
    for b_name, (f_lo, f_hi) in VOCAL_MIX_BANDS.items():
        if f_lo <= freq_hz < f_hi:
            return b_name
    if freq_hz >= 16000.0:
        return "air_sheen"
    return None


def _map_send_name(friendly_name: str) -> str | None:
    low = friendly_name.strip().lower()
    for canonical, tokens in _SEND_ALIAS_RULES:
        if canonical == "plate_reverb":
            if low == "r2" or "plate" in low:
                return canonical
        elif any(tok in low for tok in tokens):
            return canonical
    return None


def _read_mixer_state(mixer_root: ET.Element) -> dict:
    """Extract channel fader dB levels and vocal send dB levels from `Devices/audiomixer.xml`."""
    channel_gains_db: dict[str, float] = {}
    sends_by_owner: dict[str, dict[str, float]] = {}

    for elem in mixer_root.iter():
        if elem.tag not in {"AudioTrackChannel", "AudioGroupChannel", "AudioEffectChannel", "AudioOutputChannel"}:
            continue
        lbl = elem.attrib.get("label", "").strip()
        if not lbl:
            continue
        gain_lin = float(elem.attrib.get("gain", "1.0"))
        channel_gains_db[lbl] = _lin_to_db(gain_lin)

        sends_elem = None
        for child in elem:
            if child.tag == "Attributes" and any(v == "Sends" for k, v in child.attrib.items() if k.endswith("id")):
                sends_elem = child
                break
        if sends_elem is not None:
            owner_sends: dict[str, float] = {}
            for s_node in sends_elem:
                if s_node.tag != "Attributes":
                    continue
                s_name = s_node.attrib.get("name", "")
                if not s_name.startswith("Send"):
                    continue
                bypassed = s_node.attrib.get("bypass", "0") == "1"
                conn = s_node.find("Connection")
                friendly = conn.attrib.get("friendlyName", "") if conn is not None else ""
                canonical = _map_send_name(friendly)
                if canonical is None:
                    continue
                lvl_lin = 1e-4 if bypassed else float(s_node.attrib.get("level", "0.0"))
                owner_sends[canonical] = _lin_to_db(lvl_lin)
            if owner_sends:
                sends_by_owner[lbl] = owner_sends

    merged_sends: dict[str, float] = {}
    for owner in ("main", "mainvocal"):
        for k, v in sends_by_owner.get(owner, {}).items():
            merged_sends.setdefault(k, v)

    return {"channel_gains_db": channel_gains_db, "sends_db": merged_sends}


def _read_mainvocal_events(song_root: ET.Element) -> list[dict]:
    """Read ordered `AudioEvent` records (`name`, `start_beats`, `level_lin`, `level_db`) on `mainvocal`."""
    events: list[dict] = []
    for mtrack in song_root.iter("MediaTrack"):
        if mtrack.attrib.get("name") != "mainvocal":
            continue
        for ev in mtrack.iter("AudioEvent"):
            vc = None
            for child in ev:
                if child.tag == "Attributes" and any(v == "volumeCurve" for k, v in child.attrib.items() if k.endswith("id")):
                    vc = child
                    break
            if vc is None:
                vc = ev.find("Attributes")
            lvl = float(vc.attrib.get("level", "1.0")) if vc is not None else 1.0
            events.append(
                {
                    "name": ev.attrib.get("name", f"event_{len(events)}"),
                    "start_beats": float(ev.attrib.get("start", "0.0")),
                    "level_lin": lvl,
                    "level_db": _lin_to_db(lvl),
                }
            )
    return events


def _extract_comp_chunk(preset_bytes: bytes) -> bytes:
    """Extract Steinberg `Comp` chunk bytes (or raw XML fallback) from a `.vstpreset` binary."""
    if len(preset_bytes) >= 48 and preset_bytes[:4] == b"VST3":
        _, _, _, list_offset = struct.unpack("<4si32sq", preset_bytes[:48])
        if 48 <= list_offset <= len(preset_bytes) - 8 and preset_bytes[list_offset : list_offset + 4] == b"List":
            (entry_count,) = struct.unpack("<i", preset_bytes[list_offset + 4 : list_offset + 8])
            pos = list_offset + 8
            for _ in range(entry_count):
                if pos + 20 > len(preset_bytes):
                    break
                cid, c_off, c_size = struct.unpack("<4sqq", preset_bytes[pos : pos + 20])
                pos += 20
                if cid == b"Comp" and 0 <= c_off and c_off + c_size <= len(preset_bytes):
                    comp = preset_bytes[c_off : c_off + c_size]
                    if b"<VST3PluginState>" in comp:
                        comp, _ = extract_steinberg_comp_and_cont(comp)
                    return comp
    if b"<VST3PluginState>" in preset_bytes:
        comp, _ = extract_steinberg_comp_and_cont(preset_bytes)
        return comp
    return preset_bytes


def _parse_persistent_q_bands_from_vstpreset(preset_bytes: bytes) -> list[dict]:
    """Parse `<PersistantQState>` band frequencies & gains from a `Persistent Q` `.vstpreset`."""
    comp = _extract_comp_chunk(preset_bytes)
    start = comp.find(b"<PersistantQState")
    if start == -1:
        return []
    end = comp.find(b"</PersistantQState>", start)
    if end != -1:
        xml_bytes = comp[start : end + len(b"</PersistantQState>")]
    else:
        nul = comp.find(b"\x00", start)
        xml_bytes = comp[start:nul] if nul != -1 else comp[start:]

    xml_str = xml_bytes.decode("utf-8", errors="ignore")
    bands: list[dict] = []
    try:
        root = ET.fromstring(xml_str)
        for node in root.findall("Band"):
            bands.append(
                {
                    "id": int(node.attrib.get("id", len(bands) + 1)),
                    "enabled": node.attrib.get("enabled", "0") == "1",
                    "type": int(node.attrib.get("type", "0")),
                    "freq_hz": float(node.attrib.get("frequencyHz", "1000.0")),
                    "gain_db": float(node.attrib.get("gainDb", "0.0")),
                    "dynamic_enabled": node.attrib.get("dynamicEnabled", "0") == "1",
                    "spectral_enabled": node.attrib.get("spectralEnabled", "0") == "1",
                }
            )
        if bands:
            return bands
    except Exception:
        pass

    for m in re.finditer(r"<Band\b([^>]+)/?>", xml_str):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        try:
            bands.append(
                {
                    "id": int(attrs.get("id", len(bands) + 1)),
                    "enabled": attrs.get("enabled", "0") == "1",
                    "type": int(attrs.get("type", "0")),
                    "freq_hz": float(attrs.get("frequencyHz", "1000.0")),
                    "gain_db": float(attrs.get("gainDb", "0.0")),
                    "dynamic_enabled": attrs.get("dynamicEnabled", "0") == "1",
                    "spectral_enabled": attrs.get("spectralEnabled", "0") == "1",
                }
            )
        except ValueError:
            continue
    return bands


def _compute_pq_band_deltas(auto_preset: bytes, user_preset: bytes) -> tuple[list[dict], dict[str, float]]:
    """Compute per-band and 7-band aggregated `Persistent Q` EQ gain deltas (`user - auto`)."""
    auto_bands = {b["id"]: b for b in _parse_persistent_q_bands_from_vstpreset(auto_preset)}
    user_bands = {b["id"]: b for b in _parse_persistent_q_bands_from_vstpreset(user_preset)}
    if not auto_bands and not user_bands:
        return [], dict(_EMPTY_MIX_BANDS)

    band_deltas: list[dict] = []
    grouped: dict[str, list[float]] = {b: [] for b in VOCAL_MIX_BANDS}

    for bid in sorted(set(auto_bands) | set(user_bands)):
        ab = auto_bands.get(bid)
        ub = user_bands.get(bid)
        # Skip HPF/LPF filter bands (type 1=Highpass, 2=Lowpass) and dynamic/spectral bands
        b_type = ub["type"] if ub else (ab["type"] if ab else 0)
        if b_type in (1, 2):
            continue
        is_dyn_or_spec = bool(
            (ab and (ab.get("dynamic_enabled") or ab.get("spectral_enabled")))
            or (ub and (ub.get("dynamic_enabled") or ub.get("spectral_enabled")))
        )
        if is_dyn_or_spec:
            continue
        a_eff = ab["gain_db"] if (ab and ab["enabled"]) else 0.0
        u_eff = ub["gain_db"] if (ub and ub["enabled"]) else 0.0
        delta = round(float(u_eff - a_eff), 4)
        freq_hz = ub["freq_hz"] if ub else (ab["freq_hz"] if ab else 1000.0)
        mix_band = _map_freq_to_mix_band(freq_hz)
        band_deltas.append(
            {
                "id": bid,
                "freq_hz": round(freq_hz, 1),
                "mix_band": mix_band,
                "auto_gain_db": round(a_eff, 4),
                "user_gain_db": round(u_eff, 4),
                "delta_db": delta,
            }
        )
        if mix_band is not None and ((ab and ab["enabled"]) or (ub and ub["enabled"])):
            grouped[mix_band].append(delta)

    mix_band_deltas = {
        b: round(float(np.mean(vals)), 4) if vals else 0.0
        for b, vals in grouped.items()
    }
    return band_deltas, mix_band_deltas


def extract_song_tweak_deltas(auto_song_path: Path, user_song_path: Path) -> dict:
    """Compute exact DAW tweak deltas (`user - auto`) between two Studio One 5 `.song` archives."""
    with zipfile.ZipFile(auto_song_path, "r") as za, zipfile.ZipFile(user_song_path, "r") as zu:
        auto_files = {n: za.read(n) for n in za.namelist()}
        user_files = {n: zu.read(n) for n in zu.namelist()}

    auto_mixer = _read_mixer_state(_safe_parse_xml(auto_files["Devices/audiomixer.xml"]))
    user_mixer = _read_mixer_state(_safe_parse_xml(user_files["Devices/audiomixer.xml"]))

    ag = auto_mixer["channel_gains_db"]
    ug = user_mixer["channel_gains_db"]

    main_delta = round(ug.get("mainvocal", 0.0) - ag.get("mainvocal", 0.0), 4)
    main_bus_delta = round(ug.get("main", 0.0) - ag.get("main", 0.0), 4)

    harm_deltas = [
        ug[lbl] - ag[lbl]
        for lbl in ("harmvocal", "harmvocal2")
        if lbl in ag and lbl in ug
    ]
    harmvocal_delta = round(float(ug.get("harmvocal", 0.0) - ag.get("harmvocal", 0.0)), 4)
    mean_harm_delta = round(float(np.mean(harm_deltas)) if harm_deltas else harmvocal_delta, 4)

    send_keys = ("doubler", "delay", "plate_reverb", "saturation_parallel", "early_room")
    send_deltas: dict[str, float] = {}
    for sk in send_keys:
        if sk in auto_mixer["sends_db"] and sk in user_mixer["sends_db"]:
            send_deltas[sk] = round(float(user_mixer["sends_db"][sk] - auto_mixer["sends_db"][sk]), 4)
        else:
            send_deltas[sk] = 0.0

    auto_events = _read_mainvocal_events(_safe_parse_xml(auto_files["Song/song.xml"]))
    user_events = _read_mainvocal_events(_safe_parse_xml(user_files["Song/song.xml"]))

    clip_gain_deltas: dict[str, float] = {}
    user_by_name = {ev["name"]: ev for ev in user_events}
    if len(auto_events) == len(user_events) and not all(ev["name"] in user_by_name for ev in auto_events):
        pairs = list(zip(auto_events, user_events))
    else:
        pairs = [(ev, user_by_name[ev["name"]]) for ev in auto_events if ev["name"] in user_by_name]

    for a_ev, u_ev in pairs:
        clip_gain_deltas[a_ev["name"]] = round(float(u_ev["level_db"] - a_ev["level_db"]), 4)
    mean_clip_delta = round(float(np.mean(list(clip_gain_deltas.values()))) if clip_gain_deltas else 0.0, 4)

    pq_rel = "Presets/Channels/mainvocal/1 - Persistent Q.vstpreset"
    pq_bands: list[dict] = []
    pq_mix_deltas = dict(_EMPTY_MIX_BANDS)
    if pq_rel in auto_files and pq_rel in user_files:
        pq_bands, pq_mix_deltas = _compute_pq_band_deltas(auto_files[pq_rel], user_files[pq_rel])

    return {
        "mainvocal_fader_delta_db": main_delta,
        "main_bus_fader_delta_db": main_bus_delta,
        "harmvocal_fader_delta_db": harmvocal_delta,
        "mean_harmony_fader_delta_db": mean_harm_delta,
        "send_deltas_db": send_deltas,
        "clip_gain_deltas_db": clip_gain_deltas,
        "mean_clip_gain_delta_db": mean_clip_delta,
        "persistent_q_band_deltas": pq_bands,
        "persistent_q_mix_band_deltas_db": pq_mix_deltas,
    }


def update_user_preference_offsets(
    auto_song_path: Path,
    user_song_path: Path,
    offsets_json_path: Path,
    alpha: float = 0.35,
) -> dict:
    """Update bounded EMA user preference offsets JSON from Studio One tweak deltas."""
    deltas = extract_song_tweak_deltas(auto_song_path, user_song_path)
    alpha_c = float(np.clip(alpha, 0.05, 1.0))

    prev: dict = {}
    if offsets_json_path.exists():
        try:
            prev = json.loads(offsets_json_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    sends = deltas["send_deltas_db"]
    k_lufs_sample = float(
        np.clip(
            deltas["mainvocal_fader_delta_db"]
            + deltas["main_bus_fader_delta_db"]
            + deltas["mean_clip_gain_delta_db"],
            -4.0,
            4.0,
        )
    )
    wet_tail_sample = float(
        np.clip(
            0.65 * sends.get("plate_reverb", 0.0)
            + 0.20 * sends.get("early_room", 0.0)
            + 0.15 * sends.get("delay", 0.0),
            -4.0,
            4.0,
        )
    )
    side_mid_sample = float(
        np.clip(
            0.55 * deltas["harmvocal_fader_delta_db"] + 0.45 * sends.get("doubler", 0.0),
            -4.0,
            4.0,
        )
    )

    def _ema(prev_val: float, sample_val: float) -> float:
        updated = (1.0 - alpha_c) * float(prev_val) + alpha_c * float(sample_val)
        return round(float(np.clip(updated, -2.5, 2.5)), 4)

    prev_bands = prev.get("intrinsic_band_offsets_db", {})
    pq_mix = deltas["persistent_q_mix_band_deltas_db"]
    updated_bands = {
        b: _ema(prev_bands.get(b, 0.0), pq_mix.get(b, 0.0))
        for b in VOCAL_MIX_BANDS
    }

    record = {
        "schema_version": "1.0",
        "num_sessions": int(prev.get("num_sessions", 0)) + 1,
        "alpha": round(alpha_c, 3),
        "k_lufs_vi_offset_db": _ema(prev.get("k_lufs_vi_offset_db", 0.0), k_lufs_sample),
        "wet_tail_offset_db": _ema(prev.get("wet_tail_offset_db", 0.0), wet_tail_sample),
        "vocal_side_mid_offset_db": _ema(prev.get("vocal_side_mid_offset_db", 0.0), side_mid_sample),
        "intrinsic_band_offsets_db": updated_bands,
        "last_session": {
            "auto_song": str(auto_song_path),
            "user_song": str(user_song_path),
            "extracted_deltas": deltas,
        },
    }
    offsets_json_path.parent.mkdir(parents=True, exist_ok=True)
    offsets_json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def apply_user_preference_offsets(
    profile: SectionRelativeProfile,
    offsets_json_path: Path | None,
) -> SectionRelativeProfile:
    """Apply learned user preference offsets (when `num_sessions >= 1`) to a target profile."""
    if offsets_json_path is None or not offsets_json_path.exists():
        return profile

    try:
        data = json.loads(offsets_json_path.read_text(encoding="utf-8"))
    except Exception:
        return profile

    if int(data.get("num_sessions", 0)) < 1:
        return profile

    k_off = float(np.clip(data.get("k_lufs_vi_offset_db", 0.0), -2.5, 2.5))
    w_off = float(np.clip(data.get("wet_tail_offset_db", 0.0), -2.5, 2.5))
    s_off = float(np.clip(data.get("vocal_side_mid_offset_db", 0.0), -2.5, 2.5))
    raw_bands = data.get("intrinsic_band_offsets_db", {})
    band_offs = {b: float(np.clip(raw_bands.get(b, 0.0), -2.5, 2.5)) for b in VOCAL_MIX_BANDS}

    has_change = any(abs(x) > 1e-4 for x in (k_off, w_off, s_off, *band_offs.values()))
    arch_lbl = profile.backing_archetype + ("+pref" if has_change and "+pref" not in profile.backing_archetype else "")

    return SectionRelativeProfile(
        vocal_rms_db=round(profile.vocal_rms_db + k_off, 3),
        inst_rms_db=profile.inst_rms_db,
        broadband_vi_ratio_db=round(profile.broadband_vi_ratio_db + k_off, 3),
        k_lufs_vi_ratio_db=round(profile.k_lufs_vi_ratio_db + k_off, 3),
        mid_k_lufs_vi_ratio_db=round(profile.mid_k_lufs_vi_ratio_db + k_off, 3),
        partial_loudness_margin_db=round(profile.partial_loudness_margin_db + k_off, 3),
        vocal_crest_db=profile.vocal_crest_db,
        inst_crest_db=profile.inst_crest_db,
        crest_difference_db=profile.crest_difference_db,
        position_stability_std_db=profile.position_stability_std_db,
        harsh_resonance_index_db=profile.harsh_resonance_index_db,
        syllabic_clarity_index=profile.syllabic_clarity_index,
        wet_tail_ratio_db=round(float(np.clip(profile.wet_tail_ratio_db + w_off, -34.0, -10.0)), 2),
        vocal_side_mid_db=round(float(np.clip(profile.vocal_side_mid_db + s_off, -28.0, 0.0)), 2),
        inst_side_mid_db=profile.inst_side_mid_db,
        inst_mid_density_index=profile.inst_mid_density_index,
        inst_high_rolloff_db=profile.inst_high_rolloff_db,
        backing_archetype=arch_lbl,
        vocal_intrinsic_bands_db={
            b: round(profile.vocal_intrinsic_bands_db[b] + band_offs[b], 3)
            for b in VOCAL_MIX_BANDS
        },
        band_margins_db={
            b: round(profile.band_margins_db[b] + k_off + band_offs[b], 3)
            for b in VOCAL_MIX_BANDS
        },
        inst_band_levels_db=dict(profile.inst_band_levels_db),
        active_frame_ratio=profile.active_frame_ratio,
    )


def _resolve_mediapool_wavs(pool_root: ET.Element, song_dir: Path) -> tuple[Path | None, Path | None]:
    """Resolve `mainvocal` and `offvocal` WAV paths from `Song/mediapool.xml` if they exist on disk."""
    from urllib.parse import unquote, urlparse

    main_wav: Path | None = None
    inst_wav: Path | None = None
    for clip in pool_root.iter("AudioClip"):
        url_el = clip.find("Url")
        if url_el is None:
            continue
        url = url_el.attrib.get("url", "")
        if not url.startswith("file://"):
            continue
        parsed = urlparse(url)
        raw_path = unquote(parsed.path)
        if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        cand = Path(raw_path)
        if not cand.exists():
            alt = song_dir / "Media" / cand.name
            cand = alt if alt.exists() else cand
        if not cand.exists():
            continue
        low = cand.name.lower()
        if "mainvocal" in low:
            main_wav = cand
        elif "offvocal_unmasked" in low or ("offvocal" in low and inst_wav is None):
            inst_wav = cand
    return main_wav, inst_wav


def _synthetic_section_profile(
    base_profile: SectionRelativeProfile,
    inst_rms_offset_db: float,
    vocal_gain_offset_db: float,
) -> SectionRelativeProfile:
    """Construct a section-scaled `SectionRelativeProfile` fallback when WAVs are unavailable."""
    vi_shift = vocal_gain_offset_db - inst_rms_offset_db
    return SectionRelativeProfile(
        vocal_rms_db=base_profile.vocal_rms_db + vocal_gain_offset_db,
        inst_rms_db=base_profile.inst_rms_db + inst_rms_offset_db,
        broadband_vi_ratio_db=base_profile.broadband_vi_ratio_db + vi_shift,
        k_lufs_vi_ratio_db=base_profile.k_lufs_vi_ratio_db + vi_shift,
        mid_k_lufs_vi_ratio_db=base_profile.mid_k_lufs_vi_ratio_db + vi_shift,
        partial_loudness_margin_db=base_profile.partial_loudness_margin_db + vi_shift,
        vocal_crest_db=base_profile.vocal_crest_db,
        inst_crest_db=base_profile.inst_crest_db,
        crest_difference_db=base_profile.crest_difference_db,
        position_stability_std_db=base_profile.position_stability_std_db,
        harsh_resonance_index_db=base_profile.harsh_resonance_index_db,
        syllabic_clarity_index=base_profile.syllabic_clarity_index,
        wet_tail_ratio_db=base_profile.wet_tail_ratio_db,
        vocal_side_mid_db=base_profile.vocal_side_mid_db,
        inst_side_mid_db=base_profile.inst_side_mid_db,
        inst_mid_density_index=base_profile.inst_mid_density_index,
        inst_high_rolloff_db=base_profile.inst_high_rolloff_db,
        backing_archetype=base_profile.backing_archetype,
        vocal_intrinsic_bands_db=dict(base_profile.vocal_intrinsic_bands_db),
        band_margins_db={b: v + vi_shift for b, v in base_profile.band_margins_db.items()},
        inst_band_levels_db={b: v + inst_rms_offset_db for b, v in base_profile.inst_band_levels_db.items()},
        active_frame_ratio=base_profile.active_frame_ratio,
    )


def _update_mainvocal_event_level_xml(song_xml: str, event_name: str, new_lin: float) -> str:
    """Update `<Attributes x:id="volumeCurve" level="..."/>` inside the named `mainvocal` `<AudioEvent>`."""
    mt_pos = song_xml.find('name="mainvocal"')
    if mt_pos == -1:
        return song_xml
    mt_end = song_xml.find("</MediaTrack>", mt_pos)
    if mt_end == -1:
        mt_end = len(song_xml)

    ev_pat = re.compile(
        r'(<AudioEvent\b[^>]*\bname="' + re.escape(event_name) + r'"[^>]*>.*?</AudioEvent>)',
        re.S,
    )
    track_slice = song_xml[mt_pos:mt_end]
    m = ev_pat.search(track_slice)
    if not m:
        return song_xml

    block = m.group(1)
    updated_block = re.sub(r'(\blevel=")[^"]+(")', rf"\g<1>{new_lin:.8f}\g<2>", block, count=1)
    abs_s = mt_pos + m.start(1)
    abs_e = mt_pos + m.end(1)
    return song_xml[:abs_s] + updated_block + song_xml[abs_e:]


def _scale_channel_gain_xml(mixer_xml: str, label: str, delta_db: float) -> tuple[str, float | None]:
    """Scale `<AudioTrackChannel gain="..." ... label="<label>">` by `delta_db` and return new dB."""
    pat = re.compile(
        r'(<AudioTrackChannel\b[^>]*\bgain=")([^"]+)("[^>]*\blabel="' + re.escape(label) + r'")'
    )
    m = pat.search(mixer_xml)
    if not m:
        return mixer_xml, None
    old_db = _lin_to_db(float(m.group(2)))
    new_db = float(np.clip(old_db + delta_db, -36.0, 6.0))
    new_lin = round(float(10.0 ** (new_db / 20.0)), 6)
    updated = mixer_xml[: m.start(2)] + f"{new_lin}" + mixer_xml[m.end(2) :]
    return updated, round(new_db, 3)


def propagate_anchor_advanced(
    song_path: Path,
    anchor_event_index: int = 0,
) -> tuple[Path, dict]:
    """Propagate user's anchor clip-gain & fader tweak across sections and scale harmonies proportionally."""
    with zipfile.ZipFile(song_path, "r") as zin:
        files = {n: zin.read(n) for n in zin.namelist()}

    song_xml_text = files["Song/song.xml"].decode("utf-8")
    mixer_xml_text = files["Devices/audiomixer.xml"].decode("utf-8")
    song_root = _safe_parse_xml(files["Song/song.xml"])
    mixer_root = _safe_parse_xml(files["Devices/audiomixer.xml"])

    events = _read_mainvocal_events(song_root)
    if not events or anchor_event_index < 0 or anchor_event_index >= len(events):
        raise ValueError(f"Could not find event index {anchor_event_index} on mainvocal track.")

    report_path = song_path.parent / "mix_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    orig_sections: list[dict] = report.get("sections", [])

    anchor_ev = events[anchor_event_index]
    orig_anchor_clip_db = (
        float(orig_sections[anchor_event_index]["clip_gain_db"])
        if orig_sections and anchor_event_index < len(orig_sections)
        else 0.0
    )
    anchor_clip_delta_db = float(anchor_ev["level_db"] - orig_anchor_clip_db)

    mixer_state = _read_mixer_state(mixer_root)
    current_main_fader_db = mixer_state["channel_gains_db"].get("mainvocal", 0.0)
    orig_main_fader_db = float(report.get("optimized_chain_params", {}).get("vocal_fader_db", 0.0))
    mainvocal_fader_delta_db = float(current_main_fader_db - orig_main_fader_db)

    # Attempt perceptual audio-backed `compute_anchor_propagation` ratios if WAVs exist on disk
    main_wav: Path | None = None
    inst_wav: Path | None = None
    if "Song/mediapool.xml" in files:
        pool_root = _safe_parse_xml(files["Song/mediapool.xml"])
        main_wav, inst_wav = _resolve_mediapool_wavs(pool_root, song_path.parent)

    use_audio_profiles = bool(
        main_wav is not None and inst_wav is not None and orig_sections and anchor_event_index < len(orig_sections)
    )
    v_audio = i_audio = None
    sr_audio = 44100
    anchor_prof_orig = anchor_prof_new = None
    prop_mode = "report_ratio"

    if use_audio_profiles and main_wav is not None and inst_wav is not None:
        try:
            v_audio, sr_audio = load_audio(main_wav, target_sr=44100, mono=True)
            i_audio, _ = load_audio(inst_wav, target_sr=44100, mono=False)
            a_sec = orig_sections[anchor_event_index]
            a0 = max(0, int(float(a_sec["start_sec"]) * sr_audio))
            a1 = min(len(v_audio), i_audio.shape[1], int(float(a_sec["end_sec"]) * sr_audio))
            if a1 - a0 >= sr_audio:
                v_a_orig = v_audio[a0:a1] * float(10.0 ** (orig_anchor_clip_db / 20.0))
                v_a_new = v_audio[a0:a1] * float(10.0 ** (anchor_ev["level_db"] / 20.0))
                anchor_prof_orig = compute_section_relative_profile(v_a_orig, i_audio[:, a0:a1], sr=sr_audio)
                anchor_prof_new = compute_section_relative_profile(v_a_new, i_audio[:, a0:a1], sr=sr_audio)
                prop_mode = "audio_perceptual"
        except Exception:
            use_audio_profiles = False

    # Construct synthetic fallback reference profile for `compute_anchor_propagation` when WAVs are absent
    if anchor_prof_orig is None or anchor_prof_new is None:
        core_dict = report.get("comparison_anchor_chorus", {}).get("automix_anchor_core")
        if core_dict:
            base_prof = SectionRelativeProfile(**core_dict)
        else:
            base_prof = SectionRelativeProfile(
                vocal_rms_db=-18.0,
                inst_rms_db=-16.0,
                broadband_vi_ratio_db=-2.0,
                k_lufs_vi_ratio_db=1.0,
                mid_k_lufs_vi_ratio_db=1.5,
                partial_loudness_margin_db=1.8,
                vocal_crest_db=10.5,
                inst_crest_db=9.5,
                crest_difference_db=1.0,
                position_stability_std_db=2.5,
                harsh_resonance_index_db=3.5,
                syllabic_clarity_index=0.22,
                wet_tail_ratio_db=-22.0,
                vocal_side_mid_db=-10.0,
                inst_side_mid_db=-7.5,
                inst_mid_density_index=0.55,
                inst_high_rolloff_db=-8.5,
                backing_archetype="consensus",
                vocal_intrinsic_bands_db={b: -10.0 for b in VOCAL_MIX_BANDS},
                band_margins_db={b: 0.0 for b in VOCAL_MIX_BANDS},
                inst_band_levels_db={b: -20.0 for b in VOCAL_MIX_BANDS},
                active_frame_ratio=0.75,
            )
        anchor_prof_orig = _synthetic_section_profile(base_prof, 0.0, 0.0)
        anchor_prof_new = _synthetic_section_profile(base_prof, 0.0, anchor_clip_delta_db)

    updated_song_xml = song_xml_text
    updated_events_summary: list[dict] = []

    for idx, ev in enumerate(events):
        ev_name = ev["name"]
        sec_info = orig_sections[idx] if orig_sections and idx < len(orig_sections) else None
        is_inst = "instrumental" in ev_name.lower() or (sec_info is not None and "instrumental" in sec_info["label"].lower())
        orig_clip_db = float(sec_info["clip_gain_db"]) if sec_info is not None else float(ev["level_db"])

        if idx == anchor_event_index or is_inst:
            updated_events_summary.append(
                {
                    "name": ev_name,
                    "old_db": round(float(ev["level_db"]), 3),
                    "new_db": round(float(ev["level_db"]), 3),
                    "delta_db": 0.0,
                    "is_instrumental": is_inst,
                    "is_anchor": idx == anchor_event_index,
                }
            )
            continue

        sec_prof = None
        if prop_mode == "audio_perceptual" and v_audio is not None and i_audio is not None and sec_info is not None:
            s0 = max(0, int(float(sec_info["start_sec"]) * sr_audio))
            s1 = min(len(v_audio), i_audio.shape[1], int(float(sec_info["end_sec"]) * sr_audio))
            if s1 - s0 >= sr_audio:
                v_s = v_audio[s0:s1] * float(10.0 ** (orig_clip_db / 20.0))
                sec_prof = compute_section_relative_profile(v_s, i_audio[:, s0:s1], sr=sr_audio)

        if sec_prof is None:
            inst_off = (
                float(sec_info["inst_rms_db"]) - float(orig_sections[anchor_event_index]["inst_rms_db"])
                if sec_info is not None and orig_sections
                else 0.0
            )
            sec_prof = _synthetic_section_profile(anchor_prof_orig, inst_off, orig_clip_db - orig_anchor_clip_db)

        adj_orig = compute_anchor_propagation(anchor_prof_orig, sec_prof)
        adj_new = compute_anchor_propagation(anchor_prof_new, sec_prof)
        prop_delta_db = float(adj_new.input_gain_delta_db - adj_orig.input_gain_delta_db)

        # Gentle intimacy taper in quiet sections so a heavy chorus push doesn't make verses float
        section_scale = 0.85 if "quiet" in ev_name.lower() else 1.0
        eff_delta_db = round(float(np.clip(prop_delta_db * section_scale, -6.0, 6.0)), 3)
        new_db = round(float(np.clip(orig_clip_db + eff_delta_db, -18.0, 12.0)), 3)
        new_lin = float(10.0 ** (new_db / 20.0))

        updated_song_xml = _update_mainvocal_event_level_xml(updated_song_xml, ev_name, new_lin)
        updated_events_summary.append(
            {
                "name": ev_name,
                "old_db": round(float(ev["level_db"]), 3),
                "new_db": new_db,
                "delta_db": round(new_db - float(ev["level_db"]), 3),
                "is_instrumental": False,
                "is_anchor": False,
            }
        )

    total_anchor_delta_db = float(anchor_clip_delta_db + mainvocal_fader_delta_db)
    updated_mixer_xml = mixer_xml_text
    harm_delta_applied = 0.0
    harmony_new_gains: dict[str, float] = {}
    if abs(total_anchor_delta_db) >= 0.25:
        harm_delta_applied = round(float(np.clip(total_anchor_delta_db, -6.0, 6.0)), 3)
        for h_lbl in ("harmvocal", "harmvocal2"):
            updated_mixer_xml, h_new_db = _scale_channel_gain_xml(updated_mixer_xml, h_lbl, harm_delta_applied)
            if h_new_db is not None:
                harmony_new_gains[h_lbl] = h_new_db

    out_path = song_path.with_name(song_path.stem + "_propagated.song")
    files["Song/song.xml"] = updated_song_xml.encode("utf-8")
    files["Devices/audiomixer.xml"] = updated_mixer_xml.encode("utf-8")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for k, v in files.items():
            zout.writestr(k, v)

    summary = {
        "output_song_path": str(out_path),
        "anchor_event_index": anchor_event_index,
        "anchor_event_name": anchor_ev["name"],
        "anchor_clip_delta_db": round(anchor_clip_delta_db, 3),
        "mainvocal_fader_delta_db": round(mainvocal_fader_delta_db, 3),
        "total_anchor_delta_db": round(total_anchor_delta_db, 3),
        "harmony_gain_delta_db": harm_delta_applied,
        "harmony_new_gains_db": harmony_new_gains,
        "propagation_mode": prop_mode,
        "updated_events": updated_events_summary,
    }
    return out_path, summary
