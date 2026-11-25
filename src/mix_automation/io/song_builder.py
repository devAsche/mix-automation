"""v6 Studio One 5.5 True 1:1 Native Steinberg `.vstpreset` Synthesizer & Ultra-Precision Console Builder.

Exports EVERY v6 ultra-precision DSP module 1:1 into Studio One 5 via decoded native Steinberg
`Comp`/`Cont` binaries:
  - `mainvocal`:
      1. `Persistent Q` (16-Band Static + Surgical Notches + Dynamic EQ + Spectral EQ!)
      2. `VCA Compressor` (Fast clean peak-tamer `attack=0.55`, `output_drive=0.1`)
      3. `UADx LA-2A Tube Compressor` (Zero-noise optical body leveler across ALL archetypes)
      4. `TDR VOS SlickEQ` / `UADx Pultec EQP-1A` (when ballad/pop color applies)
      5. `soothe2` (Post-Comp / Post-Sat 4-Band Surgical Resonance & HF Sizzle Polish!)
  - `harmvocal` & `harmvocal2`:
      1. `Persistent Q` (Lead-Formant-Adaptive Hollow-Core EQ + Dynamic Harmony Sibilance Control)
      2. `VCA Compressor` (Ultra-Fast Attack `0.25` + Slow Release `3.40` `6:1` Onset-Crushing Glue)
      + Fuller audible harmony fader (`~ -8.6 dB`) & shared `PhoenixVerb`/`R2` depth sends!
  - `Saturation Parallel`:
      1. `Persistent Q` (`240 Hz` HPF + `4.0 kHz` LPF Midrange-Only Bandpass — zero HF sizzle!)
      2. `Saturation Knob` (`KEEP LOW`)
  - `メイン`: `VCA Compressor` bus glue + `Wave Breaker` true-peak ceiling.
  - Also registers newly licensed `VocalSynth 2` (`{56535456-7853-3256-6F63-616C53796E74}`).
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import struct
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import soundfile as sf

from mix_automation.config.plugin_registry import VST3_ROOT, get_plugin_map
from mix_automation.dsp.pitch_timing import ReviewMarker
from mix_automation.dsp.vst3_optimizer import SongSection, VocalChainParams

_JUCE_B64_ALPHABET = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
_JUCE_B64_LUT = {ch: idx for idx, ch in enumerate(_JUCE_B64_ALPHABET)}


@dataclass(frozen=True)
class TrackMediaSpec:
    """Audio track specification to embed into the generated Studio One 5 session."""

    name: str
    wav_path: Path
    stereo: bool
    color: str = "FF4A90E2"
    gain_db: float = 0.0
    mute: bool = False
    pan: float = 0.5
    insert_slots: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class S1PluginClassSpec:
    """Studio One 5 VST3 plugin descriptor matching `Devices/audiomixer.xml` & `Plugins-ja.settings`."""

    name: str
    class_id: str          # "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}"
    sub_category: str      # e.g. "VST3/Dynamics"
    stereo_ports: bool = True

    @property
    def hex32(self) -> str:
        return self.class_id.strip("{}").replace("-", "").upper()


DEFAULT_S1_PLUGIN_SPECS: dict[str, S1PluginClassSpec] = {
    "persistent_q": S1PluginClassSpec("Persistent Q", "{ABCDEF01-9182-FAEB-4B61-706550713031}", "VST3", False),
    "vca_compressor": S1PluginClassSpec("VCA Compressor", "{24DCAB96-A33C-41E5-B2C9-7A9953B51CCD}", "VST3/Dynamics", True),
    "mv2_mono": S1PluginClassSpec("MV2 Mono", "{5653544D-5632-4D6D-7632-206D6F6E6F00}", "VST3/Dynamics", False),
    "uad_la2a": S1PluginClassSpec("UADx LA-2A Tube Compressor", "{ABCDEF01-9182-FAEB-5541-44785533444A}", "VST3/Dynamics", True),
    "uad_pultec": S1PluginClassSpec("UADx Pultec EQP-1A EQ", "{ABCDEF01-9182-FAEB-5541-44785533414E}", "VST3/EQ", True),
    "uad_pultec_meq5": S1PluginClassSpec("UADx Pultec MEQ-5 EQ", "{ABCDEF01-9182-FAEB-5541-447855334152}", "VST3/EQ", True),
    "slick_eq": S1PluginClassSpec("TDR VOS SlickEQ", "{56535454-6431-3074-6472-20766F732073}", "VST3/EQ", True),
    "soothe2": S1PluginClassSpec("soothe2", "{ABCDEF01-9182-FAEB-4F65-6B5353746842}", "VST3/Dynamics", True),
    "initial_dynamic_eq": S1PluginClassSpec("Initial Dynamic EQ", "{ABCDEF01-9182-FAEB-4961-756464596E33}", "VST3", True),
    "vocalsynth2": S1PluginClassSpec("VocalSynth 2", "{56535456-7853-3256-6F63-616C53796E74}", "VST3/Mixing", True),
    "mautopitch": S1PluginClassSpec("MAutoPitch", "{4D656C64-6170-726F-4D61-75544D617554}", "VST3/Pitch & Time", True),
    "izotope_trash2": S1PluginClassSpec("iZotope Trash 2", "{5653545A-6E54-3269-5A6F-746F70652054}", "VST3/Distortion", True),
    "saturation_knob": S1PluginClassSpec("Saturation Knob", "{F73F44A6-96F4-40AF-B187-1EAFA5E2AF01}", "VST3/Distortion", True),
    "phoenix_verb": S1PluginClassSpec("PhoenixVerb", "{5653544D-4373-7270-686F-656E69787665}", "VST3/Reverb", True),
    "rev_plate_140": S1PluginClassSpec("Rev PLATE-140", "{41727475-4156-4953-5265-506C50726F63}", "VST3/Reverb", True),
    "r2": S1PluginClassSpec("R2", "{5653544D-4373-3272-3200-000000000000}", "VST3/Reverb", True),
    "ozone9_elements": S1PluginClassSpec("Ozone 9 Elements", "{5653545A-6E45-394F-7A6F-6E6520392045}", "VST3/Mastering", True),
    "wave_breaker": S1PluginClassSpec("Wave Breaker", "{F2AEE70D-00DE-4F4E-5050-6C7950507762}", "VST3/Dynamics", True),
    "h_delay": S1PluginClassSpec("H-Delay Stereo", "{56535448-4244-5368-2D64-656C61792073}", "VST3/Delay", True),
    "khs_dual_delay": S1PluginClassSpec("kHs Dual Delay", "{00ED3532-00A9-4557-0028-037A00175A0E}", "VST3/Spatial", True),
    "doubler2": S1PluginClassSpec("Doubler2 Stereo", "{56535457-4432-5364-6F75-626C65723220}", "VST3/Modulation", True),
    "vocal_doubler": S1PluginClassSpec("Vocal Doubler", "{56535449-5A56-4456-6F63-616C20446F75}", "VST3/Modulation", True),
}


def decode_juce_base64(encoded: str) -> bytes:
    """Decode JUCE's `MemoryBlock::toBase64Encoding()` format (`'{num_bytes}.{64-char-stream}'`)."""
    s = encoded.strip()
    if "." not in s:
        return b""
    n_str, payload = s.split(".", 1)
    n_bytes = int(n_str)
    out = bytearray()
    pos = 0
    plen = len(payload)
    while len(out) < n_bytes and pos < plen:
        chunk = payload[pos : pos + 4]
        pos += 4
        val = 0
        for idx, ch in enumerate(chunk):
            val |= _JUCE_B64_LUT[ch] << (6 * idx)
        for b in range(3):
            if len(out) < n_bytes:
                out.append((val >> (8 * b)) & 0xFF)
    return bytes(out)


def extract_steinberg_comp_and_cont(raw_state: bytes) -> tuple[bytes, bytes]:
    """Unwrap Pedalboard/JUCE `<VST3PluginState>` XML into native Steinberg `(Comp, Cont)` byte streams."""
    if b"<VST3PluginState>" not in raw_state:
        return raw_state, b""

    m_comp = re.search(rb"<IComponent>([^<]*)</IComponent>", raw_state)
    m_cont = re.search(rb"<IEditController>([^<]*)</IEditController>", raw_state)
    comp = decode_juce_base64(m_comp.group(1).decode("ascii")) if (m_comp and m_comp.group(1).strip()) else raw_state
    cont = decode_juce_base64(m_cont.group(1).decode("ascii")) if (m_cont and m_cont.group(1).strip()) else b""
    return comp, cont


def load_s1_plugin_specs() -> dict[str, S1PluginClassSpec]:
    """Load Studio One 5 VST3 class descriptors, refreshing from local `Plugins-*.settings` when available."""
    specs = dict(DEFAULT_S1_PLUGIN_SPECS)
    s1_dir = Path(os.path.expandvars(r"%APPDATA%\PreSonus\Studio One 5\x64"))
    if not s1_dir.exists():
        return specs

    settings_files = sorted(s1_dir.glob("Plugins-*.settings"))
    if not settings_files:
        return specs

    try:
        txt = settings_files[0].read_text(encoding="utf-8", errors="ignore")
        by_name: dict[str, tuple[str, str]] = {}
        for m in re.finditer(r"<ClassDescription\s+([^>]+)>", txt, re.S):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            if "VST3" in attrs.get("subCategory", "") and attrs.get("classID"):
                p_name = html.unescape(attrs.get("name", "")).lower()
                by_name[p_name] = (html.unescape(attrs["classID"]), html.unescape(attrs["subCategory"]))

        for key, spec in list(specs.items()):
            hit = by_name.get(spec.name.lower())
            if hit:
                specs[key] = S1PluginClassSpec(spec.name, hit[0], hit[1], spec.stereo_ports)
    except Exception:
        pass
    return specs


def create_vstpreset_bytes(
    class_id_guid: str,
    comp_state: bytes,
    cont_state: bytes = b"",
) -> bytes:
    """Synthesize a genuine Studio One 5-native Steinberg `.vstpreset` binary (`Comp` + `Cont` + `Info`)."""
    hex32 = class_id_guid.strip("{}").replace("-", "").upper()
    if len(hex32) != 32:
        raise ValueError(f"Expected 32-hex VST3 class ID, got {class_id_guid!r}")

    if b"<VST3PluginState>" in comp_state:
        comp_state, extracted_cont = extract_steinberg_comp_and_cont(comp_state)
        if not cont_state:
            cont_state = extracted_cont

    info_xml = (
        '\ufeff<?xml version="1.0" encoding="UTF-8"?>\r\n'
        "<MetaInfo>\r\n"
        f'\t<Attribute id="VST3UniqueID" value="{hex32}" type="string"/>\r\n'
        "</MetaInfo>\r\n"
    ).encode("utf-8")

    chunks = [
        (b"Comp", comp_state),
        (b"Cont", cont_state),
        (b"Info", info_xml),
    ]
    data_blobs = bytearray()
    list_entries = bytearray()
    cursor = 48
    for cid, payload in chunks:
        size = len(payload)
        list_entries.extend(struct.pack("<4sqq", cid, cursor, size))
        data_blobs.extend(payload)
        cursor += size

    header = struct.pack("<4si32sq", b"VST3", 1, hex32.encode("ascii"), cursor)
    list_hdr = struct.pack("<4si", b"List", len(chunks))
    return bytes(header) + bytes(data_blobs) + bytes(list_hdr) + bytes(list_entries)


def patch_vstpreset_component_state(orig_vstpreset: bytes, new_comp_state: bytes) -> bytes:
    """Replace the `Comp` (and `Cont` if JUCE-wrapped) chunk inside an existing Steinberg `.vstpreset`."""
    if len(orig_vstpreset) < 48 or orig_vstpreset[:4] != b"VST3":
        raise ValueError("Original bytes are not a valid Steinberg .vstpreset file.")

    new_cont_state: bytes | None = None
    if b"<VST3PluginState>" in new_comp_state:
        new_comp_state, extracted_cont = extract_steinberg_comp_and_cont(new_comp_state)
        if extracted_cont:
            new_cont_state = extracted_cont

    magic, version, class_id_bytes, list_offset = struct.unpack("<4si32sq", orig_vstpreset[:48])
    if orig_vstpreset[list_offset : list_offset + 4] != b"List":
        raise ValueError("Invalid .vstpreset chunk list marker.")

    (entry_count,) = struct.unpack("<i", orig_vstpreset[list_offset + 4 : list_offset + 8])
    pos = list_offset + 8
    chunks: list[tuple[bytes, bytes]] = []
    comp_replaced = False

    for _ in range(entry_count):
        cid, c_off, c_size = struct.unpack("<4sqq", orig_vstpreset[pos : pos + 20])
        pos += 20
        if cid == b"Comp":
            chunks.append((b"Comp", new_comp_state))
            comp_replaced = True
        elif cid == b"Cont" and new_cont_state is not None:
            chunks.append((b"Cont", new_cont_state))
        else:
            chunks.append((cid, orig_vstpreset[c_off : c_off + c_size]))

    if not comp_replaced:
        chunks.insert(0, (b"Comp", new_comp_state))

    data_blobs = bytearray()
    list_entries = bytearray()
    cursor = 48
    for cid, payload in chunks:
        size = len(payload)
        list_entries.extend(struct.pack("<4sqq", cid, cursor, size))
        data_blobs.extend(payload)
        cursor += size

    new_header = struct.pack("<4si32sq", magic, version, class_id_bytes, cursor)
    new_list_header = struct.pack("<4si", b"List", len(chunks))
    return bytes(new_header) + bytes(data_blobs) + bytes(new_list_header) + bytes(list_entries)


def write_pcm24_wav(path: Path, audio: np.ndarray, sr: int) -> tuple[int, int, float]:
    """Write float32 `(samples,)` or `(channels, samples)` audio as 24-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -0.9999, 0.9999)
    peak = float(np.max(np.abs(clipped))) if clipped.size > 0 else 0.0

    if clipped.ndim == 1:
        sf.write(str(path), clipped, sr, subtype="PCM_24")
        return len(clipped), 1, peak
    else:
        data = clipped.T if clipped.shape[0] <= 2 and clipped.shape[0] < clipped.shape[1] else clipped
        sf.write(str(path), data, sr, subtype="PCM_24")
        return data.shape[0], data.shape[1], peak


def _new_uid() -> str:
    return "{" + str(uuid.uuid4()).upper() + "}"


def _db_to_s1_fader_gain(db: float) -> float:
    return round(float(10.0 ** (db / 20.0)), 6)


def _safe_xml_attr(val: str) -> str:
    return xml_escape(val, {'"': "&quot;", "'": "&apos;"})


def _render_fx_slot_xml(
    slot_index: int,
    spec: S1PluginClassSpec,
    preset_path: str,
    stereo_channel: bool = False,
) -> str:
    """Render an exact Studio One 5 `<Attributes name="FX0N">` insert block for `Devices/audiomixer.xml`."""
    sub_type = "3" if (stereo_channel or spec.stereo_ports) else "1"
    in_lbl = "Stereo In" if sub_type == "3" else "Mono In"
    out_lbl = "Stereo Out" if sub_type == "3" else "Mono Out"
    safe_nm = _safe_xml_attr(spec.name)
    safe_sub = _safe_xml_attr(spec.sub_category)
    safe_pre = _safe_xml_attr(preset_path)

    return (
        f'\t\t\t\t\t<Attributes name="FX{slot_index:02d}">\n'
        f'\t\t\t\t\t\t<UID x:id="uniqueID" uid="{_new_uid()}"/>\n'
        f'\t\t\t\t\t\t<UID x:id="deviceClassID" uid="{spec.class_id}"/>\n'
        f'\t\t\t\t\t\t<Attributes x:id="deviceData" name="{safe_nm}">\n'
        f'\t\t\t\t\t\t\t<UID x:id="uniqueID" uid="{_new_uid()}"/>\n'
        f"\t\t\t\t\t\t</Attributes>\n"
        f'\t\t\t\t\t\t<Attributes x:id="ghostData" presetType="vstpreset">\n'
        f'\t\t\t\t\t\t\t<Attributes x:id="classInfo" classID="{spec.class_id}" name="{safe_nm}"\n'
        f'\t\t\t\t\t\t\t            category="AudioEffect" subCategory="{safe_sub}"/>\n'
        f'\t\t\t\t\t\t\t<Attributes x:id="IO">\n'
        f'\t\t\t\t\t\t\t\t<List x:id="audioInputs">\n'
        f'\t\t\t\t\t\t\t\t\t<PortDescription flags="3" subType="{sub_type}" label="{in_lbl}"/>\n'
        f"\t\t\t\t\t\t\t\t</List>\n"
        f'\t\t\t\t\t\t\t\t<List x:id="audioOutputs">\n'
        f'\t\t\t\t\t\t\t\t\t<PortDescription flags="3" subType="{sub_type}" label="{out_lbl}"/>\n'
        f"\t\t\t\t\t\t\t\t</List>\n"
        f"\t\t\t\t\t\t\t</Attributes>\n"
        f"\t\t\t\t\t\t</Attributes>\n"
        f'\t\t\t\t\t\t<Attributes x:id="Automation" mode="0"/>\n'
        f'\t\t\t\t\t\t<Attributes x:id="Presets" pname="{safe_nm}" dirty="0"/>\n'
        f'\t\t\t\t\t\t<Attributes x:id="SlotRemoteFX" syncActive="0"/>\n'
        f'\t\t\t\t\t\t<String x:id="presetPath" text="{safe_pre}"/>\n'
        f"\t\t\t\t\t</Attributes>"
    )


def _replace_channel_inserts(mixer_xml: str, channel_label: str, slots_xml: list[str], preset_name: str) -> str:
    """Replace the `<Attributes x:id="Inserts">...</Attributes>` block for an exact channel label."""
    if not slots_xml:
        return mixer_xml
    anchor = f'label="{channel_label}"'
    c_start = mixer_xml.find(anchor)
    if c_start == -1:
        return mixer_xml
    ins_s = mixer_xml.find('\n\t\t\t\t<Attributes x:id="Inserts">', c_start)
    ins_e = mixer_xml.find("\n\t\t\t\t</Attributes>\n", ins_s)
    if ins_s == -1 or ins_e == -1:
        return mixer_xml

    new_block = (
        '\t\t\t\t<Attributes x:id="Inserts">\n'
        + "\n".join(slots_xml)
        + f'\n\t\t\t\t\t<Attributes x:id="Presets" pname="{_safe_xml_attr(preset_name)}" dirty="0"/>\n'
        + "\t\t\t\t</Attributes>"
    )
    return mixer_xml[: ins_s + 1] + new_block + mixer_xml[ins_e + len("\n\t\t\t\t</Attributes>") :]


_VST3_DUMP_WORKER_SCRIPT = (
    "import base64, json, sys\n"
    "from pedalboard import load_plugin\n"
    "from mix_automation.dsp.vst3_adapter import (\n"
    "    configure_clean_plate_r2, configure_early_room_phoenix,\n"
    "    configure_persistent_q_5band, configure_persistent_q_harmony_hollow,\n"
    "    configure_persistent_q_precision, configure_slick_eq,\n"
    "    configure_softube_saturation, configure_softube_vca,\n"
    "    configure_soothe2_precision, configure_uad_la2a, configure_uad_pultec,\n"
    "    configure_vocalsynth2_radio, set_vst3_param_strict\n"
    ")\n"
    "spec = json.loads(sys.argv[1])\n"
    "p = load_plugin(spec['path'], plugin_name=spec.get('plugin_name')) if spec.get('plugin_name') else load_plugin(spec['path'])\n"
    "fn = spec.get('helper')\n"
    "kw = spec.get('kwargs', {})\n"
    "if fn == 'pq_prec':\n"
    "    kw['surgical_notches'] = [tuple(x) for x in kw.get('surgical_notches', [])]\n"
    "    configure_persistent_q_precision(p, **kw)\n"
    "elif fn == 'pq_harm': configure_persistent_q_harmony_hollow(p, **kw)\n"
    "elif fn == 'pq_bp':\n"
    "    set_vst3_param_strict(p, 'bypass', False)\n"
    "    set_vst3_param_strict(p, 'mix', 100.0)\n"
    "    set_vst3_param_strict(p, 'band_1_enabled', True)\n"
    "    set_vst3_param_strict(p, 'band_1_type', 'Highpass')\n"
    "    set_vst3_param_strict(p, 'band_1_frequency', 240.0)\n"
    "    set_vst3_param_strict(p, 'band_1_slope', 18.0)\n"
    "    set_vst3_param_strict(p, 'band_2_enabled', True)\n"
    "    set_vst3_param_strict(p, 'band_2_type', 'Lowpass')\n"
    "    set_vst3_param_strict(p, 'band_2_frequency', 4000.0)\n"
    "    set_vst3_param_strict(p, 'band_2_slope', 18.0)\n"
    "elif fn == 'pq': configure_persistent_q_5band(p, **kw)\n"
    "elif fn == 'vca': configure_softube_vca(p, **kw)\n"
    "elif fn == 'la2a': configure_uad_la2a(p, **kw)\n"
    "elif fn == 'pultec': configure_uad_pultec(p, **kw)\n"
    "elif fn == 'slick': configure_slick_eq(p, **kw)\n"
    "elif fn == 'soothe2': configure_soothe2_precision(p, **kw)\n"
    "elif fn == 'sat': configure_softube_saturation(p, **kw)\n"
    "elif fn == 'phoenix': configure_early_room_phoenix(p, **kw)\n"
    "elif fn == 'r2': configure_clean_plate_r2(p, **kw)\n"
    "elif fn == 'vs2_radio': configure_vocalsynth2_radio(p, **kw)\n"
    "else:\n"
    "    for k, v in kw.items():\n"
    "        if k in p.parameters: set_vst3_param_strict(p, k, v)\n"
    "print('STATE_B64:' + base64.b64encode(p.raw_state).decode('ascii'))\n"
)


def _dump_vst3_preset(
    s1_spec: S1PluginClassSpec,
    vst_path: str,
    plugin_name: str | None,
    helper: str | None,
    kwargs: dict,
) -> bytes | None:
    """Load a VST3 plugin in an isolated subprocess and unwrap its state into a native `.vstpreset`."""
    if not Path(vst_path).exists():
        return None
    payload = json.dumps({"path": vst_path, "plugin_name": plugin_name, "helper": helper, "kwargs": kwargs})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _VST3_DUMP_WORKER_SCRIPT, payload],
            check=False,
            capture_output=True,
            text=True,
            timeout=25,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("STATE_B64:"):
                juce_blob = base64.b64decode(line[len("STATE_B64:") :])
                comp_bytes, cont_bytes = extract_steinberg_comp_and_cont(juce_blob)
                return create_vstpreset_bytes(s1_spec.class_id, comp_bytes, cont_bytes)
    except Exception:
        return None
    return None


def export_template_vst3_states(params: VocalChainParams, bpm: float) -> dict[str, bytes]:
    """Load every real commercial VST3 plugin in isolated subprocesses and synthesize native Steinberg `.vstpreset`s."""
    pmap = get_plugin_map()
    s1_specs = load_s1_plugin_specs()
    waves_shell = str(VST3_ROOT / "WaveShell1-VST3 14.12_x64.vst3")
    presets: dict[str, bytes] = {}

    def _dump_preset(
        spec_key: str,
        vst_path: str,
        plugin_name: str | None,
        helper: str | None,
        kwargs: dict,
    ) -> bytes | None:
        return _dump_vst3_preset(s1_specs[spec_key], vst_path, plugin_name, helper, kwargs)

    # --- 1. Main Vocal True 1:1 v6 Ultra-Precision VST3 Chain ---
    pq_main = _dump_preset(
        "persistent_q",
        str(pmap["persistent_q"].loadable_path),
        None,
        "pq_prec",
        {
            "hpf_hz": float(np.clip(params.hpf_hz, 60.0, 160.0)),
            "band_gains_db": dict(params.precision_eq_gains_db),
            "surgical_notches": list(params.surgical_notches),
            "dynamic_mud_thresh_db": float(params.dynamic_mud_thresh_db),
            "dynamic_mud_range_db": float(params.dynamic_mud_range_db),
            "dynamic_bite_hz": float(params.dynamic_bite_hz),
            "dynamic_bite_thresh_db": float(params.dynamic_bite_thresh_db),
            "dynamic_bite_range_db": float(params.dynamic_bite_range_db),
            "spectral_hf_hz": float(params.spectral_hf_hz),
            "spectral_hf_thresh_db": float(params.spectral_hf_thresh_db),
            "spectral_hf_range_db": float(params.spectral_hf_range_db),
            "enable_dynamic_bands": True,
        },
    )
    if pq_main:
        presets["Presets/Channels/mainvocal/1 - Persistent Q.vstpreset"] = pq_main

    vca_main = _dump_preset(
        "vca_compressor",
        str(pmap["vca_compressor"].loadable_path),
        None,
        "vca",
        {
            "threshold_db": float(params.vca_threshold_db),
            "ratio": 3.5 if params.archetype_chain == "dense_rock_edm" else 3.0,
            "attack_norm": float(params.vca_attack_norm),
            "release_norm": float(params.vca_release_norm),
            "sc_hpf_hz": 120.0,
            "output_db": float(np.clip(params.vca_measured_gr_db, 0.0, 7.0)),
            "dry_wet": 100.0,
            "output_drive": float(params.vca_output_drive),
        },
    )
    if vca_main:
        presets["Presets/Channels/mainvocal/2 - VCA Compressor.vstpreset"] = vca_main

    la2a_main = _dump_preset(
        "uad_la2a",
        str(pmap["uad_la2a"].loadable_path),
        None,
        "la2a",
        {"peak_reduct": float(np.clip(params.comp2_param, 18.0, 80.0)), "gain": float(np.clip(params.la2a_gain_knob, 20.0, 72.0))},
    )
    if la2a_main:
        presets["Presets/Channels/mainvocal/3 - UADx LA-2A Tube Compressor.vstpreset"] = la2a_main

    next_slot = 4
    if params.archetype_chain == "intimate_ballad_sparse":
        slick = _dump_preset(
            "slick_eq",
            str(pmap["slick_eq"].loadable_path),
            None,
            "slick",
            {"out_stage": "Silky", "drive_db": 3.2},
        )
        if slick:
            presets[f"Presets/Channels/mainvocal/{next_slot} - TDR VOS SlickEQ.vstpreset"] = slick
            next_slot += 1
    elif params.archetype_chain == "groovy_mid_pop":
        pultec = _dump_preset(
            "uad_pultec",
            str(pmap["uad_pultec_eqp1a"].loadable_path),
            None,
            "pultec",
            {"lf_boost": 0.45, "hf_boost": 0.0, "hf_freq": "12 KCS"},
        )
        if pultec:
            presets[f"Presets/Channels/mainvocal/{next_slot} - UADx Pultec EQP-1A EQ.vstpreset"] = pultec
            next_slot += 1

    if pmap["soothe2"].exists and params.soothe2_depth > 0.15:
        s2_main = _dump_preset(
            "soothe2",
            str(pmap["soothe2"].loadable_path),
            None,
            "soothe2",
            {
                "depth": float(params.soothe2_depth),
                "bite_sens_db": float(params.soothe2_bite_sens),
                "sibilance_sens_db": float(params.soothe2_sib_sens),
                "hf_sizzle_sens_db": float(params.soothe2_hf_sens),
            },
        )
        if s2_main:
            presets[f"Presets/Channels/mainvocal/{next_slot} - soothe2.vstpreset"] = s2_main

    # --- 2. Main Harmony Channels (`harmvocal` / `harmvocal2`): Hollow-Core EQ + Onset-Crushing Comp! ---
    pq_harm = _dump_preset(
        "persistent_q",
        str(pmap["persistent_q"].loadable_path),
        None,
        "pq_harm",
        {"lead_core_hz": float(params.lead_core_formant_hz), "hpf_hz": 240.0, "lpf_hz": 8500.0},
    )
    if pq_harm:
        presets["Presets/Channels/harmvocal/1 - Persistent Q.vstpreset"] = pq_harm
        presets["Presets/Channels/harmvocal2/1 - Persistent Q.vstpreset"] = pq_harm

    vca_harm = _dump_preset(
        "vca_compressor",
        str(pmap["vca_compressor"].loadable_path),
        None,
        "vca",
        {
            "threshold_db": -27.0,
            "ratio": 6.0,
            "attack_norm": 0.25,   # Ultra-fast onset crush ("語頭をつぶす")!
            "release_norm": 3.40,  # Slow smooth release keeping harmony glued behind lead!
            "sc_hpf_hz": 160.0,
            "output_db": 4.2,
            "dry_wet": 100.0,
            "output_drive": 0.0,
        },
    )
    if vca_harm:
        presets["Presets/Channels/harmvocal/2 - VCA Compressor.vstpreset"] = vca_harm
        presets["Presets/Channels/harmvocal2/2 - VCA Compressor.vstpreset"] = vca_harm

    # --- 3. FX Channels (exact Studio One labels + Midrange-Only Bandpass before Saturation Knob!) ---
    hd_pre = _dump_preset(
        "h_delay",
        waves_shell,
        "H-Delay Stereo",
        None,
        {
            "mix": 100.0,
            "bpm": float(np.clip(bpm, 40.0, 240.0)),
            "feedback": 20.0,
            "hipass": 450.0,
            "lopass": 3800.0,
        },
    )
    if hd_pre:
        presets["Presets/Channels/H-Delay Stereo/1 - H-Delay Stereo.vstpreset"] = hd_pre

    db2_pre = _dump_preset("doubler2", waves_shell, "Doubler2 Stereo", None, {"direct_onoff": False})
    if db2_pre:
        presets["Presets/Channels/Doubler2 Stereo/1 - Doubler2 Stereo.vstpreset"] = db2_pre

    r_time = 1.25 if params.archetype_chain == "intimate_ballad_sparse" else 0.94
    predly = round(float(np.clip(60000.0 / max(bpm, 60.0) / 16.0, 18.0, 36.0)), 1)
    r2_pre = _dump_preset(
        "r2",
        str(pmap["exponential_r2"].loadable_path),
        None,
        "r2",
        {"rvb_time": r_time, "predelay_ms": predly, "lpf_hz": 5400.0},
    )
    if r2_pre:
        presets["Presets/Channels/R2/1 - R2.vstpreset"] = r2_pre

    if pmap["phoenix_verb"].exists:
        ph_pre = _dump_preset(
            "phoenix_verb",
            str(pmap["phoenix_verb"].loadable_path),
            None,
            "phoenix",
            {"size_m": 8.0, "rvb_time": 0.44, "predelay_ms": 10.0},
        )
        if ph_pre:
            presets["Presets/Channels/Phoenix EarlyRoom/1 - PhoenixVerb.vstpreset"] = ph_pre

    if pmap["saturation_knob"].exists:
        pq_sat_bp = _dump_preset("persistent_q", str(pmap["persistent_q"].loadable_path), None, "pq_bp", {})
        if pq_sat_bp:
            presets["Presets/Channels/Saturation Parallel/1 - Persistent Q.vstpreset"] = pq_sat_bp
        sat_pre = _dump_preset(
            "saturation_knob",
            str(pmap["saturation_knob"].loadable_path),
            None,
            "sat",
            {"amount": float(params.parallel_sat_amount), "sat_type": "KEEP LOW"},
        )
        if sat_pre:
            presets["Presets/Channels/Saturation Parallel/2 - Saturation Knob.vstpreset"] = sat_pre

    # --- 4. Master Output (`メイン`) 1:1 Bus Glue & True-Peak Limiter ---
    vca_master = _dump_preset(
        "vca_compressor",
        str(pmap["vca_compressor"].loadable_path),
        None,
        "vca",
        {
            "threshold_db": -16.0,
            "ratio": 1.6,
            "attack_norm": 2.10,
            "release_norm": 0.70,
            "sc_hpf_hz": 110.0,
            "output_db": 1.0,
            "dry_wet": 100.0,
            "output_drive": 0.2,
        },
    )
    if vca_master:
        presets["Presets/Channels/メイン/1 - VCA Compressor.vstpreset"] = vca_master

    if pmap["wave_breaker"].exists:
        wb_master = _dump_preset(
            "wave_breaker",
            str(pmap["wave_breaker"].loadable_path),
            None,
            None,
            {"threshold_db": -0.8},
        )
        if wb_master:
            presets["Presets/Channels/メイン/2 - Wave Breaker.vstpreset"] = wb_master

    return presets


def export_stage3_fx_vst3_states(
    active_fx_types: set[str],
    bpm: float = 120.0,
    scale_mask: list[bool] | None = None,
) -> dict[str, bytes]:
    """Export native Steinberg `.vstpreset` binaries for active Stage 3 spot-FX tracks."""
    active = {str(x).strip().lower() for x in active_fx_types}
    if not active:
        return {}

    pmap = get_plugin_map()
    s1_specs = load_s1_plugin_specs()
    presets: dict[str, bytes] = {}

    if "radio_voice" in active and pmap["izotope_vocalsynth2"].exists:
        vs_pre = _dump_vst3_preset(
            s1_specs["vocalsynth2"],
            str(pmap["izotope_vocalsynth2"].loadable_path),
            None,
            "vs2_radio",
            {"freq_hz": 1650.0, "drive": 3.8, "distort_mix": 42.0},
        )
        if vs_pre:
            presets["Presets/Channels/fx_radio_vocalsynth2/1 - VocalSynth 2.vstpreset"] = vs_pre

    if "hard_autotune" in active and pmap["mautopitch"].exists:
        mp_kw: dict[str, object] = {}
        if scale_mask is not None and len(scale_mask) >= 12:
            note_keys = ("c", "c_sharp", "d", "d_sharp", "e", "f", "f_sharp", "g", "g_sharp", "a", "a_sharp", "b")
            for idx_n, nk in enumerate(note_keys):
                mp_kw[nk] = bool(scale_mask[idx_n])
        mp_kw.update(
            {
                "depth_automatic_tuning": 100.0,
                "speed_automatic_tuning": 95.0,
                "keep_formants_effects": 100.0,
                "dry_wet_effects": "100% wet, 0% dry",
            }
        )
        mp_pre = _dump_vst3_preset(
            s1_specs["mautopitch"],
            str(pmap["mautopitch"].loadable_path),
            None,
            None,
            mp_kw,
        )
        if mp_pre:
            presets["Presets/Channels/fx_autotune_mautopitch/1 - MAutoPitch.vstpreset"] = mp_pre

    if "distortion_grit" in active and pmap["izotope_trash2"].exists:
        t2_pre = _dump_vst3_preset(
            s1_specs["izotope_trash2"],
            str(pmap["izotope_trash2"].loadable_path),
            None,
            None,
            {
                "global_bypass": False,
                "trash_module_bypass": False,
                "trash_bypass": False,
                "trash_stg1_enable": True,
                "trash_stg1_drive": 3.6,
                "trash_stg1_filter_enable": True,
                "trash_stg1_filter_low_freq": 380.0,
                "trash_stg1_filter_high_freq": 4200.0,
                "global_mix": 55.0,
            },
        )
        if t2_pre:
            presets["Presets/Channels/fx_distortion_trash2/1 - iZotope Trash 2.vstpreset"] = t2_pre

    if "spot_delay_throw" in active and pmap["khs_dual_delay"].exists:
        khs_pre = _dump_vst3_preset(
            s1_specs["khs_dual_delay"],
            str(pmap["khs_dual_delay"].loadable_path),
            None,
            None,
            {
                "bypass": False,
                "feedback": 36.0,
                "spread": 75.0,
                "duck": 25.0,
                "mix": 100.0,
                "sync_mode": "sync",
                "delay_time_steps": 1.0,
                "delay_denominator": "8",
            },
        )
        if khs_pre:
            presets["Presets/Channels/fx_spot_delay/1 - kHs Dual Delay.vstpreset"] = khs_pre

    return presets


def build_studio_one_song(
    template_song_path: Path,
    output_song_dir: Path,
    song_name: str,
    bpm: float,
    sr: int,
    main_vocal_wav: Path,
    sections: list[SongSection],
    extra_tracks: list[TrackMediaSpec],
    harm1_wav: Path | None = None,
    harm2_wav: Path | None = None,
    patched_presets: dict[str, bytes] | None = None,
    chain_params: VocalChainParams | None = None,
    review_markers: list[ReviewMarker] | None = None,
    harmony_fader_db: float = -8.6,
    fine_main_events: list[tuple[float, float, float, str]] | None = None,
    tempo_segments: list[tuple[float, float]] | None = None,
) -> Path:
    """Build a complete Studio One 5 `.song` file with native Steinberg `.vstpreset`s & v6 exact console parity."""
    output_song_dir.mkdir(parents=True, exist_ok=True)
    output_song_path = output_song_dir / f"{song_name}.song"
    patched_presets = patched_presets or {}
    review_markers = review_markers or []
    s1_specs = load_s1_plugin_specs()

    norm_tempo_segs: list[tuple[float, float]] = []
    if tempo_segments:
        sorted_segs = sorted((max(0.0, float(b)), max(40.0, float(tb))) for b, tb in tempo_segments)
        if sorted_segs[0][0] > 1e-6:
            norm_tempo_segs.append((0.0, max(float(bpm), 40.0)))
        for b_pos, seg_bpm in sorted_segs:
            if norm_tempo_segs and abs(b_pos - norm_tempo_segs[-1][0]) <= 1e-6:
                norm_tempo_segs[-1] = (norm_tempo_segs[-1][0], seg_bpm)
            else:
                norm_tempo_segs.append((b_pos, seg_bpm))
    if not norm_tempo_segs:
        norm_tempo_segs = [(0.0, max(float(bpm), 40.0))]

    init_bpm = norm_tempo_segs[0][1]
    sec_per_beat = 60.0 / init_bpm

    def _sec_to_beat(t_sec: float) -> float:
        if t_sec <= 0.0:
            return 0.0
        rem = float(t_sec)
        for idx_seg, (b0, seg_bpm) in enumerate(norm_tempo_segs):
            spb = 60.0 / seg_bpm
            if idx_seg + 1 < len(norm_tempo_segs):
                b1 = norm_tempo_segs[idx_seg + 1][0]
                seg_dur = max(0.0, (b1 - b0) * spb)
                if rem <= seg_dur:
                    return b0 + rem / spb
                rem -= seg_dur
            else:
                return b0 + rem / spb
        return t_sec / sec_per_beat

    with zipfile.ZipFile(template_song_path, "r") as zin:
        template_files = {name: zin.read(name) for name in zin.namelist()}

    ET.register_namespace("x", "http://www.presonus.com/xml")
    song_xml_str = template_files["Song/song.xml"].decode("utf-8").replace("\r\n", "\n")
    pool_xml_str = template_files["Song/mediapool.xml"].decode("utf-8").replace("\r\n", "\n")
    mixer_xml_str = template_files["Devices/audiomixer.xml"].decode("utf-8").replace("\r\n", "\n")
    console_xml_str = template_files.get("Devices/mixerconsole.xml", b"").decode("utf-8").replace("\r\n", "\n")

    now_str = datetime.now().strftime("%Y/%m/%d %H:%M:%S.000")
    clip_records: list[dict] = []

    def _register_wav(key: str, wav_file: Path) -> dict:
        info = sf.info(str(wav_file))
        data, _ = sf.read(str(wav_file), dtype="float32", always_2d=True)
        peak = float(np.max(np.abs(data))) if data.size > 0 else 0.01
        rec = {
            "key": key,
            "media_id": _new_uid(),
            "url": wav_file.resolve().as_uri(),
            "frames": info.frames,
            "sr": info.samplerate,
            "channels": info.channels,
            "peak": peak,
            "duration_sec": info.frames / info.samplerate,
        }
        clip_records.append(rec)
        return rec

    main_clip = _register_wav("mainvocal", main_vocal_wav)
    harm1_clip = _register_wav("harmvocal", harm1_wav) if harm1_wav and harm1_wav.exists() else None
    harm2_clip = _register_wav("harmvocal2", harm2_wav) if harm2_wav and harm2_wav.exists() else None
    extra_clips = [(spec, _register_wav(spec.name, spec.wav_path)) for spec in extra_tracks if spec.wav_path.exists()]

    max_dur_sec = max((r["duration_sec"] for r in clip_records), default=240.0) + 12.0
    max_beats = _sec_to_beat(max_dur_sec)

    main_events_xml: list[str] = []
    if fine_main_events:
        for s_raw, e_raw, cg_db, ev_label in fine_main_events:
            s_sec = max(0.0, min(float(s_raw), main_clip["duration_sec"]))
            e_sec = max(s_sec, min(float(e_raw), main_clip["duration_sec"]))
            if e_sec - s_sec < 0.05:
                continue
            start_beats = _sec_to_beat(s_sec)
            len_beats = max(0.01, _sec_to_beat(e_sec) - start_beats)
            offset_beats = s_sec / sec_per_beat
            lin_level = _db_to_s1_fader_gain(float(cg_db))
            ev_name = _safe_xml_attr(str(ev_label))
            main_events_xml.append(
                f'\t\t\t\t\t<AudioEvent clipID="{main_clip["media_id"]}" timeFormat="2" '
                f'start="{start_beats:.8f}" offset="{offset_beats:.8f}" length="{len_beats:.8f}" '
                f'name="{ev_name}" speed="1" transpose="0" tune="0" normalize="0" modification="0">\n'
                f'\t\t\t\t\t\t<Attributes x:id="volumeCurve" level="{lin_level:.8f}"/>\n'
                f'\t\t\t\t\t</AudioEvent>'
            )
    elif sections:
        for sec in sections:
            s_sec = max(0.0, min(sec.start_sec, main_clip["duration_sec"]))
            e_sec = max(s_sec, min(sec.end_sec, main_clip["duration_sec"]))
            if e_sec - s_sec < 0.05:
                continue
            start_beats = _sec_to_beat(s_sec)
            len_beats = max(0.01, _sec_to_beat(e_sec) - start_beats)
            offset_beats = s_sec / sec_per_beat
            lin_level = _db_to_s1_fader_gain(sec.clip_gain_db)
            ev_name = _safe_xml_attr(f"main_{sec.label}")
            main_events_xml.append(
                f'\t\t\t\t\t<AudioEvent clipID="{main_clip["media_id"]}" timeFormat="2" '
                f'start="{start_beats:.8f}" offset="{offset_beats:.8f}" length="{len_beats:.8f}" '
                f'name="{ev_name}" speed="1" transpose="0" tune="0" normalize="0" modification="0">\n'
                f'\t\t\t\t\t\t<Attributes x:id="volumeCurve" level="{lin_level:.8f}"/>\n'
                f'\t\t\t\t\t</AudioEvent>'
            )
    if not main_events_xml:
        len_beats = _sec_to_beat(main_clip["duration_sec"])
        main_events_xml.append(
            f'\t\t\t\t\t<AudioEvent clipID="{main_clip["media_id"]}" timeFormat="2" '
            f'length="{len_beats:.8f}" name="mainvocal" speed="1" transpose="0" tune="0" normalize="0" modification="0">\n'
            f'\t\t\t\t\t\t<Attributes x:id="volumeCurve" level="1"/>\n'
            f'\t\t\t\t\t</AudioEvent>'
        )

    clip_xml_blocks: list[str] = []
    for rec in clip_records:
        use_cnt = max(1, len(main_events_xml)) if rec["key"] == "mainvocal" else 1
        clip_xml_blocks.append(
            f'\t\t\t<AudioClip mediaID="{rec["media_id"]}" useCount="{use_cnt}" peak="{rec["peak"]:.10f}">\n'
            f'\t\t\t\t<Url x:id="path" type="1" url="{rec["url"]}"/>\n'
            f'\t\t\t\t<DateTime x:id="recordTime" time="{now_str}"/>\n'
            f'\t\t\t\t<AudioTempoMap x:id="tempoMap" tempoApproved="1">\n'
            f'\t\t\t\t\t<AudioTempoMapSegment offset="0" tempo="{sec_per_beat:.16f}"/>\n'
            f'\t\t\t\t</AudioTempoMap>\n'
            f'\t\t\t\t<Attributes x:id="format" frameCount="{rec["frames"]}" sampleRate="{rec["sr"]}" '
            f'numChannels="{rec["channels"]}" bitDepth="24"/>\n'
            f'\t\t\t\t<List x:id="modifications">\n'
            f'\t\t\t\t\t<AudioModification id="0">\n'
            f'\t\t\t\t\t\t<List x:id="propertySets">\n'
            f'\t\t\t\t\t\t\t<AudioBendProperties threshold="0.2" visible="0"/>\n'
            f'\t\t\t\t\t\t\t<AudioChordProperties tuningFrequency="440"/>\n'
            f'\t\t\t\t\t\t</List>\n'
            f'\t\t\t\t\t</AudioModification>\n'
            f'\t\t\t\t</List>\n'
            f'\t\t\t</AudioClip>'
        )

    media_folder_xml = (
        '\t<Attributes x:id="rootFolder">\n\t\t<MediaFolder name="Audio">\n'
        + "\n".join(clip_xml_blocks)
        + "\n\t\t</MediaFolder>\n\t</Attributes>"
    )
    pool_xml_out = pool_xml_str.replace('<Attributes x:id="rootFolder"/>', media_folder_xml)
    pool_xml_out = pool_xml_out.replace(
        'url="file:///C:/MIX_automation/data/templates/test20261124/test20261124.song"',
        f'url="{output_song_path.resolve().as_uri()}"',
    )

    def _single_event_list_xml(rec: dict, name: str) -> str:
        lb = _sec_to_beat(rec["duration_sec"])
        safe_n = _safe_xml_attr(name)
        return (
            f'\t\t\t\t<List x:id="Events">\n'
            f'\t\t\t\t\t<AudioEvent clipID="{rec["media_id"]}" timeFormat="2" length="{lb:.8f}" '
            f'name="{safe_n}" speed="1" transpose="0" tune="0" normalize="0" modification="0">\n'
            f'\t\t\t\t\t\t<Attributes x:id="volumeCurve" level="1"/>\n'
            f'\t\t\t\t\t</AudioEvent>\n'
            f'\t\t\t\t</List>'
        )

    env_points_xml: list[str] = []
    seg_blocks_xml: list[str] = []
    for idx_seg, (b0, seg_bpm) in enumerate(norm_tempo_segs):
        t_norm = (seg_bpm - 10.0) / 390.0
        spb = 60.0 / seg_bpm
        if idx_seg == 0:
            env_points_xml.append(f'\t\t\t\t\t<EnvelopePoint value="{t_norm:.16f}"/>')
            s_attr = "0"
        else:
            env_points_xml.append(f'\t\t\t\t\t<EnvelopePoint time="{b0:.6f}" value="{t_norm:.16f}"/>')
            s_attr = f"{b0:.6f}"
        e_attr = (
            f"{norm_tempo_segs[idx_seg + 1][0]:.6f}"
            if idx_seg + 1 < len(norm_tempo_segs)
            else "1.7976931348623156386368421785227527018228425601508e+200"
        )
        seg_blocks_xml.append(
            f'\t\t\t\t<TempoMapSegment curveType="0" start="{s_attr}" end="{e_attr}"\n'
            f'\t\t\t\t                 tempo="{spb:.16f}"/>'
        )
    tempo_map_xml = (
        '\t\t\t<TempoMap x:id="tempoMap">\n'
        '\t\t\t\t<Attributes x:id="envelope">\n'
        + "\n".join(env_points_xml)
        + "\n\t\t\t\t</Attributes>\n"
        + "\n".join(seg_blocks_xml)
        + "\n\t\t\t</TempoMap>"
    )

    song_xml_out = song_xml_str
    song_xml_out = song_xml_out.replace('sampleRate="44100"', f'sampleRate="{sr}"')
    song_xml_out = song_xml_out.replace('length="300"', f'length="{max_dur_sec:.2f}"')
    song_xml_out = re.sub(
        r'\t\t\t<TempoMap x:id="tempoMap">.*?</TempoMap>',
        tempo_map_xml,
        song_xml_out,
        count=1,
        flags=re.S,
    )
    song_xml_out = song_xml_out.replace(
        'start="730.0000000000001136868377216160297393798828125"',
        f'start="{max_beats:.4f}"',
    )

    all_marker_lines: list[str] = []
    for sec in sections:
        if sec.start_sec > 0.5:
            b_pos = _sec_to_beat(sec.start_sec)
            tag = f"[{sec.label}]" + (" ★Anchor" if sec.is_anchor else "")
            all_marker_lines.append(
                f'\t\t\t\t<MarkerEvent start="{b_pos:.6f}" timeFormat="2" name="{_safe_xml_attr(tag)}"/>'
            )
    for rm in review_markers:
        b_pos = max(0.05, _sec_to_beat(rm.time_sec))
        all_marker_lines.append(
            f'\t\t\t\t<MarkerEvent start="{b_pos:.6f}" timeFormat="2" name="{_safe_xml_attr(rm.name)}"/>'
        )

    if all_marker_lines:
        song_xml_out = song_xml_out.replace(
            '<MarkerTrack version="1" timeFormat="2">\n\t\t\t\t<Attributes x:id="attributes" hidden="1"/>',
            '<MarkerTrack version="1" timeFormat="2">\n\t\t\t\t<Attributes x:id="attributes" hidden="0"/>',
        )
        start_marker_tag = '<MarkerEvent markerType="2" timeFormat="2" name="スタート"/>'
        song_xml_out = song_xml_out.replace(
            start_marker_tag,
            start_marker_tag + "\n" + "\n".join(all_marker_lines),
        )

    main_ev_block = '\t\t\t\t<List x:id="Events">\n' + "\n".join(main_events_xml) + "\n\t\t\t\t</List>"
    target_main_close = (
        '<Url x:id="points" type="1" url="media:///Envelopes/mainvocal/パン.envelopex"/>\n'
        "\t\t\t\t\t</AutomationRegion>\n\t\t\t\t</Attributes>"
    )
    song_xml_out = song_xml_out.replace(target_main_close, target_main_close + "\n" + main_ev_block)

    if harm1_clip is not None:
        t_h1 = (
            '<Url x:id="points" type="1" url="media:///Envelopes/harmvocal/パン.envelopex"/>\n'
            "\t\t\t\t\t</AutomationRegion>\n\t\t\t\t</Attributes>"
        )
        song_xml_out = song_xml_out.replace(t_h1, t_h1 + "\n" + _single_event_list_xml(harm1_clip, "harmvocal"))

    if harm2_clip is not None:
        t_h2 = (
            '<Url x:id="points" type="1" url="media:///Envelopes/harmvocal2/パン.envelopex"/>\n'
            "\t\t\t\t\t</AutomationRegion>\n\t\t\t\t</Attributes>"
        )
        song_xml_out = song_xml_out.replace(t_h2, t_h2 + "\n" + _single_event_list_xml(harm2_clip, "harmvocal2"))

    for spec in extra_tracks:
        template_files[f"Envelopes/{spec.name}/ボリューム.envelopex"] = b"{i\x06events[]}"
        template_files[f"Envelopes/{spec.name}/パン.envelopex"] = b"{i\x07bipolari\x01i\x06events[]}"

    extra_media_tracks_xml: list[str] = []
    extra_mixer_channels_xml: list[str] = []
    new_console_uids: list[str] = []

    for idx_e, (spec, rec) in enumerate(extra_clips, start=4):
        track_uid = _new_uid()
        chan_uid = _new_uid()
        new_console_uids.append(chan_uid)
        spk_type = "Stereo" if spec.stereo else "Mono"
        mute_flag = "1" if spec.mute else "0"
        fader_lin = _db_to_s1_fader_gain(spec.gain_db)
        safe_spec_name = _safe_xml_attr(spec.name)

        extra_media_tracks_xml.append(
            f'\t\t\t<MediaTrack mediaType="Audio" tempoFollow="2" materialMode="0" layersFollow="1" '
            f'version="1" trackID="{track_uid}" timeFormat="2" name="{safe_spec_name}" color="{spec.color}">\n'
            f'\t\t\t\t<SpeakerSetup x:id="trackFormat" type="{spk_type}"/>\n'
            f'\t\t\t\t<UID x:id="channelID" uid="{chan_uid}"/>\n'
            f'\t\t\t\t<Attributes x:id="AutomationRegionList" visible="0">\n'
            f'\t\t\t\t\t<AutomationRegion title="ボリューム" identity="param:///AudioMixer/{chan_uid}/volume" '
            f'metaIdentity="AudioTrackChannel/volume" color="FFFFD829" mute="0">\n'
            f'\t\t\t\t\t\t<Url x:id="points" type="1" url="media:///Envelopes/{safe_spec_name}/ボリューム.envelopex"/>\n'
            f'\t\t\t\t\t</AutomationRegion>\n'
            f'\t\t\t\t\t<AutomationRegion title="パン" identity="param:///AudioMixer/{chan_uid}/pan" '
            f'metaIdentity="AudioTrackChannel/pan" color="FFC7FF3A" mute="0">\n'
            f'\t\t\t\t\t\t<Url x:id="points" type="1" url="media:///Envelopes/{safe_spec_name}/パン.envelopex"/>\n'
            f'\t\t\t\t\t</AutomationRegion>\n'
            f'\t\t\t\t</Attributes>\n'
            f"{_single_event_list_xml(rec, spec.name)}\n"
            f"\t\t\t</MediaTrack>"
        )

        extra_slots: list[str] = []
        for s_key, p_rel in spec.insert_slots:
            if p_rel in patched_presets and s_key in s1_specs:
                extra_slots.append(
                    _render_fx_slot_xml(len(extra_slots) + 1, s1_specs[s_key], p_rel, stereo_channel=spec.stereo)
                )
        if extra_slots:
            inserts_block = (
                '\t\t\t\t<Attributes x:id="Inserts">\n'
                + "\n".join(extra_slots)
                + '\n\t\t\t\t\t<Attributes x:id="Presets" pname="AutoMIX Stage3 FX" dirty="0"/>\n'
                + "\t\t\t\t</Attributes>"
            )
        else:
            inserts_block = '\t\t\t\t<Attributes x:id="Inserts"><Attributes x:id="Presets" pname="デフォルト" dirty="0"/></Attributes>'

        extra_mixer_channels_xml.append(
            f'\t\t\t<AudioTrackChannel gain="{fader_lin}" pan="{spec.pan}" name="Channel{idx_e:02d}" '
            f'label="{safe_spec_name}" mute="{mute_flag}" solo="0" soloSafe="0" color="{spec.color}">\n'
            f'\t\t\t\t<SpeakerSetup x:id="speakerType" type="{spk_type}"/>\n'
            f'\t\t\t\t<UID x:id="uniqueID" uid="{chan_uid}"/>\n'
            f'\t\t\t\t<Connection x:id="destination" objectID="{{636828DD-7358-4765-B2A2-106A2780EE51}}/Input" friendlyName="メイン"/>\n'
            f'\t\t\t\t<Attributes x:id="Automation" mode="0"/>\n'
            f"{inserts_block}\n"
            f'\t\t\t\t<Attributes x:id="InputFX" gain="0" invertPhaseL="0" invertPhaseR="0"/>\n'
            f'\t\t\t\t<Attributes x:id="Sends"><Attributes x:id="Presets" pname="デフォルト" dirty="0"/></Attributes>\n'
            f"\t\t\t</AudioTrackChannel>"
        )

    if extra_media_tracks_xml:
        song_xml_out = song_xml_out.replace(
            "\t\t</List>\n\t</Attributes>\n</Song>",
            "\n".join(extra_media_tracks_xml) + "\n\t\t</List>\n\t</Attributes>\n</Song>",
        )

    mixer_xml_out = mixer_xml_str

    if chain_params is not None:
        # 1. `mainvocal` Inserts: Exact v6 chain (16-Band PQ -> Fast VCA -> LA-2A -> Color -> Post-Soothe2!)
        main_fx_slots: list[str] = []
        slot_candidates: list[tuple[str, str]] = [
            ("persistent_q", "Presets/Channels/mainvocal/1 - Persistent Q.vstpreset"),
            ("vca_compressor", "Presets/Channels/mainvocal/2 - VCA Compressor.vstpreset"),
            ("uad_la2a", "Presets/Channels/mainvocal/3 - UADx LA-2A Tube Compressor.vstpreset"),
            ("slick_eq", "Presets/Channels/mainvocal/4 - TDR VOS SlickEQ.vstpreset"),
            ("uad_pultec", "Presets/Channels/mainvocal/4 - UADx Pultec EQP-1A EQ.vstpreset"),
            ("soothe2", "Presets/Channels/mainvocal/4 - soothe2.vstpreset"),
            ("soothe2", "Presets/Channels/mainvocal/5 - soothe2.vstpreset"),
        ]
        s_idx = 1
        for s_key, p_rel in slot_candidates:
            if p_rel in patched_presets:
                main_fx_slots.append(_render_fx_slot_xml(s_idx, s1_specs[s_key], p_rel, stereo_channel=False))
                s_idx += 1
        mixer_xml_out = _replace_channel_inserts(mixer_xml_out, "mainvocal", main_fx_slots, "AutoMIX v6 Main")

        # 2. `harmvocal` & `harmvocal2` Inserts: Hollow-Core PQ + Onset-Crushing VCA!
        for h_label in ("harmvocal", "harmvocal2"):
            h_slots: list[str] = []
            p_pq = f"Presets/Channels/{h_label}/1 - Persistent Q.vstpreset"
            p_vca = f"Presets/Channels/{h_label}/2 - VCA Compressor.vstpreset"
            if p_pq in patched_presets:
                h_slots.append(_render_fx_slot_xml(1, s1_specs["persistent_q"], p_pq, stereo_channel=False))
            if p_vca in patched_presets:
                h_slots.append(_render_fx_slot_xml(2, s1_specs["vca_compressor"], p_vca, stereo_channel=False))
            mixer_xml_out = _replace_channel_inserts(mixer_xml_out, h_label, h_slots, "AutoMIX v6 HollowHarmony")

        # 3. `メイン` Master Output Inserts
        master_slots: list[str] = []
        if "Presets/Channels/メイン/1 - VCA Compressor.vstpreset" in patched_presets:
            master_slots.append(
                _render_fx_slot_xml(1, s1_specs["vca_compressor"], "Presets/Channels/メイン/1 - VCA Compressor.vstpreset", True)
            )
        if "Presets/Channels/メイン/2 - Wave Breaker.vstpreset" in patched_presets:
            master_slots.append(
                _render_fx_slot_xml(2, s1_specs["wave_breaker"], "Presets/Channels/メイン/2 - Wave Breaker.vstpreset", True)
            )
        mixer_xml_out = _replace_channel_inserts(mixer_xml_out, "メイン", master_slots, "Master Glue")

        # 4. Dedicated FX Channels (`Saturation Parallel` with Mid-Only Bandpass + `Phoenix EarlyRoom`)
        extra_fx_channels: list[str] = []
        extra_sends_for_main: list[str] = []
        fx_ch_num = 4
        send_num = 4
        ph_uid: str | None = None

        sat_slots: list[str] = []
        if "Presets/Channels/Saturation Parallel/1 - Persistent Q.vstpreset" in patched_presets:
            sat_slots.append(
                _render_fx_slot_xml(
                    1, s1_specs["persistent_q"], "Presets/Channels/Saturation Parallel/1 - Persistent Q.vstpreset", True
                )
            )
        sat_knob_path = (
            "Presets/Channels/Saturation Parallel/2 - Saturation Knob.vstpreset"
            if "Presets/Channels/Saturation Parallel/2 - Saturation Knob.vstpreset" in patched_presets
            else "Presets/Channels/Saturation Parallel/1 - Saturation Knob.vstpreset"
        )
        if sat_knob_path in patched_presets:
            sat_slots.append(_render_fx_slot_xml(len(sat_slots) + 1, s1_specs["saturation_knob"], sat_knob_path, True))

        if sat_slots:
            sat_uid = _new_uid()
            new_console_uids.append(sat_uid)
            extra_fx_channels.append(
                f'\t\t\t<AudioEffectChannel gain="1" pan="0.5" name="Channel{fx_ch_num:02d}" label="Saturation Parallel" '
                f'mute="0" solo="0" soloSafe="1" color="FFD35400">\n'
                f'\t\t\t\t<SpeakerSetup x:id="speakerType" type="Stereo"/>\n'
                f'\t\t\t\t<UID x:id="uniqueID" uid="{sat_uid}"/>\n'
                f'\t\t\t\t<Connection x:id="destination" objectID="{{636828DD-7358-4765-B2A2-106A2780EE51}}/Input" friendlyName="メイン"/>\n'
                f'\t\t\t\t<Attributes x:id="Automation" mode="0"/>\n'
                f'\t\t\t\t<Attributes x:id="Inserts">\n' + "\n".join(sat_slots) + "\n"
                '\t\t\t\t\t<Attributes x:id="Presets" pname="MidOnly Sat" dirty="0"/>\n'
                "\t\t\t\t</Attributes>\n"
                '\t\t\t\t<Attributes x:id="CueMix"><Attributes x:id="slots" version="1"/></Attributes>\n'
                '\t\t\t\t<Attributes x:id="InputFX" gain="0" invertPhaseL="0" invertPhaseR="0"/>\n'
                "\t\t\t</AudioEffectChannel>"
            )
            w = float(np.clip(chain_params.parallel_sat_wet_pct / 100.0, 0.04, 0.22))
            sat_send_lin = round(float(np.clip((w / max(1.0 - w, 0.5)) * 0.52, 0.03, 0.14)), 6)
            extra_sends_for_main.append(
                f'\t\t\t\t\t<Attributes name="Send{send_num:02d}" bypass="0" prefader="0" level="{sat_send_lin}" pan="0.5">\n'
                f'\t\t\t\t\t\t<Connection x:id="destination" objectID="{sat_uid}/Input" friendlyName="Saturation Parallel"/>\n'
                f"\t\t\t\t\t</Attributes>"
            )
            fx_ch_num += 1
            send_num += 1

        if "Presets/Channels/Phoenix EarlyRoom/1 - PhoenixVerb.vstpreset" in patched_presets:
            ph_uid = _new_uid()
            new_console_uids.append(ph_uid)
            ph_slot = _render_fx_slot_xml(
                1,
                s1_specs["phoenix_verb"],
                "Presets/Channels/Phoenix EarlyRoom/1 - PhoenixVerb.vstpreset",
                True,
            )
            extra_fx_channels.append(
                f'\t\t\t<AudioEffectChannel gain="1" pan="0.5" name="Channel{fx_ch_num:02d}" label="Phoenix EarlyRoom" '
                f'mute="0" solo="0" soloSafe="1" color="FF2980B9">\n'
                f'\t\t\t\t<SpeakerSetup x:id="speakerType" type="Stereo"/>\n'
                f'\t\t\t\t<UID x:id="uniqueID" uid="{ph_uid}"/>\n'
                f'\t\t\t\t<Connection x:id="destination" objectID="{{636828DD-7358-4765-B2A2-106A2780EE51}}/Input" friendlyName="メイン"/>\n'
                f'\t\t\t\t<Attributes x:id="Automation" mode="0"/>\n'
                f'\t\t\t\t<Attributes x:id="Inserts">\n{ph_slot}\n'
                f'\t\t\t\t\t<Attributes x:id="Presets" pname="デフォルト" dirty="0"/>\n'
                f'\t\t\t\t</Attributes>\n'
                f'\t\t\t\t<Attributes x:id="CueMix"><Attributes x:id="slots" version="1"/></Attributes>\n'
                f'\t\t\t\t<Attributes x:id="InputFX" gain="0" invertPhaseL="0" invertPhaseR="0"/>\n'
                f"\t\t\t</AudioEffectChannel>"
            )
            er_send_lin = _db_to_s1_fader_gain(chain_params.early_room_send_db)
            extra_sends_for_main.append(
                f'\t\t\t\t\t<Attributes name="Send{send_num:02d}" bypass="0" prefader="0" level="{er_send_lin}" pan="0.5">\n'
                f'\t\t\t\t\t\t<Connection x:id="destination" objectID="{ph_uid}/Input" friendlyName="Phoenix EarlyRoom"/>\n'
                f"\t\t\t\t\t</Attributes>"
            )

        if extra_fx_channels:
            mixer_xml_out = mixer_xml_out.replace(
                "\t\t\t</AudioEffectChannel>\n\t\t</ChannelGroup>\n\t</Attributes>\n</AudioMixer>",
                "\t\t\t</AudioEffectChannel>\n"
                + "\n".join(extra_fx_channels)
                + "\n\t\t</ChannelGroup>\n\t</Attributes>\n</AudioMixer>",
            )

        if extra_sends_for_main:
            r2_send_anchor = (
                '\t\t\t\t\t\t<Connection x:id="destination" objectID="{9E67D54D-94FA-4A21-8BFA-151F51BBA20D}/Input" friendlyName="R2"/>\n'
                "\t\t\t\t\t</Attributes>"
            )
            mixer_xml_out = mixer_xml_out.replace(
                r2_send_anchor,
                r2_send_anchor + "\n" + "\n".join(extra_sends_for_main),
                1,
            )

        # 5. Depth-pushing Room + Plate + Delay sends on `harm` & `harm2` buses!
        harm_sends_list = [
            '\t\t\t\t\t<Attributes name="Send01" bypass="0" prefader="0" level="0.24" pan="0.5">\n'
            '\t\t\t\t\t\t<Connection x:id="destination" objectID="{9E67D54D-94FA-4A21-8BFA-151F51BBA20D}/Input" friendlyName="R2"/>\n'
            "\t\t\t\t\t</Attributes>"
        ]
        if ph_uid:
            harm_sends_list.append(
                f'\t\t\t\t\t<Attributes name="Send02" bypass="0" prefader="0" level="0.22" pan="0.5">\n'
                f'\t\t\t\t\t\t<Connection x:id="destination" objectID="{ph_uid}/Input" friendlyName="Phoenix EarlyRoom"/>\n'
                f"\t\t\t\t\t</Attributes>"
            )
        harm_sends_block = (
            '\t\t\t\t<Attributes x:id="Sends">\n'
            + "\n".join(harm_sends_list)
            + '\n\t\t\t\t\t<Attributes x:id="Presets" pname="DepthPush" dirty="0"/>\n'
            + "\t\t\t\t</Attributes>"
        )
        for h_bus_lbl in ("harm", "harm2"):
            hb_s = mixer_xml_out.find(f'label="{h_bus_lbl}"')
            if hb_s != -1:
                empty_sends = (
                    '\t\t\t\t<Attributes x:id="Sends">\n'
                    '\t\t\t\t\t<Attributes x:id="Presets" pname="デフォルト" dirty="0"/>\n'
                    "\t\t\t\t</Attributes>"
                )
                es_pos = mixer_xml_out.find(empty_sends, hb_s)
                if es_pos != -1 and (es_pos - hb_s) < 900:
                    mixer_xml_out = mixer_xml_out[:es_pos] + harm_sends_block + mixer_xml_out[es_pos + len(empty_sends) :]

        # 6. Calibrate Faders, Balanced Harmony Pans & Existing Main Send Levels
        main_lin = _db_to_s1_fader_gain(chain_params.vocal_fader_db)
        mixer_xml_out = mixer_xml_out.replace(
            '<AudioTrackChannel gain="1" pan="0.5" name="Channel01" label="mainvocal"',
            f'<AudioTrackChannel gain="{main_lin}" pan="0.5" name="Channel01" label="mainvocal"',
        )
        h_lin = _db_to_s1_fader_gain(harmony_fader_db)
        mixer_xml_out = mixer_xml_out.replace(
            '<AudioTrackChannel gain="1" pan="0.5" name="Channel02" label="harmvocal"',
            f'<AudioTrackChannel gain="{h_lin}" pan="0.22" name="Channel02" label="harmvocal"',
        )
        mixer_xml_out = mixer_xml_out.replace(
            '<AudioTrackChannel gain="1" pan="0.5" name="Channel03" label="harmvocal2"',
            f'<AudioTrackChannel gain="{h_lin}" pan="0.78" name="Channel03" label="harmvocal2"',
        )
        doub_lv = _db_to_s1_fader_gain(chain_params.doubler_send_db)
        dly_lv = _db_to_s1_fader_gain(chain_params.tempo_delay_send_db)
        rev_lv = _db_to_s1_fader_gain(chain_params.plate_reverb_send_db)
        mixer_xml_out = mixer_xml_out.replace(
            '<Attributes name="Send01" bypass="0" prefader="0" level="0.5" pan="0.5">',
            f'<Attributes name="Send01" bypass="0" prefader="0" level="{doub_lv}" pan="0.5">',
        )
        mixer_xml_out = mixer_xml_out.replace(
            '<Attributes name="Send02" bypass="0" prefader="0" level="0.5" pan="0.5">',
            f'<Attributes name="Send02" bypass="0" prefader="0" level="{dly_lv}" pan="0.5">',
        )
        mixer_xml_out = mixer_xml_out.replace(
            '<Attributes name="Send03" bypass="0" prefader="0" level="0.5" pan="0.5">',
            f'<Attributes name="Send03" bypass="0" prefader="0" level="{rev_lv}" pan="0.5">',
        )

    if extra_mixer_channels_xml:
        target_end_audiotrack = "\t\t\t</AudioTrackChannel>\n\t\t</ChannelGroup>"
        mixer_xml_out = mixer_xml_out.replace(
            target_end_audiotrack,
            "\t\t\t</AudioTrackChannel>\n" + "\n".join(extra_mixer_channels_xml) + "\n\t\t</ChannelGroup>",
            1,
        )

    if console_xml_str:
        for expand_uid in (
            "636828DD73584765B2A2106A2780EE51",  # メイン Master Output
            "B65FF637689748AE850F1DB2294F9113",  # mainvocal
            "9088293D13894C9EBC6DE6E19FCE0915",  # main bus
        ):
            sec_tag = f'<Section path="{expand_uid}">'
            pos_s = console_xml_str.find(sec_tag)
            if pos_s != -1:
                pos_exp = console_xml_str.find('expanded="0"', pos_s)
                if pos_exp != -1 and (pos_exp - pos_s) < 120:
                    console_xml_str = console_xml_str[:pos_exp] + 'expanded="1"' + console_xml_str[pos_exp + 12 :]

        if new_console_uids:
            uid_lines = "\n".join(f'\t\t\t\t<UID uid="{u}"/>' for u in new_console_uids)
            sb_anchor = '<ChannelShowHidePreset x:id="ScreenBank" name="" version="1">\n\t\t\t<List x:id="visible">'
            if sb_anchor in console_xml_str:
                console_xml_str = console_xml_str.replace(sb_anchor, sb_anchor + "\n" + uid_lines, 1)

        template_files["Devices/mixerconsole.xml"] = console_xml_str.replace("\n", "\r\n").encode("utf-8")

    if chain_params is not None:
        for obsolete in list(template_files.keys()):
            if obsolete.startswith(("Presets/Channels/mainvocal/", "Presets/Channels/harmvocal/", "Presets/Channels/harmvocal2/")):
                del template_files[obsolete]

    template_files["Song/mediapool.xml"] = pool_xml_out.replace("\n", "\r\n").encode("utf-8")
    template_files["Song/song.xml"] = song_xml_out.replace("\n", "\r\n").encode("utf-8")
    template_files["Devices/audiomixer.xml"] = mixer_xml_out.replace("\n", "\r\n").encode("utf-8")

    for preset_rel_path, preset_bytes in patched_presets.items():
        if len(preset_bytes) >= 48 and preset_bytes[:4] == b"VST3":
            template_files[preset_rel_path] = preset_bytes
        elif preset_rel_path in template_files:
            template_files[preset_rel_path] = patch_vstpreset_component_state(
                template_files[preset_rel_path], preset_bytes
            )

    with zipfile.ZipFile(output_song_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for arcname, raw_bytes in template_files.items():
            zout.writestr(arcname, raw_bytes)

    return output_song_path
