"""v8 R5 part B: Studio One plugin states (.vstpreset inside a .song) <-> pedalboard raw_state.

Studio One stores each insert's state as `Presets/Channels/<channel label>/<n> - <plugin>.vstpreset`
inside the .song zip (Steinberg preset: header 'VST3' + class id, chunks Comp / Cont / Info, chunk list
at the end). pedalboard's `raw_state` for a VST3 is JUCE's `<VST3PluginState>` XML holding the same
component / controller streams in JUCE base64. Writing goes through song_builder.create_vstpreset_bytes
(v7); this module adds the inverse and the lookup of a channel's insert in Devices/audiomixer.xml.

Checked on 2026-11-28 against a Studio One 5.5 autosave (data/outputs/エバ_auto/History/
エバ_AutoMIX 20261127-175228 (Autosaved).song): the Persistent Q preset has Comp = 'VC2!' + size + XML
(<PersistantQState ...>), an empty Cont and an Info chunk.
"""

from __future__ import annotations

import re
import struct
import zipfile
from pathlib import Path
from typing import Any

from mix_automation.io.song_builder import (
    _JUCE_B64_ALPHABET,
    create_vstpreset_bytes,
    decode_juce_base64,
    extract_steinberg_comp_and_cont,
)

PRESONUS_NS = "http://www.presonus.com/xml"


class S1PresetError(ValueError):
    pass


# ------------------------------------------------------------------ JUCE base64 / raw_state


def encode_juce_base64(data: bytes) -> str:
    """Inverse of song_builder.decode_juce_base64 (JUCE MemoryBlock::toBase64Encoding: '<n>.' then
    6-bit groups, least significant bit first)."""
    n_chars = (len(data) * 8 + 5) // 6
    val = int.from_bytes(data, "little")
    out = [_JUCE_B64_ALPHABET[(val >> (6 * i)) & 0x3F] for i in range(n_chars)]
    return f"{len(data)}." + "".join(out)


def raw_state_with_streams(template_raw_state: bytes, comp: bytes, cont: bytes) -> bytes:
    """A pedalboard raw_state carrying `comp` / `cont`, built from the plugin's own current raw_state
    (so the XML wrapper is exactly what this pedalboard writes). An empty `cont` keeps the template's
    controller stream (Studio One saves an empty Cont for Persistent Q)."""
    if b"<VST3PluginState>" not in template_raw_state:
        raise S1PresetError("template raw_state is not a JUCE <VST3PluginState>")
    # pedalboard 2026-11-28: raw_state = b'VC2!' + uint32 LE length of the XML + XML + b'\0'
    # (JUCE copyXmlToBinary; seen on Wave Breaker, VCA Compressor, Saturation Knob, Persistent Q).
    wrapped = template_raw_state[:4] == b"VC2!"
    xml = template_raw_state[8:].rstrip(b"\x00") if wrapped else template_raw_state
    xml = re.sub(rb"<IComponent>[^<]*</IComponent>",
                 b"<IComponent>" + encode_juce_base64(comp).encode("ascii") + b"</IComponent>", xml, count=1)
    if cont:
        enc = b"<IEditController>" + encode_juce_base64(cont).encode("ascii") + b"</IEditController>"
        if b"<IEditController>" in xml:
            xml = re.sub(rb"<IEditController>[^<]*</IEditController>", enc, xml, count=1)
        else:
            xml = xml.replace(b"</VST3PluginState>", enc + b"</VST3PluginState>", 1)
    if not wrapped:
        return xml
    return b"VC2!" + struct.pack("<I", len(xml)) + xml + b"\x00"


# ------------------------------------------------------------------ .vstpreset


def parse_vstpreset(data: bytes) -> dict[str, Any]:
    """{'class_id': 32 hex, 'chunks': {id: bytes}} of a Steinberg .vstpreset."""
    if len(data) < 48 or data[:4] != b"VST3":
        raise S1PresetError("not a Steinberg .vstpreset (missing 'VST3' header)")
    _, _ver, cid, list_off = struct.unpack("<4si32sq", data[:48])
    if not (48 <= list_off <= len(data) - 8) or data[list_off:list_off + 4] != b"List":
        raise S1PresetError("bad .vstpreset chunk list")
    (count,) = struct.unpack("<i", data[list_off + 4:list_off + 8])
    pos = list_off + 8
    chunks: dict[str, bytes] = {}
    for _ in range(count):
        if pos + 20 > len(data):
            raise S1PresetError("truncated .vstpreset chunk list")
        c, off, size = struct.unpack("<4sqq", data[pos:pos + 20])
        pos += 20
        if off < 0 or off + size > len(data):
            raise S1PresetError(f"chunk {c!r} outside the file")
        chunks[c.decode("ascii", "replace")] = data[off:off + size]
    if "Comp" not in chunks:
        raise S1PresetError("no Comp chunk")
    return {"class_id": cid.decode("ascii"), "chunks": chunks}


def vstpreset_from_plugin(plug: Any, class_id: str) -> bytes:
    comp, cont = extract_steinberg_comp_and_cont(bytes(plug.raw_state))
    return create_vstpreset_bytes(class_id, comp, cont)


def load_vstpreset_into(plug: Any, data: bytes) -> None:
    """Set a loaded pedalboard plugin to the state stored in a .vstpreset."""
    p = parse_vstpreset(data)
    plug.raw_state = raw_state_with_streams(bytes(plug.raw_state), p["chunks"]["Comp"], p["chunks"].get("Cont", b""))


def check_class(data: bytes, class_id: str) -> None:
    want = class_id.strip("{}").replace("-", "").upper()
    got = parse_vstpreset(data)["class_id"].upper()
    if got != want:
        raise S1PresetError(f".vstpreset class {got} is not the expected {want}")


# ------------------------------------------------------------------ .song zip


def zip_member_name(info: zipfile.ZipInfo) -> str:
    """Real member name. Studio One writes some Japanese names without the UTF-8 flag, in cp932
    (seen: 'Presets/Channels/メイン/...' in the autosave above); zipfile then decodes them as cp437."""
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode("cp437")
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return info.filename


def read_song_members(song: Path | str) -> dict[str, bytes]:
    with zipfile.ZipFile(song) as z:
        return {zip_member_name(i): z.read(i) for i in z.infolist()}


def parse_s1_xml(data: bytes | str):
    """ElementTree root of a Studio One XML file (they use the 'x:' prefix without declaring it)."""
    import xml.etree.ElementTree as ET

    txt = data.decode("utf-8") if isinstance(data, bytes) else data
    txt = txt.lstrip("﻿")
    if "xmlns:x=" not in txt:
        txt = re.sub(r"<(?![?!])([A-Za-z0-9_]+)(\s|>|/)", rf'<\1 xmlns:x="{PRESONUS_NS}"\2', txt, count=1)
    return ET.fromstring(txt)


XID = f"{{{PRESONUS_NS}}}id"


def channel_inserts(mixer_root) -> dict[str, list[dict]]:
    """{channel label: [{'slot': 'FX01', 'plugin': name, 'class_id': uid, 'preset_path': path}]}."""
    out: dict[str, list[dict]] = {}
    for ch in mixer_root.iter():
        label = ch.get("label")
        if label is None or not ch.tag.endswith("Channel"):
            continue
        slots = []
        for ins in ch:
            if ins.get(XID) != "Inserts":
                continue
            for fx in ins:
                name = fx.get("name") or ""
                if not re.fullmatch(r"FX\d+", name):
                    continue
                dev = next((e for e in fx if e.get(XID) == "deviceData"), None)
                cls = next((e for e in fx if e.get(XID) == "deviceClassID"), None)
                pp = next((e for e in fx if e.get(XID) == "presetPath"), None)
                slots.append({"slot": name, "plugin": dev.get("name") if dev is not None else None,
                              "class_id": cls.get("uid") if cls is not None else None,
                              "preset_path": pp.get("text") if pp is not None else None})
        out[label] = slots
    return out


def find_insert_preset(members: dict[str, bytes], track: str, plugin: str) -> tuple[dict, bytes]:
    """The (slot info, .vstpreset bytes) of `plugin` on the channel labelled `track`. Exactly one such
    insert must exist."""
    mixer = parse_s1_xml(members["Devices/audiomixer.xml"])
    ins = channel_inserts(mixer)
    if track not in ins:
        raise S1PresetError(f"no channel labelled {track!r}; channels: {sorted(ins)}")
    hits = [s for s in ins[track] if s["plugin"] == plugin]
    if len(hits) != 1:
        raise S1PresetError(f"channel {track!r} has {len(hits)} {plugin!r} inserts "
                            f"(inserts: {[s['plugin'] for s in ins[track]]})")
    slot = hits[0]
    path = slot["preset_path"]
    if path not in members:
        raise S1PresetError(f"preset {path!r} named in audiomixer.xml is not in the .song")
    return slot, members[path]


def persistent_q_state_xml(data: bytes):
    """<PersistantQState> element from a Persistent Q .vstpreset Comp chunk (the plugin's own XML; used
    to cross-check the values pedalboard reports)."""
    import xml.etree.ElementTree as ET

    comp = parse_vstpreset(data)["chunks"]["Comp"]
    s = comp.find(b"<PersistantQState")
    e = comp.find(b"</PersistantQState>", s)
    if s < 0:
        raise S1PresetError("no <PersistantQState> in the Comp chunk")
    if e < 0:
        e2 = comp.find(b"/>", s)
        blob = comp[s:e2 + 2]
    else:
        blob = comp[s:e + len(b"</PersistantQState>")]
    return ET.fromstring(blob.decode("utf-8"))


# ------------------------------------------------------------------ correction EQ out of a .song

DEFAULT_TRACKS = ("Vocal CorrEQ", "mainvocal")  # v8 export track first, then the v7 template's lead


def crosscheck_persistent_q(cfg: dict, state_xml) -> list[str]:
    """Differences between the bands pedalboard reports and the plugin's own <PersistantQState> XML
    (enabled set, type, frequency, gain, q). Empty list = they agree. The XML type index follows
    correction_eq.BAND_TYPES (Peaking=0, Highpass=1: matched on the autosave above)."""
    from mix_automation.v8 import correction_eq as ceq

    issues = []
    xml_on = {int(b.get("id")): b for b in state_xml if b.tag == "Band" and b.get("enabled") == "1"}
    got = {b["band"]: b for b in cfg["bands"]}
    if set(xml_on) != set(got):
        issues.append(f"enabled bands differ: xml {sorted(xml_on)} vs plugin {sorted(got)}")
    for n in sorted(set(xml_on) & set(got)):
        x, b = xml_on[n], got[n]
        t = int(x.get("type", -1))
        if not (0 <= t < len(ceq.BAND_TYPES)) or ceq.BAND_TYPES[t] != b["type"]:
            issues.append(f"band {n}: type xml {t} vs {b['type']}")
        for key, attr, tol in (("frequency_hz", "frequencyHz", 0.01), ("gain_db", "gainDb", 0.01), ("q", "q", 0.002)):
            xv = float(x.get(attr))
            if abs(xv - b[key]) > tol * max(1.0, abs(xv)):
                issues.append(f"band {n}: {key} xml {xv:.4f} vs {b[key]}")
    for key, attr in (("output_gain", "outputGainDb"), ("input_gain", "inputGainDb")):
        xv = float(state_xml.get(attr, 0.0))
        if abs(xv - float(cfg["global"].get(key, 0.0))) > 0.01:
            issues.append(f"{key}: xml {xv} vs {cfg['global'].get(key, 0.0)}")
    return issues


def correction_eq_from_song(song: Path | str, track: str | None = None, voice: str | None = None) -> dict:
    """Read the Persistent Q on `track` (default: 'Vocal CorrEQ', else 'mainvocal') of a .song into a
    correction EQ dict (correction_eq format) with a source block. Raises when the XML cross-check fails."""
    import hashlib
    from datetime import datetime

    from pedalboard import load_plugin

    from mix_automation.config.plugin_registry import get_plugin_map
    from mix_automation.v8 import correction_eq as ceq

    song = Path(song)
    members = read_song_members(song)
    labels = channel_inserts(parse_s1_xml(members["Devices/audiomixer.xml"]))
    if track is None:
        track = next((t for t in DEFAULT_TRACKS if t in labels), DEFAULT_TRACKS[0])
    slot, data = find_insert_preset(members, track, ceq.PLUGIN)
    plug = load_plugin(str(get_plugin_map()["persistent_q"].loadable_path))
    reference = {k: ceq._value(plug, k) for k in plug.parameters}  # fresh-load values, before the state
    load_vstpreset_into(plug, data)
    source = {"song": str(song.resolve()), "track": track, "slot": slot["slot"], "preset_path": slot["preset_path"],
              "song_sha256": hashlib.sha256(song.read_bytes()).hexdigest(),
              "read_at": datetime.now().isoformat(timespec="seconds")}
    cfg = ceq.bands_from_plugin(plug, reference, voice=voice, source=source)
    issues = crosscheck_persistent_q(cfg, persistent_q_state_xml(data))
    if issues:
        raise S1PresetError("pedalboard and <PersistantQState> disagree: " + "; ".join(issues))
    return cfg


__all__ = [
    "S1PresetError",
    "channel_inserts",
    "check_class",
    "correction_eq_from_song",
    "crosscheck_persistent_q",
    "decode_juce_base64",
    "encode_juce_base64",
    "find_insert_preset",
    "load_vstpreset_into",
    "parse_s1_xml",
    "parse_vstpreset",
    "persistent_q_state_xml",
    "raw_state_with_streams",
    "read_song_members",
    "vstpreset_from_plugin",
    "zip_member_name",
]
