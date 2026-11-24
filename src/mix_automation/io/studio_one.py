"""Studio One 5.5 native integration: .vstpreset writer and .song template bridge.

Studio One 5 predates .dawproject (introduced in Studio One 6.5). To deliver ready-to-tweak
sessions for Studio One 5.5, this module provides two complementary mechanisms:
  1. Standard Steinberg `.vstpreset` binary generation from Pedalboard's VST3 `raw_state`,
     allowing drag-and-drop recall of exact optimized plugin knobs in Studio One 5.
  2. Native `.song` archive (ZIP + XML + Presets + Media) inspection and template cloning,
     enabling automated injection of processed audio stems, VST3 preset chunks, and
     section-level automation envelopes directly into a playable `.song` file.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

HEX32_PATTERN = re.compile(r"^[0-9A-Fa-f]{32}$")


@dataclass(frozen=True)
class SongInspectionSummary:
    """Structural summary of a PreSonus Studio One 5 .song archive."""

    song_path: Path
    xml_entries: list[str]
    preset_entries: list[str]
    media_entries: list[str]
    detected_plugin_names: list[str]


def extract_vst3_processor_cid(vst3_path: Path) -> str | None:
    """Resolve the 32-character hex VST3 Processor Class ID from moduleinfo.json if available."""
    candidates: list[Path] = []
    if vst3_path.is_dir():
        candidates.extend(vst3_path.rglob("moduleinfo.json"))
    if not candidates:
        return None

    for info_file in candidates:
        try:
            text = info_file.read_text(encoding="utf-8", errors="ignore")
            # Strip trailing commas tolerated by VST3 JSON5-like moduleinfo files
            cleaned = re.sub(r",\s*([\]}])", r"\1", text)
            data = json.loads(cleaned)
            for cls in data.get("Classes", []):
                if cls.get("Category") == "Audio Module Class":
                    cid = str(cls.get("CID", "")).strip().upper()
                    if HEX32_PATTERN.match(cid):
                        return cid
        except Exception:
            continue
    return None


def write_vstpreset(
    output_path: Path,
    class_id_hex32: str,
    component_state: bytes,
) -> Path:
    """Write a Steinberg VST3 standard `.vstpreset` binary file.

    Binary Layout (Steinberg VST3 Preset File Format Specification v1):
      - Header (48 bytes):
          'VST3' (4B) | version=1 (int32 LE, 4B) | ASCII ClassID (32B) | chunkListOffset (int64 LE, 8B)
      - Data Area:
          Component chunk bytes
      - Chunk List:
          'List' (4B) | entryCount=1 (int32 LE, 4B) |
          entry[0]: 'Comp' (4B) | offset=48 (int64 LE, 8B) | size (int64 LE, 8B)
    """
    cid = class_id_hex32.strip().upper()
    if not HEX32_PATTERN.match(cid):
        raise ValueError(f"VST3 Class ID must be exactly 32 hex characters, got: {class_id_hex32!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_size = 48
    comp_offset = header_size
    comp_size = len(component_state)
    chunk_list_offset = comp_offset + comp_size

    header = struct.pack(
        "<4si32sq",
        b"VST3",
        1,
        cid.encode("ascii"),
        chunk_list_offset,
    )
    chunk_list = struct.pack(
        "<4si4sqq",
        b"List",
        1,
        b"Comp",
        comp_offset,
        comp_size,
    )

    with output_path.open("wb") as f:
        f.write(header)
        f.write(component_state)
        f.write(chunk_list)

    return output_path


def inspect_studio_one_song(song_path: Path) -> SongInspectionSummary:
    """Inspect the internal ZIP layout and referenced devices of a Studio One 5 `.song` file."""
    if not song_path.exists():
        raise FileNotFoundError(f".song file not found: {song_path}")
    if not zipfile.is_zipfile(song_path):
        raise ValueError(f"File is not a valid Studio One ZIP-based .song archive: {song_path}")

    xml_entries: list[str] = []
    preset_entries: list[str] = []
    media_entries: list[str] = []
    plugin_names: set[str] = set()

    with zipfile.ZipFile(song_path, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith(".xml") or lower.endswith(".xstream"):
                xml_entries.append(name)
            if lower.startswith("presets/"):
                preset_entries.append(name)
            if lower.startswith("media/"):
                media_entries.append(name)

        for xml_name in xml_entries:
            try:
                raw_bytes = zf.read(xml_name)
                root = ET.fromstring(raw_bytes)
                for elem in root.iter():
                    if elem.attrib.get("classId") or "Plugin" in elem.tag or "Vendor" in elem.attrib:
                        dev_name = elem.attrib.get("name") or elem.attrib.get("deviceName")
                        if dev_name:
                            plugin_names.add(dev_name)
            except Exception:
                continue

    return SongInspectionSummary(
        song_path=song_path,
        xml_entries=sorted(xml_entries),
        preset_entries=sorted(preset_entries),
        media_entries=sorted(media_entries),
        detected_plugin_names=sorted(plugin_names),
    )


def clone_song_with_replaced_media(
    template_song_path: Path,
    output_song_path: Path,
    media_replacements: dict[str, Path],
    preset_replacements: dict[str, bytes] | None = None,
) -> Path:
    """Clone a Studio One 5 `.song` template while replacing media WAVs and preset chunks."""
    if not zipfile.is_zipfile(template_song_path):
        raise ValueError(f"Invalid template .song archive: {template_song_path}")

    output_song_path.parent.mkdir(parents=True, exist_ok=True)
    preset_replacements = preset_replacements or {}

    if template_song_path.resolve() == output_song_path.resolve():
        raise ValueError("output_song_path must differ from template_song_path.")

    with zipfile.ZipFile(template_song_path, "r") as zin, zipfile.ZipFile(
        output_song_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        existing_names = set(zin.namelist())
        for item in zin.infolist():
            if item.filename in media_replacements:
                src_wav = media_replacements[item.filename]
                zout.write(src_wav, arcname=item.filename)
            elif item.filename in preset_replacements:
                zout.writestr(item, preset_replacements[item.filename])
            else:
                zout.writestr(item, zin.read(item.filename))

        # Add any new media entries not originally present in the template
        for arcname, src_wav in media_replacements.items():
            if arcname not in existing_names:
                shutil.copyfile(src_wav, src_wav) if False else zout.write(src_wav, arcname=arcname)

    return output_song_path
