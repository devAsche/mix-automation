"""Studio One 5.5 Complete `.song` Session Builder & `.vstpreset` Chunk Patcher.

Generates a ready-to-open Studio One 5 project folder (`<SongName>.song` + `Media/*.wav`)
from the base `test20261124.song` template:
  - Preserves `ClassID`, `Cont`, and `Info` chunks inside every `.vstpreset` while injecting
    newly optimized VST3 `Comp` binary states extracted from Pedalboard.
  - Registers all processed vocal, instrumental, FX throw, and muted harmony guide WAV clips
    in `Song/mediapool.xml`.
  - Splits `mainvocal` into structural section `<AudioEvent>` clips with pre-computed
    anchor-propagated `<Attributes x:id="volumeCurve" level="..."/>` clip gains so the user
    can inspect and tweak per-section vocal balance effortlessly in Studio One 5!
"""

from __future__ import annotations

import struct
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import load_plugin

from mix_automation.config.plugin_registry import VST3_ROOT
from mix_automation.dsp.vst3_optimizer import SongSection, VocalChainParams


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


def patch_vstpreset_component_state(orig_vstpreset: bytes, new_comp_state: bytes) -> bytes:
    """Replace the `Comp` chunk inside an existing Steinberg `.vstpreset` while keeping `Cont` & `Info`."""
    if len(orig_vstpreset) < 48 or orig_vstpreset[:4] != b"VST3":
        raise ValueError("Original bytes are not a valid Steinberg .vstpreset file.")

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
        else:
            chunks.append((cid, orig_vstpreset[c_off : c_off + c_size]))

    if not comp_replaced:
        chunks.insert(0, (b"Comp", new_comp_state))

    # Rebuild binary layout
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
    """Write float32 `(samples,)` or `(channels, samples)` audio as 24-bit PCM WAV.

    Returns `(frame_count, num_channels, peak_amplitude)`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -0.9999, 0.9999)
    peak = float(np.max(np.abs(clipped))) if clipped.size > 0 else 0.0

    if clipped.ndim == 1:
        sf.write(str(path), clipped, sr, subtype="PCM_24")
        return len(clipped), 1, peak
    else:
        # (channels, samples) -> (samples, channels)
        data = clipped.T if clipped.shape[0] <= 2 and clipped.shape[0] < clipped.shape[1] else clipped
        sf.write(str(path), data, sr, subtype="PCM_24")
        return data.shape[0], data.shape[1], peak


def _new_uid() -> str:
    return "{" + str(uuid.uuid4()).upper() + "}"


def _db_to_s1_fader_gain(db: float) -> float:
    """Convert dB offset to linear Studio One channel gain multiplier."""
    return round(float(10.0 ** (db / 20.0)), 6)


def export_template_vst3_states(params: VocalChainParams, bpm: float) -> dict[str, bytes]:
    """Load each VST3 plugin in an isolated timeout-guarded subprocess and dump optimized `Comp` states.

    Subprocess isolation prevents WaveShell shared-memory locks from ever hanging batch runs.
    """
    import base64
    import json
    import subprocess
    import sys

    states: dict[str, bytes] = {}
    waves_shell = str(VST3_ROOT / "WaveShell1-VST3 14.12_x64.vst3")
    pq_path = str(VST3_ROOT / "Persistent Q.vst3")
    r2_path = str(VST3_ROOT / "R2.vst3")

    worker_script = (
        "import base64, json, sys\n"
        "from pedalboard import load_plugin\n"
        "spec = json.loads(sys.argv[1])\n"
        "p = load_plugin(spec['path'], plugin_name=spec.get('plugin_name')) if spec.get('plugin_name') else load_plugin(spec['path'])\n"
        "for k, v in spec['settings'].items():\n"
        "    if k in p.parameters:\n"
        "        try: setattr(p, k, v)\n"
        "        except Exception: pass\n"
        "print('STATE_B64:' + base64.b64encode(p.raw_state).decode('ascii'))\n"
    )

    def _dump_isolated(vst_path: str, plugin_name: str | None, settings: dict) -> bytes | None:
        payload = json.dumps({"path": vst_path, "plugin_name": plugin_name, "settings": settings})
        try:
            proc = subprocess.run(
                [sys.executable, "-c", worker_script, payload],
                capture_output=True,
                text=True,
                timeout=8,
            )
            for line in proc.stdout.splitlines():
                if line.startswith("STATE_B64:"):
                    return base64.b64decode(line[len("STATE_B64:") :])
        except Exception:
            return None
        return None

    # 1. Persistent Q (Main Vocal & Harmony Vocal EQs)
    pq_main_raw = _dump_isolated(
        pq_path,
        None,
        {
            "band_1_enabled": True,
            "band_1_type": "Highpass",
            "band_1_frequency": float(np.clip(params.hpf_hz, 40.0, 250.0)),
            "band_1_slope": 18.0,
            "band_2_enabled": True,
            "band_2_type": "Peaking",
            "band_2_frequency": 240.0,
            "band_2_gain": float(params.body_gain_db),
            "band_2_q": 0.9,
            "band_3_enabled": True,
            "band_3_type": "Peaking",
            "band_3_frequency": 560.0,
            "band_3_gain": float(params.mud_cut_db),
            "band_3_q": 1.4,
            "band_4_enabled": True,
            "band_4_type": "Peaking",
            "band_4_frequency": 2850.0,
            "band_4_gain": float(params.presence_gain_db),
            "band_4_q": 0.85,
            "band_5_enabled": True,
            "band_5_type": "HighShelf",
            "band_5_frequency": 10500.0,
            "band_5_gain": float(params.air_shelf_db),
            "band_5_q": 0.7,
        },
    )
    if pq_main_raw:
        states["Presets/Channels/mainvocal/2 - Persistent Q.vstpreset"] = pq_main_raw

    pq_harm_raw = _dump_isolated(
        pq_path,
        None,
        {
            "band_1_enabled": True,
            "band_1_type": "Highpass",
            "band_1_frequency": 165.0,
            "band_1_slope": 18.0,
            "band_2_enabled": True,
            "band_2_type": "Peaking",
            "band_2_frequency": 560.0,
            "band_2_gain": float(params.mud_cut_db - 1.0),
            "band_3_enabled": True,
            "band_3_type": "Peaking",
            "band_3_frequency": 2850.0,
            "band_3_gain": -2.0,
            "band_4_enabled": True,
            "band_4_type": "HighShelf",
            "band_4_frequency": 10000.0,
            "band_4_gain": float(max(0.0, params.air_shelf_db * 0.7)),
        },
    )
    if pq_harm_raw:
        states["Presets/Channels/harmvocal/2 - Persistent Q.vstpreset"] = pq_harm_raw
        states["Presets/Channels/harmvocal2/2 - Persistent Q.vstpreset"] = pq_harm_raw

    # 2. Waves MV2, C1, L1, H-Delay, Doubler2 (each isolated with 8s timeout guard)
    mv2_raw = _dump_isolated(
        waves_shell,
        "MV2 Mono",
        {"low_level": float(params.mv2_low_level), "high_level": float(params.mv2_high_level), "output_gain": 0.0},
    )
    if mv2_raw:
        states["Presets/Channels/mainvocal/1 - MV2 Mono.vstpreset"] = mv2_raw
        states["Presets/Channels/harmvocal/1 - MV2 Mono.vstpreset"] = mv2_raw
        states["Presets/Channels/harmvocal2/1 - MV2 Mono.vstpreset"] = mv2_raw

    c1_raw = _dump_isolated(
        waves_shell,
        "C1 comp Mono",
        {
            "threshold": float(params.comp_threshold_db),
            "ratio": float(params.comp_ratio),
            "attack": 6.0,
            "release": 75.0,
            "makeup": 2.0,
        },
    )
    if c1_raw:
        states["Presets/Channels/mainvocal/3 - C1 comp Mono.vstpreset"] = c1_raw
        states["Presets/Channels/harmvocal/3 - C1 comp Mono.vstpreset"] = c1_raw
        states["Presets/Channels/harmvocal2/3 - C1 comp Mono.vstpreset"] = c1_raw

    l1_raw = _dump_isolated(waves_shell, "L1 limiter Mono", {"threshold": -3.5, "ceiling": -0.3})
    if l1_raw:
        states["Presets/Channels/mainvocal/4 - L1 limiter Mono.vstpreset"] = l1_raw

    hd_raw = _dump_isolated(
        waves_shell,
        "H-Delay Stereo",
        {
            "mix": 100.0,
            "bpm": float(np.clip(bpm, 40.0, 240.0)),
            "pingpong": True,
            "feedback": 26.0,
            "hipass": 350.0,
            "lopass": 4200.0,
        },
    )
    if hd_raw:
        states["Presets/Channels/H-Delay Stereo/1 - H-Delay Stereo.vstpreset"] = hd_raw

    db2_raw = _dump_isolated(waves_shell, "Doubler2 Stereo", {"direct_onoff": False})
    if db2_raw:
        states["Presets/Channels/Doubler2 Stereo/1 - Doubler2 Stereo.vstpreset"] = db2_raw

    # 3. Exponential Audio R2 (Stereo Reverb Send: 100% Wet)
    r2_raw = _dump_isolated(r2_path, None, {"mix": 100.0})
    if r2_raw:
        states["Presets/Channels/R2/1 - R2.vstpreset"] = r2_raw

    return states


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
) -> Path:
    """Build a complete Studio One 5 `.song` file with section-split clips, audio pool, and presets."""
    output_song_dir.mkdir(parents=True, exist_ok=True)
    output_song_path = output_song_dir / f"{song_name}.song"
    patched_presets = patched_presets or {}

    sec_per_beat = 60.0 / max(bpm, 40.0)
    tempo_norm = (bpm - 10.0) / 390.0

    with zipfile.ZipFile(template_song_path, "r") as zin:
        template_files = {name: zin.read(name) for name in zin.namelist()}

    # Normalize CRLF -> LF before structural XML block injection, then restore CRLF on save
    ET.register_namespace("x", "http://www.presonus.com/xml")
    song_xml_str = template_files["Song/song.xml"].decode("utf-8").replace("\r\n", "\n")
    pool_xml_str = template_files["Song/mediapool.xml"].decode("utf-8").replace("\r\n", "\n")
    mixer_xml_str = template_files["Devices/audiomixer.xml"].decode("utf-8").replace("\r\n", "\n")

    # Collect all WAV files to register in MediaPool
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
    max_beats = max_dur_sec / sec_per_beat

    # --- 1. Build `Song/mediapool.xml` ---
    clip_xml_blocks: list[str] = []
    for rec in clip_records:
        use_cnt = len(sections) if rec["key"] == "mainvocal" and sections else 1
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

    media_folder_xml = '\t<Attributes x:id="rootFolder">\n\t\t<MediaFolder name="Audio">\n' + "\n".join(clip_xml_blocks) + "\n\t\t</MediaFolder>\n\t</Attributes>"
    pool_xml_out = pool_xml_str.replace('<Attributes x:id="rootFolder"/>', media_folder_xml)
    pool_xml_out = pool_xml_out.replace(
        'url="file:///C:/MIX_automation/data/templates/test20261124/test20261124.song"',
        f'url="{output_song_path.resolve().as_uri()}"',
    )

    # --- 2. Build `<List x:id="Events">` for mainvocal (split per SongSection with clip gain!) ---
    main_events_xml: list[str] = []
    if sections:
        for sec in sections:
            s_sec = max(0.0, min(sec.start_sec, main_clip["duration_sec"]))
            e_sec = max(s_sec, min(sec.end_sec, main_clip["duration_sec"]))
            if e_sec - s_sec < 0.05:
                continue
            start_beats = s_sec / sec_per_beat
            len_beats = (e_sec - s_sec) / sec_per_beat
            lin_level = _db_to_s1_fader_gain(sec.clip_gain_db)
            ev_name = f"main_{sec.label}"
            main_events_xml.append(
                f'\t\t\t\t\t<AudioEvent clipID="{main_clip["media_id"]}" timeFormat="2" '
                f'start="{start_beats:.8f}" offset="{s_sec:.8f}" length="{len_beats:.8f}" '
                f'name="{ev_name}" speed="1" transpose="0" tune="0" normalize="0" modification="0">\n'
                f'\t\t\t\t\t\t<Attributes x:id="volumeCurve" level="{lin_level:.8f}"/>\n'
                f'\t\t\t\t\t</AudioEvent>'
            )
    else:
        len_beats = main_clip["duration_sec"] / sec_per_beat
        main_events_xml.append(
            f'\t\t\t\t\t<AudioEvent clipID="{main_clip["media_id"]}" timeFormat="2" '
            f'length="{len_beats:.8f}" name="mainvocal" speed="1" transpose="0" tune="0" normalize="0" modification="0">\n'
            f'\t\t\t\t\t\t<Attributes x:id="volumeCurve" level="1"/>\n'
            f'\t\t\t\t\t</AudioEvent>'
        )

    def _single_event_list_xml(rec: dict, name: str) -> str:
        lb = rec["duration_sec"] / sec_per_beat
        return (
            f'\t\t\t\t<List x:id="Events">\n'
            f'\t\t\t\t\t<AudioEvent clipID="{rec["media_id"]}" timeFormat="2" length="{lb:.8f}" '
            f'name="{name}" speed="1" transpose="0" tune="0" normalize="0" modification="0">\n'
            f'\t\t\t\t\t\t<Attributes x:id="volumeCurve" level="1"/>\n'
            f'\t\t\t\t\t</AudioEvent>\n'
            f'\t\t\t\t</List>'
        )

    # Inject tempo, sampleRate, song length, and events into `Song/song.xml`
    song_xml_out = song_xml_str
    song_xml_out = song_xml_out.replace('sampleRate="44100"', f'sampleRate="{sr}"')
    song_xml_out = song_xml_out.replace('length="300"', f'length="{max_dur_sec:.2f}"')
    song_xml_out = song_xml_out.replace(
        'value="0.34871794871794875581016981413995381444692611694336"',
        f'value="{tempo_norm:.16f}"',
    )
    song_xml_out = song_xml_out.replace(
        'tempo="0.41095890410958896277193730384169612079858779907227"',
        f'tempo="{sec_per_beat:.16f}"',
    )
    song_xml_out = song_xml_out.replace(
        'start="730.0000000000001136868377216160297393798828125"',
        f'start="{max_beats:.4f}"',
    )

    main_ev_block = '\t\t\t\t<List x:id="Events">\n' + "\n".join(main_events_xml) + "\n\t\t\t\t</List>"
    # Insert `<List x:id="Events">` into mainvocal, harmvocal, harmvocal2 tracks
    target_main_close = '<Url x:id="points" type="1" url="media:///Envelopes/mainvocal/パン.envelopex"/>\n\t\t\t\t\t</AutomationRegion>\n\t\t\t\t</Attributes>'
    song_xml_out = song_xml_out.replace(target_main_close, target_main_close + "\n" + main_ev_block)

    if harm1_clip is not None:
        t_h1 = '<Url x:id="points" type="1" url="media:///Envelopes/harmvocal/パン.envelopex"/>\n\t\t\t\t\t</AutomationRegion>\n\t\t\t\t</Attributes>'
        song_xml_out = song_xml_out.replace(t_h1, t_h1 + "\n" + _single_event_list_xml(harm1_clip, "harmvocal"))

    if harm2_clip is not None:
        t_h2 = '<Url x:id="points" type="1" url="media:///Envelopes/harmvocal2/パン.envelopex"/>\n\t\t\t\t\t</AutomationRegion>\n\t\t\t\t</Attributes>'
        song_xml_out = song_xml_out.replace(t_h2, t_h2 + "\n" + _single_event_list_xml(harm2_clip, "harmvocal2"))

    # Add extra tracks (offvocal, ethereal FX throws, muted director harmony guides) to song.xml & audiomixer.xml
    extra_media_tracks_xml: list[str] = []
    extra_mixer_channels_xml: list[str] = []

    for idx_e, (spec, rec) in enumerate(extra_clips, start=4):
        track_uid = _new_uid()
        chan_uid = _new_uid()
        spk_type = "Stereo" if spec.stereo else "Mono"
        mute_flag = "1" if spec.mute else "0"
        fader_lin = _db_to_s1_fader_gain(spec.gain_db)

        extra_media_tracks_xml.append(
            f'\t\t\t<MediaTrack mediaType="Audio" tempoFollow="2" materialMode="0" layersFollow="1" '
            f'version="1" trackID="{track_uid}" timeFormat="2" name="{spec.name}" color="{spec.color}">\n'
            f'\t\t\t\t<SpeakerSetup x:id="trackFormat" type="{spk_type}"/>\n'
            f'\t\t\t\t<UID x:id="channelID" uid="{chan_uid}"/>\n'
            f'{_single_event_list_xml(rec, spec.name)}\n'
            f'\t\t\t</MediaTrack>'
        )

        # Route directly to Master Output `{636828DD-7358-4765-B2A2-106A2780EE51}/Input`
        extra_mixer_channels_xml.append(
            f'\t\t\t<AudioTrackChannel gain="{fader_lin}" pan="{spec.pan}" name="Channel{idx_e:02d}" '
            f'label="{spec.name}" mute="{mute_flag}" solo="0" soloSafe="0" color="{spec.color}">\n'
            f'\t\t\t\t<SpeakerSetup x:id="speakerType" type="{spk_type}"/>\n'
            f'\t\t\t\t<UID x:id="uniqueID" uid="{chan_uid}"/>\n'
            f'\t\t\t\t<Connection x:id="destination" objectID="{{636828DD-7358-4765-B2A2-106A2780EE51}}/Input" friendlyName="メイン"/>\n'
            f'\t\t\t\t<Attributes x:id="Automation" mode="0"/>\n'
            f'\t\t\t\t<Attributes x:id="Inserts"><Attributes x:id="Presets" pname="デフォルト" dirty="0"/></Attributes>\n'
            f'\t\t\t\t<Attributes x:id="InputFX" gain="0" invertPhaseL="0" invertPhaseR="0"/>\n'
            f'\t\t\t\t<Attributes x:id="Sends"><Attributes x:id="Presets" pname="デフォルト" dirty="0"/></Attributes>\n'
            f'\t\t\t</AudioTrackChannel>'
        )

    if extra_media_tracks_xml:
        song_xml_out = song_xml_out.replace(
            "\t\t</List>\n\t</Attributes>\n</Song>",
            "\n".join(extra_media_tracks_xml) + "\n\t\t</List>\n\t</Attributes>\n</Song>",
        )

    mixer_xml_out = mixer_xml_str
    if chain_params is not None:
        # Update mainvocal fader gain & FX send levels in audiomixer.xml
        main_lin = _db_to_s1_fader_gain(chain_params.vocal_fader_db)
        mixer_xml_out = mixer_xml_out.replace(
            '<AudioTrackChannel gain="1" pan="0.5" name="Channel01" label="mainvocal"',
            f'<AudioTrackChannel gain="{main_lin}" pan="0.5" name="Channel01" label="mainvocal"',
        )
        doub_lv = _db_to_s1_fader_gain(chain_params.doubler_send_db)
        dly_lv = _db_to_s1_fader_gain(chain_params.delay_send_db)
        rev_lv = _db_to_s1_fader_gain(chain_params.reverb_send_db)
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

    # Write updated files into the output `.song` ZIP archive (restoring Windows CRLF in XMLs)
    template_files["Song/mediapool.xml"] = pool_xml_out.replace("\n", "\r\n").encode("utf-8")
    template_files["Song/song.xml"] = song_xml_out.replace("\n", "\r\n").encode("utf-8")
    template_files["Devices/audiomixer.xml"] = mixer_xml_out.replace("\n", "\r\n").encode("utf-8")

    for preset_rel_path, new_comp_bytes in patched_presets.items():
        if preset_rel_path in template_files:
            template_files[preset_rel_path] = patch_vstpreset_component_state(
                template_files[preset_rel_path], new_comp_bytes
            )

    with zipfile.ZipFile(output_song_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for arcname, raw_bytes in template_files.items():
            zout.writestr(arcname, raw_bytes)

    return output_song_path
