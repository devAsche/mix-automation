"""v7 Loudness-Normalized Multi-Song Blind A/B & Regression Audit Harness.

Provides:
  1. `extract_loudness_normalized_ab_clips`: Scans completed song outputs, locates 3 diagnostic
     12s listening spans (`anchor_chorus`, `quiet_drop`, `high_sibilance`), loudness-normalizes
     AutoMIX Master and User Self-Mix excerpts (-14.0 LUFS, -1.0 dBFS true-peak ceiling) for
     level-matched listening, and runs automated Do-Not-Regress acoustic + Studio One `.song` checks.
  2. `generate_ab_blind_console_html`: Builds a self-contained dark-mode interactive A/B console
     with regression pass/fail badges, metric tables, synchronized audio players, and Blind X/Y
     randomized listening mode with Reveal Identity & preference tracking.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal

from mix_automation.analysis.relative_profile import compute_section_relative_profile
from mix_automation.io.audio_loader import load_audio
from mix_automation.io.song_builder import write_pcm24_wav

_CORE_SONG_XML_PATHS: tuple[str, ...] = (
    "Song/song.xml",
    "Song/mediapool.xml",
    "Devices/audiomixer.xml",
    "Devices/mixerconsole.xml",
)


def _to_stereo_2ch(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=0).astype(np.float32)
    if audio.shape[0] <= 2 and audio.shape[0] < audio.shape[1]:
        return audio.astype(np.float32) if audio.shape[0] == 2 else np.repeat(audio, 2, axis=0).astype(np.float32)
    if audio.shape[1] == 1:
        return np.repeat(audio.T, 2, axis=0).astype(np.float32)
    return audio.T[:2].astype(np.float32)


def _measure_true_peak_db(audio_st: np.ndarray) -> float:
    if audio_st.size == 0:
        return -120.0
    if audio_st.shape[1] >= 16:
        up = signal.resample_poly(audio_st.astype(np.float64), 4, 1, axis=-1)
        pk = float(np.max(np.abs(up)))
    else:
        pk = float(np.max(np.abs(audio_st)))
    return float(20.0 * np.log10(max(pk, 1e-12)))


def _measure_lufs(audio_st: np.ndarray, sr: int) -> float:
    min_len = max(int(sr * 0.45), 1024)
    buf = audio_st.astype(np.float64)
    if buf.shape[1] < min_len:
        buf = np.pad(buf, ((0, 0), (0, min_len - buf.shape[1])))
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(buf.T))
    return lufs if np.isfinite(lufs) else -70.0


def _loudness_normalize_clip(
    audio_st: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
    true_peak_ceiling_dbfs: float = -1.0,
) -> tuple[np.ndarray, float, float]:
    """Normalize stereo excerpt to `target_lufs` LUFS while strictly guarding true peak <= ceiling."""
    st = _to_stereo_2ch(audio_st)
    if st.shape[1] == 0 or float(np.max(np.abs(st))) < 1e-7:
        return st.astype(np.float32), -70.0, -120.0

    cur_lufs = _measure_lufs(st, sr)
    if cur_lufs > -68.0:
        gain_db = float(np.clip(target_lufs - cur_lufs, -24.0, 30.0))
        norm = (st.astype(np.float64) * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    else:
        norm = st.copy()

    tp_db = _measure_true_peak_db(norm)
    if tp_db > true_peak_ceiling_dbfs:
        trim_db = true_peak_ceiling_dbfs - tp_db
        norm = (norm.astype(np.float64) * (10.0 ** (trim_db / 20.0))).astype(np.float32)
        tp_db = _measure_true_peak_db(norm)

    achieved_lufs = _measure_lufs(norm, sr)
    return norm.astype(np.float32), round(achieved_lufs, 2), round(tp_db, 2)


def _align_self_mix_to_master(
    self_mix_st: np.ndarray,
    master_st: np.ndarray,
    sr: int,
    max_offset_sec: float = 8.0,
) -> tuple[np.ndarray, float]:
    """Align user self-mix to AutoMIX Master within `±max_offset_sec` using 10ms RMS onset correlation."""
    hop = max(int(sr * 0.01), 128)
    scan_len = min(self_mix_st.shape[1], master_st.shape[1], sr * 90)
    n_f = scan_len // hop
    if n_f < 100:
        out = np.zeros_like(master_st)
        c = min(master_st.shape[1], self_mix_st.shape[1])
        out[:, :c] = self_mix_st[:, :c]
        return out, 0.0

    m_mono = 0.5 * (master_st[0, : n_f * hop] + master_st[1, : n_f * hop])
    s_mono = 0.5 * (self_mix_st[0, : n_f * hop] + self_mix_st[1, : n_f * hop])
    m_env = np.sqrt(np.mean(m_mono.reshape(n_f, hop) ** 2, axis=1) + 1e-12)
    s_env = np.sqrt(np.mean(s_mono.reshape(n_f, hop) ** 2, axis=1) + 1e-12)
    m_diff = np.maximum(np.diff(m_env, prepend=0.0), 0.0)
    s_diff = np.maximum(np.diff(s_env, prepend=0.0), 0.0)

    corr = signal.correlate(m_diff, s_diff, mode="full")
    lags = signal.correlation_lags(len(m_diff), len(s_diff), mode="full")
    max_lag_f = int(max_offset_sec / (hop / sr))
    valid_m = np.abs(lags) <= max_lag_f
    best_lag_f = int(lags[valid_m][int(np.argmax(corr[valid_m]))])
    shift_samples = best_lag_f * hop

    target_len = master_st.shape[1]
    aligned = np.zeros((2, target_len), dtype=np.float32)
    if shift_samples >= 0:
        copy_n = min(target_len - shift_samples, self_mix_st.shape[1])
        if copy_n > 0:
            aligned[:, shift_samples : shift_samples + copy_n] = self_mix_st[:, :copy_n]
    else:
        src_start = -shift_samples
        copy_n = min(target_len, max(0, self_mix_st.shape[1] - src_start))
        if copy_n > 0:
            aligned[:, :copy_n] = self_mix_st[:, src_start : src_start + copy_n]

    return aligned, round(best_lag_f * (hop / sr), 3)


def _find_self_mix_path(song_dir: Path, raw_root: Path | None, song_name: str) -> Path | None:
    candidates: list[Path] = []
    if raw_root is not None:
        raw_dir = raw_root / song_name
        if raw_dir.exists():
            candidates.extend(sorted(raw_dir.iterdir()))
    candidates.extend(sorted(song_dir.iterdir()))
    for f in candidates:
        if not f.is_file():
            continue
        low = f.name.lower()
        if f.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"} and "自mix" in low and "オケなし" not in low:
            return f
    return None


def _resolve_song_artifacts(song_dir: Path, report: dict) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    master_wav = Path(report["master_wav"]) if report.get("master_wav") else None
    if master_wav is None or not master_wav.exists():
        master_wav = next(song_dir.glob("*_AutoMIX_Master.wav"), None)

    vocal_bus_wav = Path(report["vocal_bus_wav"]) if report.get("vocal_bus_wav") else None
    if vocal_bus_wav is None or not vocal_bus_wav.exists():
        vocal_bus_wav = next(song_dir.glob("*_AutoMIX_VocalBus.wav"), None)

    inst_wav = song_dir / "Media" / "01_offvocal_unmasked.wav"
    if not inst_wav.exists():
        inst_wav = None

    song_file = Path(report["studio_one_song"]) if report.get("studio_one_song") else None
    if song_file is None or not song_file.exists():
        song_file = next(song_dir.glob("*.song"), None)

    return master_wav, vocal_bus_wav, inst_wav, song_file


def _verify_studio_one_song_integrity(song_file: Path | None) -> tuple[bool, int, bool, bool]:
    """Verify `.song` ZIP has 0 `<VST3PluginState>` leaks, valid XMLs, `tempoFollow=0` & `start == offset` sync."""
    if song_file is None or not song_file.exists():
        return False, -1, False, False

    juce_leak_count = 0
    xml_valid = True
    sync_locked = True
    try:
        with zipfile.ZipFile(song_file, "r") as zf:
            names = set(zf.namelist())
            for name in names:
                raw = zf.read(name)
                if b"<VST3PluginState>" in raw:
                    juce_leak_count += raw.count(b"<VST3PluginState>")
            for core_xml in _CORE_SONG_XML_PATHS:
                if core_xml not in names:
                    if core_xml != "Devices/mixerconsole.xml":
                        xml_valid = False
                    continue
                txt = zf.read(core_xml).decode("utf-8").lstrip("\ufeff")
                if "xmlns:x=" not in txt:
                    txt = re.sub(
                        r"^(\s*(?:<\?xml[^?]*\?>\s*)?(?:<!--[\s\S]*?-->\s*)*<[A-Za-z0-9_:.-]+)(?=[\s>/])",
                        r'\1 xmlns:x="http://www.presonus.com/xml"',
                        txt,
                        count=1,
                    )
                root_el = ET.fromstring(txt)
                if core_xml == "Song/song.xml":
                    for mtrk in root_el.iter("MediaTrack"):
                        if mtrk.attrib.get("mediaType") == "Audio" and mtrk.attrib.get("tempoFollow", "0") != "0":
                            sync_locked = False
                        for aev in mtrk.iter("AudioEvent"):
                            if "start" in aev.attrib and "offset" in aev.attrib:
                                if abs(float(aev.attrib["start"]) - float(aev.attrib["offset"])) > 1e-3:
                                    sync_locked = False
    except Exception:
        return False, juce_leak_count, False, False

    return (juce_leak_count == 0 and xml_valid and sync_locked), juce_leak_count, xml_valid, sync_locked


def _window_around_span(
    start_sec: float,
    end_sec: float,
    clip_sec: float,
    total_dur: float,
) -> tuple[float, float]:
    eff_clip = min(clip_sec, total_dur)
    mid = 0.5 * (start_sec + end_sec)
    w0 = max(0.0, mid - 0.5 * eff_clip)
    w1 = w0 + eff_clip
    if w1 > total_dur:
        w1 = total_dur
        w0 = max(0.0, w1 - eff_clip)
    return round(w0, 3), round(w1, 3)


def _locate_diagnostic_spans(
    report: dict,
    vocal_bus_st: np.ndarray,
    inst_st: np.ndarray | None,
    sr: int,
    clip_sec: float,
    total_dur: float,
) -> dict[str, dict[str, object]]:
    sections: list[dict] = list(report.get("sections", []))
    anchor_lbl = report.get("anchor_section", "")

    # 1. Anchor chorus span
    anchor_sec = next((s for s in sections if s.get("is_anchor")), None)
    if anchor_sec is None and anchor_lbl:
        anchor_sec = next((s for s in sections if s.get("label") == anchor_lbl), None)
    if anchor_sec is None and sections:
        anchor_sec = sections[0]

    if anchor_sec is not None:
        a_s, a_e = _window_around_span(
            float(anchor_sec["start_sec"]), float(anchor_sec["end_sec"]), clip_sec, total_dur
        )
        a_name = str(anchor_sec.get("label", "Anchor_Chorus"))
    else:
        a_s, a_e = _window_around_span(0.0, min(clip_sec, total_dur), clip_sec, total_dur)
        a_name = "Anchor_Chorus"

    # 2. Quietest sung breakdown/verse section (lowest inst_rms_db among non-instrumental sections)
    non_inst = [s for s in sections if "instrumental" not in str(s.get("label", "")).lower()]
    breakdown_cands = [
        s
        for s in non_inst
        if not s.get("is_anchor")
        and not ("chorus" in str(s.get("label", "")).lower() and "prechorus" not in str(s.get("label", "")).lower())
    ]
    pool_q = breakdown_cands or [s for s in non_inst if not s.get("is_anchor")] or non_inst
    if pool_q:
        quiet_sec = min(pool_q, key=lambda s: float(s.get("inst_rms_db", 0.0)))
        q_s, q_e = _window_around_span(
            float(quiet_sec["start_sec"]), float(quiet_sec["end_sec"]), clip_sec, total_dur
        )
        q_name = str(quiet_sec.get("label", "Quiet_Drop"))
    else:
        q_s, q_e = _window_around_span(0.0, min(clip_sec, total_dur), clip_sec, total_dur)
        q_name = "Quiet_Drop"

    # 3. High sibilance window: 12s active window with highest 5.5-9.5 kHz sibilant energy on vocal bus
    vb_mono = 0.5 * (vocal_bus_st[0] + vocal_bus_st[1])
    n_vb = len(vb_mono)
    win_samp = min(n_vb, max(1, int(round(clip_sec * sr))))
    hop_samp = max(int(round(0.5 * sr)), 256)
    sib_hi = min(9500.0, sr * 0.45)
    sib_lo = min(5500.0, sib_hi * 0.75)

    if n_vb <= win_samp or sib_hi <= sib_lo + 100.0:
        s_s, s_e = 0.0, round(min(clip_sec, total_dur), 3)
    else:
        sos_sib = signal.butter(3, [sib_lo, sib_hi], btype="bandpass", fs=sr, output="sos")
        sib_sig = signal.sosfilt(sos_sib, vb_mono.astype(np.float64))
        csum_sib = np.cumsum(np.insert(sib_sig**2, 0, 0.0))
        csum_broad = np.cumsum(np.insert(vb_mono.astype(np.float64) ** 2, 0, 0.0))

        starts = np.arange(0, n_vb - win_samp + 1, hop_samp)
        sib_means = (csum_sib[starts + win_samp] - csum_sib[starts]) / float(win_samp)
        broad_means = (csum_broad[starts + win_samp] - csum_broad[starts]) / float(win_samp)
        act_thresh = float(np.percentile(broad_means, 90)) * 0.04
        valid = broad_means >= act_thresh
        cand_scores = np.where(valid, sib_means, -1.0)
        best_idx = int(np.argmax(cand_scores)) if np.any(valid) else int(np.argmax(sib_means))
        best_s0 = int(starts[best_idx])
        s_s = round(best_s0 / float(sr), 3)
        s_e = round(min(total_dur, (best_s0 + win_samp) / float(sr)), 3)

    # Suppress unused-argument lint while reserving inst_st for future mid-band fallback
    _ = inst_st

    return {
        "anchor_chorus": {"start_sec": a_s, "end_sec": a_e, "section_label": a_name},
        "quiet_drop": {"start_sec": q_s, "end_sec": q_e, "section_label": q_name},
        "high_sibilance": {"start_sec": s_s, "end_sec": s_e, "section_label": "Peak_Sibilance_Window"},
    }


def _evaluate_health_checks(
    report: dict,
    vocal_bus_st: np.ndarray,
    inst_st: np.ndarray | None,
    sr: int,
    anchor_span_sec: tuple[float, float],
    song_file: Path | None,
) -> tuple[dict[str, bool], dict[str, object]]:
    comp = report.get("comparison_anchor_chorus", {})
    tgt = dict(comp.get("pro_target", {}))
    bus = dict(comp.get("automix_full_vocal_bus") or comp.get("automix_anchor_core") or {})

    # Fallback to direct audio profile measurement when report lacks v6 K-LUFS fields
    if ("k_lufs_vi_ratio_db" not in bus or "vocal_intrinsic_bands_db" not in bus) and inst_st is not None:
        i0 = max(0, int(anchor_span_sec[0] * sr))
        i1 = min(vocal_bus_st.shape[1], inst_st.shape[1], int(anchor_span_sec[1] * sr))
        if i1 - i0 >= int(0.5 * sr):
            prof = compute_section_relative_profile(vocal_bus_st[:, i0:i1], inst_st[:, i0:i1], sr=sr)
            bus = prof.to_dict()

    achieved_klufs = float(bus.get("k_lufs_vi_ratio_db", 0.0))
    target_klufs = float(tgt.get("k_lufs_vi_ratio_db", achieved_klufs))
    klufs_abs_err = abs(achieved_klufs - target_klufs)

    bus_intr = bus.get("vocal_intrinsic_bands_db", {})
    tgt_intr = tgt.get("vocal_intrinsic_bands_db", {})
    achieved_body = float(bus_intr.get("body_warmth", -7.0))
    target_body = float(tgt_intr.get("body_warmth", achieved_body))
    achieved_mud = float(bus_intr.get("boxiness_mud", -4.2))
    target_mud = float(tgt_intr.get("boxiness_mud", achieved_mud))
    achieved_air = float(bus_intr.get("air_sheen", -20.0))
    target_air = float(tgt_intr.get("air_sheen", -19.0))
    achieved_sib = float(bus_intr.get("sibilance", -19.0))
    target_sib = float(tgt_intr.get("sibilance", -18.0))
    achieved_pres = float(bus_intr.get("core_presence", -10.0))

    s1_ok, juce_leak_count, xml_valid, sync_locked = _verify_studio_one_song_integrity(song_file)

    no_muffling = (
        achieved_air >= max(target_air - 3.2, -22.5)
        and achieved_sib >= max(target_sib - 3.2, -21.0)
        and achieved_mud <= target_mud + 1.15
    )

    checks = {
        "klufs_error_db": bool(klufs_abs_err <= 0.85),
        "no_lower_mid_scoop": bool(achieved_body >= target_body - 1.8),
        "no_hf_sizzle": bool(achieved_air <= -17.5),
        "no_vocal_muffling": bool(no_muffling),
        "studio_one_zero_juce_leak": bool(s1_ok),
    }
    metrics: dict[str, object] = {
        "achieved_klufs": round(achieved_klufs, 3),
        "target_klufs": round(target_klufs, 3),
        "klufs_abs_error_db": round(klufs_abs_err, 3),
        "achieved_mid_klufs": round(float(bus.get("mid_k_lufs_vi_ratio_db", 0.0)), 3),
        "target_mid_klufs": round(float(tgt.get("mid_k_lufs_vi_ratio_db", 0.0)), 3),
        "partial_loudness_margin_db": round(float(bus.get("partial_loudness_margin_db", 0.0)), 3),
        "achieved_body_warmth": round(achieved_body, 3),
        "target_body_warmth": round(target_body, 3),
        "achieved_boxiness_mud": round(achieved_mud, 3),
        "target_boxiness_mud": round(target_mud, 3),
        "achieved_core_presence": round(achieved_pres, 3),
        "target_core_presence": round(float(tgt_intr.get("core_presence", achieved_pres)), 3),
        "achieved_sibilance": round(achieved_sib, 3),
        "target_sibilance": round(target_sib, 3),
        "achieved_air_sheen": round(achieved_air, 3),
        "target_air_sheen": round(target_air, 3),
        "achieved_wet_tail_db": round(float(bus.get("wet_tail_ratio_db", -22.0)), 2),
        "achieved_side_mid_db": round(float(bus.get("vocal_side_mid_db", -10.0)), 2),
        "juce_leak_count": int(juce_leak_count),
        "song_xml_valid": bool(xml_valid),
        "song_sync_locked": bool(sync_locked),
        "studio_one_song": str(song_file) if song_file is not None else None,
    }
    return checks, metrics


def extract_loudness_normalized_ab_clips(
    outputs_root: Path,
    out_ab_dir: Path,
    target_lufs: float = -14.0,
    clip_sec: float = 12.0,
    raw_root: Path | None = None,
) -> dict:
    """Extract -14 LUFS loudness-normalized diagnostic A/B excerpts & verify Do-Not-Regress health checks."""
    outputs_root = Path(outputs_root)
    out_ab_dir = Path(out_ab_dir)
    out_ab_dir.mkdir(parents=True, exist_ok=True)
    eff_raw_root = Path(raw_root) if raw_root is not None else (outputs_root.parent / "raw")

    songs_summary: list[dict] = []
    if not outputs_root.exists():
        return {
            "schema_version": "7.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "outputs_root": str(outputs_root),
            "out_ab_dir": str(out_ab_dir),
            "target_lufs": target_lufs,
            "true_peak_ceiling_dbfs": -1.0,
            "clip_sec": clip_sec,
            "songs_evaluated": 0,
            "songs_passed": 0,
            "all_passed": False,
            "songs": [],
        }

    for song_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        report_path = song_dir / "mix_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        master_wav, vocal_bus_wav, inst_wav, song_file = _resolve_song_artifacts(song_dir, report)
        if master_wav is None or not master_wav.exists() or vocal_bus_wav is None or not vocal_bus_wav.exists():
            continue

        sr = int(sf.info(str(master_wav)).samplerate)
        master_st, _ = load_audio(master_wav, target_sr=sr, mono=False)
        vocal_bus_st, _ = load_audio(vocal_bus_wav, target_sr=sr, mono=False)
        inst_st, _ = load_audio(inst_wav, target_sr=sr, mono=False) if inst_wav is not None else (None, sr)
        total_dur = master_st.shape[1] / float(sr)

        song_name = str(report.get("song_name", song_dir.name))
        self_mix_path = _find_self_mix_path(song_dir, eff_raw_root if eff_raw_root.exists() else None, song_name)
        self_mix_aligned: np.ndarray | None = None
        self_mix_offset_sec: float | None = None
        if self_mix_path is not None:
            raw_sm_st, _ = load_audio(self_mix_path, target_sr=sr, mono=False)
            self_mix_aligned, self_mix_offset_sec = _align_self_mix_to_master(raw_sm_st, master_st, sr=sr)

        spans = _locate_diagnostic_spans(report, vocal_bus_st, inst_st, sr=sr, clip_sec=clip_sec, total_dur=total_dur)
        song_ab_dir = out_ab_dir / song_dir.name
        song_ab_dir.mkdir(parents=True, exist_ok=True)

        clips_meta: dict[str, dict[str, object]] = {}
        for clip_key, span in spans.items():
            s_sec = float(span["start_sec"])
            e_sec = float(span["end_sec"])
            i0 = max(0, min(master_st.shape[1], int(round(s_sec * sr))))
            i1 = max(i0 + 1, min(master_st.shape[1], int(round(e_sec * sr))))

            am_norm, am_lufs, am_tp = _loudness_normalize_clip(
                master_st[:, i0:i1], sr=sr, target_lufs=target_lufs, true_peak_ceiling_dbfs=-1.0
            )
            am_path = song_ab_dir / f"{clip_key}_automix.wav"
            write_pcm24_wav(am_path, am_norm, sr)

            sm_path_str: str | None = None
            sm_lufs: float | None = None
            sm_tp: float | None = None
            if self_mix_aligned is not None:
                sm_norm, sm_lufs, sm_tp = _loudness_normalize_clip(
                    self_mix_aligned[:, i0:i1], sr=sr, target_lufs=target_lufs, true_peak_ceiling_dbfs=-1.0
                )
                sm_path = song_ab_dir / f"{clip_key}_selfmix.wav"
                write_pcm24_wav(sm_path, sm_norm, sr)
                sm_path_str = str(sm_path)

            clips_meta[clip_key] = {
                "label": clip_key,
                "section_label": span["section_label"],
                "start_sec": s_sec,
                "end_sec": e_sec,
                "duration_sec": round((i1 - i0) / float(sr), 2),
                "automix_wav": str(am_path),
                "automix_lufs": am_lufs,
                "automix_true_peak_dbfs": am_tp,
                "selfmix_wav": sm_path_str,
                "selfmix_lufs": sm_lufs,
                "selfmix_true_peak_dbfs": sm_tp,
            }

        anchor_span = (
            float(spans["anchor_chorus"]["start_sec"]),
            float(spans["anchor_chorus"]["end_sec"]),
        )
        checks, health_metrics = _evaluate_health_checks(
            report=report,
            vocal_bus_st=vocal_bus_st,
            inst_st=inst_st,
            sr=sr,
            anchor_span_sec=anchor_span,
            song_file=song_file,
        )
        chain_p = report.get("optimized_chain_params", {})
        songs_summary.append(
            {
                "run_id": song_dir.name,
                "song_name": song_name,
                "schema_version": str(report.get("schema_version", "legacy")),
                "resolved_style": str(report.get("resolved_target_profile", report.get("requested_style", "auto"))),
                "archetype": str(chain_p.get("archetype_chain", "unknown")),
                "bpm": float(report.get("bpm", 120.0)),
                "detected_key": str(report.get("detected_key", "-")),
                "has_self_mix": self_mix_aligned is not None,
                "self_mix_offset_sec": self_mix_offset_sec,
                "health_checks": checks,
                "all_health_checks_passed": all(checks.values()),
                "health_metrics": health_metrics,
                "clips": clips_meta,
            }
        )

    songs_passed = sum(1 for s in songs_summary if s["all_health_checks_passed"])
    summary = {
        "schema_version": "7.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "outputs_root": str(outputs_root),
        "out_ab_dir": str(out_ab_dir),
        "target_lufs": target_lufs,
        "true_peak_ceiling_dbfs": -1.0,
        "clip_sec": clip_sec,
        "songs_evaluated": len(songs_summary),
        "songs_passed": songs_passed,
        "all_passed": bool(songs_summary) and songs_passed == len(songs_summary),
        "songs": songs_summary,
    }
    (out_ab_dir / "ab_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _audio_src_for_html(wav_path: str | None, html_dir: Path) -> str | None:
    if not wav_path:
        return None
    p = Path(wav_path)
    try:
        return os.path.relpath(p.resolve(), start=html_dir.resolve()).replace("\\", "/")
    except ValueError:
        return p.resolve().as_uri()


def generate_ab_blind_console_html(summary: dict, html_path: Path) -> Path:
    """Generate a self-contained dark-mode A/B blind listening & regression audit console HTML."""
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_dir = html_path.parent

    payload_songs: list[dict] = []
    for s in summary.get("songs", []):
        clips_rel = {}
        for ck, cv in s.get("clips", {}).items():
            clips_rel[ck] = {
                **cv,
                "automix_src": _audio_src_for_html(cv.get("automix_wav"), html_dir),
                "selfmix_src": _audio_src_for_html(cv.get("selfmix_wav"), html_dir),
            }
        payload_songs.append({**s, "clips": clips_rel})

    client_data = {
        "generated_at": summary.get("generated_at", ""),
        "target_lufs": summary.get("target_lufs", -14.0),
        "songs_evaluated": summary.get("songs_evaluated", len(payload_songs)),
        "songs_passed": summary.get("songs_passed", 0),
        "songs": payload_songs,
    }
    json_blob = json.dumps(client_data, ensure_ascii=False).replace("</", "<\\/")

    html_doc = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AutoMIX v7 Blind A/B Listening &amp; Regression Console</title>
  <style>
    :root {{
      --bg: #0a0d14;
      --card: #121826;
      --card-2: #182032;
      --border: #253047;
      --text: #e6edf7;
      --muted: #8b9bb4;
      --accent: #38bdf8;
      --accent-2: #818cf8;
      --ok: #10b981;
      --fail: #f43f5e;
      --warn: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 20px;
      background: radial-gradient(circle at top right, #162038 0%, var(--bg) 55%);
      color: var(--text);
      font-family: "Inter", "Segoe UI", "Hiragino Kaku Gothic ProN", sans-serif;
      line-height: 1.45;
    }}
    .container {{ max-width: 1220px; margin: 0 auto; display: grid; gap: 16px; }}
    .header {{
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
      padding: 16px 20px; border-radius: 14px; background: var(--card); border: 1px solid var(--border);
    }}
    .title {{ font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em; }}
    .sub {{ font-size: 0.82rem; color: var(--muted); margin-top: 2px; }}
    .kpi-row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .kpi {{
      padding: 8px 14px; border-radius: 10px; background: var(--card-2); border: 1px solid var(--border);
      font-size: 0.78rem; color: var(--muted);
    }}
    .kpi strong {{ display: block; font-size: 1.05rem; color: var(--text); font-family: monospace; }}
    .card {{
      padding: 16px 20px; border-radius: 14px; background: var(--card); border: 1px solid var(--border);
    }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
    .btn {{
      padding: 7px 14px; border-radius: 8px; border: 1px solid var(--border); background: var(--card-2);
      color: var(--text); font-size: 0.82rem; font-weight: 600; cursor: pointer; transition: 0.15s;
    }}
    .btn:hover {{ border-color: var(--accent); }}
    .btn.active {{ background: var(--accent); color: #05111d; border-color: var(--accent); }}
    .btn.ok {{ background: rgba(16, 185, 129, 0.16); border-color: var(--ok); color: #6ee7b7; }}
    .btn.warn {{ background: rgba(245, 158, 11, 0.16); border-color: var(--warn); color: #fcd34d; }}
    .badge {{
      display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
      font-family: monospace;
    }}
    .badge.pass {{ background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .badge.fail {{ background: rgba(244, 63, 94, 0.18); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.4); }}
    .badge.neutral {{ background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid var(--border); }}
    .ab-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; margin-top: 12px; }}
    .deck {{
      padding: 14px; border-radius: 12px; background: var(--card-2); border: 2px solid var(--border);
      display: grid; gap: 10px; transition: border-color 0.15s;
    }}
    .deck.listening {{ border-color: var(--accent); box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.35); }}
    .deck-head {{ display: flex; justify-content: space-between; align-items: center; }}
    .deck-title {{ font-size: 1rem; font-weight: 700; }}
    .transport {{
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 12px 14px;
      border-radius: 10px; background: #0d121d; border: 1px solid var(--border); margin-top: 12px;
    }}
    input[type="range"] {{ flex: 1; min-width: 160px; accent-color: var(--accent); cursor: pointer; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th, td {{ padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em; }}
    tr.selected-row {{ background: rgba(56, 189, 248, 0.08); }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .controls-row {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; }}
    .vote-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 10px; font-size: 0.82rem; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="title">AutoMIX v7 Blind A/B Listening &amp; Regression Audit Console</div>
        <div class="sub">Loudness-matched at <span class="mono" id="hdr-lufs">-14.0 LUFS</span> (True-Peak &le; -1.0 dBFS) &middot; Generated <span id="hdr-date">-</span></div>
      </div>
      <div class="kpi-row">
        <div class="kpi">Songs Audited<strong id="kpi-songs">0</strong></div>
        <div class="kpi">Do-Not-Regress Pass<strong id="kpi-pass">0 / 0</strong></div>
        <div class="kpi">Blind Votes<strong id="kpi-votes">0</strong></div>
      </div>
    </div>

    <div class="card">
      <div class="controls-row">
        <div class="tabs" id="song-tabs"></div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <button class="btn active" id="btn-blind-mode">Blind X/Y Mode: ON</button>
          <button class="btn" id="btn-reshuffle">Re-Randomize X/Y</button>
          <button class="btn warn" id="btn-reveal">Reveal Identity</button>
        </div>
      </div>

      <div class="tabs" id="clip-tabs"></div>
      <div id="clip-info" class="sub"></div>

      <div class="transport">
        <button class="btn active" id="btn-play">Play / Pause [Space]</button>
        <button class="btn" id="btn-switch">Switch X &harr; Y [S]</button>
        <input type="range" id="seek-bar" min="0" max="12" step="0.02" value="0" />
        <span class="mono" id="time-readout" style="font-size:0.82rem;">0.00s / 0.00s</span>
      </div>

      <div class="ab-grid" id="decks-container"></div>

      <div class="vote-bar" id="vote-bar">
        <span style="color:var(--muted);">Blind Preference:</span>
        <button class="btn" data-vote="X">Prefer X</button>
        <button class="btn" data-vote="Y">Prefer Y</button>
        <button class="btn" data-vote="Tie">Tie / Equal</button>
        <span id="vote-status" class="mono" style="color:var(--muted);"></span>
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Multi-Song Do-Not-Regress Acoustic &amp; Studio One 5 Audit Matrix</div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Song / Run</th>
              <th>Archetype / BPM</th>
              <th>K-LUFS Err (&le;0.85)</th>
              <th>No Scoop (Body)</th>
              <th>No Muffling (Sib/Mud)</th>
              <th>No HF Sizzle (&le;-17.5)</th>
              <th>S1 Sync &amp; Zero JUCE</th>
              <th>Presence / WetTail / Side-Mid</th>
            </tr>
          </thead>
          <tbody id="audit-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const DATA = {json_blob};
    const CLIP_LABELS = {{
      anchor_chorus: "Anchor Chorus",
      quiet_drop: "Quiet Breakdown / Drop",
      high_sibilance: "High Sibilance Window"
    }};

    let curSongIdx = 0;
    let curClipKey = "anchor_chorus";
    let blindMode = true;
    let revealed = false;
    let activeSlot = "X";
    let isPlaying = false;
    const blindOrders = {{}};
    const votes = {{}};

    let audioX = null;
    let audioY = null;
    let rafId = null;

    function orderKey(sIdx, cKey) {{ return sIdx + "::" + cKey; }}

    function ensureOrder(sIdx, cKey, forceShuffle) {{
      const k = orderKey(sIdx, cKey);
      if (!blindOrders[k] || forceShuffle) {{
        blindOrders[k] = Math.random() < 0.5 ? ["automix", "selfmix"] : ["selfmix", "automix"];
      }}
      return blindOrders[k];
    }}

    function stopPlayback() {{
      if (audioX) audioX.pause();
      if (audioY) audioY.pause();
      isPlaying = false;
      if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
    }}

    function syncMuteState() {{
      if (audioX) audioX.muted = (activeSlot !== "X");
      if (audioY) audioY.muted = (activeSlot !== "Y");
      document.querySelectorAll(".deck").forEach(el => {{
        el.classList.toggle("listening", el.dataset.slot === activeSlot);
      }});
    }}

    function updateTimeLoop() {{
      const masterEl = activeSlot === "X" ? audioX : audioY;
      const slaveEl = activeSlot === "X" ? audioY : audioX;
      if (masterEl) {{
        const cur = masterEl.currentTime || 0;
        const dur = masterEl.duration || 0;
        const bar = document.getElementById("seek-bar");
        if (dur > 0 && isFinite(dur)) bar.max = dur.toFixed(2);
        bar.value = cur.toFixed(2);
        document.getElementById("time-readout").textContent = cur.toFixed(2) + "s / " + (isFinite(dur) ? dur.toFixed(2) : "0.00") + "s";
        if (slaveEl && isPlaying && Math.abs((slaveEl.currentTime || 0) - cur) > 0.05) {{
          try {{ slaveEl.currentTime = cur; }} catch (e) {{}}
        }}
      }}
      if (isPlaying) rafId = requestAnimationFrame(updateTimeLoop);
    }}

    function togglePlay() {{
      if (!audioX) return;
      if (isPlaying) {{
        stopPlayback();
      }} else {{
        isPlaying = true;
        audioX.play().catch(() => {{}});
        if (audioY) audioY.play().catch(() => {{}});
        syncMuteState();
        updateTimeLoop();
      }}
    }}

    function switchSlot(targetSlot) {{
      if (!audioY && targetSlot === "Y") return;
      const prevEl = activeSlot === "X" ? audioX : audioY;
      const nextEl = targetSlot === "X" ? audioX : audioY;
      if (prevEl && nextEl && Math.abs(nextEl.currentTime - prevEl.currentTime) > 0.03) {{
        try {{ nextEl.currentTime = prevEl.currentTime; }} catch (e) {{}}
      }}
      activeSlot = targetSlot;
      syncMuteState();
    }}

    function renderDecks() {{
      stopPlayback();
      const song = DATA.songs[curSongIdx];
      if (!song) return;
      const clip = song.clips[curClipKey];
      const container = document.getElementById("decks-container");
      if (!clip) {{ container.innerHTML = "<div class='sub'>Clip unavailable.</div>"; return; }}

      document.getElementById("clip-info").innerHTML =
        `Section: <strong style="color:var(--text)">${{clip.section_label}}</strong> &middot; Span <span class="mono">${{clip.start_sec.toFixed(2)}}s - ${{clip.end_sec.toFixed(2)}}s</span> (${{clip.duration_sec}}s)`;

      const hasSelf = Boolean(clip.selfmix_src);
      const order = hasSelf ? ensureOrder(curSongIdx, curClipKey, false) : ["automix"];
      const slots = hasSelf ? ["X", "Y"] : ["X"];

      container.innerHTML = slots.map((slot, idx) => {{
        const role = order[idx];
        const isAuto = role === "automix";
        const src = isAuto ? clip.automix_src : clip.selfmix_src;
        const lufs = isAuto ? clip.automix_lufs : clip.selfmix_lufs;
        const tp = isAuto ? clip.automix_true_peak_dbfs : clip.selfmix_true_peak_dbfs;
        let title = blindMode && !revealed ? `Candidate ${{slot}}` : (isAuto ? "AutoMIX Master" : "User Self-Mix");
        if (blindMode && revealed) title = `${{slot}} = ${{isAuto ? "AutoMIX Master" : "User Self-Mix"}}`;
        const badgeCls = (!blindMode || revealed) ? (isAuto ? "pass" : "neutral") : "neutral";
        const badgeTxt = (!blindMode || revealed) ? (isAuto ? "AUTOMIX" : "SELF-MIX") : "BLIND";
        return `
          <div class="deck ${{slot === activeSlot ? "listening" : ""}}" data-slot="${{slot}}">
            <div class="deck-head">
              <div class="deck-title">${{title}}</div>
              <span class="badge ${{badgeCls}}">${{badgeTxt}}</span>
            </div>
            <div class="sub mono">Loudness: ${{lufs}} LUFS &middot; True Peak: ${{tp}} dBFS</div>
            <audio id="audio-${{slot}}" controls loop preload="auto" src="${{src}}" style="width:100%; height:36px;"></audio>
            <button class="btn ${{slot === activeSlot ? "ok" : ""}}" onclick="window._listenSlot('${{slot}}')">Listen ${{slot}}</button>
          </div>
        `;
      }}).join("");

      audioX = document.getElementById("audio-X");
      audioY = document.getElementById("audio-Y");
      activeSlot = "X";
      syncMuteState();
      document.getElementById("seek-bar").value = 0;
      document.getElementById("time-readout").textContent = "0.00s / " + clip.duration_sec.toFixed(2) + "s";

      const vKey = orderKey(curSongIdx, curClipKey);
      const vVal = votes[vKey];
      document.getElementById("vote-bar").style.display = hasSelf ? "flex" : "none";
      document.getElementById("vote-status").textContent = vVal ? ("Recorded: " + vVal) : "";
    }}

    window._listenSlot = function(slot) {{
      switchSlot(slot);
      if (!isPlaying) togglePlay();
    }};

    function renderAll() {{
      document.getElementById("hdr-lufs").textContent = DATA.target_lufs.toFixed(1) + " LUFS";
      document.getElementById("hdr-date").textContent = DATA.generated_at || "-";
      document.getElementById("kpi-songs").textContent = String(DATA.songs_evaluated);
      document.getElementById("kpi-pass").textContent = `${{DATA.songs_passed}} / ${{DATA.songs_evaluated}}`;
      document.getElementById("kpi-votes").textContent = String(Object.keys(votes).length);

      document.getElementById("song-tabs").innerHTML = DATA.songs.map((s, idx) => {{
        const cls = idx === curSongIdx ? "btn active" : "btn";
        const mark = s.all_health_checks_passed ? "&#10003;" : "&#9888;";
        return `<button class="${{cls}}" onclick="window._selectSong(${{idx}})">${{mark}} ${{s.run_id}}</button>`;
      }}).join("");

      document.getElementById("clip-tabs").innerHTML = Object.keys(CLIP_LABELS).map(ck => {{
        const cls = ck === curClipKey ? "btn active" : "btn";
        return `<button class="${{cls}}" onclick="window._selectClip('${{ck}}')">${{CLIP_LABELS[ck]}}</button>`;
      }}).join("");

      document.getElementById("btn-blind-mode").textContent = "Blind X/Y Mode: " + (blindMode ? "ON" : "OFF");
      document.getElementById("btn-blind-mode").className = blindMode ? "btn active" : "btn";
      document.getElementById("btn-reveal").textContent = revealed ? "Hide Identity" : "Reveal Identity";

      document.getElementById("audit-tbody").innerHTML = DATA.songs.map((s, idx) => {{
        const hc = s.health_checks || {{}};
        const hm = s.health_metrics || {{}};
        const badge = (ok, txt) => `<span class="badge ${{ok ? "pass" : "fail"}}">${{ok ? "PASS" : "FAIL"}} ${{txt}}</span>`;
        return `
          <tr class="${{idx === curSongIdx ? "selected-row" : ""}}" onclick="window._selectSong(${{idx}})" style="cursor:pointer;">
            <td><strong>${{s.song_name}}</strong><div class="sub mono">${{s.run_id}}</div></td>
            <td class="mono">${{s.archetype}}<div class="sub">${{s.bpm}} BPM &middot; ${{s.detected_key}}</div></td>
            <td>${{badge(hc.klufs_error_db, (hm.klufs_abs_error_db ?? 0).toFixed(2) + "dB")}}</td>
            <td>${{badge(hc.no_lower_mid_scoop, (hm.achieved_body_warmth ?? 0).toFixed(1) + " vs " + (hm.target_body_warmth ?? 0).toFixed(1))}}</td>
            <td>${{badge(hc.no_vocal_muffling !== false, "sib=" + (hm.achieved_sibilance ?? 0).toFixed(1) + " mud=" + (hm.achieved_boxiness_mud ?? 0).toFixed(1))}}</td>
            <td>${{badge(hc.no_hf_sizzle, (hm.achieved_air_sheen ?? 0).toFixed(1) + "dB")}}</td>
            <td>${{badge(hc.studio_one_zero_juce_leak, "leak=" + (hm.juce_leak_count ?? "?") + " sync=" + String(hm.song_sync_locked ?? true))}}</td>
            <td class="mono">${{(hm.achieved_core_presence ?? 0).toFixed(1)}} / ${{(hm.achieved_wet_tail_db ?? 0).toFixed(1)}} / ${{(hm.achieved_side_mid_db ?? 0).toFixed(1)}} dB</td>
          </tr>
        `;
      }}).join("");

      renderDecks();
    }}

    window._selectSong = function(idx) {{ curSongIdx = idx; revealed = false; renderAll(); }};
    window._selectClip = function(ck) {{ curClipKey = ck; renderAll(); }};

    document.getElementById("btn-blind-mode").addEventListener("click", () => {{
      blindMode = !blindMode;
      revealed = !blindMode;
      renderAll();
    }});
    document.getElementById("btn-reshuffle").addEventListener("click", () => {{
      ensureOrder(curSongIdx, curClipKey, true);
      revealed = false;
      renderAll();
    }});
    document.getElementById("btn-reveal").addEventListener("click", () => {{
      revealed = !revealed;
      renderAll();
    }});
    document.getElementById("btn-play").addEventListener("click", togglePlay);
    document.getElementById("btn-switch").addEventListener("click", () => switchSlot(activeSlot === "X" ? "Y" : "X"));
    document.getElementById("seek-bar").addEventListener("input", e => {{
      const t = parseFloat(e.target.value) || 0;
      if (audioX) audioX.currentTime = t;
      if (audioY) audioY.currentTime = t;
    }});

    document.querySelectorAll("[data-vote]").forEach(btn => {{
      btn.addEventListener("click", () => {{
        const choice = btn.dataset.vote;
        const order = ensureOrder(curSongIdx, curClipKey, false);
        let mapped = choice;
        if (choice === "X") mapped = "X (" + order[0] + ")";
        if (choice === "Y") mapped = "Y (" + order[1] + ")";
        votes[orderKey(curSongIdx, curClipKey)] = mapped;
        document.getElementById("kpi-votes").textContent = String(Object.keys(votes).length);
        document.getElementById("vote-status").textContent = "Recorded: " + (blindMode && !revealed ? choice : mapped);
      }});
    }});

    document.addEventListener("keydown", e => {{
      if (e.target.tagName === "INPUT") return;
      if (e.code === "Space") {{ e.preventDefault(); togglePlay(); }}
      if (e.key === "s" || e.key === "S") switchSlot(activeSlot === "X" ? "Y" : "X");
    }});

    renderAll();
  </script>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")
    return html_path
