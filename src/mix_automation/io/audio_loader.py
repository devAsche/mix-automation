"""v3 Intelligent Stem Consolidation: Acoustic Deduplication, Role Inference & Wide Harmony Staging.

Fixes every v2 harmony/stem weakness for arbitrary real-world DAW exports:
  1. Acoustic Duplicate Rejection: If a user re-exports mono tracks alongside earlier dual-mono
     stereo exports (e.g., `choir1.wav` + `mono - choir1.wav`), waveform envelope correlation
     (`r >= 0.985`) automatically drops the duplicate and prefers the clean 1-channel mono export.
  2. Universal Role Classification (Name + Acoustic Pitch-Offset Fallback): Recognizes Japanese
     and English Utattemita naming (`+3`, `-3`, `+5`, `-5`, `上ハモ`, `下ハモ`, `ハモ`, `harm`,
     `choir`, `chorus`, `suboc`, `superoc`, `oct`, `double`, `dbl`), and classifies unknown
     auxiliary filenames via median semitone interval relative to the lead vocal!
  3. Individual Active-RMS Leveling BEFORE Panning: Every auxiliary stem is noise-gated and
     staged to consistent active RMS (`-20 dBFS`) prior to equal-power L/R stereo distribution,
     preventing quiet `-45..-60 dBFS` exports from vanishing or skewing the stereo image!
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import ndimage, signal


def _find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    winget_path = Path(
        r"C:\Users\user\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
    )
    if winget_path.exists():
        return str(winget_path)
    raise FileNotFoundError("ffmpeg executable not found.")


def load_audio(
    path: Path,
    target_sr: int = 44100,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Load any audio file (WAV, FLAC, MP3, WebM/Opus, M4A) as float32 ndarray."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext in {".wav", ".flac", ".ogg"}:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = data.T
        if sr != target_sr:
            gcd = np.gcd(sr, target_sr)
            audio = signal.resample_poly(audio, target_sr // gcd, sr // gcd, axis=-1).astype(np.float32)
            sr = target_sr
    else:
        ffmpeg_exe = _find_ffmpeg()
        channels = 1 if mono else 2
        cmd = [
            ffmpeg_exe,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(target_sr),
            "-ac",
            str(channels),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        raw = np.frombuffer(proc.stdout, dtype=np.float32)
        audio = raw[np.newaxis, :] if channels == 1 else raw.reshape(-1, 2).T
        sr = target_sr

    if mono:
        return np.mean(audio, axis=0, dtype=np.float32), sr
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return audio.astype(np.float32, copy=False), sr


def strip_silence_gate(
    mono_audio: np.ndarray,
    sr: int,
    rel_gate_db: float = -54.0,
    abs_floor_db: float = -72.0,
    hold_ms: float = 160.0,
    fade_ms: float = 28.0,
) -> np.ndarray:
    """Gently mute pure mic noise between phrases while keeping breathy tails & whispers intact."""
    x = mono_audio.astype(np.float32, copy=True)
    peak_abs = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    if peak_abs < 10.0 ** (-78.0 / 20.0):
        return np.zeros_like(x)

    hop = max(int(sr * 0.01), 128)
    win = hop * 2
    pad = np.pad(x.astype(np.float64) ** 2, (win // 2, win - win // 2 - 1), mode="edge")
    csum = np.cumsum(np.insert(pad, 0, 0.0))
    rms_env = np.sqrt(np.maximum((csum[win:] - csum[:-win]) / win, 1e-12))[: len(x)]

    peak_rms = float(np.percentile(rms_env, 98))
    if peak_rms < 1e-6:
        return np.zeros_like(x)

    thresh = max(peak_rms * (10.0 ** (rel_gate_db / 20.0)), 10.0 ** (abs_floor_db / 20.0))
    open_mask = rms_env >= thresh

    hold_samples = max(int(sr * hold_ms / 1000.0), 1)
    w_hold = hold_samples * 2 + 1
    m_pad = np.pad(open_mask.astype(np.int32), (hold_samples, hold_samples), mode="edge")
    m_csum = np.cumsum(np.insert(m_pad, 0, 0))
    dilated = (m_csum[w_hold:] - m_csum[:-w_hold]) > 0

    fade_samples = max(int(sr * fade_ms / 1000.0), 4)
    smooth_kernel = np.hanning(fade_samples * 2 + 1).astype(np.float32)
    smooth_kernel /= np.sum(smooth_kernel)
    gain_curve = np.clip(signal.fftconvolve(dilated.astype(np.float32), smooth_kernel, mode="same"), 0.0, 1.0)

    return (x * gain_curve).astype(np.float32)


def apply_equal_power_pan(mono_sig: np.ndarray, pan: float) -> np.ndarray:
    """Pan a 1D mono signal into `(2, samples)` stereo using equal-power law (`pan` in `[-1.0, +1.0]`)."""
    angle = (np.clip(pan, -1.0, 1.0) + 1.0) * (np.pi / 4.0)
    g_l = float(np.cos(angle))
    g_r = float(np.sin(angle))
    return np.stack([mono_sig * g_l, mono_sig * g_r], axis=0).astype(np.float32)


def _active_level_normalize(sig: np.ndarray, target_active_rms_db: float = -20.0, max_boost_db: float = 38.0) -> np.ndarray:
    """Normalize a single vocal stem's active singing frames to consistent RMS (handles -58 dBFS stems!)."""
    if sig.size == 0:
        return sig
    env = np.abs(sig)
    pk = float(np.percentile(env, 99.5))
    if pk < 10.0 ** (-72.0 / 20.0):
        return np.zeros_like(sig)
    act = sig[env >= pk * (10.0 ** (-22.0 / 20.0))]
    if act.size < 256:
        return sig
    cur_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(act**2)) + 1e-12))
    gain_db = float(np.clip(target_active_rms_db - cur_rms_db, -18.0, max_boost_db))
    out = sig * (10.0 ** (gain_db / 20.0))
    out_pk = float(np.max(np.abs(out))) + 1e-9
    if out_pk > 0.88:
        out *= 0.88 / out_pk
    return out.astype(np.float32)


def _delay_samples(x: np.ndarray, shift_samples: int) -> np.ndarray:
    if shift_samples <= 0:
        return x
    out = np.zeros_like(x)
    if shift_samples < len(x):
        out[shift_samples:] = x[:-shift_samples]
    return out


def _deduplicate_stem_records(records: list[dict], sr: int) -> tuple[list[dict], list[str]]:
    """Drop acoustic duplicates (e.g. stereo dual-mono + mono re-export of same track), keeping 1ch mono."""
    hop = max(int(sr * 0.02), 256)
    # Sort so true 1-channel mono exports and newer files are evaluated first (kept as canonical!)
    ordered = sorted(records, key=lambda r: (0 if r["orig_channels"] == 1 else 1, -r["mtime"], r["path"].name))

    kept: list[dict] = []
    kept_sigs: list[np.ndarray] = []
    skipped_names: list[str] = []

    for rec in ordered:
        m = rec["mono"]
        n_f = len(m) // hop
        if n_f < 10:
            continue
        env = np.sqrt(np.mean(m[: n_f * hop].reshape(n_f, hop) ** 2, axis=1) + 1e-12)
        env_norm = env - np.mean(env)
        denom = float(np.linalg.norm(env_norm)) + 1e-12

        is_dup = False
        for k_rec, k_env in zip(kept, kept_sigs):
            c_len = min(len(env_norm), len(k_env))
            if c_len < 10:
                continue
            r_env = float(np.dot(env_norm[:c_len], k_env[:c_len]) / (denom * (float(np.linalg.norm(k_env[:c_len])) + 1e-12)))
            if r_env >= 0.985:
                # Confirm fine waveform correlation on active region to be 100% sure it's identical audio
                cm = min(len(m), len(k_rec["mono"]))
                a = m[:cm]
                b = k_rec["mono"][:cm]
                act = (np.abs(a) > 1e-4) | (np.abs(b) > 1e-4)
                if np.sum(act) > 500:
                    r_wav = float(np.corrcoef(a[act][:50000], b[act][:50000])[0, 1])
                    if abs(r_wav) >= 0.96:
                        is_dup = True
                        break
        if is_dup:
            skipped_names.append(rec["path"].name)
        else:
            kept.append(rec)
            kept_sigs.append(env_norm)

    return kept, skipped_names


def _classify_filename_role(stem_name: str) -> str | None:
    """Classify stem role from Japanese & English Utattemita naming conventions."""
    low = stem_name.lower()
    if any(k in low for k in ("suboc", "oct-", "low_oct", "下オク", "オクターブ下", "-12")):
        return "sub_octave"
    if any(k in low for k in ("superoc", "oct+", "high_oct", "上オク", "オクターブ上", "+12")):
        return "super_octave"
    if any(k in low for k in ("choir", "chorus", "コーラス")):
        return "choir"
    if any(k in low for k in ("+3", "+4", "+5", "上ハモ", "harm_up", "hamo_up", "upper")):
        return "upper_harm"
    if any(k in low for k in ("-3", "-4", "-5", "下ハモ", "harm_low", "hamo_low", "lower")):
        return "lower_harm"
    if any(k in low for k in ("harm", "hamo", "ハモ")):
        return "harmony"
    if any(k in low for k in ("double", "dbl", "ダブル")):
        return "double"
    if any(k in low for k in ("main", "lead", "メイン")) or re.search(r"(^|[^a-z])vo([^a-z]|$)", low):
        return "main"
    return None


def _infer_role_by_pitch_offset(aux_mono: np.ndarray, lead_mono: np.ndarray, sr: int) -> str:
    """Fallback acoustic role classifier using median semitone interval vs lead vocal."""
    n = min(len(aux_mono), len(lead_mono), sr * 60)
    if n < sr * 2:
        return "harmony"
    hop = 1024
    f_l, v_l, _ = librosa.pyin(lead_mono[:n], fmin=75.0, fmax=880.0, sr=sr, hop_length=hop)
    f_a, v_a, _ = librosa.pyin(aux_mono[:n], fmin=75.0, fmax=880.0, sr=sr, hop_length=hop)
    both = v_l & v_a & (~np.isnan(f_l)) & (~np.isnan(f_a))
    if np.sum(both) < 12:
        return "harmony"
    st_diff = float(np.median(librosa.hz_to_midi(f_a[both]) - librosa.hz_to_midi(f_l[both])))
    if st_diff >= 9.5:
        return "super_octave"
    if st_diff <= -9.5:
        return "sub_octave"
    if st_diff >= 1.5:
        return "upper_harm"
    if st_diff <= -1.5:
        return "lower_harm"
    return "double"


@dataclass(frozen=True)
class ConsolidatedStems:
    """Losslessly consolidated & acoustically deduplicated vocal stems with separate Main Harmony & Choir buses."""

    lead_mono: np.ndarray
    double_stereo: np.ndarray | None
    sub_octave_stereo: np.ndarray | None
    super_octave_stereo: np.ndarray | None
    harmonies_stereo_bus: np.ndarray | None
    harm1_mono: np.ndarray | None
    harm2_mono: np.ndarray | None
    has_user_harmonies: bool
    user_harmony_track_count: int
    deduplicated_files: list[str]
    stem_roles_summary: dict[str, list[str]]
    sr: int
    main_harmonies_stereo_bus: np.ndarray | None = None
    choir_stereo_bus: np.ndarray | None = None


def consolidate_vocal_stem_folder(
    stem_files: list[Path],
    target_sr: int = 44100,
) -> ConsolidatedStems:
    """Consolidate vocal stems with acoustic dedup, universal role inference, and leveled wide stereo staging."""
    raw_records: list[dict] = []
    for p in sorted(stem_files):
        try:
            info = sf.info(str(p))
            orig_ch = info.channels
        except Exception:
            orig_ch = 2
        st, _ = load_audio(p, target_sr=target_sr, mono=False)
        if float(np.max(np.abs(st))) < 10.0 ** (-75.0 / 20.0):
            continue
        mono = np.mean(st, axis=0, dtype=np.float32)
        lr_diff = float(np.sqrt(np.mean((st[0] - st[1]) ** 2)))
        raw_records.append(
            {
                "path": p,
                "orig_channels": orig_ch,
                "mtime": p.stat().st_mtime,
                "stereo": st,
                "mono": mono,
                "is_true_stereo": (orig_ch >= 2 and lr_diff > 1e-3),
                "role": _classify_filename_role(p.stem),
            }
        )

    deduped, skipped_dups = _deduplicate_stem_records(raw_records, sr=target_sr)

    main_recs = [r for r in deduped if r["role"] == "main"]
    if not main_recs:
        raise ValueError("No main vocal stem files (`main*`) found.")

    max_len = max(len(r["mono"]) for r in deduped)
    padded_mains = [
        np.pad(strip_silence_gate(r["mono"], target_sr), (0, max_len - len(r["mono"]))) for r in main_recs
    ]

    # Lossless summation for punch-ins/phrase handoffs; route >= 1.4s sustained overdubs to `double_stereo`
    if len(padded_mains) == 1:
        lead = padded_mains[0]
        double_st = None
    else:
        hop = int(target_sr * 0.05)
        n_f = int(np.ceil(max_len / hop))
        stack = np.stack(padded_mains, axis=0)
        pad_s = np.pad(stack, ((0, 0), (0, n_f * hop - max_len)))
        rms_grid = np.sqrt(np.mean(pad_s.reshape(len(padded_mains), n_f, hop) ** 2, axis=-1))

        active_grid = rms_grid > 1.5e-3
        multi_mask = np.sum(active_grid, axis=0) >= 2
        min_sustained = int(1.4 / 0.05)
        labels, n_ov = ndimage.label(multi_mask)
        sustained_ov = np.zeros_like(multi_mask)
        for idx_l in range(1, n_ov + 1):
            pts = np.where(labels == idx_l)[0]
            if len(pts) >= min_sustained:
                sustained_ov[pts] = True

        if not np.any(sustained_ov):
            lead = np.sum(stack, axis=0).astype(np.float32)
            double_st = None
        else:
            lead = np.zeros(max_len, dtype=np.float32)
            dbl_l = np.zeros(max_len, dtype=np.float32)
            dbl_r = np.zeros(max_len, dtype=np.float32)
            lead_w = np.ones_like(rms_grid, dtype=np.float32)
            dbl_w = np.zeros_like(rms_grid, dtype=np.float32)

            for f_i in np.where(sustained_ov)[0]:
                col = rms_grid[:, f_i]
                act_t = np.where(active_grid[:, f_i])[0]
                primary = act_t[int(np.argmax(col[act_t]))]
                for t_i in act_t:
                    if t_i != primary:
                        lead_w[t_i, f_i] = 0.0
                        dbl_w[t_i, f_i] = 1.0

            fade_k = np.hanning(int(target_sr * 0.08) * 2 + 1).astype(np.float32)
            fade_k /= np.sum(fade_k)
            for t_i in range(len(padded_mains)):
                lw_s = np.clip(signal.fftconvolve(np.repeat(lead_w[t_i], hop)[:max_len], fade_k, mode="same"), 0.0, 1.0)
                dw_s = np.clip(signal.fftconvolve(np.repeat(dbl_w[t_i], hop)[:max_len], fade_k, mode="same"), 0.0, 1.0)
                lead += padded_mains[t_i] * lw_s
                if t_i % 2 == 0:
                    dbl_l += padded_mains[t_i] * dw_s
                else:
                    dbl_r += padded_mains[t_i] * dw_s
            double_st = apply_equal_power_pan(dbl_l, -0.48) + apply_equal_power_pan(dbl_r, +0.48)

    # Classify any remaining unnamed auxiliary tracks via acoustic pitch interval vs lead
    role_buckets: dict[str, list[dict]] = {
        "main": main_recs,
        "double": [],
        "sub_octave": [],
        "super_octave": [],
        "upper_harm": [],
        "lower_harm": [],
        "harmony": [],
        "choir": [],
    }
    for r in deduped:
        if r["role"] == "main":
            continue
        role = r["role"] or _infer_role_by_pitch_offset(r["mono"], lead, sr=target_sr)
        role_buckets[role].append(r)

    def _stage_leveled_stereo_group(
        recs: list[dict],
        pan_positions: list[float],
        micro_haas_ms: float = 0.0,
    ) -> np.ndarray | None:
        if not recs:
            return None
        acc = np.zeros((2, max_len), dtype=np.float32)
        pan_idx = 0
        for rec in recs:
            n = min(len(rec["mono"]), max_len)
            if rec["is_true_stereo"]:
                L = _active_level_normalize(strip_silence_gate(rec["stereo"][0, :n], target_sr))
                R = _active_level_normalize(strip_silence_gate(rec["stereo"][1, :n], target_sr))
                acc[0, :n] += L
                acc[1, :n] += R
            else:
                m_clean = _active_level_normalize(strip_silence_gate(rec["mono"][:n], target_sr))
                if float(np.max(np.abs(m_clean))) < 1e-5:
                    continue
                pan_val = pan_positions[pan_idx % len(pan_positions)]
                panned = apply_equal_power_pan(m_clean, pan_val)
                if micro_haas_ms > 0.0:
                    # Subtle opposite-ear 6-11ms micro-delay creates natural choir ensemble width!
                    shift = int(target_sr * (micro_haas_ms + (pan_idx % 3) * 2.5) / 1000.0)
                    if pan_val < 0:
                        panned[1] = _delay_samples(panned[1], shift)
                    else:
                        panned[0] = _delay_samples(panned[0], shift)
                acc[:, :n] += panned
                pan_idx += 1
        if float(np.max(np.abs(acc))) < 1e-5:
            return None
        # Normalize stacked bus back to consistent -19 dBFS active RMS so N tracks don't explode headroom
        bus_m = 0.5 * (acc[0] + acc[1])
        pk = float(np.percentile(np.abs(bus_m), 99.5))
        if pk > 1e-5:
            act_m = bus_m[np.abs(bus_m) >= pk * 0.1]
            if act_m.size > 256:
                cur_db = float(20.0 * np.log10(np.sqrt(np.mean(act_m**2)) + 1e-12))
                acc *= float(10.0 ** ((-19.0 - cur_db) / 20.0))
        return acc.astype(np.float32)

    # Explicit double tracks merged with any sustained main overdub
    explicit_dbl = _stage_leveled_stereo_group(role_buckets["double"], [-0.55, +0.55], micro_haas_ms=5.0)
    if explicit_dbl is not None:
        double_st = explicit_dbl if double_st is None else (double_st + explicit_dbl)

    sub_st = _stage_leveled_stereo_group(role_buckets["sub_octave"], [-0.25, +0.25])
    sup_st = _stage_leveled_stereo_group(role_buckets["super_octave"], [-0.42, +0.42], micro_haas_ms=4.0)

    # Stage Main Harmonies (+3/-3/harm) SEPARATELY from Choirs (choir1..N):
    # Main harmonies use focused pans (±0.38..0.52) with 0ms Haas so they lock tightly inside the lead vocal
    # without binaural detachment; Choirs get lush wide ensemble panning (±0.38..0.88) + 8.5ms Haas!
    main_harm_parts: list[np.ndarray] = []
    if role_buckets["upper_harm"]:
        up_st = _stage_leveled_stereo_group(role_buckets["upper_harm"], [-0.40, +0.40, -0.52, +0.52], micro_haas_ms=0.0)
        if up_st is not None:
            main_harm_parts.append(up_st)
    if role_buckets["lower_harm"]:
        lo_st = _stage_leveled_stereo_group(role_buckets["lower_harm"], [+0.38, -0.38, +0.50, -0.50], micro_haas_ms=0.0)
        if lo_st is not None:
            main_harm_parts.append(lo_st)
    if role_buckets["harmony"]:
        # If there is only 1 mono harmony stem (e.g. Fall in Love `harm1.wav`), wrap symmetrically L/R via 9ms Haas
        # rather than hard-panning lopsided to one ear!
        if len(role_buckets["harmony"]) == 1 and not role_buckets["harmony"][0]["is_true_stereo"]:
            rec0 = role_buckets["harmony"][0]
            n0 = min(len(rec0["mono"]), max_len)
            m0 = _active_level_normalize(strip_silence_gate(rec0["mono"][:n0], target_sr), target_active_rms_db=-19.0)
            if float(np.max(np.abs(m0))) >= 1e-5:
                sym = np.zeros((2, max_len), dtype=np.float32)
                sym[0, :n0] = m0
                sym[1, :n0] = _delay_samples(m0, int(target_sr * 0.009)) * 0.94
                main_harm_parts.append(sym)
        else:
            gen_st = _stage_leveled_stereo_group(role_buckets["harmony"], [-0.42, +0.42, -0.54, +0.54], micro_haas_ms=0.0)
            if gen_st is not None:
                main_harm_parts.append(gen_st)

    def _center_lr_bus(parts: list[np.ndarray]) -> np.ndarray | None:
        if not parts:
            return None
        bus = np.sum(parts, axis=0).astype(np.float32)
        l_rms = float(np.sqrt(np.mean(bus[0] ** 2) + 1e-12))
        r_rms = float(np.sqrt(np.mean(bus[1] ** 2) + 1e-12))
        if l_rms > 1e-5 and r_rms > 1e-5:
            lr_ratio = float(np.clip(np.sqrt(r_rms / l_rms), 0.72, 1.38))
            bus[0] *= lr_ratio
            bus[1] /= lr_ratio
        return bus

    main_harm_bus = _center_lr_bus(main_harm_parts)

    choir_bus: np.ndarray | None = None
    if role_buckets["choir"]:
        ch_st = _stage_leveled_stereo_group(
            role_buckets["choir"],
            [-0.88, +0.88, -0.72, +0.72, -0.54, +0.54, -0.38, +0.38],
            micro_haas_ms=8.5,
        )
        if ch_st is not None:
            choir_bus = _center_lr_bus([ch_st])

    combined_parts = [b for b in (main_harm_bus, choir_bus) if b is not None]
    harm_bus = _center_lr_bus(combined_parts)
    primary_s1_harm = main_harm_bus if main_harm_bus is not None else choir_bus

    user_harm_cnt = (
        len(role_buckets["upper_harm"])
        + len(role_buckets["lower_harm"])
        + len(role_buckets["harmony"])
        + len(role_buckets["choir"])
    )

    roles_summary = {k: [r["path"].name for r in v] for k, v in role_buckets.items() if v}

    return ConsolidatedStems(
        lead_mono=lead,
        double_stereo=double_st,
        sub_octave_stereo=sub_st,
        super_octave_stereo=sup_st,
        harmonies_stereo_bus=harm_bus,
        harm1_mono=(primary_s1_harm[0] if primary_s1_harm is not None else None),
        harm2_mono=(primary_s1_harm[1] if primary_s1_harm is not None else None),
        has_user_harmonies=(harm_bus is not None),
        user_harmony_track_count=user_harm_cnt,
        deduplicated_files=skipped_dups,
        stem_roles_summary=roles_summary,
        sr=target_sr,
        main_harmonies_stereo_bus=main_harm_bus,
        choir_stereo_bus=choir_bus,
    )
