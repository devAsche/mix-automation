"""v2 Lossless Stem Consolidation & Stereo Stage Panning.

Eliminates the v1 flaws:
  1. Drops completely dead/empty stems (`peak < -75 dBFS`, e.g., empty export tracks).
  2. Preserves punch-in crossfades and phrase overlaps 100% intact instead of chopping
     25ms frames into a secondary track. Only sustained simultaneous vocal overdubs
     (`>= 1.4s` continuous overlap) are routed to `double_mono`.
  3. Automatically pans dual-mono harmony/choir tracks symmetrically across the stereo
     image (`L/R ±52%`, `±74%`, `±36%`) using equal-power panning so the center Mid
     channel stays crystal-clear for the Lead Vocal!
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
    abs_floor_db: float = -65.0,
    hold_ms: float = 160.0,
    fade_ms: float = 28.0,
) -> np.ndarray:
    """Gently mute pure mic noise between phrases while keeping breathy tails & whispers intact."""
    x = mono_audio.astype(np.float32, copy=True)
    peak_abs = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    if peak_abs < 10.0 ** (-75.0 / 20.0):
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


@dataclass(frozen=True)
class ConsolidatedStems:
    """Losslessly consolidated vocal stems with stereo-staged harmonies."""

    lead_mono: np.ndarray
    double_stereo: np.ndarray | None
    sub_octave_stereo: np.ndarray | None
    super_octave_stereo: np.ndarray | None
    harmonies_stereo_bus: np.ndarray | None
    harm1_mono: np.ndarray | None
    harm2_mono: np.ndarray | None
    has_user_harmonies: bool
    sr: int


def consolidate_vocal_stem_folder(
    stem_files: list[Path],
    target_sr: int = 44100,
) -> ConsolidatedStems:
    """Consolidate multi-take vocal stems without chopping phrase transitions, and pan harmonies wide."""
    main_paths: list[Path] = []
    suboc_paths: list[Path] = []
    superoc_paths: list[Path] = []
    harm_paths: list[Path] = []

    for p in sorted(stem_files):
        lname = p.stem.lower()
        if "suboc" in lname:
            suboc_paths.append(p)
        elif "superoc" in lname:
            superoc_paths.append(p)
        elif "harm" in lname or "choir" in lname or "chorus" in lname:
            harm_paths.append(p)
        elif "main" in lname or "lead" in lname or "vo" in lname:
            main_paths.append(p)

    if not main_paths:
        raise ValueError("No main vocal stem files (`main*`) found.")

    # Filter out silent/empty tracks (`peak < -75 dBFS`)
    valid_mains: list[np.ndarray] = []
    for p in main_paths:
        m_raw, _ = load_audio(p, target_sr=target_sr, mono=True)
        if float(np.max(np.abs(m_raw))) >= 10.0 ** (-75.0 / 20.0):
            valid_mains.append(strip_silence_gate(m_raw, target_sr))

    if not valid_mains:
        raise ValueError("All main vocal stem files are silent.")

    max_len = max(len(m) for m in valid_mains)
    padded_mains = [np.pad(m, (0, max_len - len(m))) for m in valid_mains]

    # Lossless summation for punch-ins/phrase handoffs!
    # Only separate sustained simultaneous overdubs (>= 1.4 seconds continuous overlap) into `double_stereo`.
    if len(padded_mains) == 1:
        lead = padded_mains[0]
        double_st = None
    else:
        hop = int(target_sr * 0.05)  # 50ms coarse activity grid
        n_f = int(np.ceil(max_len / hop))
        stack = np.stack(padded_mains, axis=0)
        pad_s = np.pad(stack, ((0, 0), (0, n_f * hop - max_len)))
        rms_grid = np.sqrt(np.mean(pad_s.reshape(len(padded_mains), n_f, hop) ** 2, axis=-1))

        active_grid = rms_grid > 1.5e-3
        multi_mask = np.sum(active_grid, axis=0) >= 2
        # Keep ONLY long sustained overlaps (>= 28 frames = 1.4s); short phrase crossfades stay 100% in lead!
        min_sustained = int(1.4 / 0.05)
        labels, n_ov = ndimage.label(multi_mask)
        sustained_ov = np.zeros_like(multi_mask)
        for idx_l in range(1, n_ov + 1):
            pts = np.where(labels == idx_l)[0]
            if len(pts) >= min_sustained:
                sustained_ov[pts] = True

        if not np.any(sustained_ov):
            # Pure punch-in / phrase-split tracks -> clean lossless sum!
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
            double_st = apply_equal_power_pan(dbl_l, -0.45) + apply_equal_power_pan(dbl_r, +0.45)

    def _stage_stereo_group(paths: list[Path], pan_positions: list[float]) -> np.ndarray | None:
        if not paths:
            return None
        acc = np.zeros((2, max_len), dtype=np.float32)
        valid_idx = 0
        for p in paths:
            st, _ = load_audio(p, target_sr=target_sr, mono=False)
            if float(np.max(np.abs(st))) < 10.0 ** (-75.0 / 20.0):
                continue
            L = strip_silence_gate(st[0], target_sr)
            R = strip_silence_gate(st[1], target_sr)
            n = min(len(L), max_len)
            lr_diff = float(np.sqrt(np.mean((L[:n] - R[:n]) ** 2)))
            if lr_diff > 1e-3:
                # Already genuinely stereo-panned by the user -> preserve exact stereo image!
                acc[0, :n] += L[:n]
                acc[1, :n] += R[:n]
            else:
                # Dual-mono center track -> pan symmetrically wide across L/R stage!
                pan_val = pan_positions[valid_idx % len(pan_positions)]
                panned = apply_equal_power_pan(L[:n], pan_val)
                acc[:, :n] += panned
                valid_idx += 1
        return acc if float(np.max(np.abs(acc))) > 1e-5 else None

    harm_pans = [-0.58, +0.58, -0.78, +0.78, -0.40, +0.40, -0.88, +0.88]
    harm_bus = _stage_stereo_group(harm_paths, harm_pans)
    sub_st = _stage_stereo_group(suboc_paths, [-0.22, +0.22])
    sup_st = _stage_stereo_group(superoc_paths, [-0.35, +0.35])

    harm1_m = None
    harm2_m = None
    if harm_bus is not None:
        harm1_m = harm_bus[0]
        harm2_m = harm_bus[1]

    return ConsolidatedStems(
        lead_mono=lead,
        double_stereo=double_st,
        sub_octave_stereo=sub_st,
        super_octave_stereo=sup_st,
        harmonies_stereo_bus=harm_bus,
        harm1_mono=harm1_m,
        harm2_mono=harm2_m,
        has_user_harmonies=(harm_bus is not None),
        sr=target_sr,
    )
