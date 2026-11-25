"""Japanese/English Lyrics & Phoneme Forced Alignment + Vowel/Consonant Protection Masks.

Provides two aligned capabilities for Stage 2 (phoneme-protected pitch/timing & gain riding)
and Stage 3/4 (lyric-indexed creative FX & director reports):
  1. Mixed Japanese/English lyric file parsing via `pykakasi` (preserving original text,
     classifying `"ja" | "en" | "mixed"`, and normalizing romaji `[a-z' ]` for CTC alignment).
  2. Acoustic vowel-core (`180-3600 Hz`) vs consonant-attack protection (`4.8-11 kHz` unvoiced
     fricatives + `1.8-5.5 kHz` sharp plosive/affricate bursts + breath edges), always computed
     even when `lyrics_path is None`.
  3. Forced alignment supporting BOTH:
     - Zero-download acoustic syllable/phrase Dynamic Programming (default fast path), mapping
       mora-weighted lyric lines onto active vocal phrases & intra-phrase valleys.
     - Optional `torchaudio.pipelines.MMS_FA` (`torchaudio.functional.forced_align` on GPU/CPU)
       when `use_mms_fa=True` and the local checkpoint cache is available, with automatic
       graceful fallback to acoustic DP so offline runs never hang or crash.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pykakasi
from scipy import ndimage, signal

_SECTION_TAG_RE = re.compile(
    r"^\s*(?:"
    r"\[[^\]]+\]|"
    r"【[^】]+】|"
    r"〈[^〉]+〉|"
    r"《[^》]+》|"
    r"<[^>]+>|"
    r"[-=_*#~]{2,}.*|"
    r"\((?:chorus|verse|intro|outro|bridge|pre-?chorus|hook|interlude|solo|rap|inst|ending|"
    r"サビ|ラスサビ|大サビ|落ちサビ|aメロ|bメロ|cメロ|イントロ|アウトロ|間奏|前奏|後奏|1番|2番|台詞)[^)]*\)"
    r")\s*$",
    re.IGNORECASE,
)
_METADATA_LINE_RE = re.compile(r"\bfeat\s*\.|\b(?:作詞|作曲|編曲|歌|vocal|music|lyrics)\s*[:：]", re.IGNORECASE)
_JA_CHAR_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff65-\uff9f]")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮヵヶ")


@dataclass(frozen=True)
class AlignedLyricLine:
    """Single lyric line aligned to vocal timeline."""

    index: int
    text: str
    romaji: str
    language: str  # "ja" | "en" | "mixed"
    start_sec: float
    end_sec: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LyricsPhonemeAlignment:
    """Complete lyric alignment and acoustic vowel/consonant protection masks."""

    lines: list[AlignedLyricLine]
    vowel_core_mask: np.ndarray
    consonant_attack_protect_mask: np.ndarray
    hop_samples: int
    sr: int
    method_used: str


def _safe_bandpass(x: np.ndarray, lo_hz: float, hi_hz: float, sr: int) -> np.ndarray:
    """Apply zero-phase bandpass safely even on low sample rates or short signals."""
    nyq = 0.5 * sr
    lo = max(20.0, min(lo_hz, nyq * 0.85))
    hi = max(lo + 40.0, min(hi_hz, nyq * 0.92))
    if len(x) < 32 or hi >= nyq * 0.98 or lo >= hi:
        return np.zeros_like(x, dtype=np.float32)
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    padlen = min(len(x) - 1, 3 * (2 * sos.shape[0] + 1))
    return signal.sosfiltfilt(sos, x.astype(np.float64), padlen=padlen).astype(np.float32)


def compute_vowel_consonant_masks(
    vocal_mono: np.ndarray,
    sr: int,
    hop_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Segment frame-level `vowel_core_mask` (180-3600 Hz) and `consonant_attack_protect_mask`.

    Protected regions combine:
      - Unvoiced fricatives (`4.8-11 kHz` energy rivaling or exceeding vowel core)
      - Sharp plosive/affricate transient bursts (`1.8-5.5 kHz` rapid upward onset)
      - Breath & phrase entry/exit edges adjacent to vowel nuclei
    """
    hop = hop_samples or max(128, int(round(sr * 0.01)))
    n = len(vocal_mono)
    n_frames = max(1, int(np.ceil(n / hop)))
    if n < 64 or float(np.max(np.abs(vocal_mono))) < 1e-6:
        return np.zeros(n_frames, dtype=bool), np.zeros(n_frames, dtype=bool), hop

    v_vowel = _safe_bandpass(vocal_mono, 180.0, 3600.0, sr)
    v_fric = _safe_bandpass(vocal_mono, 4800.0, 11000.0, sr)
    v_plos = _safe_bandpass(vocal_mono, 1800.0, 5500.0, sr)

    pad_len = n_frames * hop - n
    tot_f = np.pad(vocal_mono.astype(np.float32), (0, pad_len)).reshape(n_frames, hop)
    vow_f = np.pad(v_vowel, (0, pad_len)).reshape(n_frames, hop)
    frc_f = np.pad(v_fric, (0, pad_len)).reshape(n_frames, hop)
    pls_f = np.pad(v_plos, (0, pad_len)).reshape(n_frames, hop)

    tot_db = 20.0 * np.log10(np.sqrt(np.mean(tot_f**2, axis=1) + 1e-12))
    vow_db = 20.0 * np.log10(np.sqrt(np.mean(vow_f**2, axis=1) + 1e-12))
    frc_db = 20.0 * np.log10(np.sqrt(np.mean(frc_f**2, axis=1) + 1e-12))
    pls_db = 20.0 * np.log10(np.sqrt(np.mean(pls_f**2, axis=1) + 1e-12))

    # Zero-crossing rate helps separate unvoiced fricatives/breaths from periodic vowel nuclei
    zcr = np.mean(np.abs(np.diff(np.signbit(tot_f), axis=1)), axis=1)

    tot_peak = float(np.percentile(tot_db, 96))
    vow_peak = float(np.percentile(vow_db, 96))
    active = tot_db >= (tot_peak - 34.0)

    # 1. Unvoiced fricatives (4.8-11 kHz strong relative to vowel core, or high ZCR + active HF)
    hf_dominates = (frc_db - vow_db) > -6.5
    hf_burst = (frc_db >= float(np.percentile(frc_db, 92)) - 10.0) & ((frc_db - vow_db) > -11.0) & (zcr > 0.16)
    fric_mask = active & (hf_dominates | hf_burst)

    # 2. Plosive / affricate sharp transient bursts (1.8-5.5 kHz steep attack)
    pls_flux = np.maximum(np.diff(pls_db, prepend=pls_db[0]), 0.0)
    vow_local = ndimage.uniform_filter1d(vow_db, size=max(5, int(round(0.08 * sr / hop))))
    pls_spike = (pls_flux >= 5.5) & (pls_db >= float(np.percentile(pls_db, 85)) - 14.0)
    pls_burst = (pls_db - vow_local > -3.5) & (pls_flux >= 3.0) & (zcr > 0.11)
    raw_plos = active & (pls_spike | pls_burst)
    plos_dilate = max(2, int(round(0.025 * sr / hop)))
    plos_mask = ndimage.binary_dilation(raw_plos, structure=np.ones(plos_dilate, dtype=bool))

    # 3. Voiced harmonic vowel nuclei (180-3600 Hz strong relative to HF/noise)
    raw_vowel = active & (vow_db >= (vow_peak - 26.0)) & ((vow_db - frc_db) > 3.0)
    raw_vowel = ndimage.binary_opening(raw_vowel, structure=np.ones(max(2, int(round(0.025 * sr / hop))), dtype=bool))

    # 4. Breath & phrase entry/exit edges adjacent to active vowels without strong harmonic core
    near_vowel = ndimage.binary_dilation(raw_vowel, structure=np.ones(max(3, int(round(0.06 * sr / hop))), dtype=bool))
    breath_edge = active & near_vowel & (~raw_vowel) & (vow_db < (vow_peak - 14.0))

    consonant_protect = (fric_mask | plos_mask | breath_edge) & active
    vowel_core = raw_vowel & (~consonant_protect)

    # Keep interior of sustained steady vowels protected from isolated 1-frame glitches
    if np.any(vowel_core):
        vowel_core = ndimage.binary_closing(vowel_core, structure=np.ones(2, dtype=bool)) & (~consonant_protect)

    return vowel_core.astype(bool), consonant_protect.astype(bool), hop


def _classify_and_romanize_line(line: str, converter: pykakasi.kakasi) -> tuple[str, str, int]:
    """Return `(romaji, language, mora_count)` for a single cleaned lyric line."""
    norm = unicodedata.normalize("NFKC", line).strip()
    has_ja = bool(_JA_CHAR_RE.search(norm))
    has_en = bool(_EN_WORD_RE.search(norm))
    if has_ja and has_en:
        lang = "mixed"
    elif has_ja:
        lang = "ja"
    elif has_en:
        lang = "en"
    else:
        lang = "ja"

    items = converter.convert(norm)
    romaji_tokens: list[str] = []
    mora_count = 0

    for item in items:
        orig_seg = item.get("orig", "")
        hira_seg = item.get("hira", "")
        hep = item.get("hepburn", "")

        if _JA_CHAR_RE.search(orig_seg):
            for ch in hira_seg:
                if "\u3041" <= ch <= "\u3096" or ch in {"ー", "っ", "ん"}:
                    if ch not in _SMALL_KANA:
                        mora_count += 1
            cleaned = re.sub(r"[^a-z']+", " ", hep.lower()).strip()
            if cleaned:
                romaji_tokens.extend(cleaned.split())
        else:
            # English / alphanumeric segments
            for word in _EN_WORD_RE.findall(orig_seg.lower()):
                cleaned = re.sub(r"[^a-z']+", "", word)
                if cleaned:
                    romaji_tokens.append(cleaned)
                    sylls = len(re.findall(r"[aeiouy]+", cleaned))
                    mora_count += max(1, sylls)

    romaji = re.sub(r"\s+", " ", " ".join(romaji_tokens)).strip()
    if mora_count == 0 and romaji:
        mora_count = max(1, len(re.findall(r"[aeiouy]+", romaji)))
    return romaji, lang, max(1, mora_count) if romaji else 0


def parse_lyrics_lines(lyrics_path: Path) -> list[tuple[str, str, str, int]]:
    """Parse mixed Japanese/English `.txt` lyrics into `(text, romaji, language, mora_count)`."""
    if not lyrics_path.exists():
        return []
    raw_text = lyrics_path.read_text(encoding="utf-8", errors="ignore")
    converter = pykakasi.kakasi()
    parsed: list[tuple[str, str, str, int]] = []

    for idx, raw_line in enumerate(raw_text.splitlines()):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if _SECTION_TAG_RE.match(stripped):
            continue
        if idx == 0 and _METADATA_LINE_RE.search(stripped):
            continue
        if _METADATA_LINE_RE.match(stripped):
            continue
        romaji, lang, mora = _classify_and_romanize_line(stripped, converter)
        if not romaji:
            continue
        parsed.append((stripped, romaji, lang, mora))
    return parsed


def _try_mms_fa_align(
    vocal_mono: np.ndarray,
    sr: int,
    parsed_lines: list[tuple[str, str, str, int]],
) -> list[AlignedLyricLine] | None:
    """Run `torchaudio.pipelines.MMS_FA` + `forced_align` strictly if local cache exists."""
    try:
        import torch
        import torchaudio

        hub_dir = Path(torch.hub.get_dir())
        model_url = getattr(torchaudio.pipelines.MMS_FA, "_path", "")
        ckpt_name = Path(model_url).name if model_url else "model.pt"
        cached_candidates = [
            hub_dir / "checkpoints" / ckpt_name,
            hub_dir / "torchaudio" / "models" / ckpt_name,
        ]
        if not any(p.exists() for p in cached_candidates):
            return None

        bundle = torchaudio.pipelines.MMS_FA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = bundle.get_model(with_star=False).to(device)
        vocab = bundle.get_dict()

        target_sr = int(bundle.sample_rate)
        wav_t = torch.from_numpy(vocal_mono.astype(np.float32)).unsqueeze(0)
        if sr != target_sr:
            wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        wav_t = wav_t.to(device)

        with torch.inference_mode():
            emission, _ = model(wav_t)  # (1, T, C)

        char_ids: list[int] = []
        line_char_spans: list[tuple[int, int]] = []
        for _, romaji, _, _ in parsed_lines:
            c_start = len(char_ids)
            for ch in romaji.replace(" ", ""):
                if ch in vocab and ch != "-":
                    char_ids.append(int(vocab[ch]))
            c_end = len(char_ids)
            line_char_spans.append((c_start, c_end))

        if not char_ids or emission.shape[1] < len(char_ids):
            return None

        targets = torch.tensor([char_ids], dtype=torch.int32, device=device)
        aligned_tokens, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
        toks = aligned_tokens[0].cpu().numpy()
        frame_scores = torch.exp(scores[0]).cpu().numpy()
        dur_sec = len(vocal_mono) / float(sr)
        frame_dur = dur_sec / max(1, len(toks))

        # Walk CTC path monotonically to map each char occurrence to frame range
        char_frames: list[tuple[int, int, float]] = []
        cur_char_idx = 0
        prev_tok = 0
        in_char = False
        s_f = 0
        for f_i, tok in enumerate(toks):
            if tok != 0 and (not in_char or tok != prev_tok):
                if in_char:
                    char_frames.append((s_f, f_i, float(np.mean(frame_scores[s_f:f_i]))))
                    cur_char_idx += 1
                s_f = f_i
                in_char = True
            elif tok == 0 and in_char:
                char_frames.append((s_f, f_i, float(np.mean(frame_scores[s_f:f_i]))))
                cur_char_idx += 1
                in_char = False
            prev_tok = int(tok)
        if in_char:
            char_frames.append((s_f, len(toks), float(np.mean(frame_scores[s_f:]))))

        if len(char_frames) < len(char_ids):
            return None

        aligned: list[AlignedLyricLine] = []
        for idx, ((orig, romaji, lang, _), (cs, ce)) in enumerate(zip(parsed_lines, line_char_spans)):
            if ce <= cs:
                t0 = aligned[-1].end_sec if aligned else 0.0
                t1 = t0
                conf = 0.25
            else:
                f0 = char_frames[cs][0]
                f1 = char_frames[ce - 1][1]
                t0 = round(float(f0 * frame_dur), 3)
                t1 = round(float(max(f0 + 1, f1) * frame_dur), 3)
                conf = round(float(np.clip(np.mean([c[2] for c in char_frames[cs:ce]]), 0.1, 0.99)), 2)
            aligned.append(
                AlignedLyricLine(
                    index=idx,
                    text=orig,
                    romaji=romaji,
                    language=lang,
                    start_sec=t0,
                    end_sec=t1,
                    confidence=conf,
                )
            )
        return aligned
    except Exception:
        return None


def _align_acoustic_dp(
    vocal_mono: np.ndarray,
    sr: int,
    vowel_core_mask: np.ndarray,
    hop_samples: int,
    parsed_lines: list[tuple[str, str, str, int]],
) -> list[AlignedLyricLine]:
    """Zero-download DP aligning mora-weighted lyric lines onto active singing phrases & valleys."""
    n_lines = len(parsed_lines)
    if n_lines == 0:
        return []

    n = len(vocal_mono)
    n_frames = len(vowel_core_mask)
    dur_sec = max(0.05, n / float(sr))
    frame_dt = hop_samples / float(sr)

    pad_len = n_frames * hop_samples - n
    padded = np.pad(vocal_mono.astype(np.float32), (0, pad_len)).reshape(n_frames, hop_samples)
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(padded**2, axis=1) + 1e-12))
    rms_peak = float(np.percentile(rms_db, 96))
    active = vowel_core_mask | (rms_db >= (rms_peak - 28.0))

    # Bridge short gaps (< 0.22s) inside words, keep real phrase pauses (>= 0.28s)
    close_len = max(2, int(round(0.22 / frame_dt)))
    phrase_act = ndimage.binary_closing(active, structure=np.ones(close_len, dtype=bool))
    if not np.any(phrase_act):
        # Fallback to uniform timing when vocal track is completely silent
        step = dur_sec / n_lines
        return [
            AlignedLyricLine(
                index=i,
                text=txt,
                romaji=rom,
                language=lang,
                start_sec=round(i * step, 3),
                end_sec=round((i + 1) * step, 3),
                confidence=0.15,
            )
            for i, (txt, rom, lang, _) in enumerate(parsed_lines)
        ]

    # Build cumulative active-singing duration & energy-valley boundary candidates
    act_weight = active.astype(np.float64)
    cum_act = np.concatenate([[0.0], np.cumsum(act_weight)]) * frame_dt
    total_act_sec = float(cum_act[-1])

    # Identify candidate cut frames: all phrase entry/exit edges + local energy valleys inside phrases
    labels, n_phrases = ndimage.label(phrase_act)
    cand_set: set[int] = set()
    gap_anchors: set[int] = set()

    smooth_db = ndimage.gaussian_filter1d(rms_db.astype(np.float64), sigma=max(1.5, 0.04 / frame_dt))
    for p_i in range(1, n_phrases + 1):
        p_idx = np.where(labels == p_i)[0]
        if len(p_idx) * frame_dt < 0.08:
            continue
        p0, p1 = int(p_idx[0]), int(p_idx[-1]) + 1
        cand_set.add(p0)
        cand_set.add(p1)
        gap_anchors.add(p0)
        gap_anchors.add(p1)

        # Add internal dips (~every >= 0.22s) so multi-line rapid phrases split cleanly
        if (p1 - p0) * frame_dt >= 0.45:
            valleys, _ = signal.find_peaks(-smooth_db[p0:p1], distance=max(4, int(round(0.22 / frame_dt))), prominence=2.5)
            for v_rel in valleys:
                cand_set.add(p0 + int(v_rel))

    first_act = int(np.where(phrase_act)[0][0])
    last_act = int(np.where(phrase_act)[0][-1]) + 1
    cand_set.add(first_act)
    cand_set.add(last_act)

    # Ensure sufficient candidate grid granularity when a continuous phrase has few valleys
    min_cands = max(n_lines * 3, 24)
    if len(cand_set) < min_cands:
        active_indices = np.where(phrase_act)[0]
        step_idx = max(1, len(active_indices) // min_cands)
        for k in active_indices[::step_idx]:
            cand_set.add(int(k))

    cands = sorted(cand_set)
    # Guarantee at least n_lines + 1 monotonic points
    if len(cands) < n_lines + 1:
        cands = sorted(set(np.linspace(first_act, last_act, n_lines + 1, dtype=int).tolist()))

    B = len(cands)
    mora_counts = np.array([max(1, m) for _, _, _, m in parsed_lines], dtype=np.float64)
    cum_mora = np.concatenate([[0.0], np.cumsum(mora_counts)])
    total_mora = float(cum_mora[-1])
    sec_per_mora = total_act_sec / max(total_mora, 1.0)

    # DP state: dp[i, j] = min cost aligning first i lines ending at candidate boundary cands[j]
    inf = 1e12
    dp = np.full((n_lines + 1, B), inf, dtype=np.float64)
    back = np.full((n_lines + 1, B), -1, dtype=int)
    dp[0, 0] = 0.0

    cand_cum_act = np.array([cum_act[c] for c in cands], dtype=np.float64)
    boundary_bonus = np.array([0.0 if c in gap_anchors else 0.12 for c in cands], dtype=np.float64)
    target_cum_act = (cum_mora / max(total_mora, 1.0)) * total_act_sec

    for i in range(1, n_lines + 1):
        exp_dur = mora_counts[i - 1] * sec_per_mora
        exp_cum = target_cum_act[i]
        # Valid range for j leaves room for remaining lines
        j_min = i
        j_max = B - (n_lines - i)
        for j in range(j_min, j_max):
            # Evaluate predecessor k < j
            prev_vals = dp[i - 1, i - 1 : j]
            seg_durs = cand_cum_act[j] - cand_cum_act[i - 1 : j]
            # Penalize empty zero-speech segments & mismatch against proportional mora duration
            empty_pen = np.where(seg_durs < 0.04, 8.0, 0.0)
            dur_err = np.abs(seg_durs - exp_dur) / max(exp_dur, 0.25)
            cum_err = 0.45 * abs(cand_cum_act[j] - exp_cum) / max(sec_per_mora * 4.0, 0.5)
            costs = prev_vals + dur_err + empty_pen + cum_err + boundary_bonus[j]
            best_rel = int(np.argmin(costs))
            dp[i, j] = float(costs[best_rel])
            back[i, j] = (i - 1) + best_rel

    # Trace back from final boundary B - 1
    bounds_idx = [B - 1]
    cur_j = B - 1
    for i in range(n_lines, 0, -1):
        prev_j = int(back[i, cur_j])
        if prev_j < 0:
            prev_j = max(0, cur_j - 1)
        bounds_idx.append(prev_j)
        cur_j = prev_j
    bounds_idx.reverse()

    aligned: list[AlignedLyricLine] = []
    for i, (txt, rom, lang, mora) in enumerate(parsed_lines):
        f0 = cands[bounds_idx[i]]
        f1 = cands[bounds_idx[i + 1]]
        # Trim leading/trailing pure silence within the assigned segment for tight vocal bounds
        seg_act = np.where(phrase_act[f0:f1])[0]
        if len(seg_act) > 0:
            t0_f = f0 + int(seg_act[0])
            t1_f = f0 + int(seg_act[-1]) + 1
        else:
            t0_f, t1_f = f0, max(f0 + 1, f1)

        t0 = round(float(min(t0_f * frame_dt, dur_sec)), 3)
        t1 = round(float(min(max(t1_f * frame_dt, t0 + 0.05), dur_sec)), 3)
        act_d = float(cum_act[t1_f] - cum_act[t0_f])
        exp_d = max(0.15, mora * sec_per_mora)
        ratio_err = abs(act_d - exp_d) / exp_d
        conf = round(float(np.clip(0.92 - 0.35 * ratio_err, 0.30, 0.94)), 2)

        aligned.append(
            AlignedLyricLine(
                index=i,
                text=txt,
                romaji=rom,
                language=lang,
                start_sec=t0,
                end_sec=t1,
                confidence=conf,
            )
        )
    return aligned


def align_lyrics_and_phonemes(
    vocal_mono: np.ndarray,
    sr: int,
    lyrics_path: Path | None = None,
    use_mms_fa: bool = False,
) -> LyricsPhonemeAlignment:
    """Align lyrics (when provided) and compute vowel-core vs consonant-attack protection masks."""
    vowel_mask, consonant_mask, hop = compute_vowel_consonant_masks(vocal_mono, sr=sr)

    if lyrics_path is None:
        return LyricsPhonemeAlignment(
            lines=[],
            vowel_core_mask=vowel_mask,
            consonant_attack_protect_mask=consonant_mask,
            hop_samples=hop,
            sr=sr,
            method_used="acoustic_segmentation_only",
        )

    parsed_lines = parse_lyrics_lines(Path(lyrics_path))
    if not parsed_lines:
        return LyricsPhonemeAlignment(
            lines=[],
            vowel_core_mask=vowel_mask,
            consonant_attack_protect_mask=consonant_mask,
            hop_samples=hop,
            sr=sr,
            method_used="acoustic_segmentation_only",
        )

    if use_mms_fa:
        mms_lines = _try_mms_fa_align(vocal_mono, sr=sr, parsed_lines=parsed_lines)
        if mms_lines is not None:
            return LyricsPhonemeAlignment(
                lines=mms_lines,
                vowel_core_mask=vowel_mask,
                consonant_attack_protect_mask=consonant_mask,
                hop_samples=hop,
                sr=sr,
                method_used="mms_fa",
            )
        method = "acoustic_dp_fallback"
    else:
        method = "acoustic_dp"

    dp_lines = _align_acoustic_dp(
        vocal_mono=vocal_mono,
        sr=sr,
        vowel_core_mask=vowel_mask,
        hop_samples=hop,
        parsed_lines=parsed_lines,
    )
    return LyricsPhonemeAlignment(
        lines=dp_lines,
        vowel_core_mask=vowel_mask,
        consonant_attack_protect_mask=consonant_mask,
        hop_samples=hop,
        sr=sr,
        method_used=method,
    )


def find_lyric_snippet_at_span(
    alignment: LyricsPhonemeAlignment | None,
    start_sec: float,
    end_sec: float,
    max_chars: int = 24,
) -> str:
    """Return a concise lyric snippet overlapping `[start_sec, end_sec]` (or nearest phrase)."""
    if alignment is None or not alignment.lines:
        return ""

    best_line: AlignedLyricLine | None = None
    best_overlap = 0.0
    for ln in alignment.lines:
        ov = min(end_sec + 0.15, ln.end_sec) - max(start_sec - 0.15, ln.start_sec)
        if ov > best_overlap:
            best_overlap = ov
            best_line = ln

    if best_line is None:
        mid = 0.5 * (start_sec + end_sec)
        nearest = min(
            alignment.lines,
            key=lambda ln: min(abs(ln.start_sec - mid), abs(ln.end_sec - mid)),
        )
        dist = min(abs(nearest.start_sec - mid), abs(nearest.end_sec - mid))
        if dist <= 0.85:
            best_line = nearest

    if best_line is None:
        return ""

    txt = best_line.text.strip()
    if len(txt) > max_chars:
        return txt[: max_chars - 1].rstrip() + "…"
    return txt
