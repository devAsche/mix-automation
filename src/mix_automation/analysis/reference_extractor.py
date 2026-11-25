"""v3 Reference Extractor: Cached GPU BS-RoFormer 35-Song Profiler + Official Off-Vocal Calibration.

Upgrades over v2:
  1. Aggregates all v3 psychoacoustic invariants (`k_lufs_vi_ratio_db`, `mid_k_lufs_vi_ratio_db`,
     `partial_loudness_margin_db`, 7-band `vocal_intrinsic_bands_db`, `inst_mid_density_index`,
     `inst_high_rolloff_db`, `inst_side_mid_db`) across all 35 pro reference covers.
  2. Performs Official Off-Vocal Calibration: wherever a reference folder contains the official
     distributed off-vocal WAV/MP3, aligns it against the `BS-RoFormer`-separated instrumental
     to quantify pro 2-mix mastering glue loudness delta, spectral tilt shift, and source-separation
     bias without guesswork!
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal

from mix_automation.analysis.relative_profile import (
    VOCAL_MIX_BANDS,
    SectionRelativeProfile,
    classify_backing_archetype,
    compute_backing_full_descriptors,
    compute_section_relative_profile,
)
from mix_automation.io.audio_loader import load_audio


class BSRoFormerSeparator:
    """Resident CUDA BS-RoFormer vocal/instrumental separator (`sdr_12.9755`)."""

    def __init__(self, cache_dir: Path, model_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir or (cache_dir.parent / "roformer_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.cache_dir / "_tmp_sep"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._sep = None

    def _ensure_loaded(self) -> None:
        if self._sep is not None:
            return
        from audio_separator.separator import Separator

        sep = Separator(
            model_file_dir=str(self.model_dir),
            output_dir=str(self.tmp_dir),
            output_format="WAV",
            use_autocast=True,
            use_soundfile=True,
            mdxc_params={
                "segment_size": 256,
                "override_model_segment_size": False,
                "batch_size": 8,
                "overlap": 2,
                "pitch_shift": 0,
            },
        )
        sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        self._sep = sep

    def separate(
        self,
        mix_path: Path,
        slug: str,
        target_sr: int = 44100,
    ) -> tuple[np.ndarray, np.ndarray, int, str]:
        """Separate `mix_path` into `(vocal_stereo, inst_stereo, sr, backend)` with disk caching."""
        voc_cache = self.cache_dir / f"{slug}_vocals.wav"
        inst_cache = self.cache_dir / f"{slug}_inst.wav"

        if voc_cache.exists() and inst_cache.exists():
            v_st, _ = load_audio(voc_cache, target_sr=target_sr, mono=False)
            i_st, _ = load_audio(inst_cache, target_sr=target_sr, mono=False)
            return v_st, i_st, target_sr, "bs_roformer_cached"

        self._ensure_loaded()
        assert self._sep is not None

        mix_st, _ = load_audio(mix_path, target_sr=target_sr, mono=False)
        tmp_in = self.tmp_dir / f"{slug}_in.wav"
        sf.write(str(tmp_in), mix_st.T, target_sr, subtype="PCM_24")

        out_files = self._sep.separate(str(tmp_in))
        tmp_in.unlink(missing_ok=True)

        voc_file: Path | None = None
        inst_file: Path | None = None
        for f_str in out_files:
            fp = Path(f_str)
            if not fp.is_absolute():
                fp = self.tmp_dir / fp
            low = fp.name.lower()
            if "(vocals)" in low or ("vocal" in low and "inst" not in low):
                voc_file = fp
            else:
                inst_file = fp

        if voc_file is None or inst_file is None or not voc_file.exists() or not inst_file.exists():
            raise RuntimeError(f"BS-RoFormer did not produce expected vocal/inst files for {slug}: {out_files}")

        v_tmp, _ = load_audio(voc_file, target_sr=target_sr, mono=False)
        i_tmp, _ = load_audio(inst_file, target_sr=target_sr, mono=False)
        sf.write(str(voc_cache), v_tmp.T, target_sr, subtype="PCM_16")
        sf.write(str(inst_cache), i_tmp.T, target_sr, subtype="PCM_16")
        voc_file.unlink(missing_ok=True)
        inst_file.unlink(missing_ok=True)

        return v_tmp, i_tmp, target_sr, "bs_roformer_cuda"


def _estimate_fast_median_f0(vocal_mono: np.ndarray, sr: int) -> float:
    chunk_len = min(len(vocal_mono), sr * 45)
    start = max(0, (len(vocal_mono) - chunk_len) // 2)
    seg = vocal_mono[start : start + chunk_len]
    f0, voiced, vp = librosa.pyin(seg, fmin=75.0, fmax=880.0, sr=sr, hop_length=1024)
    ok = (~np.isnan(f0)) & voiced & (vp > 0.72)
    if not np.any(ok):
        return 260.0
    return round(float(np.median(f0[ok])), 1)


def _find_primary_official_offvocal(song_dir: Path) -> Path | None:
    off_dir = song_dir / "offvocal"
    if not off_dir.exists():
        return None
    candidates = [
        p
        for p in off_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".webm"}
        and not any(
            x in p.name.lower()
            for x in ("+1", "+2", "+3", "+4", "+5", "+6", "-1", "-2", "-3", "-4", "-5", "hamo", "drums-only", "no-drums", "bass_less", "_part", "vo 0")
        )
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: (
            0 if p.suffix.lower() == ".wav" else 1,
            1 if "nomst" in p.name.lower() or "pre-master" in p.name.lower() else 0,
            len(p.name),
        )
    )
    return candidates[0]


def _pick_loudest_chorus_span(v_mono: np.ndarray, i_mono: np.ndarray, sr: int) -> tuple[int, int]:
    n = min(len(v_mono), len(i_mono))
    win = min(sr * 20, n)
    if n <= win:
        return 0, n
    sos = signal.butter(3, [1500.0, 9500.0], btype="bandpass", fs=sr, output="sos")
    ih = signal.sosfilt(sos, i_mono[:n])
    step = sr * 4
    best_s = 0
    best_sc = -999.0
    for s in range(0, n - win + 1, step):
        v_db = 20.0 * np.log10(np.sqrt(np.mean(v_mono[s : s + win] ** 2) + 1e-12))
        ih_db = 20.0 * np.log10(np.sqrt(np.mean(ih[s : s + win] ** 2) + 1e-12))
        if v_db > -38.0 and (ih_db + 0.3 * v_db) > best_sc:
            best_sc = ih_db + 0.3 * v_db
            best_s = s
    return best_s, best_s + win


def _calibrate_official_offvocal_vs_separated(
    sep_inst_st: np.ndarray,
    off_st: np.ndarray,
    sr: int,
    meter: pyln.Meter,
) -> dict | None:
    """Measure 2-mix mastering glue & spectral tilt delta between official distributed off-vocal and pro mix."""
    try:
        off_lufs = float(meter.integrated_loudness(off_st.T.astype(np.float64)))
        sep_lufs = float(meter.integrated_loudness(sep_inst_st.T.astype(np.float64)))
        if not (np.isfinite(off_lufs) and np.isfinite(sep_lufs)):
            return None

        # Compare high-vs-mid spectral tilt on loudest 30s core
        f_off, p_off = signal.welch(np.mean(off_st, axis=0), fs=sr, nperseg=4096)
        f_sep, p_sep = signal.welch(np.mean(sep_inst_st, axis=0), fs=sr, nperseg=4096)

        def _tilt(f, p):
            lo = float(10.0 * np.log10(np.sum(p[(f >= 160) & (f < 900)]) + 1e-18))
            hi = float(10.0 * np.log10(np.sum(p[(f >= 2000) & (f < 12000)]) + 1e-18))
            return hi - lo

        def _crest(st):
            m = np.mean(st, axis=0)
            pk = float(20.0 * np.log10(np.percentile(np.abs(m), 99.5) + 1e-12))
            rm = float(20.0 * np.log10(np.sqrt(np.mean(m**2)) + 1e-12))
            return pk - rm

        return {
            "official_offvocal_lufs": round(off_lufs, 2),
            "sep_inst_lufs": round(sep_lufs, 2),
            "lufs_delta_sep_minus_off_db": round(sep_lufs - off_lufs, 2),
            "crest_delta_sep_minus_off_db": round(_crest(sep_inst_st) - _crest(off_st), 2),
            "tilt_delta_sep_minus_off_db": round(_tilt(f_sep, p_sep) - _tilt(f_off, p_off), 2),
        }
    except Exception:
        return None


def aggregate_profiles_median(
    profiles: list[SectionRelativeProfile],
    archetype_label: str = "consensus",
) -> SectionRelativeProfile:
    """Compute robust median consensus across a subset of v3 reference profiles."""
    if not profiles:
        raise ValueError("Profile list is empty.")

    intrinsic = {
        b: round(float(np.median([p.vocal_intrinsic_bands_db[b] for p in profiles])), 3)
        for b in VOCAL_MIX_BANDS
    }
    band_margins = {
        b: round(float(np.median([p.band_margins_db[b] for p in profiles])), 3)
        for b in VOCAL_MIX_BANDS
    }
    inst_bands = {
        b: round(float(np.median([p.inst_band_levels_db[b] for p in profiles])), 3)
        for b in VOCAL_MIX_BANDS
    }

    return SectionRelativeProfile(
        vocal_rms_db=round(float(np.median([p.vocal_rms_db for p in profiles])), 3),
        inst_rms_db=round(float(np.median([p.inst_rms_db for p in profiles])), 3),
        broadband_vi_ratio_db=round(float(np.median([p.broadband_vi_ratio_db for p in profiles])), 3),
        k_lufs_vi_ratio_db=round(float(np.median([p.k_lufs_vi_ratio_db for p in profiles])), 3),
        mid_k_lufs_vi_ratio_db=round(float(np.median([p.mid_k_lufs_vi_ratio_db for p in profiles])), 3),
        partial_loudness_margin_db=round(float(np.median([p.partial_loudness_margin_db for p in profiles])), 3),
        vocal_crest_db=round(float(np.median([p.vocal_crest_db for p in profiles])), 3),
        inst_crest_db=round(float(np.median([p.inst_crest_db for p in profiles])), 3),
        crest_difference_db=round(float(np.median([p.crest_difference_db for p in profiles])), 3),
        position_stability_std_db=round(float(np.median([p.position_stability_std_db for p in profiles])), 3),
        harsh_resonance_index_db=round(float(np.median([p.harsh_resonance_index_db for p in profiles])), 3),
        syllabic_clarity_index=round(float(np.median([p.syllabic_clarity_index for p in profiles])), 4),
        wet_tail_ratio_db=round(float(np.median([p.wet_tail_ratio_db for p in profiles])), 2),
        vocal_side_mid_db=round(float(np.median([p.vocal_side_mid_db for p in profiles])), 2),
        inst_side_mid_db=round(float(np.median([p.inst_side_mid_db for p in profiles])), 2),
        inst_mid_density_index=round(float(np.median([p.inst_mid_density_index for p in profiles])), 3),
        inst_high_rolloff_db=round(float(np.median([p.inst_high_rolloff_db for p in profiles])), 2),
        backing_archetype=archetype_label,
        vocal_intrinsic_bands_db=intrinsic,
        band_margins_db=band_margins,
        inst_band_levels_db=inst_bands,
        active_frame_ratio=round(float(np.median([p.active_frame_ratio for p in profiles])), 3),
    )


def build_all_reference_profiles(
    references_root: Path,
    cache_dir: Path,
    output_json_path: Path,
    target_sr: int = 44100,
) -> dict:
    """Process all reference covers (`manifest*.json`) into psychoacoustic target profiles."""
    info_dir = references_root / "_info"
    manifest: list[dict] = []
    seen_dirs: set[str] = set()
    for mp in sorted(info_dir.glob("manifest*.json")):
        for entry in json.loads(mp.read_text(encoding="utf-8")):
            d_key = str(entry.get("dir", "")).replace("/", "\\").lower()
            if d_key and d_key not in seen_dirs:
                seen_dirs.add(d_key)
                manifest.append(entry)

    separator = BSRoFormerSeparator(cache_dir=cache_dir)
    meter = pyln.Meter(target_sr)

    per_song_records: list[dict] = []
    by_group: dict[str, list[SectionRelativeProfile]] = {}
    by_register_and_arch: dict[str, list[SectionRelativeProfile]] = {}
    by_register: dict[str, list[SectionRelativeProfile]] = {}
    all_chorus_profiles: list[SectionRelativeProfile] = []
    offvocal_calibs: list[dict] = []

    for idx, item in enumerate(manifest, start=1):
        rel_dir = Path(item["dir"])
        song_dir = references_root / rel_dir
        cover_files = list(song_dir.glob("cover_*.*"))
        if not cover_files:
            continue

        cover_path = cover_files[0]
        group = item["group"]
        slug = f"{group}_{rel_dir.name}"
        print(f"[{idx:02d}/{len(manifest):02d}] v3 Profiling {slug} ({item['title']}) ...", flush=True)

        v_st, i_st, sr, backend = separator.separate(cover_path, slug=slug, target_sr=target_sr)
        v_mono = np.mean(v_st, axis=0, dtype=np.float32)
        i_mono = np.mean(i_st, axis=0, dtype=np.float32)

        med_f0 = _estimate_fast_median_f0(v_mono, sr=sr)
        reg_key = "low_mid" if med_f0 < 295.0 else "high"

        s0, s1 = _pick_loudest_chorus_span(v_mono, i_mono, sr=sr)
        chorus_prof = compute_section_relative_profile(v_st[:, s0:s1], i_st[:, s0:s1], sr=sr)
        full_prof = compute_section_relative_profile(v_st, i_st, sr=sr)
        full_desc = compute_backing_full_descriptors(i_st, sr=sr)
        verse_intimacy_shift = round(
            float(np.clip(full_prof.k_lufs_vi_ratio_db - chorus_prof.k_lufs_vi_ratio_db, -1.5, 2.5)), 3
        )

        cover_2mix = np.clip(v_st + i_st, -1.0, 1.0)
        cover_lufs = round(float(meter.integrated_loudness(cover_2mix.T.astype(np.float64))), 2)

        off_file = _find_primary_official_offvocal(song_dir)
        off_calib = None
        if off_file is not None:
            try:
                off_st, _ = load_audio(off_file, target_sr=sr, mono=False)
                off_calib = _calibrate_official_offvocal_vs_separated(i_st, off_st, sr=sr, meter=meter)
                if off_calib is not None:
                    offvocal_calibs.append(off_calib)
            except Exception:
                off_calib = None

        arch = classify_backing_archetype(
            inst_crest_db=chorus_prof.inst_crest_db,
            inst_band_levels_db=chorus_prof.inst_band_levels_db,
            full_inst_mid_density_index=full_desc["full_inst_mid_density_index"],
            backing_dynamic_range_db=full_desc["backing_dynamic_range_db"],
            full_inst_crest_db=full_desc["full_inst_crest_db"],
            chorus_inst_mid_density_index=chorus_prof.inst_mid_density_index,
        )
        by_group.setdefault(group.lower(), []).append(chorus_prof)
        by_register.setdefault(reg_key, []).append(chorus_prof)
        by_register_and_arch.setdefault(f"{reg_key}_{arch}", []).append(chorus_prof)
        by_register_and_arch.setdefault(arch, []).append(chorus_prof)
        all_chorus_profiles.append(chorus_prof)

        per_song_records.append(
            {
                "slug": slug,
                "group": group,
                "mixer": item.get("mixer", ""),
                "vocal": item.get("vocal", ""),
                "title": item["title"],
                "genre_hint": item.get("genre_hint", ""),
                "separator_backend": backend,
                "vocal_median_f0_hz": med_f0,
                "vocal_register": reg_key,
                "backing_archetype": arch,
                "cover_master_lufs": cover_lufs,
                "verse_intimacy_shift_db": verse_intimacy_shift,
                "backing_full_descriptors": full_desc,
                "official_offvocal_calibration": off_calib,
                "chorus_profile": chorus_prof.to_dict(),
                "full_song_profile": full_prof.to_dict(),
            }
        )

    targets: dict[str, dict] = {
        "consensus": aggregate_profiles_median(all_chorus_profiles, "consensus").to_dict(),
    }
    for reg_k, plist in by_register.items():
        targets[f"register_{reg_k}"] = aggregate_profiles_median(plist, f"register_{reg_k}").to_dict()
    for ra_k, plist in by_register_and_arch.items():
        targets[ra_k] = aggregate_profiles_median(plist, ra_k).to_dict()
    for grp, plist in by_group.items():
        targets[grp] = aggregate_profiles_median(plist, grp).to_dict()

    calib_summary = {}
    if offvocal_calibs:
        calib_summary = {
            "num_paired_official_offvocals": len(offvocal_calibs),
            "median_master_minus_offvocal_lufs_db": round(
                float(np.median([r["cover_master_lufs"] - r["official_offvocal_calibration"]["official_offvocal_lufs"] for r in per_song_records if r["official_offvocal_calibration"]])),
                2,
            ),
            "median_sep_minus_offvocal_lufs_db": round(
                float(np.median([c["lufs_delta_sep_minus_off_db"] for c in offvocal_calibs])), 2
            ),
            "median_sep_minus_offvocal_crest_db": round(
                float(np.median([c["crest_delta_sep_minus_off_db"] for c in offvocal_calibs])), 2
            ),
            "median_sep_minus_offvocal_tilt_db": round(
                float(np.median([c["tilt_delta_sep_minus_off_db"] for c in offvocal_calibs])), 2
            ),
        }

    result = {
        "schema_version": "4.0",
        "separator_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "num_reference_songs": len(per_song_records),
        "official_offvocal_calibration": calib_summary,
        "target_profiles": targets,
        "songs": per_song_records,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute robust weighted median resistant to single-track production outliers."""
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    v_s = values[order]
    w_s = weights[order]
    cdf = np.cumsum(w_s) / (np.sum(w_s) + 1e-12)
    idx = int(np.searchsorted(cdf, 0.5))
    return float(v_s[min(idx, len(v_s) - 1)])


def synthesize_acoustic_knn_target(
    songs_records: list[dict],
    query_chorus_prof: SectionRelativeProfile,
    query_full_desc: dict[str, float],
    query_median_f0_hz: float,
    k: int = 9,
    taste_profile: SectionRelativeProfile | None = None,
    taste_weight: float = 0.0,
) -> tuple[SectionRelativeProfile, list[dict], float]:
    """Synthesize a continuous song-specific target profile via distance-weighted acoustic k-NN across all pro covers.

    Returns `(target_profile, top_neighbors_summary, learned_verse_intimacy_shift_db)`.
    """
    if not songs_records:
        raise ValueError("Cannot run acoustic k-NN with empty reference song list.")

    def _feat_vec(cp: dict, fd: dict, f0_hz: float) -> np.ndarray:
        return np.array(
            [
                float(cp.get("inst_mid_density_index", 0.55)),
                float(fd.get("full_inst_mid_density_index", cp.get("inst_mid_density_index", 0.50))),
                float(cp.get("inst_crest_db", 9.5)),
                float(fd.get("backing_dynamic_range_db", 3.0)),
                float(cp.get("inst_side_mid_db", -7.5)),
                float(cp.get("inst_high_rolloff_db", -8.5)),
                float(np.log2(max(f0_hz, 80.0))),
            ],
            dtype=np.float64,
        )

    # Feature importance weights: full/chorus density, dynamic range & vocal register matter most!
    feat_weights = np.array([1.35, 1.50, 0.90, 1.15, 0.70, 0.70, 1.40], dtype=np.float64)

    ref_feats = np.stack(
        [
            _feat_vec(
                s["chorus_profile"],
                s.get("backing_full_descriptors", {}),
                float(s.get("vocal_median_f0_hz", 260.0)),
            )
            for s in songs_records
        ],
        axis=0,
    )
    q_feat = _feat_vec(query_chorus_prof.to_dict(), query_full_desc, query_median_f0_hz)

    mu = np.mean(ref_feats, axis=0)
    sd = np.std(ref_feats, axis=0) + 1e-6
    z_ref = (ref_feats - mu) / sd
    z_q = (q_feat - mu) / sd

    dists = np.sqrt(np.sum(feat_weights * ((z_ref - z_q) ** 2), axis=1))
    k_eff = min(max(3, k), len(songs_records))
    nn_idx = np.argsort(dists)[:k_eff]
    nn_dists = dists[nn_idx]

    sigma = max(float(np.median(nn_dists)), 0.75)
    raw_w = np.exp(-0.5 * (nn_dists / sigma) ** 2)
    weights = raw_w / (np.sum(raw_w) + 1e-12)

    nn_records = [songs_records[int(i)] for i in nn_idx]
    nn_profs = [r["chorus_profile"] for r in nn_records]

    def _agg(field: str, digits: int = 3) -> float:
        vals = np.array([float(p[field]) for p in nn_profs], dtype=np.float64)
        # Blend weighted median (70%) + weighted mean (30%) for smooth continuity & outlier immunity
        val = 0.70 * _weighted_median(vals, weights) + 0.30 * float(np.sum(vals * weights))
        return round(val, digits)

    intrinsic = {}
    margins = {}
    inst_bands = {}
    for b in VOCAL_MIX_BANDS:
        v_i = np.array([float(p["vocal_intrinsic_bands_db"][b]) for p in nn_profs], dtype=np.float64)
        v_m = np.array([float(p["band_margins_db"][b]) for p in nn_profs], dtype=np.float64)
        v_ib = np.array([float(p["inst_band_levels_db"][b]) for p in nn_profs], dtype=np.float64)
        intrinsic[b] = round(0.70 * _weighted_median(v_i, weights) + 0.30 * float(np.sum(v_i * weights)), 3)
        margins[b] = round(0.70 * _weighted_median(v_m, weights) + 0.30 * float(np.sum(v_m * weights)), 3)
        inst_bands[b] = round(0.70 * _weighted_median(v_ib, weights) + 0.30 * float(np.sum(v_ib * weights)), 3)

    verse_shifts = np.array([float(r.get("verse_intimacy_shift_db", 0.5)) for r in nn_records], dtype=np.float64)
    learned_verse_shift = round(float(np.clip(np.sum(verse_shifts * weights), 0.0, 1.35)), 2)

    knn_prof = SectionRelativeProfile(
        vocal_rms_db=_agg("vocal_rms_db"),
        inst_rms_db=_agg("inst_rms_db"),
        broadband_vi_ratio_db=_agg("broadband_vi_ratio_db"),
        k_lufs_vi_ratio_db=_agg("k_lufs_vi_ratio_db"),
        mid_k_lufs_vi_ratio_db=_agg("mid_k_lufs_vi_ratio_db"),
        partial_loudness_margin_db=_agg("partial_loudness_margin_db"),
        vocal_crest_db=_agg("vocal_crest_db"),
        inst_crest_db=_agg("inst_crest_db"),
        crest_difference_db=_agg("crest_difference_db"),
        position_stability_std_db=_agg("position_stability_std_db"),
        harsh_resonance_index_db=_agg("harsh_resonance_index_db"),
        syllabic_clarity_index=_agg("syllabic_clarity_index", 4),
        wet_tail_ratio_db=_agg("wet_tail_ratio_db", 2),
        vocal_side_mid_db=_agg("vocal_side_mid_db", 2),
        inst_side_mid_db=_agg("inst_side_mid_db", 2),
        inst_mid_density_index=_agg("inst_mid_density_index", 3),
        inst_high_rolloff_db=_agg("inst_high_rolloff_db", 2),
        backing_archetype=f"knn_{k_eff}_continuous",
        vocal_intrinsic_bands_db=intrinsic,
        band_margins_db=margins,
        inst_band_levels_db=inst_bands,
        active_frame_ratio=_agg("active_frame_ratio"),
    )

    # Optional engineer taste blend (e.g. 35% Azusagawa or Sou aesthetic on top of acoustic k-NN!)
    tw = float(np.clip(taste_weight, 0.0, 0.85))
    if taste_profile is not None and tw > 0.01:
        def _mix(a: float, b: float, d: int = 3) -> float:
            return round((1.0 - tw) * a + tw * b, d)

        knn_prof = SectionRelativeProfile(
            vocal_rms_db=_mix(knn_prof.vocal_rms_db, taste_profile.vocal_rms_db),
            inst_rms_db=knn_prof.inst_rms_db,
            broadband_vi_ratio_db=_mix(knn_prof.broadband_vi_ratio_db, taste_profile.broadband_vi_ratio_db),
            k_lufs_vi_ratio_db=_mix(knn_prof.k_lufs_vi_ratio_db, taste_profile.k_lufs_vi_ratio_db),
            mid_k_lufs_vi_ratio_db=_mix(knn_prof.mid_k_lufs_vi_ratio_db, taste_profile.mid_k_lufs_vi_ratio_db),
            partial_loudness_margin_db=_mix(knn_prof.partial_loudness_margin_db, taste_profile.partial_loudness_margin_db),
            vocal_crest_db=_mix(knn_prof.vocal_crest_db, taste_profile.vocal_crest_db),
            inst_crest_db=knn_prof.inst_crest_db,
            crest_difference_db=_mix(knn_prof.crest_difference_db, taste_profile.crest_difference_db),
            position_stability_std_db=_mix(knn_prof.position_stability_std_db, taste_profile.position_stability_std_db),
            harsh_resonance_index_db=_mix(knn_prof.harsh_resonance_index_db, taste_profile.harsh_resonance_index_db),
            syllabic_clarity_index=_mix(knn_prof.syllabic_clarity_index, taste_profile.syllabic_clarity_index, 4),
            wet_tail_ratio_db=_mix(knn_prof.wet_tail_ratio_db, taste_profile.wet_tail_ratio_db, 2),
            vocal_side_mid_db=_mix(knn_prof.vocal_side_mid_db, taste_profile.vocal_side_mid_db, 2),
            inst_side_mid_db=knn_prof.inst_side_mid_db,
            inst_mid_density_index=knn_prof.inst_mid_density_index,
            inst_high_rolloff_db=knn_prof.inst_high_rolloff_db,
            backing_archetype=f"{knn_prof.backing_archetype}+taste({int(tw * 100)}%)",
            vocal_intrinsic_bands_db={
                b: _mix(knn_prof.vocal_intrinsic_bands_db[b], taste_profile.vocal_intrinsic_bands_db[b])
                for b in VOCAL_MIX_BANDS
            },
            band_margins_db={
                b: _mix(knn_prof.band_margins_db[b], taste_profile.band_margins_db[b])
                for b in VOCAL_MIX_BANDS
            },
            inst_band_levels_db=knn_prof.inst_band_levels_db,
            active_frame_ratio=knn_prof.active_frame_ratio,
        )

    neighbors_summary = [
        {
            "slug": r["slug"],
            "title": r["title"],
            "group": r["group"],
            "archetype": r.get("backing_archetype", ""),
            "weight_pct": round(float(w * 100.0), 1),
            "distance": round(float(d), 2),
        }
        for r, w, d in zip(nn_records[:6], weights[:6], nn_dists[:6])
    ]
    return knn_prof, neighbors_summary, learned_verse_shift
