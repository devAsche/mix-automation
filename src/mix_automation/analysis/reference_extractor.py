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
    offvocal_calibration: dict | None = None,
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

    # Gentle bounded official off-vocal separation-bias calibration (<= 0.35 dB, never boosts HF hiss)
    if offvocal_calibration:
        tilt_bias = float(offvocal_calibration.get("median_sep_minus_offvocal_tilt_db", 0.0) or 0.0)
        crest_bias = float(offvocal_calibration.get("median_sep_minus_offvocal_crest_db", 0.0) or 0.0)
        tilt_corr = float(np.clip(tilt_bias * 0.5, -0.35, 0.35))
        crest_corr = float(np.clip(crest_bias * 0.5, -0.35, 0.35))
        lufs_nudge = float(np.clip(-0.5 * crest_corr, -0.25, 0.25))

        adj_margins = dict(knn_prof.band_margins_db)
        adj_margins["body_warmth"] = round(
            adj_margins["body_warmth"] + float(np.clip(-0.35 * tilt_corr, -0.35, 0.35)), 3
        )
        adj_margins["boxiness_mud"] = round(
            adj_margins["boxiness_mud"] + float(np.clip(-0.25 * tilt_corr, -0.35, 0.35)), 3
        )
        adj_margins["core_presence"] = round(
            adj_margins["core_presence"] + float(np.clip(0.50 * tilt_corr, -0.35, 0.35)), 3
        )
        hf_m_delta = float(np.clip(-0.50 * tilt_corr, -0.35, 0.0))
        adj_margins["sibilance"] = round(adj_margins["sibilance"] + hf_m_delta, 3)
        adj_margins["air_sheen"] = round(adj_margins["air_sheen"] + hf_m_delta, 3)

        knn_prof = SectionRelativeProfile(
            vocal_rms_db=knn_prof.vocal_rms_db,
            inst_rms_db=knn_prof.inst_rms_db,
            broadband_vi_ratio_db=round(knn_prof.broadband_vi_ratio_db + lufs_nudge, 3),
            k_lufs_vi_ratio_db=round(knn_prof.k_lufs_vi_ratio_db + lufs_nudge, 3),
            mid_k_lufs_vi_ratio_db=round(knn_prof.mid_k_lufs_vi_ratio_db + lufs_nudge, 3),
            partial_loudness_margin_db=round(knn_prof.partial_loudness_margin_db + lufs_nudge, 3),
            vocal_crest_db=knn_prof.vocal_crest_db,
            inst_crest_db=round(knn_prof.inst_crest_db - crest_corr, 3),
            crest_difference_db=round(knn_prof.crest_difference_db + crest_corr, 3),
            position_stability_std_db=knn_prof.position_stability_std_db,
            harsh_resonance_index_db=knn_prof.harsh_resonance_index_db,
            syllabic_clarity_index=knn_prof.syllabic_clarity_index,
            wet_tail_ratio_db=knn_prof.wet_tail_ratio_db,
            vocal_side_mid_db=knn_prof.vocal_side_mid_db,
            inst_side_mid_db=knn_prof.inst_side_mid_db,
            inst_mid_density_index=knn_prof.inst_mid_density_index,
            inst_high_rolloff_db=round(knn_prof.inst_high_rolloff_db - tilt_corr, 2),
            backing_archetype=knn_prof.backing_archetype,
            vocal_intrinsic_bands_db=knn_prof.vocal_intrinsic_bands_db,
            band_margins_db=adj_margins,
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


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".webm", ".ogg"}


def _estimate_bandwidth_and_integrity(audio_st: np.ndarray, audio_path: Path, sr: int) -> dict:
    """Inspect high-frequency brickwall cutoff, clipping, and stereo width of an audio candidate."""
    native_sr = sr
    try:
        native_sr = int(sf.info(str(audio_path)).samplerate)
    except Exception:
        pass

    mono = np.mean(audio_st, axis=0, dtype=np.float64)
    clip_ratio = float(np.mean(np.abs(audio_st) >= 0.999)) if audio_st.size > 0 else 0.0

    if audio_st.shape[0] >= 2:
        mid = 0.5 * (audio_st[0].astype(np.float64) + audio_st[1].astype(np.float64))
        side = 0.5 * (audio_st[0].astype(np.float64) - audio_st[1].astype(np.float64))
        side_mid_db = float(10.0 * np.log10((np.mean(side**2) + 1e-12) / (np.mean(mid**2) + 1e-12)))
    else:
        side_mid_db = -80.0
    is_mono_collapsed = side_mid_db < -42.0

    # Compute Welch PSD across contiguous 1.0s active frames (never skip samples, avoiding click aliasing)
    nperseg = min(8192, max(256, len(mono)))
    if len(mono) >= sr * 2:
        step = sr
        n_blk = len(mono) // step
        blocks = mono[: n_blk * step].reshape(n_blk, step)
        blk_rms = np.sqrt(np.mean(blocks**2, axis=1) + 1e-18)
        act_blks = blocks[blk_rms >= float(np.percentile(blk_rms, 95)) * 0.15]
        if len(act_blks) == 0:
            act_blks = blocks
        f, psds = signal.welch(act_blks, fs=sr, nperseg=min(8192, step), axis=1)
        psd = np.median(psds, axis=0)
    else:
        f, psd = signal.welch(mono, fs=sr, nperseg=nperseg)

    def _band_db(lo: float, hi: float) -> float:
        m = (f >= lo) & (f < hi)
        if not np.any(m):
            return -120.0
        return float(10.0 * np.log10(np.mean(psd[m]) + 1e-18))

    ref_6_12 = _band_db(6000.0, 12000.0)
    midhi_12_155 = _band_db(12000.0, 15500.0)
    hi_155_195 = _band_db(15500.0, 19500.0)

    est_cutoff_hz = 20000.0
    for f_edge in range(8000, 19750, 250):
        b_db = _band_db(float(f_edge), float(f_edge + 250))
        if b_db < ref_6_12 - 30.0:
            est_cutoff_hz = float(f_edge)
            break

    if native_sr < 32000:
        est_cutoff_hz = min(est_cutoff_hz, float(native_sr) * 0.5)

    brickwall_cutoff = (native_sr < 32000) or (
        est_cutoff_hz < 15500.0 and ((hi_155_195 - ref_6_12 < -34.0) or (hi_155_195 - midhi_12_155 < -20.0))
    )
    severe_clipping = clip_ratio > 0.005

    reasons: list[str] = []
    if brickwall_cutoff:
        reasons.append(f"brickwall_cutoff_below_15.5khz ({est_cutoff_hz:.0f} Hz)")
    if severe_clipping:
        reasons.append(f"severe_clipping ({clip_ratio * 100:.2f}% samples >= 0.999)")
    if is_mono_collapsed:
        reasons.append(f"mono_collapsed_stereo (side/mid={side_mid_db:.1f} dB)")

    return {
        "native_sr": native_sr,
        "estimated_cutoff_hz": round(est_cutoff_hz, 0),
        "hf_155_195_vs_6_12_db": round(hi_155_195 - ref_6_12, 2),
        "hf_155_195_vs_12_155_db": round(hi_155_195 - midhi_12_155, 2),
        "clip_ratio": round(clip_ratio, 6),
        "side_mid_db": round(side_mid_db, 2),
        "bandwidth_ok": not brickwall_cutoff,
        "clipping_ok": not severe_clipping,
        "stereo_ok": not is_mono_collapsed,
        "rejection_reasons": reasons,
    }


def _align_and_check_key_shift(
    cover_mono: np.ndarray,
    off_mono: np.ndarray,
    sr: int,
    max_offset_sec: float = 12.0,
) -> dict:
    """Measure envelope cross-correlation alignment and 12-semitone CQT chromagram key shift."""
    hop = max(int(sr * 0.01), 128)
    scan_len = min(len(cover_mono), len(off_mono), sr * 90)
    n_f = scan_len // hop
    if n_f < 50:
        return {
            "alignment_offset_sec": 0.0,
            "alignment_corr": 0.0,
            "key_shift_semitones": 0,
            "key_shifted": False,
            "chroma_corr_best": 0.0,
            "chroma_corr_zero": 0.0,
            "aligned_ok": False,
            "paired_calibration_eligible": False,
        }

    c_env = np.log1p(25.0 * np.sqrt(np.mean(cover_mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1) + 1e-12))
    o_env = np.log1p(25.0 * np.sqrt(np.mean(off_mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1) + 1e-12))
    c_d = np.maximum(np.diff(c_env, prepend=c_env[0]), 0.0)
    o_d = np.maximum(np.diff(o_env, prepend=o_env[0]), 0.0)

    ce_z = (c_env - np.mean(c_env)) / (np.std(c_env) + 1e-9)
    oe_z = (o_env - np.mean(o_env)) / (np.std(o_env) + 1e-9)
    cd_z = (c_d - np.mean(c_d)) / (np.std(c_d) + 1e-9)
    od_z = (o_d - np.mean(o_d)) / (np.std(o_d) + 1e-9)

    corr_e = signal.correlate(ce_z, oe_z, mode="full")
    corr_d = signal.correlate(cd_z, od_z, mode="full")
    lags = signal.correlation_lags(n_f, n_f, mode="full")
    overlap = n_f - np.abs(lags)
    max_lag_f = min(int(max_offset_sec / (hop / sr)), int(n_f * 0.4))
    valid = (np.abs(lags) <= max_lag_f) & (overlap >= max(30, int(n_f * 0.55)))
    if np.any(valid):
        v_lags = lags[valid]
        score = (0.60 * corr_e[valid] + 0.40 * corr_d[valid]) / overlap[valid] - 0.0002 * np.abs(v_lags)
        best_lag_f = int(v_lags[int(np.argmax(score))])
    else:
        best_lag_f = 0
    shift_samples = best_lag_f * hop
    offset_sec = round(float(best_lag_f * (hop / sr)), 3)

    if shift_samples >= 0:
        c_al = cover_mono[shift_samples:]
        o_al = off_mono[: max(0, len(cover_mono) - shift_samples)]
    else:
        c_al = cover_mono[: max(0, len(off_mono) + shift_samples)]
        o_al = off_mono[-shift_samples:]
    ov_len = min(len(c_al), len(o_al), sr * 75)
    c_al = c_al[:ov_len]
    o_al = o_al[:ov_len]

    # Envelope Pearson correlation on aligned audio (downsampled to 25 ms RMS)
    e_hop = max(int(sr * 0.025), 256)
    e_nf = ov_len // e_hop
    if e_nf >= 12:
        ce = np.log1p(25.0 * np.sqrt(np.mean(c_al[: e_nf * e_hop].reshape(e_nf, e_hop) ** 2, axis=1) + 1e-12))
        oe = np.log1p(25.0 * np.sqrt(np.mean(o_al[: e_nf * e_hop].reshape(e_nf, e_hop) ** 2, axis=1) + 1e-12))
        ce_z = ce - np.mean(ce)
        oe_z = oe - np.mean(oe)
        align_corr = float(np.dot(ce_z, oe_z) / (np.linalg.norm(ce_z) * np.linalg.norm(oe_z) + 1e-12))
    else:
        align_corr = 0.0

    # 12-semitone CQT chromagram rotation check
    cqt_hop = 2048
    if ov_len >= sr * 2:
        c_chr = librosa.feature.chroma_cqt(y=c_al.astype(np.float32), sr=sr, hop_length=cqt_hop)
        o_chr = librosa.feature.chroma_cqt(y=o_al.astype(np.float32), sr=sr, hop_length=cqt_hop)
    else:
        n_fft = min(4096, max(512, 1 << int(np.floor(np.log2(max(ov_len, 512))))))
        c_chr = librosa.feature.chroma_stft(y=c_al.astype(np.float32), sr=sr, n_fft=n_fft, hop_length=max(256, n_fft // 4))
        o_chr = librosa.feature.chroma_stft(y=o_al.astype(np.float32), sr=sr, n_fft=n_fft, hop_length=max(256, n_fft // 4))

    t_frames = min(c_chr.shape[1], o_chr.shape[1])
    c_chr = c_chr[:, :t_frames]
    o_chr = o_chr[:, :t_frames]
    c_z = c_chr - np.mean(c_chr, axis=0, keepdims=True)
    c_norm = float(np.linalg.norm(c_z)) + 1e-12

    rot_corrs: list[float] = []
    for k in range(12):
        o_rot = np.roll(o_chr, k, axis=0)
        o_z = o_rot - np.mean(o_rot, axis=0, keepdims=True)
        r_k = float(np.sum(c_z * o_z) / (c_norm * (float(np.linalg.norm(o_z)) + 1e-12)))
        rot_corrs.append(r_k)

    best_k = int(np.argmax(rot_corrs))
    key_shift_st = best_k if best_k <= 6 else best_k - 12
    best_chr_corr = float(rot_corrs[best_k])
    zero_chr_corr = float(rot_corrs[0])

    # Guard against tiny 1-bin ambiguity when key_shift=0 is essentially tied with neighbor
    if key_shift_st != 0 and (best_chr_corr - zero_chr_corr) < 0.04:
        key_shift_st = 0
        best_chr_corr = zero_chr_corr

    key_shifted = key_shift_st != 0
    aligned_ok = (align_corr >= 0.30) and (best_chr_corr >= 0.40)
    paired_eligible = aligned_ok and (not key_shifted)

    return {
        "alignment_offset_sec": offset_sec,
        "alignment_corr": round(align_corr, 3),
        "key_shift_semitones": int(key_shift_st),
        "key_shifted": key_shifted,
        "chroma_corr_best": round(best_chr_corr, 3),
        "chroma_corr_zero": round(zero_chr_corr, 3),
        "aligned_ok": aligned_ok,
        "paired_calibration_eligible": paired_eligible,
    }


def _infer_preliminary_archetype(backing_st: np.ndarray, sr: int) -> tuple[str, dict[str, float]]:
    """Infer preliminary `backing_archetype` and full backing descriptors for candidate stratification."""
    full_desc = compute_backing_full_descriptors(backing_st, sr=sr)
    i_mono = np.mean(backing_st, axis=0, dtype=np.float64)
    freqs, psd = signal.welch(i_mono, fs=sr, nperseg=min(4096, max(256, len(i_mono))))

    inst_bands: dict[str, float] = {}
    for b_name, (f_lo, f_hi) in VOCAL_MIX_BANDS.items():
        m = (freqs >= f_lo) & (freqs < f_hi)
        inst_bands[b_name] = round(float(10.0 * np.log10(np.sum(psd[m]) + 1e-10)), 3)

    arch = classify_backing_archetype(
        inst_crest_db=full_desc["full_inst_crest_db"],
        inst_band_levels_db=inst_bands,
        full_inst_mid_density_index=full_desc["full_inst_mid_density_index"],
        backing_dynamic_range_db=full_desc["backing_dynamic_range_db"],
        full_inst_crest_db=full_desc["full_inst_crest_db"],
        chorus_inst_mid_density_index=full_desc["full_inst_mid_density_index"],
    )
    return arch, full_desc


def _extract_cover_id(cover_path: Path) -> str:
    stem = cover_path.stem
    if stem.lower().startswith("cover_"):
        return stem[6:]
    return stem


def validate_reference_song_candidate(
    song_dir: Path,
    existing_cover_ids: set[str] | None = None,
    target_sr: int = 44100,
) -> dict:
    """Enforce all 4 quality gates on a candidate reference song folder (`cover_*.*` + `offvocal/`)."""
    song_dir = Path(song_dir)
    cover_files = sorted(
        p for p in song_dir.glob("cover_*.*") if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
    )
    if not cover_files:
        return {
            "song_dir": str(song_dir),
            "group": song_dir.parent.name,
            "slug": song_dir.name,
            "cover_id": None,
            "passed": False,
            "is_duplicate": False,
            "has_offvocal": False,
            "paired_calibration_eligible": False,
            "key_shifted": False,
            "key_shift_semitones": 0,
            "backing_archetype": "unknown",
            "is_underrepresented_archetype": False,
            "rejection_reasons": ["missing_cover_audio"],
        }

    cover_path = cover_files[0]
    cover_id = _extract_cover_id(cover_path)
    orig_files = sorted(
        p for p in song_dir.glob("original_*.*") if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
    )
    off_file = _find_primary_official_offvocal(song_dir)

    cover_st, sr = load_audio(cover_path, target_sr=target_sr, mono=False)
    cover_mono = np.mean(cover_st, axis=0, dtype=np.float32)

    integrity = _estimate_bandwidth_and_integrity(cover_st, cover_path, sr=sr)
    reasons = list(integrity["rejection_reasons"])

    is_duplicate = bool(existing_cover_ids and cover_id in existing_cover_ids)
    if is_duplicate:
        reasons.append(f"duplicate_cover_id ({cover_id})")

    off_info: dict = {
        "alignment_offset_sec": 0.0,
        "alignment_corr": 0.0,
        "key_shift_semitones": 0,
        "key_shifted": False,
        "chroma_corr_best": 0.0,
        "chroma_corr_zero": 0.0,
        "aligned_ok": False,
        "paired_calibration_eligible": False,
    }
    arch_audio = cover_st
    if off_file is not None:
        try:
            off_st, _ = load_audio(off_file, target_sr=sr, mono=False)
            off_mono = np.mean(off_st, axis=0, dtype=np.float32)
            off_info = _align_and_check_key_shift(cover_mono, off_mono, sr=sr)
            if off_info["alignment_corr"] >= 0.25:
                arch_audio = off_st
            if not off_info["aligned_ok"]:
                reasons.append(
                    f"offvocal_alignment_mismatch (align_r={off_info['alignment_corr']:.2f}, "
                    f"chroma_r={off_info['chroma_corr_best']:.2f})"
                )
        except Exception as exc:
            reasons.append(f"offvocal_load_failed ({exc})")

    backing_arch, full_desc = _infer_preliminary_archetype(arch_audio, sr=sr)
    is_underrep = backing_arch == "intimate_ballad_sparse"
    passed = len(reasons) == 0

    return {
        "song_dir": str(song_dir),
        "group": song_dir.parent.name,
        "slug": song_dir.name,
        "cover_file": cover_path.name,
        "cover_id": cover_id,
        "has_original": bool(orig_files),
        "has_offvocal": off_file is not None,
        "offvocal_file": off_file.name if off_file is not None else None,
        "passed": passed,
        "is_duplicate": is_duplicate,
        "integrity": integrity,
        "offvocal_alignment": off_info,
        "key_shifted": off_info["key_shifted"],
        "key_shift_semitones": off_info["key_shift_semitones"],
        "paired_calibration_eligible": off_info["paired_calibration_eligible"],
        "backing_archetype": backing_arch,
        "backing_full_descriptors": full_desc,
        "is_underrepresented_archetype": is_underrep,
        "rejection_reasons": reasons,
    }


def scan_youtube_references_inbox(
    inbox_root: Path,
    existing_index_json: Path | None = None,
    target_sr: int = 44100,
) -> dict:
    """Scan `youtube_references/` inbox and run all 4 quality gates across all candidate folders."""
    inbox_root = Path(inbox_root)
    idx_path = existing_index_json or (inbox_root / "_info" / "existing_44_songs.json")

    known_cover_ids: set[str] = set()
    if idx_path.exists():
        try:
            idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
            for cid in idx_data.get("existing_cover_video_ids", []):
                if cid:
                    known_cover_ids.add(str(cid).strip())
            for s in idx_data.get("songs", []):
                if s.get("cover"):
                    known_cover_ids.add(str(s["cover"]).strip())
        except Exception:
            pass

    manifest_by_dir: dict[str, dict] = {}
    info_dir = inbox_root / "_info"
    if info_dir.exists():
        for mp in sorted(info_dir.glob("manifest*.json")):
            try:
                for entry in json.loads(mp.read_text(encoding="utf-8")):
                    d_norm = str(entry.get("dir", "")).replace("/", "\\").lower()
                    if d_norm:
                        manifest_by_dir[d_norm] = entry
            except Exception:
                continue

    candidate_dirs: list[Path] = []
    if inbox_root.exists():
        seen_dirs: set[Path] = set()
        for cf in sorted(inbox_root.rglob("cover_*.*")):
            if "_info" in cf.parts or cf.suffix.lower() not in _AUDIO_EXTS:
                continue
            parent = cf.parent.resolve()
            if parent not in seen_dirs:
                seen_dirs.add(parent)
                candidate_dirs.append(cf.parent)

    results: list[dict] = []
    seen_inbox_ids = set(known_cover_ids)
    archetype_counts: dict[str, int] = {}
    underrep_ballads: list[str] = []

    for cdir in candidate_dirs:
        rel_str = str(cdir.relative_to(inbox_root)).replace("/", "\\").lower()
        res = validate_reference_song_candidate(cdir, existing_cover_ids=seen_inbox_ids, target_sr=target_sr)
        if res["cover_id"]:
            seen_inbox_ids.add(str(res["cover_id"]))
        if rel_str in manifest_by_dir:
            res["manifest_entry"] = manifest_by_dir[rel_str]
        arch = res["backing_archetype"]
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        if res["passed"] and res["is_underrepresented_archetype"]:
            underrep_ballads.append(str(cdir))
        results.append(res)

    return {
        "inbox_root": str(inbox_root),
        "known_existing_cover_ids": len(known_cover_ids),
        "total_candidates": len(results),
        "passed_count": sum(1 for r in results if r["passed"]),
        "rejected_count": sum(1 for r in results if not r["passed"]),
        "key_shifted_count": sum(1 for r in results if r["key_shifted"]),
        "paired_calibration_eligible_count": sum(1 for r in results if r["paired_calibration_eligible"]),
        "archetype_counts": archetype_counts,
        "underrepresented_ballad_candidates": underrep_ballads,
        "candidates": results,
    }
