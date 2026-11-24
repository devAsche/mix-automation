"""v2 Reference Extractor: Genuine BS-RoFormer GPU separation across all 35 pro covers.

Reads both `manifest.json` (Batch 1: 15 songs) and `manifest2.json` (Batch 2: 20 songs),
separates every cover on RTX 5060 Ti using genuine `BS-RoFormer` (`sdr_12.9755`),
analyzes vocal register (`median_f0_hz`), backing track archetype (`dense_rock_edm`,
`groovy_mid_pop`, `intimate_ballad_sparse`), spatial wet-tail decay, syllabic clarity,
and official distributed off-vocal loudness, building:
  - `consensus` (all 35 songs)
  - Register x Archetype matrix (`low_mid_dense_rock_edm`, `high_dense_rock_edm`, etc.)
  - Group/Engineer profiles (`azusagawa`, `toy`, `saburou`, `aokiyo`, `aoru`, `hoshirubesho`, `nanadakiima`)
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from mix_automation.analysis.relative_profile import (
    VOCAL_MIX_BANDS,
    SectionRelativeProfile,
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

        # Save compact PCM_16 stereo cache to preserve disk space across 35+ songs while keeping full 22kHz bandwidth
        v_tmp, _ = load_audio(voc_file, target_sr=target_sr, mono=False)
        i_tmp, _ = load_audio(inst_file, target_sr=target_sr, mono=False)
        sf.write(str(voc_cache), v_tmp.T, target_sr, subtype="PCM_16")
        sf.write(str(inst_cache), i_tmp.T, target_sr, subtype="PCM_16")
        voc_file.unlink(missing_ok=True)
        inst_file.unlink(missing_ok=True)

        return v_tmp, i_tmp, target_sr, "bs_roformer_cuda"


def _estimate_fast_median_f0(vocal_mono: np.ndarray, sr: int) -> float:
    """Fast median F0 estimation over a 45-second active excerpt."""
    # Pick loudest 45s window
    chunk_len = min(len(vocal_mono), sr * 45)
    start = max(0, (len(vocal_mono) - chunk_len) // 2)
    seg = vocal_mono[start : start + chunk_len]
    f0, voiced, vp = librosa.pyin(seg, fmin=75.0, fmax=880.0, sr=sr, hop_length=1024)
    ok = (~np.isnan(f0)) & voiced & (vp > 0.72)
    if not np.any(ok):
        return 260.0
    return round(float(np.median(f0[ok])), 1)


def _find_primary_official_offvocal(song_dir: Path) -> Path | None:
    """Locate the main official distributed off-vocal file inside `offvocal/`."""
    off_dir = song_dir / "offvocal"
    if not off_dir.exists():
        return None
    candidates = [
        p
        for p in off_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".webm"}
        and not any(x in p.name.lower() for x in ("+1", "+2", "+3", "+4", "+5", "+6", "-1", "-2", "-3", "-4", "-5", "hamo", "drums-only", "no-drums", "bass_less", "_part", "vo 0"))
    ]
    if not candidates:
        return None
    # Prefer WAV over lossy, and mastered/default over pre-mastering
    candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".wav" else 1, 1 if "nomst" in p.name.lower() or "pre-master" in p.name.lower() else 0, len(p.name)))
    return candidates[0]


def _pick_loudest_chorus_span(v_mono: np.ndarray, i_mono: np.ndarray, sr: int) -> tuple[int, int]:
    """Find the 20-second window with highest combined mid-high backing density + active singing."""
    from scipy import signal

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


def aggregate_profiles_median(
    profiles: list[SectionRelativeProfile],
    archetype_label: str = "consensus",
) -> SectionRelativeProfile:
    """Compute robust median consensus across a subset of reference profiles."""
    if not profiles:
        raise ValueError("Profile list is empty.")

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
        vocal_crest_db=round(float(np.median([p.vocal_crest_db for p in profiles])), 3),
        inst_crest_db=round(float(np.median([p.inst_crest_db for p in profiles])), 3),
        crest_difference_db=round(float(np.median([p.crest_difference_db for p in profiles])), 3),
        position_stability_std_db=round(float(np.median([p.position_stability_std_db for p in profiles])), 3),
        harsh_resonance_index_db=round(float(np.median([p.harsh_resonance_index_db for p in profiles])), 3),
        syllabic_clarity_index=round(float(np.median([p.syllabic_clarity_index for p in profiles])), 4),
        wet_tail_ratio_db=round(float(np.median([p.wet_tail_ratio_db for p in profiles])), 2),
        vocal_side_mid_db=round(float(np.median([p.vocal_side_mid_db for p in profiles])), 2),
        backing_archetype=archetype_label,
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
    """Process all 35 reference covers (`manifest.json` + `manifest2.json`) with genuine BS-RoFormer."""
    info_dir = references_root / "_info"
    manifest: list[dict] = []
    for m_name in ("manifest.json", "manifest2.json"):
        mp = info_dir / m_name
        if mp.exists():
            manifest.extend(json.loads(mp.read_text(encoding="utf-8")))

    separator = BSRoFormerSeparator(cache_dir=cache_dir)
    meter = pyln.Meter(target_sr)

    per_song_records: list[dict] = []
    by_group: dict[str, list[SectionRelativeProfile]] = {}
    by_register_and_arch: dict[str, list[SectionRelativeProfile]] = {}
    by_register: dict[str, list[SectionRelativeProfile]] = {}
    all_chorus_profiles: list[SectionRelativeProfile] = []

    for idx, item in enumerate(manifest, start=1):
        rel_dir = Path(item["dir"])
        song_dir = references_root / rel_dir
        cover_files = list(song_dir.glob("cover_*.*"))
        if not cover_files:
            continue

        cover_path = cover_files[0]
        group = item["group"]
        slug = f"{group}_{rel_dir.name}"
        print(f"[{idx:02d}/{len(manifest):02d}] BS-RoFormer separating & profiling {slug} ({item['title']}) ...", flush=True)

        v_st, i_st, sr, backend = separator.separate(cover_path, slug=slug, target_sr=target_sr)
        v_mono = np.mean(v_st, axis=0, dtype=np.float32)
        i_mono = np.mean(i_st, axis=0, dtype=np.float32)

        med_f0 = _estimate_fast_median_f0(v_mono, sr=sr)
        reg_key = "low_mid" if med_f0 < 295.0 else "high"

        s0, s1 = _pick_loudest_chorus_span(v_mono, i_mono, sr=sr)
        chorus_prof = compute_section_relative_profile(v_st[:, s0:s1], i_st[:, s0:s1], sr=sr)
        full_prof = compute_section_relative_profile(v_st, i_st, sr=sr)

        # Measure mastered cover LUFS vs official distributed off-vocal LUFS
        cover_2mix = np.clip(v_st + i_st, -1.0, 1.0)
        cover_lufs = round(float(meter.integrated_loudness(cover_2mix.T.astype(np.float64))), 2)
        off_file = _find_primary_official_offvocal(song_dir)
        off_lufs = None
        if off_file is not None:
            try:
                off_st, _ = load_audio(off_file, target_sr=sr, mono=False)
                off_lufs = round(float(meter.integrated_loudness(off_st.T.astype(np.float64))), 2)
            except Exception:
                off_lufs = None

        arch = chorus_prof.backing_archetype
        by_group.setdefault(group.lower(), []).append(chorus_prof)
        by_register.setdefault(reg_key, []).append(chorus_prof)
        by_register_and_arch.setdefault(f"{reg_key}_{arch}", []).append(chorus_prof)
        all_chorus_profiles.append(chorus_prof)

        per_song_records.append(
            {
                "slug": slug,
                "group": group,
                "mixer": item.get("mixer", ""),
                "vocal": item.get("vocal", ""),
                "title": item["title"],
                "separator_backend": backend,
                "vocal_median_f0_hz": med_f0,
                "vocal_register": reg_key,
                "backing_archetype": arch,
                "cover_master_lufs": cover_lufs,
                "official_offvocal_lufs": off_lufs,
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

    result = {
        "schema_version": "2.0",
        "separator_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "num_reference_songs": len(per_song_records),
        "target_profiles": targets,
        "songs": per_song_records,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
