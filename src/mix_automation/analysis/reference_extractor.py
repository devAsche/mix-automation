"""GPU-accelerated source separation & target relative profile builder across 15 pro references.

Separates each reference cover track (`Azusagawa`, `TOY`, `Saburou`) on RTX 5060 Ti into
clean Mixed Vocal and Mastered Backing Instrumental stems (avoiding phase-cancellation artifacts),
extracts both Full-Song and Anchor-Chorus relative profiles, and builds engineer-specific
plus consensus target profiles (`consensus`, `azusagawa`, `toy`, `saburou`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from mix_automation.analysis.relative_profile import (
    VOCAL_MIX_BANDS,
    SectionRelativeProfile,
    compute_section_relative_profile,
)
from mix_automation.dsp.vst3_optimizer import detect_song_sections
from mix_automation.io.audio_loader import load_audio


class GPUSourceSeparator:
    """High-speed GPU vocal/instrumental separator keeping model weights resident in VRAM."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._roformer = None
        self._hdemucs_model = None
        self._hdemucs_sr = 44100
        self._backend = "uninitialized"

    def _init_backend(self) -> None:
        if self._backend != "uninitialized":
            return
        # Prefer BS-RoFormer via audio-separator if weights load cleanly; fallback to CUDA HDemucs
        try:
            from audio_separator.separator import Separator

            sep = Separator(
                model_file_dir=str(self.cache_dir / "models"),
                output_dir=str(self.cache_dir / "tmp_sep"),
                output_format="WAV",
            )
            sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
            self._roformer = sep
            self._backend = "bs_roformer"
            return
        except Exception:
            pass

        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        self._hdemucs_model = bundle.get_model().to(self.device).eval()
        self._hdemucs_sr = bundle.sample_rate
        self._backend = "hdemucs_cuda"

    def separate_to_vocal_and_inst(
        self,
        mix_audio_path: Path,
        slug: str,
        target_sr: int = 44100,
    ) -> tuple[np.ndarray, np.ndarray, int, str]:
        """Return `(vocal_mono, inst_mono, sr, backend_used)` using cached stems if present."""
        voc_cache = self.cache_dir / f"{slug}_vocals.wav"
        inst_cache = self.cache_dir / f"{slug}_inst.wav"

        if voc_cache.exists() and inst_cache.exists():
            v, _ = load_audio(voc_cache, target_sr=target_sr, mono=True)
            i, _ = load_audio(inst_cache, target_sr=target_sr, mono=True)
            return v, i, target_sr, "cached"

        self._init_backend()
        mix_stereo, _ = load_audio(mix_audio_path, target_sr=self._hdemucs_sr, mono=False)

        if self._backend == "bs_roformer" and self._roformer is not None:
            tmp_in = self.cache_dir / f"{slug}_decoded_input.wav"
            sf.write(str(tmp_in), mix_stereo.T, self._hdemucs_sr, subtype="PCM_24")
            out_files = self._roformer.separate(str(tmp_in))
            tmp_in.unlink(missing_ok=True)

            voc_file = None
            inst_file = None
            tmp_sep_dir = self.cache_dir / "tmp_sep"
            for f_str in out_files:
                fp = Path(f_str)
                if not fp.is_absolute():
                    fp = tmp_sep_dir / fp
                low = fp.name.lower()
                if "vocal" in low and "inst" not in low:
                    voc_file = fp
                else:
                    inst_file = fp

            if voc_file and inst_file and voc_file.exists() and inst_file.exists():
                voc_file.replace(voc_cache)
                inst_file.replace(inst_cache)
                v, _ = load_audio(voc_cache, target_sr=target_sr, mono=True)
                i, _ = load_audio(inst_cache, target_sr=target_sr, mono=True)
                return v, i, target_sr, "bs_roformer"

        # GPU chunked Hybrid Demucs separation (fast & zero external download dependency)
        v_st, i_st = self._run_hdemucs_chunked(mix_stereo)
        sf.write(str(voc_cache), v_st.T, self._hdemucs_sr, subtype="PCM_16")
        sf.write(str(inst_cache), i_st.T, self._hdemucs_sr, subtype="PCM_16")

        v, _ = load_audio(voc_cache, target_sr=target_sr, mono=True)
        i, _ = load_audio(inst_cache, target_sr=target_sr, mono=True)
        return v, i, target_sr, self._backend

    @torch.no_grad()
    def _run_hdemucs_chunked(self, mix_stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._hdemucs_model is not None
        sr = self._hdemucs_sr
        wav = torch.from_numpy(mix_stereo).to(self.device)
        ref_std = wav.std(dim=-1, keepdim=True).clamp(min=1e-5)
        wav_norm = wav / ref_std

        segment_samples = sr * 20
        overlap_samples = sr * 2
        step = segment_samples - overlap_samples
        total_len = wav_norm.shape[-1]

        voc_acc = torch.zeros_like(wav_norm)
        inst_acc = torch.zeros_like(wav_norm)
        weight_acc = torch.zeros(total_len, device=self.device)
        fade_win = torch.hann_window(segment_samples, device=self.device).clamp(min=1e-3)

        sources_list = self._hdemucs_model.sources  # ['drums', 'bass', 'other', 'vocals']
        voc_idx = sources_list.index("vocals")

        pos = 0
        while pos < total_len:
            end = min(pos + segment_samples, total_len)
            chunk = wav_norm[:, pos:end]
            pad_right = segment_samples - chunk.shape[-1]
            if pad_right > 0:
                chunk = torch.nn.functional.pad(chunk, (0, pad_right))

            # Shape (1, 2, segment_samples) -> (1, 4, 2, segment_samples)
            est = self._hdemucs_model(chunk.unsqueeze(0))[0]
            v_chunk = est[voc_idx]
            i_chunk = est.sum(dim=0) - v_chunk

            valid_len = end - pos
            w = fade_win[:valid_len]
            voc_acc[:, pos:end] += v_chunk[:, :valid_len] * w
            inst_acc[:, pos:end] += i_chunk[:, :valid_len] * w
            weight_acc[pos:end] += w

            if end == total_len:
                break
            pos += step

        voc_out = ((voc_acc / weight_acc) * ref_std).cpu().numpy()
        inst_out = ((inst_acc / weight_acc) * ref_std).cpu().numpy()
        return voc_out.astype(np.float32), inst_out.astype(np.float32)


def aggregate_profiles_median(profiles: list[SectionRelativeProfile]) -> SectionRelativeProfile:
    """Compute a robust median consensus target profile across multiple songs."""
    if not profiles:
        raise ValueError("Cannot aggregate an empty profile list.")

    band_margins = {
        b: round(float(np.median([p.band_margins_db[b] for p in profiles])), 3)
        for b in VOCAL_MIX_BANDS
    }
    inst_bands = {
        b: round(float(np.median([p.inst_band_levels_db[b] for p in profiles])), 3)
        for b in VOCAL_MIX_BANDS
    }

    # Compensate slightly (+0.8 dB) on 8.5-16kHz air_sheen for Opus/YouTube high-end roll-off
    band_margins["air_sheen"] = round(band_margins["air_sheen"] + 0.8, 3)

    return SectionRelativeProfile(
        vocal_rms_db=round(float(np.median([p.vocal_rms_db for p in profiles])), 3),
        inst_rms_db=round(float(np.median([p.inst_rms_db for p in profiles])), 3),
        broadband_vi_ratio_db=round(float(np.median([p.broadband_vi_ratio_db for p in profiles])), 3),
        vocal_crest_db=round(float(np.median([p.vocal_crest_db for p in profiles])), 3),
        inst_crest_db=round(float(np.median([p.inst_crest_db for p in profiles])), 3),
        crest_difference_db=round(float(np.median([p.crest_difference_db for p in profiles])), 3),
        position_stability_std_db=round(float(np.median([p.position_stability_std_db for p in profiles])), 3),
        harsh_resonance_index_db=round(float(np.median([p.harsh_resonance_index_db for p in profiles])), 3),
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
    """Process all 15 reference covers in `manifest.json` and save pro target profiles."""
    manifest_path = references_root / "_info" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    separator = GPUSourceSeparator(cache_dir=cache_dir)
    per_song_records: list[dict] = []
    by_group_chorus: dict[str, list[SectionRelativeProfile]] = {}
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
        print(f"[{idx:02d}/{len(manifest):02d}] Separating & profiling {slug} ({item['title']}) ...", flush=True)

        v_mono, i_mono, sr, backend = separator.separate_to_vocal_and_inst(
            cover_path, slug=slug, target_sr=target_sr
        )

        # Full-song profile & Main-Chorus Anchor profile
        full_prof = compute_section_relative_profile(v_mono, i_mono, sr=sr)
        sections, anchor_idx = detect_song_sections(v_mono, i_mono, sr=sr)
        a_start, a_end, a_lbl = sections[anchor_idx]
        s0, s1 = int(a_start * sr), int(a_end * sr)
        chorus_prof = compute_section_relative_profile(v_mono[s0:s1], i_mono[s0:s1], sr=sr)

        by_group_chorus.setdefault(group.lower(), []).append(chorus_prof)
        all_chorus_profiles.append(chorus_prof)

        per_song_records.append(
            {
                "slug": slug,
                "group": group,
                "mixer": item["mixer"],
                "vocal": item["vocal"],
                "title": item["title"],
                "separator_backend": backend,
                "anchor_section": {"label": a_lbl, "start_sec": a_start, "end_sec": a_end},
                "chorus_profile": chorus_prof.to_dict(),
                "full_song_profile": full_prof.to_dict(),
            }
        )

    targets = {
        "consensus": aggregate_profiles_median(all_chorus_profiles).to_dict(),
    }
    for grp, prof_list in by_group_chorus.items():
        targets[grp] = aggregate_profiles_median(prof_list).to_dict()

    result = {
        "num_reference_songs": len(per_song_records),
        "target_profiles": targets,
        "songs": per_song_records,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
