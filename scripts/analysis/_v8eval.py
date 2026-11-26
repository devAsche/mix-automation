"""Song-level computation of every v8 metric (Phase 2), shared by evaluate_issues.py and
build_reference_stats.py. Results are cached as JSON under data/processed/v8_eval_cache/.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from _common import ROOT, first_glob, load_ch_first, load_report

from mix_automation.v8.metrics import consonants, harmony, loudness, space, spectral, steps, timing
from mix_automation.v8.metrics._core import HOP, Segment, phrases, stft_power, to_mono

SR = 44100
CACHE = ROOT / "data" / "processed" / "v8_eval_cache"
CACHE_VERSION = 5  # bump when a metric changes
RENDER_VERSION = 9  # bump when only the render-specific part (D/F/G/H, signal assembly) changes


def _key(*parts: object) -> str:
    h = hashlib.sha1("|".join(str(p) for p in (CACHE_VERSION, *parts)).encode("utf-8")).hexdigest()[:16]
    return h


def _cached(name: str, key: str, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", name)
    p = CACHE / f"{safe}_{key}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    res = _clean(fn())
    p.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _fid(p: Path) -> str:
    st = p.stat()
    return f"{p.resolve()}:{st.st_size}:{int(st.st_mtime)}"


# --------------------------------------------------------------------------- pair (vocal + backing)


def analyze_pair(
    vocal: np.ndarray,
    backing: np.ndarray,
    sr: int = SR,
    pre: np.ndarray | None = None,
    with_consonants: bool = False,
    with_timelines: bool = False,
    with_inst_rt: bool = True,
) -> dict:
    """Metrics that need only a vocal stem and its backing (and optionally the pre vocal)."""
    n = min(vocal.shape[-1], backing.shape[-1])
    vocal, backing = vocal[..., :n], backing[..., :n]
    if pre is not None:
        pre = pre[..., :n]
    out: dict = {}
    # A. Phrases from the vocal itself (comparable across rendered / separated stems).
    segs, act, times = phrases(vocal, sr)
    tr = loudness.partial_loudness_track(vocal, backing, sr)
    ph = loudness.phrase_loudness(tr, segs)
    blocks = loudness.block_masking_loss(tr, act)
    out["A"] = {
        "phrases": [[p.seg.start, p.seg.end, p.masking_loss_phon, p.vi_1k4k_db, p.partial_sone] for p in ph],
        "blocks": blocks,
        "loss_median": float(np.median(blocks[:, 1])) if len(blocks) else None,
        "vi1k4k_median": float(np.median(blocks[:, 2])) if len(blocks) else None,
    }
    if pre is not None:
        psegs, pact, _ = phrases(pre, sr)
        pph = loudness.phrase_loudness(tr, psegs)
        out["A"]["pre_phrases"] = [[p.seg.start, p.seg.end, p.masking_loss_phon, p.vi_1k4k_db, p.partial_sone] for p in pph]
        out["A"]["pre_blocks"] = loudness.block_masking_loss(tr, pact)
    # B. Colour (active frames of the vocal itself).
    m = min(len(act), len(times))
    ci = spectral.colour_indices(vocal, backing, sr, act[:m])
    out["B"] = {
        "vi_bands_db": ci.vi_bands_db,
        "vocal_bands_db": ci.vocal_bands_db,
        "tilt": ci.tilt_db_per_oct,
        "body_rel": ci.body_rel_db,
        "air_rel": ci.air_rel_db,
        "bright_rel": ci.bright_rel_db,
    }
    if with_timelines:
        centers, tl = spectral.colour_timeline(vocal, backing, sr, act[:m], times[:m], win_s=6.0, hop_s=1.0)
        out["B"]["timeline"] = {
            "t": centers,
            "bright_rel": [c.bright_rel_db if c else None for c in tl],
            "air_rel": [c.air_rel_db if c else None for c in tl],
            "body_rel": [c.body_rel_db if c else None for c in tl],
        }
    if pre is not None:
        pact_m = min(m, len(pact))
        fz = spectral.fizz_indices(vocal, pre, sr, act[:pact_m] & pact[:pact_m])
        out["B"]["fizz"] = {"hf_rel_delta": fz.hf_rel_delta_db, "flat_delta": fz.hf_flatness_delta, "hf_rel_proc": fz.hf_rel_proc_db, "hf_rel_pre": fz.hf_rel_pre_db}
    # E. Space.
    ds = space.phrase_end_decay(vocal, sr)
    out["E"] = {
        "decay_t60": ds.decay_t60_s,
        "tail_ratio": ds.tail_ratio_db,
        "n_ends": ds.n_ends,
        "inst_rt60_blind": space.blind_rt60(backing, sr) if with_inst_rt else None,
        "side_mid": space.side_mid_db(vocal, sr),
    }
    if pre is not None:
        dsp = space.phrase_end_decay(vocal, sr, ref_for_phrases=pre)
        out["E"]["decay_t60_pre_phrases"] = dsp.decay_t60_s
        out["E"]["tail_ratio_pre_phrases"] = dsp.tail_ratio_db
    # C. Consonants.
    if with_consonants:
        det = pre if pre is not None else vocal
        feats = consonants.frame_features(det, sr)
        evs = consonants.detect_consonants(det, sr, feats)
        ms = consonants.measure_consonants(evs, vocal, pre, backing, sr)
        out["C"] = [[c.event.kind, c.event.start, c.event.end, c.delta_ratio_db, c.vi_db, c.delta_crest_db, c.ratio_proc_db] for c in ms]
    return out


# --------------------------------------------------------------------------- v7 render


def render_signals(render_dir: Path) -> dict:
    media = render_dir / "Media"
    sig = {
        "master": first_glob(render_dir, "*_Master.wav"),
        "vocal_bus": first_glob(render_dir, "*_VocalBus.wav"),
        "inst": media / "01_offvocal_unmasked.wav",
        "inst_raw": media / "00_offvocal_raw_backup.wav",
        "pre": media / "02_mainvocal_prepared.wav",
        "aux_double": media / "08_aux_double.wav",
        "aux_sub_octave": media / "09_aux_sub_octave.wav",
        "aux_super_octave": media / "10_aux_super_octave.wav",
        "choir": media / "11_choir_wide_bus.wav",
    }
    return {k: (v if v is not None and Path(v).exists() else None) for k, v in sig.items()}


HARMONY_LAYERS = ("aux_double", "aux_sub_octave", "aux_super_octave", "choir")
# Stage 3 inline-FX slice tracks: v7 cuts these spans out of 02_mainvocal_prepared.wav (Studio One
# ducking), so the pre vocal is prepared + slices (pipeline.py, inline_slice_monos / inline_duck_mask).
INLINE_FX_FILES = {
    "radio_voice": "12_fx_radio_vocalsynth2.wav",
    "hard_autotune": "13_fx_autotune_mautopitch.wav",
    "distortion_grit": "14_fx_distortion_trash2.wav",
}


def vocal_bus_gain(master: np.ndarray, vb: np.ndarray, inst: np.ndarray, block_s: float = 1.0) -> float:
    """Gain of the exported VocalBus.wav relative to the vocal bus inside the master.

    v7 peak-normalises the exported bus when it exceeds 0.89 (pipeline.py, vocal_bus_export), so the
    file can be quieter than what was mixed. Per 1 s block the master is fitted as a*inst + b*vb
    (least squares; the bus glue/limiter make it only approximately linear) and the median b/a is
    returned. Measured: 1.00 / 0.97 / 1.85 / 1.03 for FiL / エバ / シックス / ラブ&デストロイ.
    """
    n = min(master.shape[-1], vb.shape[-1], inst.shape[-1])
    b = int(block_s * SR)
    rat = []
    for k in range(0, n - b, b):
        x = np.stack([inst[:, k : k + b].ravel(), vb[:, k : k + b].ravel()], 1)
        if np.mean(x[:, 1] ** 2) < 1e-5:
            continue
        c, *_ = np.linalg.lstsq(x, master[:, k : k + b].ravel(), rcond=None)
        if c[0] > 0:
            rat.append(c[1] / c[0])
    return float(np.median(rat)) if rat else 1.0


def raw_lead_vocal(rep: dict, n: int, ref: np.ndarray) -> np.ndarray | None:
    """Lead before v7 preprocessing (de-ess, RX): the input main takes comped by v7's own
    consolidation (io.audio_loader.consolidate_vocal_stem_folder), aligned to `ref`."""
    from mix_automation.io.audio_loader import consolidate_vocal_stem_folder

    ia = rep.get("input_assets") or {}
    folder = Path(ia.get("song_folder", ""))
    mains = ((rep.get("stem_consolidation") or {}).get("roles") or {}).get("main") or []
    files = [h for m in mains for h in list(folder.rglob(m))[:1]] if folder.is_dir() else []
    if not files:
        return None
    lead = to_mono(consolidate_vocal_stem_folder(files, target_sr=SR).lead_mono)
    out = align_to(lead, ref)
    return out[:n]


def true_pre_vocal(render_dir: Path, sig: dict, rep: dict) -> np.ndarray:
    pre_path = sig["pre"]
    pre = to_mono(load_ch_first(pre_path))
    types = {e.get("fx_type") for e in rep.get("stage3_creative_fx_events") or []}
    for fx, fname in INLINE_FX_FILES.items():
        f = render_dir / "Media" / fname
        # Only slices of this render: the FX type is in the report and the file is not older than
        # the prepared vocal by more than the render duration (stale files from older renders remain).
        if fx in types and f.exists() and f.stat().st_mtime >= pre_path.stat().st_mtime - 600:
            sl = to_mono(load_ch_first(f))[: len(pre)]
            pre[: len(sl)] += sl
    return pre


def analyze_render(render_dir: Path) -> dict:
    sig = render_signals(render_dir)
    key = _key("render", RENDER_VERSION, *[_fid(p) for p in sig.values() if p is not None])
    rep = load_report(render_dir)

    def run() -> dict:
        vb = load_ch_first(sig["vocal_bus"])
        inst = load_ch_first(sig["inst"])
        pre = true_pre_vocal(render_dir, sig, rep)
        master = load_ch_first(sig["master"])
        n = min(vb.shape[1], inst.shape[1], len(pre), master.shape[1])
        vb, inst, pre, master = vb[:, :n], inst[:, :n], pre[:n], master[:, :n]
        g = vocal_bus_gain(master, vb, inst)
        # Use the bus at its in-mix level only when the export was rescaled (> 0.5 dB, i.e. beyond the
        # block-to-block spread of the fit, p10-p90 about +-0.3 dB on the unscaled songs).
        vb_gain = g if abs(20 * np.log10(g)) > 0.5 else 1.0
        vb = vb * vb_gain
        res = analyze_pair(vb, inst, SR, pre=pre, with_consonants=True, with_timelines=True)
        res["vocal_bus_gain_fit"] = g
        res["vocal_bus_gain_applied"] = vb_gain
        res["song"] = rep.get("song_name")
        res["duration"] = n / SR
        res["signals"] = {k: (v.name if v else None) for k, v in sig.items()}
        # C against the raw (pre-preprocessing) lead, same events.
        raw = raw_lead_vocal(rep, n, pre)
        if raw is not None:
            feats = consonants.frame_features(pre, SR)
            evs = consonants.detect_consonants(pre, SR, feats)
            ms = consonants.measure_consonants(evs, vb, raw, inst, SR)
            res["C_raw"] = [[c.event.kind, c.event.start, c.event.end, c.delta_ratio_db, c.vi_db, c.delta_crest_db, c.ratio_proc_db] for c in ms]
            msp = consonants.measure_consonants(evs, pre, raw, None, SR)
            res["C_prep_vs_raw"] = [[c.event.kind, c.event.start, c.event.end, c.delta_ratio_db, None, c.delta_crest_db, c.ratio_proc_db] for c in msp]
        # D. Steps on the lead estimate (bus minus exported harmony/choir/aux layers), so that a layer
        # entering at a section change is not read as the lead moving.
        layers = {k: load_ch_first(sig[k])[:, :n] for k in HARMONY_LAYERS if sig[k] is not None}
        lead_est = vb - sum(layers.values()) if layers else vb
        st = steps.step_track(lead_est, pre, SR)
        res["D"] = {"t": st.times, "far": st.far_score, "d_level": st.d_level, "d_bright": st.d_bright, "d_wet": st.d_wet}
        # F. Beat grid and delay throws.
        beats, downbeats = timing.beat_grid(master, SR)
        ioi = float(np.median(np.diff(beats))) if len(beats) > 2 else float("nan")
        tm = rep.get("tempo_map") or {}
        bpm_v7 = float(rep.get("bpm") or tm.get("global_bpm") or float("nan"))
        fx = rep.get("stage3_creative_fx_events") or []
        delays = []
        for ev in fx:
            if ev.get("fx_type") != "spot_delay_throw":
                continue
            # Delay time as rendered by v7 (creative/fx_and_harmony.render_spot_delay_throw_stem):
            # 1/4 if any throw reason says "1/4", else 1/8 of the global BPM, clipped to 120-680 ms.
            quarter = any("1/4" in (e.get("reason") or "") for e in fx if e.get("fx_type") == "spot_delay_throw")
            d_nom = float(np.clip(60.0 / max(bpm_v7, 60.0) * (1.0 if quarter else 0.5), 0.12, 0.68))
            li = timing.local_ioi(beats, ev["start_sec"])
            fit = timing.delay_vs_grid(d_nom, li)
            per = timing.echo_period(vb, SR, ev["end_sec"], ev["end_sec"] + 1.2)
            fit_m = timing.delay_vs_grid(per, li) if np.isfinite(per) else None
            delays.append({
                "start": ev["start_sec"], "end": ev["end_sec"], "delay_nominal": d_nom, "beat_ioi": li,
                "note": fit.best_note, "err": fit.error_s, "tol": fit.tolerance_s, "off_grid": fit.off_grid,
                "echo_period": per, "echo_note": fit_m.best_note if fit_m else None,
                "echo_err": fit_m.error_s if fit_m else None, "echo_off_grid": fit_m.off_grid if fit_m else None,
            })
        res["F"] = {
            "beats": beats, "downbeats": downbeats, "bpm_beatthis": 60.0 / ioi if ioi else None,
            "bpm_v7": bpm_v7, "tempo_segments_v7": tm.get("tempo_segments"), "delays": delays,
        }
        # G. Effect dropouts at note ends.
        res["G"] = []
        for ev in fx:
            if ev.get("fx_type") in ("radio_voice", "hard_autotune", "distortion_grit"):
                d = timing.fx_dropout(pre, SR, ev["start_sec"], ev["end_sec"])
                res["G"].append({"fx": ev["fx_type"], "start": ev["start_sec"], "end": ev["end_sec"], "note_end": d.note_end, "dropout": d.dropout_s, "missing": d.missing})
        # H. Harmony layers against lead estimate (vocal bus minus exported layers).
        res["H"] = {}
        if layers:
            _, ftimes, _ = stft_power(pre, SR)
            f0_lead = harmony.f0_track(pre, SR, ftimes)
            for name, lay in layers.items():
                ht = harmony.harmony_prominence(lay, lead_est, inst, SR, lead_activity_ref=pre)
                f0_h = harmony.f0_track(lay, SR, ht.times)
                m2 = min(len(ht.times), len(f0_lead))
                semis = np.full(len(ht.times), np.nan)
                ok = np.isfinite(f0_h[:m2]) & np.isfinite(f0_lead[:m2])
                semis[:m2][ok] = 12 * np.log2(f0_h[:m2][ok] / f0_lead[:m2][ok])
                # Store on a 0.1 s grid to keep the cache small.
                step = max(1, int(round(0.1 / (HOP / SR))))
                # Plain level ratio layer / lead estimate (dB) on frames where the prominence is defined:
                # energetic partial loudness cannot see unison/octave layers (their partials sit under
                # the lead's), so the level ratio is kept alongside.
                _, _, pl_ = stft_power(lay, SR)
                _, _, pe_ = stft_power(lead_est, SR)
                m3 = min(pl_.shape[1], pe_.shape[1], len(ht.times))
                rel = np.full(len(ht.times), np.nan)
                lvl = 10 * np.log10(pl_[:, :m3].sum(0) + 1e-12) - 10 * np.log10(pe_[:, :m3].sum(0) + 1e-12)
                act_h = 10 * np.log10(pl_[:, :m3].sum(0) + 1e-12) > -60
                rel[:m3] = np.where(act_h, lvl, np.nan)
                res["H"][name] = {"t": ht.times[::step], "prom": ht.prominence_phon[::step], "semis": semis[::step], "rel_db": rel[::step]}
        return res

    return _cached(f"render_{render_dir.name}", key, run)


def analyze_stem_pair(
    name: str,
    vocal_path: Path,
    inst_path: Path,
    pre: np.ndarray | None = None,
    with_consonants: bool = False,
    pre_id: str = "",
    with_inst_rt: bool = True,
) -> dict:
    key = _key("pair", _fid(vocal_path), _fid(inst_path), pre_id, with_consonants, with_inst_rt)

    def run() -> dict:
        v = load_ch_first(vocal_path)
        i = load_ch_first(inst_path)
        p = pre
        if p is not None:
            p = align_to(p, to_mono(v))
        return analyze_pair(v, i, SR, pre=p, with_consonants=with_consonants, with_inst_rt=with_inst_rt)

    return _cached(f"pair_{name}", key, run)


def align_to(x: np.ndarray, ref: np.ndarray, max_lag_s: float = 30.0) -> np.ndarray:
    """Shift x (mono) to best match ref by envelope cross-correlation (coarse) then sample xcorr."""
    from scipy.signal import correlate

    hop = 441
    def env(a: np.ndarray) -> np.ndarray:
        k = len(a) // hop
        e = np.sqrt(np.mean(a[: k * hop].reshape(k, hop) ** 2, axis=1))
        return e - e.mean()
    ex, er = env(x), env(ref)
    c = correlate(er, ex, mode="full", method="fft")
    lags = np.arange(-len(ex) + 1, len(er))
    sel = np.abs(lags) <= int(max_lag_s * SR / hop)
    lag = int(lags[sel][np.argmax(c[sel])]) * hop
    out = np.zeros(len(ref))
    if lag >= 0:
        seg = x[: max(0, len(ref) - lag)]
        out[lag : lag + len(seg)] = seg
    else:
        seg = x[-lag : -lag + len(ref)]
        out[: len(seg)] = seg
    return out


def segment(a: float, b: float) -> Segment:
    return Segment(float(a), float(b))


# --------------------------------------------------------------------------- Phase 2b: debug-stem renders

DEBUG_VERSION = 4


def debug_signals(render_dir: Path) -> dict | None:
    d = render_dir / "debug_stems"
    names = ("lead_dry", "wet", "harmony_bus", "aux_layers", "inst_unmasked")
    if not all((d / f"{n}.wav").exists() for n in names):
        return None
    return {n: d / f"{n}.wav" for n in names}


def analyze_debug_render(render_dir: Path) -> dict | None:
    """Metrics that need the vocal-bus components (render made with --export-debug-stems)."""
    ds = debug_signals(render_dir)
    if ds is None:
        return None
    sig = render_signals(render_dir)
    rep = load_report(render_dir)
    key = _key("debug", DEBUG_VERSION, *[_fid(p) for p in ds.values()], _fid(sig["pre"]))

    def run() -> dict:
        st = {k: load_ch_first(p) for k, p in ds.items()}
        pre = true_pre_vocal(render_dir, sig, rep)
        n = min(len(pre), *[a.shape[1] for a in st.values()])
        st = {k: a[:, :n] for k, a in st.items()}
        pre = pre[:n]
        lead, wet = st["lead_dry"], st["wet"]
        masker = st["inst_unmasked"] + wet + st["harmony_bus"] + st["aux_layers"]
        _, pact, ptimes = phrases(pre, SR)
        res: dict = {"song": rep.get("song_name")}
        # (a) partial loudness of the dry lead against backing + wet + harmonies
        tr = loudness.partial_loudness_track(lead, masker, SR)
        bl = loudness.block_masking_loss(tr, pact)
        res["A_dry"] = {"blocks": bl, "loss_median": float(np.median(bl[:, 1])) if len(bl) else None}
        # (b) wet / dry
        hop = 512
        m = min(len(pact), n // hop)
        mask = np.repeat(pact[:m], hop)
        e_l = float(np.mean(to_mono(lead)[: m * hop][mask] ** 2))
        e_w = float(np.mean(to_mono(wet)[: m * hop][mask] ** 2))
        e_w_all = float(np.mean(to_mono(wet) ** 2))
        e_l_all = float(np.mean(to_mono(lead) ** 2))
        dsl = space.phrase_end_decay(lead + wet, SR, ref_for_phrases=pre)
        dsd = space.phrase_end_decay(lead, SR, ref_for_phrases=pre)
        res["E_dry"] = {
            "wet_dry_db_active": 10 * np.log10(e_w / e_l), "wet_dry_db_all": 10 * np.log10(e_w_all / e_l_all),
            "decay_t60_lead_wet": dsl.decay_t60_s, "tail_ratio_lead_wet": dsl.tail_ratio_db,
            "decay_t60_dry": dsd.decay_t60_s, "tail_ratio_dry": dsd.tail_ratio_db, "n_ends": dsl.n_ends,
            "side_mid_lead_wet": space.side_mid_db(lead + wet, SR),
            "harmony_bus_rel_db": 10 * np.log10((np.mean(to_mono(st["harmony_bus"]) ** 2) + 1e-20) / e_l_all),
        }
        # (c) consonants on the dry lead
        feats = consonants.frame_features(pre, SR)
        evs = consonants.detect_consonants(pre, SR, feats)
        ms = consonants.measure_consonants(evs, lead, pre, masker, SR)
        res["C_dry"] = [[c.event.kind, c.event.start, c.event.end, c.delta_ratio_db, c.vi_db, c.delta_crest_db, c.ratio_proc_db] for c in ms]
        tr_d = consonants.deess_traces(evs, lead, pre, SR)
        res["Q_dry"] = [[t.event.start, t.event.end, t.peak_hz, t.band_att_db, t.centroid_drop_pct, t.gain_dip_db, t.resid_db, t.lsd_db, t.d_attack_ms, t.d_dur_ms] for t in tr_d]
        raw = raw_lead_vocal(rep, n, pre)
        if raw is not None:
            tr_r = consonants.deess_traces(evs, lead, raw, SR)
            res["Q_dry_raw"] = [[t.event.start, t.event.end, t.peak_hz, t.band_att_db, t.centroid_drop_pct, t.gain_dip_db, t.resid_db, t.lsd_db, t.d_attack_ms, t.d_dur_ms] for t in tr_r]
            tr_p = consonants.deess_traces(evs, pre, raw, SR)
            res["Q_prep_raw"] = [[t.event.start, t.event.end, t.peak_hz, t.band_att_db, t.centroid_drop_pct, t.gain_dip_db, t.resid_db, t.lsd_db, t.d_attack_ms, t.d_dur_ms] for t in tr_p]
        # D on lead + wet (harmony layers excluded exactly)
        stp = steps.step_track(lead + wet, pre, SR)
        res["D_dry"] = {"t": stp.times, "far": stp.far_score, "d_level": stp.d_level, "d_bright": stp.d_bright, "d_wet": stp.d_wet}
        # ESTOI per pre phrase: reference = pre, degraded = master
        from mix_automation.v8.metrics import intelligibility

        master = load_ch_first(sig["master"])[:, :n]
        psegs, _, _ = phrases(pre, SR)
        res["I_estoi"] = intelligibility.estoi_segments(pre, master, SR, psegs)
        return res

    return _cached(f"debug_{render_dir.name}", key, run)


def asr_cer(name: str, ref_audio: Path | np.ndarray, deg_audio: Path, segs_from: np.ndarray | None, transcriber, ref_id: str = "") -> dict:
    """CER per chunk: transcripts of the reference (clean vocal) and of the degraded (mix)."""
    from mix_automation.v8.metrics import intelligibility

    key = _key("asr", ref_id or (_fid(ref_audio) if isinstance(ref_audio, Path) else ""), _fid(deg_audio))

    def run() -> dict:
        ref = to_mono(load_ch_first(ref_audio)) if isinstance(ref_audio, Path) else to_mono(ref_audio)
        deg = to_mono(load_ch_first(deg_audio))
        n = min(len(ref), len(deg))
        ref, deg = ref[:n], deg[:n]
        segs, _, _ = phrases(ref if segs_from is None else segs_from[:n], SR)
        chunks = intelligibility.chunk_segments(segs)
        rows = []
        for c in chunks:
            i0, i1 = int(max(0, c.start - 0.2) * SR), int((c.end + 0.2) * SR)
            t_ref, _ = transcriber.transcribe(ref[i0:i1], SR)
            t_deg, _ = transcriber.transcribe(deg[i0:i1], SR)
            rows.append([c.start, c.end, intelligibility.cer(t_ref, t_deg), t_ref, t_deg])
        return {"chunks": rows}

    return _cached(f"asr_{name}", key, run)
