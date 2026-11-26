"""Phase 2: v8 metrics move in the right direction on synthetic signals."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from mix_automation.v8.metrics import consonants, harmony, loudness, space, spectral, steps, timing
from mix_automation.v8.metrics._core import p56_activity, phrases, stft_power

SR = 44100
RNG = np.random.default_rng(7)


def _vowel(dur: float, f0: float = 220.0, amp: float = 0.2, fmax: float = 12000.0) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    x = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 100) if f0 * k < fmax)
    env = np.minimum(1.0, np.minimum(t / 0.01, (dur - t) / 0.02))
    return amp * x * env / np.max(np.abs(x))


def _band_noise(dur: float, lo: float, hi: float, amp: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    n = sosfilt(sos, RNG.standard_normal(int(dur * SR)))
    return amp * n / np.max(np.abs(n))


def _pink(n: int, amp: float = 0.1) -> np.ndarray:
    spec = np.fft.rfft(RNG.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return amp * x / np.sqrt(np.mean(x**2))


def _sung_line(n_syll: int = 8, sib_amp: float = 0.08) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """[sibilant 0.12 s][vowel 0.35 s][silence 0.25 s] x n. Returns signal and sibilant sample spans."""
    parts, spans, pos = [], [], 0
    for i in range(n_syll):
        s = _band_noise(0.12, 4500, 10000, sib_amp)
        v = _vowel(0.35, f0=200 + 15 * i)
        gap = np.zeros(int(0.25 * SR))
        spans.append((pos, pos + len(s)))
        parts += [s, v, gap]
        pos += len(s) + len(v) + len(gap)
    return np.concatenate(parts), spans


def test_loudness_calibration_1khz_40db_is_about_one_sone():
    t = np.arange(SR * 2) / SR
    tone = 0.1 * np.sin(2 * np.pi * 1000 * t)
    tr = loudness.partial_loudness_track(tone, np.zeros_like(tone), SR, listen_db_spl=40.0)
    assert 0.8 < float(np.mean(tr.alone_sone[5:-5])) < 1.4


def test_raising_backing_buries_vocal():
    v = _vowel(3.0)
    b = _pink(len(v))
    losses = []
    for g in (0.1, 0.5, 2.0):
        tr = loudness.partial_loudness_track(v, g * b, SR)
        act, _, _ = p56_activity(v, SR)
        losses.append(float(np.nanmean(loudness.masking_loss_frames(tr, act[: len(tr.times)]))))
    assert losses[0] > losses[1] > losses[2]
    assert losses[2] < -3.0


def test_adding_highs_makes_vocal_brighter_and_airier():
    v = _vowel(3.0)
    b = _pink(len(v))
    act, _, _ = p56_activity(v, SR)
    base = spectral.colour_indices(v, b, SR, act)
    sos = butter(2, 6000, btype="high", fs=SR, output="sos")
    bright = v + 3.0 * sosfilt(sos, v)
    up = spectral.colour_indices(bright, b, SR, act)
    assert up.bright_rel_db > base.bright_rel_db + 1.0
    assert up.air_rel_db > base.air_rel_db + 1.0
    assert up.tilt_db_per_oct > base.tilt_db_per_oct


def test_adding_low_mid_raises_body():
    v = _vowel(3.0)
    b = _pink(len(v))
    act, _, _ = p56_activity(v, SR)
    sos = butter(2, [250, 600], btype="band", fs=SR, output="sos")
    thick = spectral.colour_indices(v + 2.0 * sosfilt(sos, v), b, SR, act)
    base = spectral.colour_indices(v, b, SR, act)
    assert thick.body_rel_db > base.body_rel_db + 1.0


def test_hf_noise_raises_fizz():
    v = _vowel(3.0, fmax=18000.0)
    fizzy = v + _band_noise(3.0, 9000, 15000, 0.05)
    act, _, _ = p56_activity(v, SR)
    f = spectral.fizz_indices(fizzy, v, SR, act)
    assert f.hf_rel_delta_db > 3.0
    assert f.hf_flatness_delta > 0.0


@pytest.fixture(scope="module")
def sung():
    x, spans = _sung_line()
    feats = consonants.frame_features(x, SR)
    events = consonants.detect_consonants(x, SR, feats)
    return x, spans, events


def test_sibilants_are_detected(sung):
    _x, spans, events = sung
    sib = [e for e in events if e.kind == "sibilant"]
    assert len(sib) >= len(spans) - 2


def _scale_spans(x: np.ndarray, spans: list[tuple[int, int]], g: float) -> np.ndarray:
    y = x.copy()
    for a, b in spans:
        y[a:b] *= g
    return y


def test_cutting_sibilants_reads_as_crushed_and_boosting_as_too_strong(sung):
    x, spans, events = sung
    sib = [e for e in events if e.kind == "sibilant"]
    crushed = consonants.measure_consonants(sib, _scale_spans(x, spans, 0.3), x, None, SR)
    strong = consonants.measure_consonants(sib, _scale_spans(x, spans, 2.0), x, None, SR)
    assert np.median([m.delta_ratio_db for m in crushed]) < -6.0
    assert np.median([m.delta_ratio_db for m in strong]) > 3.0
    assert np.median([m.delta_crest_db for m in strong]) > 0.0


def test_louder_backing_masks_consonants(sung):
    x, _, events = sung
    sib = [e for e in events if e.kind == "sibilant"]
    b = _pink(len(x))
    lo = consonants.measure_consonants(sib, x, x, 0.02 * b, SR)
    hi = consonants.measure_consonants(sib, x, x, 0.2 * b, SR)
    assert np.median([m.vi_db for m in hi]) < np.median([m.vi_db for m in lo]) - 10.0


def test_level_drop_in_processing_is_a_far_step():
    pre = np.concatenate([_vowel(0.6, f0=200 + 10 * (i % 5)) for i in range(40)])
    gain = np.ones_like(pre)
    k = int(12.0 * SR)
    gain[k:] = 10 ** (-3 / 20)
    tr = steps.step_track(pre * gain, pre, SR)
    best_t, best_s = steps.top_steps(tr, n=1)[0]
    assert abs(best_t - 12.0) < 1.0
    assert best_s > 2.0
    i = int(np.argmin(np.abs(tr.times - 12.0)))
    assert tr.d_level[i] < -2.0


def test_delay_grid_fit():
    ioi = 0.5  # 120 BPM
    on = timing.delay_vs_grid(0.25, ioi)
    off = timing.delay_vs_grid(0.2347, ioi)
    assert on.best_note == "1/8" and not on.off_grid
    assert off.off_grid and off.error_s < 0


def test_echo_period_is_measured():
    n = int(2.0 * SR)
    x = np.zeros(n)
    burst = _band_noise(0.03, 500, 4000, 1.0)
    for k in range(6):
        i = int(k * 0.3 * SR)
        x[i : i + len(burst)] += burst * (0.6**k)
    assert abs(timing.echo_period(x, SR, 0.0, 2.0) - 0.3) < 0.01


def test_fx_ending_before_note_end_is_a_dropout():
    pre = np.concatenate([np.zeros(SR // 2), _vowel(1.5), np.zeros(SR)])
    note_end = 0.5 + 1.5
    late = timing.fx_dropout(pre, SR, 0.5, note_end - 0.4)
    ok = timing.fx_dropout(pre, SR, 0.5, note_end)
    assert late.missing and late.dropout_s > 0.25
    assert not ok.missing


def test_louder_harmony_is_more_prominent_and_octave_interval():
    lead = _vowel(2.0, f0=220.0)
    harm = _vowel(2.0, f0=330.0)  # a fifth: partials not all hidden under the lead's
    b = _pink(len(lead), 0.03)
    quiet = harmony.harmony_prominence(0.2 * harm, lead, b, SR)
    loud = harmony.harmony_prominence(1.0 * harm, lead, b, SR)
    assert np.nanmedian(loud.prominence_phon) > np.nanmedian(quiet.prominence_phon) + 5.0
    _, times, _ = stft_power(lead, SR)
    st = harmony.interval_semitones(harmony.f0_track(harm, SR, times), harmony.f0_track(lead, SR, times))
    assert abs(st - 12.0 * np.log2(1.5)) < 0.5


def test_reverb_lengthens_vocal_decay_and_tail():
    pre = np.concatenate([np.concatenate([_vowel(0.8, f0=180 + 20 * i), np.zeros(int(0.9 * SR))]) for i in range(8)])
    ir_t = np.arange(int(1.2 * SR)) / SR
    ir = RNG.standard_normal(len(ir_t)) * np.exp(-6.9 * ir_t / 1.0)  # T60 = 1.0 s
    ir[0] = 0
    wet = pre + 0.5 * np.convolve(pre, ir)[: len(pre)] / np.sqrt(np.sum(ir**2))
    dry_s = space.phrase_end_decay(pre, SR)
    wet_s = space.phrase_end_decay(wet, SR, ref_for_phrases=pre)
    assert wet_s.tail_ratio_db > dry_s.tail_ratio_db + 6.0
    assert wet_s.decay_t60_s > dry_s.decay_t60_s


def test_side_mid_width():
    v = _vowel(2.0)
    mono = space.side_mid_db(np.stack([v, v]), SR)
    wide = space.side_mid_db(np.stack([v, 0.3 * v + _pink(len(v), 0.05)]), SR)
    assert wide > mono + 20.0


def test_phrases_split_on_gaps():
    x = np.concatenate([_vowel(1.0), np.zeros(SR), _vowel(1.0)])
    segs, _, _ = phrases(x, SR)
    assert len(segs) == 2


def test_block_masking_loss_tracks_backing_level():
    v = _vowel(4.0)
    b = _pink(len(v))
    act, _, _ = p56_activity(v, SR)
    quiet = loudness.block_masking_loss(loudness.partial_loudness_track(v, 0.1 * b, SR), act)
    loud = loudness.block_masking_loss(loudness.partial_loudness_track(v, 2.0 * b, SR), act)
    assert len(quiet) >= 8
    assert np.median(loud[:, 1]) < np.median(quiet[:, 1]) - 3.0
    assert np.median(loud[:, 2]) < np.median(quiet[:, 2])
