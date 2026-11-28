"""v8 R4: per-song knob search against the pro references (docs/v8r_match/README.md).

Pieces (pure functions are unit-tested in tests/test_v8_r4_match.py; the VST3 runs are not):

- main-only regions (2a): Ryo's non-lead stems silent, lead singing, pro vocal S/M on the low side.
  Original (v7 input) time <-> cover time through the R2 drift curve (drift.json).
- features + distance (2b): v8 voice (lead + sends, pre-master) vs the pro vocal (post-master,
  from the R2 subtraction), always relative to the backing track; each term normalised by the
  pro-to-pro spread.
- two-stage CMA-ES (2c): stage A tone/dynamics on the full lead chain, stage B space/level on a
  cached stage-A lead. Every evaluation (with the per-cover distances) goes to a jsonl.
- leave-one-out over covers (2c), from the jsonl only (no extra render).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
from scipy import signal

from mix_automation.v8.knobs import KNOB_SPEC_MAP, V8Knobs, load_knobs
from mix_automation.v8.metrics._core import HOP, N_FFT, P56_HANGOVER_S, p56_activity

ROOT = Path(__file__).resolve().parents[3]
TARGETS = ROOT / "data" / "processed" / "v8r_targets"
CACHE = ROOT / "data" / "processed" / "v8r_match"
SR = 44100
EPS = 1e-12

# Pairs whose R2 subtraction passed (docs/v8r_targets/README.md 2 章, 採用 = 引き算).
SONG_REFS: dict[str, tuple[str, ...]] = {
    "エバ": ("Ryo_Eva__cover_Ivudotto", "Ryo_Eva__cover_Junichi", "Ryo_Eva__cover_KaguraNana",
            "Ryo_Eva__cover_Ryushen"),
    "ラブ&デストロイ": ("Ryo_LoveAndDestroy__cover_Azsagawa", "Ryo_LoveAndDestroy__cover_WolpisCarter_gVT-",
                  "Ryo_LoveAndDestroy__cover_Yoei"),
    "Fall in Love": ("Ryo_FallInLove__cover_FemmeFatale_j6yGx-r",),
}


def cover_name(pair: str) -> str:
    return pair.split("__cover_", 1)[-1]


# ======================================================================= time mapping (R2 drift.json)


def load_drift_curve(pair: str, root: Path = TARGETS) -> np.ndarray:
    """[[cover_time_s, lag_samples], ...]: cover[c] ~ inst[c - lag(c)] (extract_targets.aligned_inst)."""
    d = json.loads((root / pair / "drift.json").read_text(encoding="utf-8"))
    return np.asarray(d["curve"], dtype=np.float64)


def orig_to_cover(o_samples: np.ndarray | float, curve: np.ndarray, sr: int = SR) -> np.ndarray:
    """Original (inst) sample -> cover sample: solve c - lag(c) = o by fixed-point iteration
    (lag changes by < 1 ms over the song, so 4 iterations are far below a sample)."""
    o = np.asarray(o_samples, dtype=np.float64)
    c = o.copy()
    for _ in range(4):
        c = o + np.interp(c / sr, curve[:, 0], curve[:, 1])
    return c


# ======================================================================= 2a. main-only regions

FRAME_DT = HOP / SR  # shared stft grid of v8.metrics._core
# Regions: R2 measured voc-inst in 10 s windows that were >= 50 % voice-active (extract_targets.stage_meas).
MAX_REGION_S = 10.0
MIN_LEAD_SHARE = 0.5
# >= 5 momentary-loudness blocks (400 ms) so the p10-p90 width means something.
MIN_REGION_S = 2.0
# Padding around each evaluated region, cut away before measuring: 1 s before for the compressors'
# release (LA-2A: 60 ms + slow second stage) and phrase_ride's ramps; 2 s after = longest plate decay
# knob (plate_decay_s hi) so reverb tails and the R2 tail window (100-500 ms) stay inside.
PAD_PRE_S = 1.0
PAD_POST_S = 2.0
EXCERPT_BUDGET_S = 40.0  # R4 指示: excerpts (with padding) <= 40 s per song


@dataclass(frozen=True)
class Region:
    t0: float  # evaluated part, original (inst) time in seconds
    t1: float
    lead_share: float = 1.0

    @property
    def pad0(self) -> float:
        return self.t0 - PAD_PRE_S

    @property
    def pad1(self) -> float:
        return self.t1 + PAD_POST_S

    @property
    def padded_s(self) -> float:
        return self.pad1 - self.pad0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def candidate_regions(lead_act: np.ndarray, nonlead_act: np.ndarray, duration_s: float,
                      frame_dt: float = FRAME_DT) -> list[Region]:
    """Runs where the non-lead stems are inactive, cut at lead pauses of >= PAD_PRE_S + PAD_POST_S (the
    padding already covers such a pause from both sides), trimmed to the lead's activity, split into
    pieces of at most MAX_REGION_S, kept when >= MIN_REGION_S long, >= MIN_LEAD_SHARE lead-active and the
    padding fits inside the song."""
    n = min(len(lead_act), len(nonlead_act))
    lead = np.asarray(lead_act[:n], bool)
    free = ~np.asarray(nonlead_act[:n], bool)
    max_gap = round((PAD_PRE_S + PAD_POST_S) / frame_dt)
    out: list[Region] = []
    for a, b in _runs(free):
        groups: list[list[int]] = []
        for s, e in _runs(lead[a:b]):
            if groups and s - groups[-1][1] < max_gap:
                groups[-1][1] = e
            else:
                groups.append([s, e])
        for gs, ge in groups:
            a2, b2 = a + gs, a + ge
            length = (b2 - a2) * frame_dt
            if length < MIN_REGION_S:
                continue
            k = int(np.ceil(length / MAX_REGION_S))
            edges = np.linspace(a2, b2, k + 1).round().astype(int)
            for s, e in pairwise(edges):
                r = Region(round(s * frame_dt, 3), round(e * frame_dt, 3), round(float(lead[s:e].mean()), 3))
                if (r.t1 - r.t0) >= MIN_REGION_S and r.lead_share >= MIN_LEAD_SHARE                         and r.pad0 >= 0.0 and r.pad1 <= duration_s:
                    out.append(r)
    return out


def frame_ms_energy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mid, side) frame mean-square energy of a stereo signal on the v8.metrics._core grid."""
    a = np.asarray(x, dtype=np.float64)
    z = (0.5 * (a[0] + a[1]), 0.5 * (a[0] - a[1]))
    out = []
    for m in z:
        if len(m) < N_FFT:
            m = np.pad(m, (0, N_FFT - len(m)))
        nf = 1 + (len(m) - N_FFT) // HOP
        c = np.concatenate([[0.0], np.cumsum(m * m)])
        st = HOP * np.arange(nf)
        out.append((c[st + N_FFT] - c[st]) / N_FFT)
    return out[0], out[1]


def sm_window_db(e_mid: np.ndarray, e_side: np.ndarray, act: np.ndarray, f0: int, f1: int) -> float:
    m = np.asarray(act[f0:f1], bool)
    if not np.any(m):
        return float("nan")
    return float(10 * np.log10((e_side[f0:f1][m].sum() + EPS) / (e_mid[f0:f1][m].sum() + EPS)))


def song_sm_reference(e_mid: np.ndarray, e_side: np.ndarray, act: np.ndarray, frame_dt: float = FRAME_DT) -> np.ndarray:
    """S/M of the pro vocal in 10 s windows (5 s hop) that are >= 50 % active: same windows as R2's
    voc-inst windows (extract_targets.stage_meas). The region criterion compares against their median."""
    w = round(MAX_REGION_S / frame_dt)
    vals = []
    for s in range(0, max(1, len(act) - w), w // 2):
        if np.mean(act[s:s + w]) >= MIN_LEAD_SHARE:
            vals.append(sm_window_db(e_mid, e_side, act, s, s + w))
    return np.array([v for v in vals if np.isfinite(v)])


def select_regions(cands: Sequence[Region], sm_by_cover: dict[str, list[float]],
                   ref_by_cover: dict[str, np.ndarray], budget_s: float = EXCERPT_BUDGET_S) -> list[dict]:
    """Keep regions whose pro S/M is at or below the cover's song median ("下位側") for at least half of
    the covers; take them by ascending mean percentile rank until the padded total reaches budget_s."""
    rows = []
    for i, r in enumerate(cands):
        ranks, low = [], 0
        for cov, sms in sm_by_cover.items():
            v, ref = sms[i], ref_by_cover[cov]
            if not np.isfinite(v) or ref.size == 0:
                continue
            ranks.append(float(np.mean(ref < v)))
            low += int(v <= np.median(ref))
        if ranks and low * 2 >= len(ranks):
            rows.append({"region": r, "rank": float(np.mean(ranks)),
                         "pro_sm_db": {c: round(float(s[i]), 2) for c, s in sm_by_cover.items()}})
    rows.sort(key=lambda d: (d["rank"], d["region"].t0))
    chosen, total = [], 0.0
    for d in rows:
        if total + d["region"].padded_s <= budget_s + 1e-9:
            chosen.append(d)
            total += d["region"].padded_s
    chosen.sort(key=lambda d: d["region"].t0)
    return chosen


# ======================================================================= excerpt layout

# ----------------------------------------------------------------------- R4b: reverb-tail spots
# R4 could not measure the tail inside the main-only regions (README 6 章: never 3 phrase ends in the
# evaluated parts). R4b looks for main-only phrase ends over the whole song and appends short cuts.
TAIL_PHRASE_MIN_S = 0.4  # R2 tail_db: phrases >= 0.4 s ...
# ... followed by a lead pause >= 1.5 s, so that the "late" window (end + 1.0..1.5 s) lies inside the pause
# (R2 used 0.6 s for its 100-500 ms window only; R4b 再設計 2026-11-28).
TAIL_GAP_MIN_S = 1.5
TAIL_MAX_SPOTS = 6       # R4b 指示: at most 6 phrase ends, longest pauses first
TAIL_PRE_S = 1.5         # cut 1.5 s before the mark (>= the 0.4 s phrase window + compressor settling)
# Fixed windows around end = mark + P.56 hangover (= end of the P.56 run), the same for v8 and the pros:
TAIL_PHRASE_WIN_S = (-0.4, 0.0)  # reference: last 0.4 s of the phrase (R2 minimum phrase length)
TAIL_WIN_S = (0.1, 0.5)          # R2 tail window (100-500 ms after the phrase end)
TAIL_LATE_WIN_S = (1.0, 1.5)     # floor estimate: the R2 subtraction residual stays here when the reverb is gone
TAIL_CUT_AFTER_END_S = 1.6       # cut runs 1.6 s past the end (late window + 0.1 s)
TAIL_POST_S = P56_HANGOVER_S + TAIL_CUT_AFTER_END_S  # cut length after the mark
# A pro spot counts as a real tail (two-sided term) when tail - late >= 6 dB. Reasoning (setting): a reverb
# just as loud as the subtraction floor adds only +3 dB to it, so a 6 dB decay means the reverb is clearly
# above the floor; below that the pro tail is only an upper bound for the reverb (one-sided term).
# R4b run 2026-11-28 (Eva): the two-sided spots (KaguraNana -8.5 dB, Ryushen -11.8 dB with 9-10 dB decay) pulled
# the plate/room sends up by 12 dB of tail. The pros are other singers with their own phrasing, so 100-500 ms
# after 凌's phrase end the pro can still be singing; a large "decay" does not prove a reverb tail. Two-sided
# comparison is therefore switched off (inf): every spot is an upper bound only.
TAIL_DECAY_MIN_DB = float("inf")
# R4b tail scale fixed to the R2 fallback (2 x 1.76^2), not the pro-pair median: most pro spots sit on the
# subtraction floor and only bound the reverb from above (one-sided), the few two-sided spots differ a
# lot between pros, so the pair median is unstable (R4b smoke 2026-11-28: pro-pair median 46.4).
R4B_FIXED_SCALES = ("tail",)
# The P.56 mask keeps P56_HANGOVER_S (0.2 s) after the last frame above its threshold, so the phrase
# end is marked that much before the end of the activity run.


def find_tail_spots(lead_act: np.ndarray, nonlead_act: np.ndarray, duration_s: float,
                    frame_dt: float = FRAME_DT, max_spots: int = TAIL_MAX_SPOTS) -> list[dict]:
    """Main-only phrase ends: a lead P.56 run >= TAIL_PHRASE_MIN_S, then a lead pause >= TAIL_GAP_MIN_S,
    the non-lead stems inactive from the phrase start to the end of the pause, and the tail cut inside
    the song. The `max_spots` with the longest pauses, returned in time order (original time, s)."""
    n = min(len(lead_act), len(nonlead_act))
    lead = np.asarray(lead_act[:n], bool)
    non = np.asarray(nonlead_act[:n], bool)
    runs = _runs(lead)
    out = []
    for i, (s, e) in enumerate(runs):
        if (e - s) * frame_dt < TAIL_PHRASE_MIN_S - 1e-9:
            continue
        nxt = runs[i + 1][0] if i + 1 < len(runs) else n
        gap = (nxt - e) * frame_dt
        if gap < TAIL_GAP_MIN_S - 1e-9 or non[s:nxt].any():
            continue
        mark = e * frame_dt - P56_HANGOVER_S
        if mark - TAIL_PRE_S < 0.0 or mark + TAIL_POST_S > duration_s:
            continue
        out.append({"t_start": round(s * frame_dt, 3), "t_end": round(e * frame_dt, 3),
                    "mark": round(mark, 3), "gap_s": round(gap, 3)})
    out.sort(key=lambda d: (-d["gap_s"], d["mark"]))
    return sorted(out[:max_spots], key=lambda d: d["mark"])


EDGE_FADE_S = 0.01  # 10 ms fades where padded excerpts are joined (avoid clicks into the chain)


@dataclass(frozen=True)
class Layout:
    """Padded excerpts laid end to end, then (R4b) the tail cuts. Slices are in samples of the
    concatenated signal. padded_slices / eval_slices cover the regions only; tail cuts have their own."""
    regions: tuple[Region, ...]
    sr: int = SR
    tails: tuple[float, ...] = ()  # R4b: phrase-end marks, original time (s)

    def _lens(self) -> list[tuple[int, int, int]]:
        out = []
        for r in self.regions:
            pre = round(PAD_PRE_S * self.sr)
            ev = round((r.t1 - r.t0) * self.sr)
            post = round(PAD_POST_S * self.sr)
            out.append((pre, ev, post))
        return out

    def _tail_len(self) -> int:
        return round(TAIL_PRE_S * self.sr) + round(TAIL_POST_S * self.sr)

    @property
    def n_regions(self) -> int:
        return sum(a + b + c for a, b, c in self._lens())

    @property
    def n(self) -> int:
        return self.n_regions + len(self.tails) * self._tail_len()

    def tail_slices(self) -> list[slice]:
        s0, ln = self.n_regions, self._tail_len()
        return [slice(s0 + i * ln, s0 + (i + 1) * ln) for i in range(len(self.tails))]

    def tail_mark_offset(self) -> int:
        """Sample offset of the phrase-end mark inside each tail cut."""
        return round(TAIL_PRE_S * self.sr)

    def _all_slices(self) -> list[slice]:
        return self.padded_slices() + self.tail_slices()

    def padded_slices(self) -> list[slice]:
        out, s = [], 0
        for a, b, c in self._lens():
            out.append(slice(s, s + a + b + c))
            s += a + b + c
        return out

    def eval_slices(self) -> list[slice]:
        out, s = [], 0
        for a, b, c in self._lens():
            out.append(slice(s + a, s + a + b))
            s += a + b + c
        return out

    def cut(self, x: np.ndarray, starts: Iterable[float]) -> np.ndarray:
        """Concatenate x[..., start : start + padded_len] for each excerpt start (in samples)."""
        a = np.asarray(x, dtype=np.float32)
        parts = []
        fade = max(1, round(EDGE_FADE_S * self.sr))
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        for st, sl in zip(starts, self._all_slices(), strict=True):
            ln = sl.stop - sl.start
            s0 = round(st)
            seg = np.zeros(a.shape[:-1] + (ln,), np.float32)
            lo, hi = max(0, s0), min(a.shape[-1], s0 + ln)
            if hi > lo:
                seg[..., lo - s0:hi - s0] = a[..., lo:hi]
            seg[..., :fade] *= ramp
            seg[..., -fade:] *= ramp[::-1]
            parts.append(seg)
        return np.concatenate(parts, axis=-1)

    def orig_starts(self) -> list[float]:
        return [r.pad0 * self.sr for r in self.regions] + [(t - TAIL_PRE_S) * self.sr for t in self.tails]


# ======================================================================= 2b. features and distance

THIRD_OCT_CENTERS = tuple(float(1000 * 2 ** (k / 3)) for k in range(-10, 12))  # 100 Hz .. 12.7 kHz
DUCK_BAND_HZ = (300.0, 6000.0)  # band the v8 multiband duck works in (R2 README 4 章: 300 Hz-6 kHz)
TAIL_MIN_PHRASES = 3  # R2 tail_db: median over >= 3 phrases


def _st(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return np.stack([a, a]) if a.ndim == 1 else a


def _lufs(x: np.ndarray, sr: int) -> float:
    from mix_automation.v8.dsp import integrated_lufs

    v = integrated_lufs(_st(x), sr)
    return v if np.isfinite(v) else float("nan")


def third_oct_db(x: np.ndarray, sr: int) -> np.ndarray:
    """1/3-octave band power (dB) of a stereo signal (sum of channels), Welch 4096."""
    a = _st(x)
    f, p = signal.welch(a, fs=sr, nperseg=min(4096, a.shape[-1]), axis=-1)
    p = p.sum(axis=0)
    out = []
    for fc in THIRD_OCT_CENTERS:
        sel = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        out.append(10 * np.log10(p[sel].sum() + EPS))
    return np.array(out)


def _shape(curve: np.ndarray) -> np.ndarray:
    return curve - np.mean(curve)


def momentary_lufs(x: np.ndarray, sr: int) -> np.ndarray:
    """EBU momentary loudness: 400 ms blocks, 100 ms hop, BS.1770 relative gate -10 LU."""
    from mix_automation.v8.dsp import k_weight

    k = k_weight(_st(x), sr)
    blk, hop = round(0.4 * sr), round(0.1 * sr)
    if k.shape[-1] < blk:
        return np.array([])
    c = np.concatenate([np.zeros((2, 1)), np.cumsum(k * k, axis=1)], axis=1)
    st = np.arange(0, k.shape[-1] - blk + 1, hop)
    ms = ((c[:, st + blk] - c[:, st]) / blk).sum(axis=0)
    lk = -0.691 + 10 * np.log10(ms + EPS)
    keep = lk > -70.0
    if not np.any(keep):
        return np.array([])
    rel = -0.691 + 10 * np.log10(np.mean(ms[keep]) + EPS) - 10.0
    return lk[keep & (lk > rel)]


def tail_values(voice: np.ndarray, sr: int, eval_lo: int, eval_hi: int) -> list[float]:
    """R2 tail (extract_targets.tail_db): energy 100-500 ms after a phrase end (level < p99 - 20 dB,
    10 ms frames) relative to the phrase; phrases >= 0.4 s followed by a gap >= 0.6 s. Only phrase ends
    inside [eval_lo, eval_hi) samples of this excerpt count."""
    h = 441
    m = _st(voice).mean(0)
    nfr = len(m) // h
    if nfr < 60:
        return []
    e = 10 * np.log10((m[: nfr * h].reshape(nfr, h) ** 2).mean(1) + EPS)
    on = e > np.percentile(e, 99) - 20
    on = signal.medfilt(on.astype(float), 9) > 0.5
    vals = []
    for k in np.flatnonzero(np.diff(on.astype(int)) == -1):
        if not (eval_lo <= k * h < eval_hi):
            continue
        st = k
        while st > 0 and on[st - 1]:
            st -= 1
        nxt = k + 1 + int(np.argmax(on[k + 1:])) if on[k + 1:].any() else nfr
        if k - st < 40 or nxt - k < 60 or k + 50 > nfr:
            continue
        ph = (10 ** (e[st:k + 1] / 10)).mean()
        tl = (10 ** (e[k + 10:k + 50] / 10)).mean()
        vals.append(float(10 * np.log10(tl / ph)))
    return vals


def _active(x: np.ndarray, sr: int) -> np.ndarray:
    mono = _st(x).mean(0).astype(np.float32)
    mask, _, _ = p56_activity(mono, sr)
    return mask


def _sm_db(x: np.ndarray, act: np.ndarray | None) -> float:
    em, es = frame_ms_energy(_st(x))
    m = np.ones(len(em), bool) if act is None else np.asarray(act[: len(em)], bool)
    if not np.any(m):
        return float("nan")
    return float(10 * np.log10((es[m].sum() + EPS) / (em[m].sum() + EPS)))


def _band_frames(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    from mix_automation.v8.metrics._core import band_power, stft_power

    a = _st(x)
    tot = None
    for ch in a:
        f, _, p = stft_power(ch, sr)
        b = band_power(p, f, lo, hi)
        tot = b if tot is None else tot + b
    return tot


def tail_windows_db(voice: np.ndarray, lay: Layout, sr: int = SR,
                    valid: Sequence[bool] | None = None) -> tuple[list[float], list[float]]:
    """R4b: per tail cut, (tail, late) = mean power of TAIL_WIN_S / TAIL_LATE_WIN_S after the phrase end
    relative to TAIL_PHRASE_WIN_S, in dB, on the channel-mean mono. NaN where `valid` is False."""
    m = _st(voice).mean(0)
    end = lay.tail_mark_offset() + round(P56_HANGOVER_S * sr)

    def pw(sl: slice, w: tuple[float, float]) -> float:
        seg = m[sl.start + end + round(w[0] * sr): sl.start + end + round(w[1] * sr)]
        return float(np.mean(seg * seg)) if seg.size else float("nan")

    tails, lates = [], []
    for i, sl in enumerate(lay.tail_slices()):
        if valid is not None and not valid[i]:
            tails.append(float("nan"))
            lates.append(float("nan"))
            continue
        ph = pw(sl, TAIL_PHRASE_WIN_S) + EPS
        tails.append(float(10 * np.log10(pw(sl, TAIL_WIN_S) / ph + EPS)))
        lates.append(float(10 * np.log10(pw(sl, TAIL_LATE_WIN_S) / ph + EPS)))
    return tails, lates


def _spot_median(vals: Sequence[float]) -> float:
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size >= TAIL_MIN_PHRASES else float("nan")


def features(voice: np.ndarray, inst: np.ndarray, inst_raw: np.ndarray, lay: Layout, sr: int = SR,
             afx: bool = False, tail_valid: Sequence[bool] | None = None) -> dict:
    """Measurements on the evaluated parts (padding cut away; ducking and tail also look at the padding).

    voice / inst / inst_raw are stereo and laid out by `lay`: v8 = (lead+sends, inst after duck,
    inst before duck); pro = (target_vocal, inst_pro, inst_aligned) from R2.
    R4b: with tail cuts in `lay`, "tail" is measured in fixed windows after the marked phrase ends
    ("tail_by_spot", "late_by_spot"; the R4 in-region value stays in "tail_r4"); `afx` adds the AFx-Rep
    embedding of the voice."""
    voice, inst, inst_raw = _st(voice), _st(inst), _st(inst_raw)
    ev = lay.eval_slices()
    cat = np.concatenate
    v_e = cat([voice[:, s] for s in ev], axis=1)
    i_e = cat([inst[:, s] for s in ev], axis=1)
    r_e = cat([inst_raw[:, s] for s in ev], axis=1)
    act_e = _active(v_e, sr)
    f: dict = {}
    f["lu"] = np.array([_lufs(voice[:, s], sr) - _lufs(inst[:, s], sr) for s in ev])
    f["spec"] = _shape(third_oct_db(v_e, sr) - third_oct_db(i_e, sr))
    blocks = cat([momentary_lufs(voice[:, s], sr) for s in ev])
    f["dyn"] = float(np.percentile(blocks, 90) - np.percentile(blocks, 10)) if blocks.size >= 5 else float("nan")
    crest = []
    for s in ev:
        mid = 0.5 * (voice[0, s] + voice[1, s])
        a = _active(voice[:, s], sr)
        em = frame_ms_energy(np.stack([mid, mid]))[0]
        m = np.asarray(a[: len(em)], bool)
        if np.any(m):
            crest.append(20 * np.log10(np.max(np.abs(mid)) + EPS) - 10 * np.log10(em[m].mean() + EPS))
    f["crest"] = float(np.median(crest)) if crest else float("nan")
    f["sm"] = _sm_db(v_e, act_e)
    tails: list[float] = []
    for ps, es in zip(lay.padded_slices(), ev, strict=True):
        tails += tail_values(voice[:, ps], sr, es.start - ps.start, es.stop - ps.start)
    f["tail_r4"] = float(np.median(tails)) if len(tails) >= TAIL_MIN_PHRASES else float("nan")
    if lay.tails:
        f["tail_by_spot"], f["late_by_spot"] = tail_windows_db(voice, lay, sr, tail_valid)
        f["tail"] = _spot_median(f["tail_by_spot"])
    else:
        f["tail"] = f["tail_r4"]
    if afx:
        from mix_automation.v8 import afx_rep

        f["afx"] = afx_rep.embed(v_e, sr) if afx_rep.available() else None
    f["inst_spec"] = _shape(third_oct_db(i_e, sr) - third_oct_db(r_e, sr))
    f["inst_sm"] = _sm_db(i_e, None) - _sm_db(r_e, None)
    pi, pr, act = [], [], []
    for ps in lay.padded_slices():
        bi = _band_frames(inst[:, ps], sr, *DUCK_BAND_HZ)
        br = _band_frames(inst_raw[:, ps], sr, *DUCK_BAND_HZ)
        a = _active(voice[:, ps], sr)
        n = min(len(bi), len(br), len(a))
        pi.append(bi[:n])
        pr.append(br[:n])
        act.append(np.asarray(a[:n], bool))
    pi_, pr_, act_ = cat(pi), cat(pr), cat(act)
    min_frames = int(0.5 / FRAME_DT)  # need >= 0.5 s of voice-free frames for the reference
    if act_.sum() >= min_frames and (~act_).sum() >= min_frames:
        on = 10 * np.log10(pi_[act_].sum() / (pr_[act_].sum() + EPS) + EPS)
        off = 10 * np.log10(pi_[~act_].sum() / (pr_[~act_].sum() + EPS) + EPS)
        f["duck"] = float(on - off)
    else:
        f["duck"] = float("nan")
    return f


TERMS_R4 = ("lu", "spec", "dyn", "crest", "sm", "tail", "inst_spec", "inst_sm", "duck")
TERMS = (*TERMS_R4, "afx")  # R4b: + AFx-Rep cosine distance of the voice embeddings

# Pro-to-pro spread used when a song has fewer than 2 references (standard deviation over the 4 Eva
# covers in R2 README 3 章 unless noted). The distance unit is the expected pro-to-pro squared
# difference 2*sd^2, so "1" = as different as two pro covers.
FALLBACK_SD: dict[str, float] = {
    "lu": 0.86,        # voc-inst LU +2.9/+3.2/+4.6/+2.7
    "spec": 1.83,      # brightness voc-inst -0.4/-4.3/-1.0/-3.2 dB
    "dyn": 0.34,       # 10 s window voc-inst p10-p90 width 0.7/1.3/1.2/1.5 LU
    "crest": 1.0,      # not measured in R2: placeholder, replaced by measured spread when >= 2 covers
    "sm": 2.38,        # vocal S/M -8.6/-11.7/-12.0/-14.4 dB
    "tail": 1.76,      # tail -19.8/-19.1/-18.7/-22.6 dB
    "inst_spec": 1.8,  # RMS over bands of the per-band sd of the inst EQ (125..12k, rel. 1 kHz)
    "inst_sm": 0.59,   # inst S/M change +1.1/+0.1/+0.4/-0.3 dB
    "duck": 0.5,       # ducking 1-3 kHz -0.7/-0.6/+0.3/+0.1 dB
}
# Floor for a measured scale: half the R2 sd, so a term on which the pros happen to agree closely does
# not take over the whole distance.
SCALE_FLOOR_RATIO = 0.5

# Weights (config, not measured): the vocal level relative to the backing track counts double because
# R2 found it to be the largest v8-vs-pro gap (README 5.1).
DEFAULT_WEIGHTS: dict[str, float] = {
    "lu": 2.0, "spec": 1.0, "dyn": 0.5, "crest": 0.5, "sm": 1.0, "tail": 1.0,
    "inst_spec": 0.5, "inst_sm": 0.5, "duck": 0.5,
}
# R4b (config): AFx-Rep enters with a modest weight; it is unvalidated against Ryo's ratings so far.
WEIGHTS_R4B: dict[str, float] = {**DEFAULT_WEIGHTS, "afx": 0.25}

# AFx-Rep scale (a cosine distance, not squared): median over pro pairs; floored so that pros that
# happen to embed almost identically do not blow the term up.
AFX_SCALE_FLOOR = 1e-4
# Songs with one reference: median of the pro-pair cosine distances of Eva (6 pairs) and Love&Destroy
# (3 pairs) on the R4b main-only regions, measured once on 2026-11-28: Eva 0.223-0.331 (median 0.309),
# Love&Destroy 0.119-0.194 (median 0.148); median of all 9 pairs = 0.271.
AFX_FALLBACK_SCALE = 0.271


def tail_spot_terms(fa: dict, fb: dict) -> list[tuple[int, float, float, float, str]]:
    """R4b per-spot tail comparison, fb = the reference (pro): (spot, a_tail, b_tail, b_decay, kind) for spots
    where both tails and the reference late value exist. kind "two" when the reference decays by
    >= TAIL_DECAY_MIN_DB (tail above the floor), else "one" (reference tail = upper bound only)."""
    a, b = np.asarray(fa["tail_by_spot"], float), np.asarray(fb["tail_by_spot"], float)
    late = np.asarray(fb.get("late_by_spot", [np.nan] * len(b)), float)
    out = []
    for i in range(min(len(a), len(b), len(late))):
        if np.isfinite(a[i]) and np.isfinite(b[i]) and np.isfinite(late[i]):
            dec = float(b[i] - late[i])
            out.append((i, float(a[i]), float(b[i]), dec, "two" if dec >= TAIL_DECAY_MIN_DB else "one"))
    return out


def _tail_raw(fa: dict, fb: dict) -> float:
    if fa.get("tail_by_spot") is not None and fb.get("tail_by_spot") is not None:
        rows = tail_spot_terms(fa, fb)
        if len(rows) < TAIL_MIN_PHRASES:
            return float("nan")
        return float(np.mean([(at - bt) ** 2 if kind == "two" else max(0.0, at - bt) ** 2
                              for _, at, bt, _, kind in rows]))
    a, b = float(fa.get("tail", np.nan)), float(fb.get("tail", np.nan))
    return (a - b) ** 2 if np.isfinite(a - b) else float("nan")


def term_raw(fa: dict, fb: dict, terms: Sequence[str] = TERMS) -> dict[str, float]:
    """Mean squared difference per term (vectors element-wise); NaN when either side lacks it.
    afx is the cosine distance of the embeddings (not squared). R4b tail: fb is the reference (pro);
    per spot two-sided or one-sided (tail_spot_terms), mean over >= 3 spots."""
    out = {}
    for t in terms:
        if t == "afx":
            va, vb = fa.get("afx"), fb.get("afx")
            if va is None or vb is None:
                out[t] = float("nan")
            else:
                from mix_automation.v8.afx_rep import cos_dist

                out[t] = cos_dist(np.asarray(va, float), np.asarray(vb, float))
            continue
        if t == "tail":
            out[t] = _tail_raw(fa, fb)
            continue
        a, b = np.asarray(fa.get(t, np.nan), float), np.asarray(fb.get(t, np.nan), float)
        if a.shape != b.shape:
            out[t] = float("nan")
            continue
        d = (a - b)[np.isfinite(a - b)] if a.ndim else np.atleast_1d(a - b)
        out[t] = float(np.mean(d ** 2)) if d.size and np.all(np.isfinite(d)) else float("nan")
    return out


def fallback_scales(terms: Sequence[str] = TERMS) -> dict[str, float]:
    return {t: AFX_FALLBACK_SCALE if t == "afx" else 2.0 * FALLBACK_SD[t] ** 2 for t in terms}


def pro_scales(pro_feats: dict[str, dict], terms: Sequence[str] = TERMS,
               fixed: Sequence[str] = ()) -> tuple[dict[str, float], str]:
    """Scale^2 per term = median over pro pairs of term_raw (floored); R2 fallback with < 2 covers.
    Terms in `fixed` always use the fallback scale."""
    fb = fallback_scales(terms)
    if len(pro_feats) < 2:
        return fb, "fallback_R2"
    out = {}
    for t in terms:
        if t in fixed:
            out[t] = fb[t]
            continue
        vals = [term_raw(a, b, (t,))[t] for a, b in combinations(pro_feats.values(), 2)]
        vals = [v for v in vals if np.isfinite(v)]
        floor = AFX_SCALE_FLOOR if t == "afx" else 2.0 * (SCALE_FLOOR_RATIO * FALLBACK_SD[t]) ** 2
        out[t] = max(float(np.median(vals)), floor) if vals else fb[t]
    return out, "pro_pairs"


def r4_view(f: dict) -> dict:
    """Features as R4 measured them: in-region tail, no per-spot tails, no AFx-Rep."""
    out = {k: v for k, v in f.items() if k not in ("tail_by_spot", "late_by_spot", "afx")}
    out["tail"] = f.get("tail_r4", f.get("tail", float("nan")))
    return out


def distance(fa: dict, fb: dict, scales: dict[str, float], weights: dict[str, float] | None = None,
             terms: Sequence[str] = TERMS) -> tuple[float, dict]:
    """Weighted mean of normalised terms over the terms both sides have."""
    w = weights or DEFAULT_WEIGHTS
    raw = term_raw(fa, fb, terms)
    norm = {t: raw[t] / scales.get(t, float("nan")) for t in terms}
    ok = [t for t in terms if np.isfinite(norm[t]) and w.get(t, 0.0) > 0]
    if not ok:
        return float("nan"), norm
    d = sum(w[t] * norm[t] for t in ok) / sum(w[t] for t in ok)
    return float(d), norm


def aggregate(per_cover: dict[str, float]) -> float:
    """Several references for one song -> median of the per-cover distances."""
    v = [x for x in per_cover.values() if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


# ======================================================================= knob normalisation

STAGE_A = ("hpf_hz", "presence_hz", "presence_db", "brightness", "air_hz", "comp_total_gr_db", "deess_db",
           "saturation", "pre_level_strength", "par_comp_mix")
# glue_wet is not searched: the master bus is skipped in matching and, with the uniform-gain VocalBus
# rule, the 2mix glue does not change any vocal-vs-backing measure.
STAGE_B = ("vocal_offset_lu", "ride_tolerance_lu", "doubler_send_db", "room_send_db", "plate_send_db",
           "plate_decay_s", "delay_send_db", "fx_duck_db", "inst_duck_max_db", "inst_side_high_db")


def to_unit(knobs: V8Knobs, names: Sequence[str]) -> np.ndarray:
    d = knobs.to_dict()
    return np.array([(d[n] - KNOB_SPEC_MAP[n].lo) / (KNOB_SPEC_MAP[n].hi - KNOB_SPEC_MAP[n].lo) for n in names])


def from_unit(u: np.ndarray, names: Sequence[str], base: V8Knobs) -> V8Knobs:
    vals = {}
    for n, x in zip(names, np.clip(np.asarray(u, float), 0.0, 1.0), strict=True):
        s = KNOB_SPEC_MAP[n]
        v = s.lo + float(x) * (s.hi - s.lo)
        vals[n] = float(round(v)) if s.integer else v
    return replace(base, **vals)


# ======================================================================= 2c. leave-one-out over covers


def loo_select(records: Sequence[dict], covers: Sequence[str]) -> dict[str, dict]:
    """For each held-out cover: among the evaluated candidates pick the one with the smallest median
    distance to the other covers, and report its distance to the held-out cover."""
    out = {}
    for held in covers:
        rest = [c for c in covers if c != held]
        best, best_v = None, float("inf")
        for i, r in enumerate(records):
            pc = r["per_cover"]
            v = aggregate({c: pc[c] for c in rest if c in pc})
            if np.isfinite(v) and v < best_v:
                best, best_v = i, v
        if best is None:
            continue
        out[held] = {"index": best, "train_median": round(best_v, 4),
                     "held_out": round(float(records[best]["per_cover"].get(held, np.nan)), 4)}
    return out


# ======================================================================= jsonl log


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=_jdefault) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut by an interruption
                continue
    return out


def best_record(records: Sequence[dict], stages: Iterable[str] | None = None) -> dict | None:
    st = set(stages) if stages else None
    cand = [r for r in records if (st is None or r["stage"] in st) and np.isfinite(r.get("dist", np.nan))]
    return min(cand, key=lambda r: r["dist"]) if cand else None


def _jdefault(o: object) -> object:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ======================================================================= song session (VST3)


def _hash(*parts: object) -> str:
    return hashlib.sha1(json.dumps(parts, default=str, sort_keys=True).encode()).hexdigest()[:12]


def _load_st(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: sr {sr} != {SR}")
    x = x.T
    return np.repeat(x, 2, axis=0) if x.shape[0] == 1 else x[:2]


class SongMatch:
    """Everything for one song: regions, excerpts, cached front end, pro features, evaluation."""

    def __init__(self, song: str, refs: Sequence[str] | None = None, raw_root: Path = ROOT / "data" / "raw",
                 cache_root: Path = CACHE, weights: dict[str, float] | None = None,
                 log: Callable[[str], None] = print, r4b: bool = False) -> None:
        """r4b=False keeps the R4 definition (scripts/match/run_match.py). r4b=True adds the tail cuts and
        the AFx-Rep term (dist), also reports the R4-style distance (dist_r4), and writes its cache
        files under new names (regions_r4b.json, scales_r4b.json)."""
        self.song = song
        self.r4b = r4b
        self.sfx = "_r4b" if r4b else ""
        self.terms = TERMS if r4b else TERMS_R4
        self.refs = tuple(refs or SONG_REFS[song])
        self.covers = [cover_name(p) for p in self.refs]
        self.song_folder = raw_root / song
        self.dir = cache_root / song
        self.dir.mkdir(parents=True, exist_ok=True)
        self.weights = dict(weights or (WEIGHTS_R4B if r4b else DEFAULT_WEIGHTS))
        self.log = log

    def __getstate__(self) -> dict:  # match_pool sends prepared songs to spawn workers
        d = dict(self.__dict__)
        d.pop("log", None)
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)
        self.log = print

    # ------------------------------------------------------------ preparation
    def prepare(self) -> None:
        from mix_automation.io.audio_loader import consolidate_vocal_stem_folder, load_audio
        from mix_automation.pipeline import discover_song_assets
        from mix_automation.v8 import chain

        assets = discover_song_assets(self.song_folder)
        stems = consolidate_vocal_stem_folder(list(assets["stems"]), target_sr=SR)
        inst, _ = load_audio(Path(assets["inst"]), target_sr=SR, mono=False)
        inst = np.asarray(inst, np.float32)
        inst = np.stack([inst, inst]) if inst.ndim == 1 else inst
        n = inst.shape[1]
        lead = chain.fit_len(np.asarray(stems.lead_mono, np.float32), n)
        nonlead = np.zeros(n, np.float32)
        for nm in ("double_stereo", "sub_octave_stereo", "super_octave_stereo", "harmonies_stereo_bus",
                   "main_harmonies_stereo_bus", "choir_stereo_bus"):
            a = getattr(stems, nm, None)
            if a is not None:
                nonlead += np.abs(chain.fit_len(np.asarray(a, np.float32).mean(0), n))
        self.n = n
        self.duration_s = n / SR
        self.stem_roles = stems.stem_roles_summary
        self.tail_spots: list[dict] = []
        self.regions_info = self._regions(lead, nonlead)
        self.layout = Layout(tuple(d["region"] for d in self.regions_info),
                             tails=tuple(d["mark"] for d in self.tail_spots))
        lay = self.layout
        log = chain.StageLog()
        self.inst_work = lay.cut(chain.prepare_inst(inst, SR, log), lay.orig_starts())
        self.lead_cut = lay.cut(lead, lay.orig_starts())
        self.tempo_bpm = self._tempo(inst)
        self.front_x, self.front_mask = self._front()
        self.pro = {cover_name(p): self._pro_features(p) for p in self.refs}
        self.scales, self.scale_source = pro_scales(self.pro, self.terms, R4B_FIXED_SCALES if self.r4b else ())
        info: dict = {"source": self.scale_source, "scales": self.scales, "weights": self.weights}
        if self.r4b:
            self.scales_r4, _ = pro_scales({c: r4_view(f) for c, f in self.pro.items()}, TERMS_R4)
            info |= {"scales_r4": self.scales_r4, "weights_r4": DEFAULT_WEIGHTS,
                     "pro_tail_by_spot": {c: f.get("tail_by_spot") for c, f in self.pro.items()},
                     "pro_late_by_spot": {c: f.get("late_by_spot") for c, f in self.pro.items()}}
        (self.dir / f"scales{self.sfx}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    def _regions(self, lead: np.ndarray, nonlead: np.ndarray) -> list[dict]:
        lead_act, _, _ = p56_activity(lead, SR)
        if np.max(np.abs(nonlead)) > 0:
            non_act, _, _ = p56_activity(nonlead, SR)
        else:
            non_act = np.zeros_like(lead_act)
        cands = candidate_regions(lead_act, non_act, self.duration_s)
        if self.r4b:
            self.tail_spots = find_tail_spots(lead_act, non_act, self.duration_s)
        sm_by, ref_by = {}, {}
        for p in self.refs:
            cov = cover_name(p)
            voc = _load_st(TARGETS / p / "target_vocal.wav")
            curve = load_drift_curve(p)
            em, es = frame_ms_energy(voc)
            act, _, _ = p56_activity(voc.mean(0), SR)
            act = act[: len(em)]
            ref_by[cov] = song_sm_reference(em, es, act)
            vals = []
            for r in cands:
                c0, c1 = orig_to_cover(np.array([r.t0 * SR, r.t1 * SR]), curve)
                f0, f1 = int(c0 // HOP), int(c1 // HOP)
                vals.append(sm_window_db(em, es, act, f0, f1) if 0 <= f0 and f1 <= len(em) else float("nan"))
            sm_by[cov] = vals
        chosen = select_regions(cands, sm_by, ref_by)
        info = {
            "song": self.song, "refs": list(self.refs),
            "criteria": {"nonlead": "P.56 activity of the summed non-lead stems = inactive",
                         "lead_share_min": MIN_LEAD_SHARE, "region_s": [MIN_REGION_S, MAX_REGION_S],
                         "pro_sm": "<= median of the cover's 10 s window S/M for >= half of the covers",
                         "pad_s": [PAD_PRE_S, PAD_POST_S], "excerpt_budget_s": EXCERPT_BUDGET_S},
            "n_candidates": len(cands),
            "pro_sm_song_median_db": {c: round(float(np.median(v)), 2) if v.size else None for c, v in ref_by.items()},
            "regions": [{**asdict(d["region"]), "rank": round(d["rank"], 3), "pro_sm_db": d["pro_sm_db"]}
                        for d in chosen],
            "eval_s": round(sum(d["region"].t1 - d["region"].t0 for d in chosen), 2),
            "excerpt_s": round(sum(d["region"].padded_s for d in chosen), 2),
        }
        if self.r4b:
            info["tail_spots"] = self._tail_spot_info()
            info["tail_criteria"] = {"phrase_min_s": TAIL_PHRASE_MIN_S, "gap_min_s": TAIL_GAP_MIN_S,
                                     "max_spots": TAIL_MAX_SPOTS, "cut_s": [TAIL_PRE_S, TAIL_POST_S],
                                     "mark": "end of the lead P.56 run - hangover (end = mark + hangover)",
                                     "windows_after_end_s": {"phrase": TAIL_PHRASE_WIN_S, "tail": TAIL_WIN_S,
                                                             "late": TAIL_LATE_WIN_S},
                                     "decay_min_db": TAIL_DECAY_MIN_DB}
        (self.dir / f"regions{self.sfx}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
        if not chosen:
            raise RuntimeError(f"R4: no main-only region found for {self.song}")
        return chosen

    def _tempo(self, inst: np.ndarray) -> float:
        p = self.dir / "tempo.json"
        key = _hash(self.song, inst.shape)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("key") == key:
                return float(d["tempo_bpm"])
        from mix_automation.v8.engine import estimate_tempo_bpm

        bpm, nb = estimate_tempo_bpm(inst, SR)
        p.write_text(json.dumps({"key": key, "tempo_bpm": bpm, "n_beats": nb}), encoding="utf-8")
        return bpm

    def _front(self) -> tuple[np.ndarray, np.ndarray]:
        from mix_automation.v8 import chain

        parts: list = [[asdict(r) for r in self.layout.regions], float(np.sum(self.lead_cut ** 2))]
        if self.layout.tails:  # R4b: the cut length changes with the tail windows, the energy may not
            parts += [list(self.layout.tails), self.layout.n]
        key = _hash(*parts)
        p = self.dir / f"front_{key}.npz"
        if p.exists():
            z = np.load(p)
            return z["x"], z["mask"]
        x, mask = chain.lead_front(self.lead_cut, SR, chain.StageLog())
        np.savez(p, x=x, mask=mask)
        return x, mask

    def _tail_cover_starts(self, pair: str, n_cover: int) -> tuple[list[float], list[bool]]:
        """Cover-time start of each tail cut and whether the cut lies inside that cover's audio."""
        curve = load_drift_curve(pair)
        starts, valid = [], []
        for t in self.layout.tails:
            c = float(orig_to_cover(t * SR, curve))
            ok = c - TAIL_PRE_S * SR >= 0 and c + TAIL_POST_S * SR <= n_cover
            starts.append(c - TAIL_PRE_S * SR if ok else 0.0)
            valid.append(bool(ok))
        return starts, valid

    def _tail_spot_info(self) -> list[dict]:
        out = []
        for d in self.tail_spots:
            row = dict(d)
            row["cover_mark_s"] = {cover_name(p): round(float(orig_to_cover(d["mark"] * SR, load_drift_curve(p))) / SR, 3)
                                   for p in self.refs}
            out.append(row)
        return out

    def _pro_features(self, pair: str) -> dict:
        curve = load_drift_curve(pair)
        starts = [float(orig_to_cover(r.pad0 * SR, curve)) for r in self.layout.regions]
        files = [_load_st(TARGETS / pair / f) for f in ("target_vocal.wav", "inst_pro.wav", "inst_aligned.wav")]
        valid = None
        if self.layout.tails:
            n_cover = min(a.shape[-1] for a in files)
            t_starts, valid = self._tail_cover_starts(pair, n_cover)
            starts += t_starts
        sig = [self.layout.cut(a, starts) for a in files]
        return features(sig[0], sig[1], sig[2], self.layout, afx=self.r4b, tail_valid=valid)

    # ------------------------------------------------------------ evaluation
    def tone(self, knobs: V8Knobs, log=None) -> np.ndarray:
        from mix_automation.v8 import chain

        return chain.lead_tone(self.front_x, self.front_mask, SR, knobs, log or chain.StageLog())

    def render(self, knobs: V8Knobs, tone: np.ndarray | None = None,
               log=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(voice, inst_after_duck, tone) on the excerpt layout; the master bus is not applied."""
        from mix_automation.v8 import chain

        log = log or chain.StageLog()
        if tone is None:
            tone = self.tone(knobs, log)
        lead_dry, _ = chain.lead_ride(tone, self.inst_work, SR, knobs, log)
        sends = chain.build_sends(lead_dry, SR, knobs, self.tempo_bpm, log)
        inst_d, _ = chain.duck_inst(self.inst_work, lead_dry, SR, knobs, log)
        n = self.layout.n
        voice = np.stack([lead_dry, lead_dry]).astype(np.float64)
        for w in sends.values():
            voice += chain.fit_len(w, n)
        return voice, inst_d, tone

    def evaluate(self, knobs: V8Knobs, tone: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
        t0 = time.time()
        from mix_automation.v8 import chain

        log = chain.StageLog()
        voice, inst_d, tone_out = self.render(knobs, tone, log)
        f = features(voice, inst_d, self.inst_work, self.layout, afx=self.r4b)
        per, terms = {}, {}
        for cov, pf in self.pro.items():
            per[cov], terms[cov] = distance(f, pf, self.scales, self.weights, self.terms)
        res = {"dist": aggregate(per), "per_cover": per, "terms": terms,
               "v8": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in f.items() if k != "afx"},
               "sec": round(time.time() - t0, 2)}
        if self.r4b:
            fr = r4_view(f)
            per_r4 = {cov: distance(fr, r4_view(pf), self.scales_r4, DEFAULT_WEIGHTS, TERMS_R4)[0]
                      for cov, pf in self.pro.items()}
            res |= {"dist_r4": aggregate(per_r4), "per_cover_r4": per_r4,
                    # calibration renders per stage (R4b warm starts should lower these)
                    "calib_evals": {st["stage"]: st["calibration"]["iterations"] for st in log.stages
                                    if "calibration" in st}}
        return res, tone_out


def write_knobs_json(knobs: V8Knobs, out_dir: Path, name: str) -> Path:
    """New file only: `<name>.json`, or `<name>_YYYYMMDD-HHMMSS.json` when that exists."""
    load_knobs(None, knobs.to_dict())  # validate
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    if p.exists():
        p = out_dir / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    d = knobs.to_dict()
    d["delay_type"] = int(d["delay_type"])
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


__all__ = [
    "SONG_REFS",
    "TERMS",
    "TERMS_R4",
    "WEIGHTS_R4B",
    "Layout",
    "Region",
    "SongMatch",
    "aggregate",
    "best_record",
    "candidate_regions",
    "distance",
    "features",
    "find_tail_spots",
    "from_unit",
    "loo_select",
    "orig_to_cover",
    "pro_scales",
    "r4_view",
    "select_regions",
    "tail_spot_terms",
    "tail_windows_db",
    "term_raw",
    "to_unit",
    "write_knobs_json",
]
