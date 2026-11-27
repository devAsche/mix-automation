"""R4b: calibration warm start, early stop, tail spots, layout tail cuts, AFx term. No plugin, no GPU."""

from __future__ import annotations

import numpy as np
import pytest

from mix_automation.v8 import dsp
from mix_automation.v8.match import (
    FRAME_DT,
    TAIL_POST_S,
    TAIL_PRE_S,
    TERMS,
    WEIGHTS_R4B,
    Layout,
    Region,
    distance,
    fallback_scales,
    find_tail_spots,
    pro_scales,
    r4_view,
    tail_spot_terms,
    tail_windows_db,
    term_raw,
)
from mix_automation.v8.match_pool import should_stop

SR = 44100


# ------------------------------------------------------------------ calibrate


def _calibrate_r4(render, measure, target, lo, hi, iters=4, tol=0.25):
    """Copy of the R4 dsp.calibrate (before the warm start) for the no-x0 equivalence check."""
    hist, best = [], {}

    def ev(p):
        audio = render(p)
        m = float(measure(audio))
        hist.append((p, m))
        if not best or abs(m - target) < abs(best["m"] - target):
            best.update(p=p, m=m)
        return m - target

    a, b = lo, hi
    fa, fb = ev(a), ev(b)
    if fa * fb < 0:
        for _ in range(iters):
            if min(abs(fa), abs(fb)) <= tol or fb == fa:
                break
            c = b - fb * (b - a) / (fb - fa)
            w = abs(b - a)
            c = float(np.clip(c, min(a, b) + 0.1 * w, max(a, b) - 0.1 * w))
            fc = ev(c)
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
    return best["p"], hist


FUNCS = [
    (lambda p: 0.3 * (-8.0 - p), 3.8, -40.0, -8.0),          # VCA-like: GR grows as threshold drops
    (lambda p: 12.0 / (1.0 + np.exp(-(p - 50) / 12)), 4.6, 10.0, 90.0),  # LA-2A-like sigmoid
    (lambda p: np.tanh((-p - 2.0) / 20.0) * 9.0, 6.0, -50.0, -2.0),
    (lambda p: 0.3 * (-8.0 - p), 99.0, -40.0, -8.0),          # unreachable target
    (lambda p: 0.5 * (-p), 3.5, -2.0, -11.0),                # limiter-like, lo/hi passed reversed
]


@pytest.mark.parametrize(("f", "target", "lo", "hi"), FUNCS)
def test_calibrate_without_x0_unchanged(f, target, lo, hi):
    p_ref, h_ref = _calibrate_r4(lambda p: p, f, target, lo, hi)
    cal = dsp.calibrate(lambda p: p, f, target, lo, hi)
    assert cal.value == p_ref
    assert cal.history == h_ref


def test_calibrate_warm_start_fewer_evals():
    f = FUNCS[0][0]  # GR = 0.3 * (-8 - thr): decreasing in the parameter
    cold = dsp.calibrate(lambda p: p, f, 3.8, -40.0, -8.0)
    near = dsp.calibrate(lambda p: p, f, 3.8, -40.0, -8.0, x0=-20.6, step=3.0, increasing=False)
    assert len(near.history) == 1 and near.converged  # x0 already within tol
    off = dsp.calibrate(lambda p: p, f, 3.8, -40.0, -8.0, x0=-17.0, step=3.0, increasing=False)
    assert off.converged and abs(off.measured - 3.8) <= 0.25
    assert len(off.history) < len(cold.history)
    assert off.history[1][0] == pytest.approx(-20.0)  # stepped towards more GR (lower threshold)


def test_calibrate_warm_start_doubling_and_fallback():
    f = FUNCS[1][0]  # increasing sigmoid on [10, 90]
    far = dsp.calibrate(lambda p: p, f, 10.0, 10.0, 90.0, x0=20.0, step=10.0, increasing=True)
    steps = [p for p, _ in far.history[:4]]
    assert steps[:3] == [20.0, 30.0, 50.0]  # x0, +10, then doubled step +20
    assert far.converged
    # unreachable: falls back to the ends and returns the nearer one, never outside [lo, hi]
    un = dsp.calibrate(lambda p: p, FUNCS[3][0], 99.0, -40.0, -8.0, x0=-30.0, step=3.0, increasing=False)
    assert not un.converged and un.value == -40.0
    assert all(-40.0 <= p <= -8.0 for p, _ in un.history)
    assert len({p for p, _ in un.history}) == len(un.history)  # the end is not rendered twice


# ------------------------------------------------------------------ early stop


def test_should_stop():
    assert not should_stop([1.0, 0.9, 0.85])
    assert should_stop([1.0, 0.99, 0.985, 0.981])  # improved 0.019 over the last 3 generations
    assert not should_stop([1.0, 0.95, 0.9, 0.85])
    assert should_stop([1.0, 1.0, 1.0, 1.0, 1.0])


# ------------------------------------------------------------------ tail spots


def _mask(spans: list[tuple[float, float]], dur: float) -> np.ndarray:
    m = np.zeros(int(dur / FRAME_DT), bool)
    for a, b in spans:
        m[int(a / FRAME_DT):int(b / FRAME_DT)] = True
    return m


def test_find_tail_spots():
    dur = 40.0
    lead = _mask([(2.0, 4.0), (4.3, 6.0), (7.0, 7.8), (12.0, 12.2), (14.0, 16.0), (20.0, 22.0), (30.0, 31.0)], dur)
    non = _mask([(21.0, 25.0)], dur)
    spots = find_tail_spots(lead, non, dur)
    ends = [round(d["t_end"], 1) for d in spots]
    assert 4.0 not in ends        # pause 0.3 s
    assert 6.0 not in ends        # pause 1.0 s < 1.5 s (late window must fit)
    assert 12.2 not in ends       # phrase 0.2 s < 0.4 s
    assert 22.0 not in ends       # non-lead active during phrase/pause
    assert 7.8 in ends and 16.0 in ends and 31.0 in ends
    assert [d["mark"] for d in spots] == sorted(d["mark"] for d in spots)
    few = find_tail_spots(lead, non, dur, max_spots=2)
    assert {round(d["t_end"], 1) for d in few} == {7.8, 31.0}  # longest pauses: 4.2 s and 9.0 s (to the end)


def test_tail_windows_measure_fixed_windows():
    lay = Layout((Region(3.0, 6.0),), tails=(20.0, 30.0, 40.0))
    x = np.zeros((2, lay.n), np.float32)
    end = lay.tail_mark_offset() + round(0.2 * SR)
    for sl in lay.tail_slices():
        x[:, sl.start + end - round(0.4 * SR): sl.start + end] = 1.0          # phrase
        x[:, sl.start + end + round(0.1 * SR): sl.start + end + round(0.5 * SR)] = 0.1   # -20 dB tail
        x[:, sl.start + end + round(1.0 * SR): sl.start + end + round(1.5 * SR)] = 0.01  # -40 dB late
    tails, lates = tail_windows_db(x, lay, SR, valid=[True, True, False])
    assert tails[0] == pytest.approx(-20.0, abs=0.01) and lates[1] == pytest.approx(-40.0, abs=0.01)
    assert np.isnan(tails[2]) and np.isnan(lates[2])


# ------------------------------------------------------------------ layout


def test_layout_tail_cuts_do_not_move_regions():
    regs = (Region(3.0, 6.0), Region(10.0, 14.5))
    a = Layout(regs)
    b = Layout(regs, tails=(20.0, 25.0, 31.0))
    assert a.padded_slices() == b.padded_slices() and a.eval_slices() == b.eval_slices()
    assert b.n == a.n + 3 * (round(TAIL_PRE_S * SR) + round(TAIL_POST_S * SR))
    ts = b.tail_slices()
    assert ts[0].start == a.n and ts[-1].stop == b.n
    x = np.random.default_rng(1).standard_normal((2, 40 * SR)).astype(np.float32)
    ca, cb = a.cut(x, a.orig_starts()), b.cut(x, b.orig_starts())
    assert np.array_equal(ca, cb[:, : a.n])
    s = ts[1]
    mk = b.tail_mark_offset()
    # sample at the mark comes from 25.0 s of the source
    assert cb[0, s.start + mk] == x[0, round(25.0 * SR)]


# ------------------------------------------------------------------ AFx term / tail pairing


def _feat(afx=None, tail_spots=None, tail=-20.0):
    f = {"lu": np.array([1.0, 2.0]), "sm": -15.0, "tail": tail, "tail_r4": float("nan")}
    if afx is not None:
        f["afx"] = afx
    if tail_spots is not None:
        f["tail_by_spot"] = tail_spots
    return f


def test_term_raw_afx_and_tail_spots():
    e1 = np.zeros(8)
    e1[0] = 1.0
    e2 = np.zeros(8)
    e2[0], e2[1] = 1.0, 1.0
    ta = term_raw(_feat(e1), _feat(e2))
    assert ta["afx"] == pytest.approx(1.0 - 1.0 / np.sqrt(2.0))  # cosine distance, not squared
    assert np.isnan(term_raw(_feat(e1), _feat())["afx"])
    # tail: spot-wise against the reference (pro = second argument)
    nan = float("nan")
    v8 = _feat(tail_spots=[-30.0, -18.0, -25.0, -20.0])
    v8["late_by_spot"] = [nan] * 4
    pro = _feat(tail_spots=[-20.0, -21.0, -21.0, nan])
    pro["late_by_spot"] = [-30.0, -22.0, -24.0, nan]  # decay 10 (two-sided), 1 and 3 (one-sided)
    rows = tail_spot_terms(v8, pro)
    assert [k for *_, k in rows] == ["two", "one", "one"]
    # (-30+20)^2 = 100 two-sided; -18 > -21 -> 9; -25 < -21 -> 0 (below the upper bound is fine)
    assert term_raw(v8, pro)["tail"] == pytest.approx((100.0 + 9.0 + 0.0) / 3)
    pro2 = dict(pro, tail_by_spot=[-20.0, nan, -21.0, nan])
    assert np.isnan(term_raw(v8, pro2)["tail"])  # fewer than 3 usable spots
    a = v8
    sc = fallback_scales()
    sc["afx"] = 0.1
    d, norm = distance(_feat(e1), _feat(e2), sc, WEIGHTS_R4B)
    assert set(norm) == set(TERMS) and np.isfinite(d)
    assert norm["afx"] == pytest.approx(ta["afx"] / 0.1)
    # R4 view drops the R4b additions
    v = r4_view(a)
    assert "tail_by_spot" not in v and "afx" not in v and np.isnan(v["tail"])


def test_pro_scales_afx_floor():
    e = np.zeros(8)
    e[0] = 1.0
    sc, src = pro_scales({"a": _feat(e), "b": _feat(e.copy())})
    assert src == "pro_pairs" and sc["afx"] == pytest.approx(1e-4)


def test_pro_scales_fixed_tail():
    a, b = _feat(tail=-20.0), _feat(tail=-40.0)
    sc, _ = pro_scales({"a": a, "b": b}, fixed=("tail",))
    assert sc["tail"] == pytest.approx(fallback_scales()["tail"])
    assert pro_scales({"a": a, "b": b})[0]["tail"] == pytest.approx(400.0)
