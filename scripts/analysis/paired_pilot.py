"""Paired pilot: measure how a pro mixed a raw main vocal, using practice stems + the finished cover.

Stages (each caches into data/outputs_paired_pilot/cache/<song>/ and prints a short table):
  align   cover <-> inst global lag + local lag curve (drift / edits)
  sub     linear "master" correction from vocal-free frames, extracted vocal, residual dB
  take    same-take check: envelope corr, lag drift, pitch diff, tuning detection
  select  main-only regions
  fit     DiffVox chain fitted per song (raw main -> pro main, pre-master domain)
  stats   7-song parameter table + v7 comparison
  listen  30 s A/B/C files for 2 songs

Usage: python scripts/analysis/paired_pilot.py <stage> [--songs A,B] [--steps N]
Audio inputs are read-only; everything is written under data/outputs_paired_pilot.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "utattemita_practicesite_reference"
OUT = ROOT / "data" / "outputs_paired_pilot"
CACHE = OUT / "cache"
SR = 44100
HOP = 1024  # activity frame hop (~23 ms)
EXCLUDE = {"Enuko_AnataKranken"}


# ---------------------------------------------------------------- io helpers
def songs(arg: str | None) -> list[str]:
    all_ = sorted(d.name for d in REF.iterdir() if d.is_dir() and d.name not in EXCLUDE)
    return [s for s in all_ if not arg or s in arg.split(",")]


def cdir(song: str) -> Path:
    d = CACHE / song
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_wav(p: Path) -> np.ndarray:
    """-> (2, n) float32 at SR."""
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.T
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g, axis=1).astype(np.float32)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    return x[:2]


def decode_cover(song: str) -> np.ndarray:
    p = cdir(song) / "cover.wav"
    if not p.exists():
        src = next((REF / song).glob("cover_*.webm"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", str(SR),
                        "-c:a", "pcm_f32le", str(p)], check=True)
    return load_wav(p)


def classify(name: str) -> str:
    n = Path(name).stem.lower()
    if "inst" in n:
        return "inst"
    if "参考" in n:
        return "ref"
    if re.fullmatch(r"m\d*|main.*", n):
        return "main"
    return "other"


def stems(song: str) -> dict[str, np.ndarray]:
    p = cdir(song) / "stems.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    d = {f.stem: load_wav(f) for f in sorted((REF / song / "practice").glob("*.wav"))}
    n = max(v.shape[1] for v in d.values())
    d = {k: np.pad(v, ((0, 0), (0, n - v.shape[1]))) for k, v in d.items()}
    np.savez(p, **d)
    return d


def frame_db(x: np.ndarray) -> np.ndarray:
    m = x.mean(0) if x.ndim == 2 else x
    n = len(m) // HOP
    r = np.sqrt((m[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(r)


def active(x: np.ndarray, rel: float = 40.0, floor: float = -60.0) -> np.ndarray:
    db = frame_db(x)
    return db > max(floor, np.percentile(db, 99.5) - rel)


def dilate(a: np.ndarray, before: int, after: int) -> np.ndarray:
    out = a.copy()
    idx = np.flatnonzero(a)
    for s in range(-before, after + 1):
        j = np.clip(idx + s, 0, len(a) - 1)
        out[j] = True
    return out


def save_json(song: str, name: str, obj: dict) -> None:
    (cdir(song) / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def load_json(song: str, name: str) -> dict:
    return json.loads((cdir(song) / f"{name}.json").read_text(encoding="utf-8"))


def table(rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    w = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    print(" | ".join(k.ljust(w[k]) for k in keys))
    for r in rows:
        print(" | ".join(str(r.get(k, "")).ljust(w[k]) for k in keys))


# ---------------------------------------------------------------- groups
def groups(song: str, st: dict[str, np.ndarray]) -> dict:
    """lead main + split parts (main) vs doubles/alts (other)."""
    kinds = {k: classify(k) for k in st}
    mains = [k for k in st if kinds[k] == "main"]
    act = {k: active(st[k]) for k in st if kinds[k] in ("main", "other")}
    lead = max(mains, key=lambda k: act[k].sum())
    main, other, alt, rel = [lead], [k for k in st if kinds[k] == "other"], [], {}
    for k in mains:
        if k == lead:
            continue
        both = act[k] & act[lead]
        ov = both.sum() / max(1, min(act[k].sum(), act[lead].sum()))
        a, b = st[k].mean(0), st[lead].mean(0)
        mask = np.repeat(both, HOP)[: len(a)]
        wc = float(np.dot(a[: len(mask)][mask], b[: len(mask)][mask]) /
                   (np.linalg.norm(a[: len(mask)][mask]) * np.linalg.norm(b[: len(mask)][mask]) + 1e-9))
        kind = "split" if ov < 0.2 else ("alt" if wc > 0.8 else "double")
        rel[k] = {"overlap": round(float(ov), 2), "wave_corr": round(wc, 2), "kind": kind}
        (main if kind == "split" else alt if kind == "alt" else other).append(k)
    return {"lead": lead, "main": main, "other": other, "alt": alt, "rel": rel,
            "inst": [k for k in st if kinds[k] == "inst"], "ref": [k for k in st if kinds[k] == "ref"]}


# ---------------------------------------------------------------- align
def xcorr_lag(a: np.ndarray, b: np.ndarray, maxlag: int) -> tuple[float, float]:
    """lag (samples, fractional) such that a[t] ~ b[t - lag]; plus normalized peak."""
    n = len(a) + len(b)
    nf = 1 << int(np.ceil(np.log2(n)))
    c = np.fft.irfft(np.fft.rfft(a, nf) * np.conj(np.fft.rfft(b, nf)), nf)
    c = np.concatenate([c[-maxlag:], c[: maxlag + 1]])
    i = int(np.argmax(c))
    y0, y1, y2 = c[max(i - 1, 0)], c[i], c[min(i + 1, len(c) - 1)]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return i - maxlag + float(np.clip(d, -1, 1)), float(y1 / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def stage_align(song: str) -> dict:
    cov = decode_cover(song)
    st = stems(song)
    g = groups(song, st)
    inst = sum(st[k] for k in g["inst"])
    # global lag at 4 kHz (inst may be keyed / shorter)
    ds = 11
    c4 = signal.decimate(cov.mean(0), ds)
    i4 = signal.decimate(inst.mean(0), ds)
    lag4, pk = xcorr_lag(c4, i4, len(c4) - 1)
    lag0 = int(round(lag4 * ds))
    # local lag curve, 4 s windows every 10 s, +-30 ms, full rate
    cm, im = cov.mean(0), inst.mean(0)
    pts = []
    for t0 in range(0, cov.shape[1] - 4 * SR, 10 * SR):
        s_i = t0 - lag0
        if s_i < 0 or s_i + 4 * SR > len(im):
            continue
        a, b = cm[t0: t0 + 4 * SR], im[s_i: s_i + 4 * SR]
        if np.std(b) < 1e-4:
            continue
        lg, p = xcorr_lag(a, b, 1300)
        if p > 0.3:
            pts.append((t0 / SR, lag0 + lg, p))
    pts = np.array(pts)
    # robust: keep windows within 2 ms of the median lag (outliers = vocal-dominated windows)
    keep = np.abs(pts[:, 1] - np.median(pts[:, 1])) < 0.002 * SR
    n_out = int((~keep).sum())
    slope, icpt = np.polyfit(pts[keep, 0], pts[keep, 1], 1) if keep.sum() >= 3 else (0.0, float(np.median(pts[:, 1])))
    resid = pts[keep, 1] - (slope * pts[keep, 0] + icpt)
    if abs(slope / SR) * cov.shape[1] < 0.5:  # negligible drift -> constant fractional lag
        icpt, slope = float(np.median(pts[keep, 1])), 0.0
    # dense lag track (sections of the cover are shifted by tens of ms: video/edit cuts)
    win, hop = int(1.5 * SR), SR // 4
    tr = []
    for t0 in range(0, cov.shape[1] - win, hop):
        s_i = t0 - lag0
        if s_i < 0 or s_i + win > len(im) or np.std(im[s_i: s_i + win]) < 1e-4:
            tr.append((t0 + win / 2, np.nan, 0.0))
            continue
        lg, p = xcorr_lag(cm[t0: t0 + win], im[s_i: s_i + win], 2000)
        tr.append((t0 + win / 2, lag0 + lg if p > 0.2 else np.nan, p))
    tr = np.array(tr)
    lag = tr[:, 1].copy()
    ok = ~np.isnan(lag)
    lag = np.interp(np.arange(len(lag)), np.flatnonzero(ok), lag[ok])
    lag = signal.medfilt(lag, 5)
    np.save(cdir(song) / "lag_track.npy", np.stack([tr[:, 0], lag, tr[:, 2]]))
    steps_ = np.abs(np.diff(lag)) > 3
    res_track = {"track_peak_med": round(float(np.median(tr[ok, 2])), 3),
                 "n_lag_steps": int((np.diff(np.r_[0, steps_.astype(int)]) == 1).sum()),
                 "lag_range_ms": round(float((lag.max() - lag.min()) / SR * 1000), 1)}
    res = {"lag_samples": lag0, "global_peak": round(pk, 3), "n_win": len(pts), **res_track,
           "drift_ppm": round(slope / SR * 1e6, 2), "lag_fit_icpt": float(icpt), "lag_slope": float(slope),
           "local_lag_resid_max": round(float(np.abs(resid).max()), 2), "n_outlier": n_out,
           "pts": np.round(pts, 3).tolist(),
           "local_peak_med": round(float(np.median(pts[:, 2])), 3),
           "cover_s": round(cov.shape[1] / SR, 1), "inst_s": round(inst.shape[1] / SR, 1), "groups": g}
    save_json(song, "align", res)
    return res


# ---------------------------------------------------------------- sub
NFFT, SHOP = 4096, 1024


def shift(x: np.ndarray, d: float, n_out: int) -> np.ndarray:
    """y[t] = x[t - d] (fractional d), length n_out."""
    from scipy import fft as sfft
    di = int(np.floor(d))
    fr = d - di
    y = np.zeros((x.shape[0], n_out), np.float32)
    s0, d0 = max(0, -di), max(0, di)
    m = min(x.shape[1] - s0, n_out - d0)
    if m > 0:
        y[:, d0: d0 + m] = x[:, s0: s0 + m]
    if abs(fr) > 1e-4:
        N = sfft.next_fast_len(n_out + 64)
        Y = sfft.rfft(y, N, axis=1)
        Y *= np.exp(-2j * np.pi * np.arange(Y.shape[1]) * fr / N)
        y = sfft.irfft(Y, N, axis=1)[:, :n_out].astype(np.float32)
    return y


def to_cover(x: np.ndarray, song: str, n_out: int) -> np.ndarray:
    """constant (or linear-drift) fractional lag. Trial 2 (dense lag tracking, to_cover_tracked) nulled worse
    on every song, so the dense track is kept only as an edit diagnostic."""
    al = load_json(song, "align")
    s = al["lag_slope"] / SR
    if abs(s) * n_out > 0.3:
        x = signal.resample(x, int(round(x.shape[1] / (1 - s))), axis=1).astype(np.float32)
    return shift(x, al["lag_fit_icpt"], n_out)


def to_cover_tracked(x: np.ndarray, song: str, n_out: int) -> np.ndarray:
    """time-varying fractional shift y[t] = x[t - lag(t)] by windowed overlap-add (lag from lag_track)."""
    tc, lag, _ = np.load(cdir(song) / "lag_track.npy")
    n_fr = n_out // SHOP + 1
    cen = np.arange(n_fr) * SHOP
    d = np.interp(cen, tc, lag)
    di = np.floor(d).astype(np.int64)
    fr = d - di
    w = np.hanning(NFFT + 1)[:-1].astype(np.float32)
    k = np.arange(NFFT // 2 + 1)
    xp = np.pad(x, ((0, 0), (NFFT, NFFT)))
    y = np.zeros((x.shape[0], n_out + 2 * NFFT), np.float32)
    for b in range(0, n_fr, 512):
        ii = np.arange(b, min(n_fr, b + 512))
        st0 = cen[ii] - di[ii] - NFFT // 2 + NFFT  # index into xp
        st0 = np.clip(st0, 0, xp.shape[1] - NFFT)
        seg = xp[:, st0[:, None] + np.arange(NFFT)[None]] * w  # (ch, f, NFFT)
        S = np.fft.rfft(seg, axis=-1) * np.exp(-2j * np.pi * k[None, None] * fr[ii, None][None] / NFFT)
        out = np.fft.irfft(S, NFFT, axis=-1) * w
        for j, i in enumerate(ii):
            o = cen[i] - NFFT // 2 + NFFT
            y[:, o: o + NFFT] += out[:, j]
    return (y[:, NFFT: NFFT + n_out] / 1.5).astype(np.float32)


def stft(x: np.ndarray) -> np.ndarray:
    return signal.stft(x, nperseg=NFFT, noverlap=NFFT - SHOP, window="hann")[2].astype(np.complex64)


def istft(X: np.ndarray, n: int) -> np.ndarray:
    y = signal.istft(X, nperseg=NFFT, noverlap=NFFT - SHOP, window="hann")[1]
    return np.pad(y, ((0, 0), (0, max(0, n - y.shape[1]))))[:, :n].astype(np.float32)


def frames_from_hop(mask_hop: np.ndarray, n_frames: int) -> np.ndarray:
    """activity mask at HOP=1024 -> STFT frames (SHOP=1024, centered)."""
    m = np.zeros(n_frames, bool)
    k = min(n_frames, len(mask_hop))
    m[:k] = mask_hop[:k]
    return m


def aligned(song: str) -> dict[str, np.ndarray]:
    p = cdir(song) / "aligned.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    al = load_json(song, "align")
    g = al["groups"]
    cov = decode_cover(song)
    st = stems(song)
    n = cov.shape[1]
    d = {"cover": cov}
    for name, keys in (("main", g["main"]), ("other", g["other"]), ("inst", g["inst"])):
        x = sum(st[k] for k in keys) if keys else np.zeros_like(st[g["inst"][0]])
        d[name] = to_cover(x, song, n)
    for k in g["alt"] + g["ref"]:
        d["x_" + k] = to_cover(st[k], song, n)
    np.savez(p, **d)
    return d


def stage_sub(song: str) -> dict:
    a = aligned(song)
    C, I = stft(a["cover"]), stft(a["inst"])
    nf = C.shape[2]
    voc = active(a["main"], rel=50, floor=-50) | active(a["other"], rel=50, floor=-50)
    free = ~dilate(voc, 4, 20)
    free = frames_from_hop(free, nf) & frames_from_hop(active(a["inst"], rel=40), nf)
    blk = (np.arange(nf) // int(2 * SR / SHOP)) % 2 == 0
    fit, ev = free & blk, free & ~blk
    num = (C[:, :, fit] * np.conj(I[:, :, fit])).sum(2)
    den = (np.abs(I[:, :, fit]) ** 2).sum(2) + 1e-9
    H = num / den  # (2, bins)
    g0 = np.sqrt((np.abs(C[:, :, fit]) ** 2).sum((1, 2)) / (np.abs(I[:, :, fit]) ** 2).sum((1, 2)))

    def rdb(P, m):
        return 10 * np.log10((np.abs(C[:, :, m] - P[:, :, m]) ** 2).sum() / (np.abs(C[:, :, m]) ** 2).sum())

    HI = H[:, :, None] * I
    # slow gain tracking from the low band (40-150 Hz: bass/kick, little vocal)
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    lb = (fb > 40) & (fb < 150)
    gt = np.real((C[:, lb] * np.conj(HI[:, lb])).sum((0, 1))) / ((np.abs(HI[:, lb]) ** 2).sum((0, 1)) + 1e-9)
    gt = np.clip(signal.medfilt(gt, 21), 0.5, 1.5)
    gt = np.convolve(gt, np.ones(21) / 21, mode="same")
    r_gain = rdb(g0[:, None, None] * I, ev)
    r_lin = rdb(HI, ev)
    r_trk = rdb(gt[None, None, :] * HI, ev)
    use_trk = r_trk < r_lin - 0.5
    G = gt[None, None, :] if use_trk else 1.0
    Hs = G * H[:, :, None]
    ok = np.abs(H) > 0.05 * np.abs(H).max()
    Vpost = C - Hs * I
    Vpre = np.where(ok[:, :, None], C / np.where(ok[:, :, None], Hs, 1) - I, 0)
    n = a["cover"].shape[1]
    np.save(cdir(song) / "vpre.npy", istft(Vpre, n))
    np.save(cdir(song) / "vpost.npy", istft(Vpost, n))
    np.savez(cdir(song) / "H.npz", H=H, gt=gt, use_trk=use_trk)
    # residual per band (eval frames, linear H)
    bands = {}
    for lo, hi in ((0, 250), (250, 2000), (2000, 8000), (8000, 22050)):
        b = (fb >= lo) & (fb < hi)
        Ce, Pe = C[:, b][:, :, ev], (G * HI)[:, b][:, :, ev] if use_trk else HI[:, b][:, :, ev]
        bands[f"{lo}-{hi}"] = round(float(10 * np.log10((np.abs(Ce - Pe) ** 2).sum() / (np.abs(Ce) ** 2).sum())), 1)
        Ib = I[:, b][:, :, free]
        Cf = C[:, b][:, :, free]
        coh = np.abs((Cf * np.conj(Ib)).sum(2)) ** 2 / ((np.abs(Cf) ** 2).sum(2) * (np.abs(Ib) ** 2).sum(2) + 1e-12)
        bands[f"coh{lo}-{hi}"] = round(float(np.median(coh)), 2)
    res = {"free_s": round(free.sum() * SHOP / SR, 1), "res_gain_db": round(float(r_gain), 1),
           "res_lin_db": round(float(r_lin), 1), "res_track_db": round(float(r_trk), 1),
           "use_track": bool(use_trk), "master_gain_db": round(float(20 * np.log10(g0.mean())), 1),
           "bands": bands}
    save_json(song, "sub", res)
    return res


# ---------------------------------------------------------------- take + select
def main_only(a: dict[str, np.ndarray]) -> np.ndarray:
    """hop-frame mask: main active, no other vocal within -0.1 s / +1.0 s, runs >= 1 s, inside inst span."""
    m = active(a["main"], rel=35)
    o = dilate(active(a["other"], rel=50, floor=-50), 3, 15) if np.abs(a["other"]).max() > 0 else np.zeros_like(m)
    inside = dilate(active(a["inst"], rel=60), 0, 0)
    mo = m & ~o[: len(m)] & inside[: len(m)]
    out = np.zeros_like(mo)
    edges = np.flatnonzero(np.diff(np.r_[0, mo.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= int(1.0 * SR / HOP):
            out[s:e] = True
    return out


def pitch_track(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    import librosa
    f0, vo, _ = librosa.pyin(x, fmin=80, fmax=1000, sr=sr, frame_length=1024, hop_length=160)
    f0[~vo] = np.nan
    return f0


def stage_take(song: str) -> dict:
    a = aligned(song)
    v = np.load(cdir(song) / "vpre.npy")
    mo = main_only(a)
    np.save(cdir(song) / "main_only.npy", mo)
    mact = active(a["main"], rel=35)
    dv, dm = frame_db(v), frame_db(a["main"])
    k = min(len(dv), len(mo))
    env_mo = float(np.corrcoef(dv[:k][mo[:k]], dm[:k][mo[:k]])[0, 1])
    env_all = float(np.corrcoef(dv[:k][mact[:k]], dm[:k][mact[:k]])[0, 1])
    # local lag (raw main -> extracted vocal), 3 s windows where main is active
    b, aa = signal.butter(4, 150, "hp", fs=SR)
    vm, mm = signal.filtfilt(b, aa, v.mean(0)), signal.filtfilt(b, aa, a["main"].mean(0))
    lags = []
    for t0 in range(0, len(mm) - 3 * SR, int(1.5 * SR)):
        f0_, f1_ = t0 // HOP, (t0 + 3 * SR) // HOP
        if mact[f0_:f1_].mean() < 0.6:
            continue
        lg, p = xcorr_lag(vm[t0: t0 + 3 * SR], mm[t0: t0 + 3 * SR], int(0.05 * SR))
        if p > 0.2:
            lags.append((t0 / SR, lg / SR * 1000, p))
    lags = np.array(lags) if lags else np.zeros((0, 3))
    med = float(np.median(lags[:, 1])) if len(lags) else float("nan")
    dev = np.abs(lags[:, 1] - med) if len(lags) else np.zeros(0)
    # pitch on main-only audio (<= 60 s)
    import librosa
    idx = np.flatnonzero(np.repeat(mo, HOP))[: 60 * SR]
    if len(idx) < 5 * SR:  # too little main-only audio: fall back to main-active frames
        idx = np.flatnonzero(np.repeat(mact, HOP))[: 60 * SR]
    vv = librosa.resample(v.mean(0)[idx], orig_sr=SR, target_sr=16000)
    mm2 = librosa.resample(a["main"].mean(0)[idx], orig_sr=SR, target_sr=16000)
    pv, pm = pitch_track(vv), pitch_track(mm2)
    both = ~np.isnan(pv) & ~np.isnan(pm)
    cents = 1200 * np.log2(pv[both] / pm[both])
    off = float(np.median(cents)) if both.any() else float("nan")
    d = cents - off
    d = d[np.abs(d) < 300]  # drop octave errors
    if len(d) == 0:
        d = np.array([np.nan])

    def grid_dev(f):
        c = 1200 * np.log2(f[~np.isnan(f)] / 440.0)
        t = np.median(((c + 50) % 100) - 50)
        return float(np.median(np.abs(((c - t + 50) % 100) - 50)))

    gv, gm = grid_dev(pv), grid_dev(pm)
    res = {"main_only_s": round(mo.sum() * HOP / SR, 1), "main_act_s": round(mact.sum() * HOP / SR, 1),
           "n_seg": int((np.diff(np.r_[0, mo.astype(int)]) == 1).sum()),
           "env_corr_mo": round(env_mo, 3), "env_corr_all": round(env_all, 3),
           "lag_med_ms": round(med, 2), "lag_dev_max_ms": round(float(dev.max()) if len(dev) else 0, 2),
           "n_lag_jump2ms": int((dev > 2).sum()), "n_lag_win": int(len(lags)),
           "pitch_off_cents": round(off, 1), "pitch_diff_med_c": round(float(np.median(np.abs(d))), 1),
           "pitch_diff_p90_c": round(float(np.percentile(np.abs(d), 90)), 1),
           "grid_dev_raw_c": round(gm, 1), "grid_dev_pro_c": round(gv, 1), "voiced_both": int(both.sum()),
           "lags": np.round(lags, 2).tolist()}
    save_json(song, "take", res)
    return res


# ---------------------------------------------------------------- sep (fallback target)
def stage_sep(song: str) -> dict:
    """BS-RoFormer vocals of the cover (subtraction failed: cover inst != practice inst above ~250 Hz).
    Target = separated vocal / master gain (|H| median 60-250 Hz, where cover and inst still cohere)."""
    p = cdir(song) / "sepvoc.npy"
    if not p.exists():
        sys.path.insert(0, str(ROOT / "src"))
        from mix_automation.analysis.reference_extractor import BSRoFormerSeparator
        sep = BSRoFormerSeparator(cache_dir=CACHE / "_sep", model_dir=ROOT / "data/processed/roformer_models")
        v, _, sr, _ = sep.separate(cdir(song) / "cover.wav", slug=song, target_sr=SR)
        v = np.asarray(v, np.float32)
        v = v.T if v.shape[0] != 2 else v
        np.save(p, v)
    v = np.load(p)
    a = aligned(song)
    n = min(v.shape[1], a["cover"].shape[1])
    v = np.pad(v[:, :n], ((0, 0), (0, a["cover"].shape[1] - n)))
    H = np.load(cdir(song) / "H.npz")["H"]
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    gm = float(np.median(np.abs(H[:, (fb > 60) & (fb < 250)])))
    np.save(cdir(song) / "vsep.npy", (v / gm).astype(np.float32))
    voc = active(a["main"], rel=50, floor=-50) | active(a["other"], rel=50, floor=-50)
    free = ~dilate(voc, 4, 20) & active(a["inst"], rel=40)
    fs = np.flatnonzero(np.repeat(free, HOP)[: a["cover"].shape[1]])
    leak = 10 * np.log10((v[:, fs] ** 2).sum() / ((a["cover"][:, fs] ** 2).sum() + 1e-12)) if len(fs) else float("nan")
    res = {"sep_leak_db": round(float(leak), 1), "master_gain_lowband_db": round(20 * np.log10(gm), 1)}
    save_json(song, "sep", res)
    return res


# ---------------------------------------------------------------- fit (DiffVox)
DIFFVOX_CFG = Path(r"C:\MIX_automation_ext\diffvox\presets\fx_config.yaml")
IN_RMS_DB = -20.0  # raw main is scaled so its active RMS is -20 dBFS before the chain
TARGET = "sep"  # "sep" (BS-RoFormer vocals of the cover) or "sub" (cover - corrected inst)


    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, r"C:\MIX_automation_ext\pilot")
    import yaml
    from _diffvox_env import inst  # DiffVox path setup + CPU shims for numba kernels
    return inst(yaml.safe_load(DIFFVOX_CFG.read_text())["model"])


def mrstft(y, t, m, sizes=(256, 1024, 4096)):
    """masked multi-resolution STFT loss on L/R and M/S. y,t: (B,2,N); m: (B,N) float."""
    import torch
    loss = 0.0
    for sig_y, sig_t, w in ((y, t, 1.0), (_ms(y), _ms(t), 0.5)):
        for n in sizes:
            h = n // 4
            win = torch.hann_window(n, device=y.device)
            Y = torch.stft(sig_y.reshape(-1, sig_y.shape[-1]), n, h, window=win, return_complex=True).abs()
            T = torch.stft(sig_t.reshape(-1, sig_t.shape[-1]), n, h, window=win, return_complex=True).abs()
            fm = m[:, ::h][:, : Y.shape[-1]].repeat_interleave(2, 0)[:, None, :]
            fm = torch.nn.functional.pad(fm, (0, Y.shape[-1] - fm.shape[-1]))
            lg = (torch.log(Y + 1e-5) - torch.log(T + 1e-5)).abs()
            loss = loss + w * ((lg * fm).sum() / (fm.sum() * Y.shape[1] + 1e-9))
            sc = torch.sqrt((((T - Y) * fm) ** 2).sum()) / torch.sqrt(((T * fm) ** 2).sum() + 1e-9)
            loss = loss + w * sc
    return loss


def _ms(x):
    return __import__("torch").stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], 1) * 0.5


def fit_data(song: str):
    a = aligned(song)
    v = np.load(cdir(song) / ("vsep.npy" if TARGET == "sep" else "vpre.npy"))
    mo = main_only(a)  # recomputed (guard after other vocals: 0.35 s)
    np.save(cdir(song) / "main_only.npy", mo)
    x = a["main"].mean(0)
    act = active(a["main"], rel=35)
    rms = np.sqrt(np.mean(np.repeat(act, HOP)[: len(x)] * x[: len(act) * HOP] ** 2) /
                  max(1e-9, np.repeat(act, HOP).mean()))
    k = 10 ** (IN_RMS_DB / 20) / rms
    ms = np.zeros(len(x), np.float32)
    rep = np.repeat(mo, HOP)
    ms[: len(rep)] = rep[: len(x)]
    return a, (x * k).astype(np.float32), v, ms, float(k), mo


def stage_fit(song: str, steps: int) -> dict:
    import torch
    torch.manual_seed(0)
    a, x, v, ms, k, mo = fit_data(song)
    if ms.sum() < 8 * SR:
        return {"target": TARGET, "skipped": f"main-only {ms.sum() / SR:.1f} s < 8 s"}
    W, Hp, pre = 8 * SR, 2 * SR, int(1.5 * SR)
    st_ = [s for s in range(0, len(x) - W, Hp) if ms[s + pre: s + W].mean() > 0.15 and s >= 0.06 * SR]
    # per-chunk timing: the pro moved/edited the vocal relative to the inst (take stage), so align raw main
    # to the target chunk by chunk (integer lag, +-50 ms)
    b, aa = signal.butter(4, 150, "hp", fs=SR)
    lags = []
    for s in st_:
        lg, _ = xcorr_lag(signal.lfilter(b, aa, v[:, s: s + W].mean(0)), signal.lfilter(b, aa, x[s: s + W]),
                          int(0.05 * SR))
        lags.append(int(round(lg)))
    dev = "cuda"
    X = torch.tensor(np.stack([x[s - lg: s - lg + W] for s, lg in zip(st_, lags)]))[:, None].to(dev)
    T = torch.tensor(np.stack([v[:, s: s + W] for s in st_])).to(dev)
    M = torch.tensor(np.stack([ms[s: s + W] for s in st_])).to(dev)
    M[:, :pre] = 0
    model = build_model().to(dev)
    g0 = 20 * np.log10(float(T[M[:, None, :].expand_as(T) > 0].std()) / float(X[M[:, None, :] > 0].std()))
    gain = torch.nn.Parameter(torch.tensor(g0, device=dev))
    opt = torch.optim.Adam(list(model.parameters()) + [gain], lr=0.01)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(steps * 0.8)], 0.3)
    bs = min(8, len(st_))
    hist = []
    for i in range(steps):
        idx = torch.randperm(len(st_))[:bs]
        y = model(X[idx]) * 10 ** (gain / 20)
        loss = mrstft(y, T[idx], M[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss))
    model.eval()
    with torch.no_grad():
        Ld, Lf = [], []
        for j in range(0, len(st_), bs):
            sl = slice(j, j + bs)
            y = model(X[sl]) * 10 ** (gain / 20)
            Lf.append(float(mrstft(y, T[sl], M[sl])))
            d = X[sl].repeat(1, 2, 1)
            gd = (T[sl] * M[sl, None]).std() / ((d * M[sl, None]).std() + 1e-9)
            Ld.append(float(mrstft(d * gd, T[sl], M[sl])))
        yfull = (model(torch.tensor(x, device=dev)[None, None]) * 10 ** (gain / 20))[0].cpu().numpy()
    torch.save({"model": model.state_dict(), "gain_db": float(gain), "k": k}, cdir(song) / "fit.pt")
    np.save(cdir(song) / "fit_out.npy", yfull.astype(np.float32))
    pr = fx_params(model)
    res = {"target": TARGET, "n_chunks": len(st_), "steps": steps, "vlag_med_ms": round(float(np.median(lags)) / SR * 1000, 1),
           "vlag_range_ms": round((max(lags) - min(lags)) / SR * 1000, 1), "loss_fit": round(float(np.mean(Lf)), 3),
           "loss_dry": round(float(np.mean(Ld)), 3), "loss_first50": round(float(np.mean(hist[:50])), 3),
           "loss_last50": round(float(np.mean(hist[-50:])), 3), "out_gain_db": round(float(gain), 2),
           "in_scale_db": round(20 * np.log10(k), 2), **measure(song, a, v, yfull, mo), "params": pr}
    save_json(song, "fit", res)
    return res


def fx_params(model) -> dict:
    import torch
    out = {}
    for mn, mod in model.named_modules():
        p = getattr(mod, "params", None)
        if not isinstance(p, torch.nn.Module):
            continue
        names = {n.split(".")[1] if n.startswith("parametrizations.") else n for n, _ in p.named_parameters()}
        for n in sorted(names):
            val = getattr(p, n).detach().cpu().numpy()
            out[f"{mn}.{n}"] = np.round(val, 4).tolist() if val.size <= 8 else f"array{list(val.shape)}"
    dev = next(model.parameters()).device
    with torch.no_grad():
        imp = torch.zeros(1, 1, 4 * SR, device=dev)
        imp[..., 0] = 1
        eq = imp
        for m in list(model)[:6]:
            eq = m(eq)
        Hm = np.abs(np.fft.rfft(eq[0, 0].cpu().numpy()))
        f = np.fft.rfftfreq(4 * SR, 1 / SR)
        out["eq_db"] = {str(b): round(float(20 * np.log10(Hm[np.argmin(np.abs(f - b))] + 1e-9)), 1)
                        for b in (60, 120, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000)}
        send = list(model)[7]
        ir = send(imp)[0].cpu().numpy()  # (2, n) dry pan + sends
        e = (ir ** 2).sum(0)
        d_n = int(0.002 * SR)
        out["wet_to_dry_db"] = round(float(10 * np.log10(e[d_n:].sum() / (e[:d_n].sum() + 1e-12))), 1)
        fdn = send.effects[1](imp)[0].cpu().numpy()
        out["fdn_t60_s"] = t60((fdn ** 2).sum(0))
        dl = send.effects[0](imp)[0].cpu().numpy()
        out["delay_db"] = round(float(10 * np.log10((dl ** 2).sum() / (e[:d_n].sum() + 1e-12))), 1)
        out["fdn_db"] = round(float(10 * np.log10((fdn ** 2).sum() / (e[:d_n].sum() + 1e-12))), 1)
    return out


def t60(e: np.ndarray) -> float:
    sch = np.cumsum(e[::-1])[::-1]
    db = 10 * np.log10(sch / (sch[0] + 1e-20) + 1e-20)
    i5, i25 = np.argmax(db < -5), np.argmax(db < -25)
    return round(float(3 * (i25 - i5) / SR), 2) if i25 > i5 else float("nan")


def lufs(x: np.ndarray) -> float:
    import pyloudnorm as pyln
    if x.shape[-1] < SR:
        return float("nan")
    return float(pyln.Meter(SR).integrated_loudness(x.T.astype(np.float64)))


def measure(song: str, a: dict, v: np.ndarray, y: np.ndarray | None, mo: np.ndarray) -> dict:
    """loudness vs inst, Side/Mid, reverb tail after phrase ends (target vs fit)."""
    rep = np.repeat(mo, HOP)
    n = min(len(rep), v.shape[1])
    sel = np.flatnonzero(rep[:n])
    mact = active(a["main"], rel=35)
    oth = dilate(active(a["other"], rel=50, floor=-50), 5, 45) if np.abs(a["other"]).max() > 0 else np.zeros_like(mact)
    inside = active(a["inst"], rel=60)
    tail = dilate(mact, 0, int(1.2 * SR / HOP)) & ~dilate(mact, 2, 4) & ~oth & inside
    free = ~dilate(mact | oth, 10, 60) & inside
    tsel = np.flatnonzero(np.repeat(tail, HOP)[:n])
    fsel = np.flatnonzero(np.repeat(free, HOP)[:n])

    def sm(z):
        mm, ss = (z[0] + z[1]) / 2, (z[0] - z[1]) / 2
        return round(float(10 * np.log10((ss[sel] ** 2).sum() / ((mm[sel] ** 2).sum() + 1e-12))), 1)

    def rel(z, s2):
        return round(float(10 * np.log10((z[:, s2] ** 2).sum() / len(s2) / ((z[:, sel] ** 2).sum() / len(sel)))), 1)

    out = {"voc_minus_inst_lu": round(lufs(v[:, sel]) - lufs(a["inst"][:, sel]), 1),
           "sm_pro_db": sm(v), "tail_pro_db": rel(v, tsel), "floor_pro_db": rel(v, fsel) if len(fsel) else None}
    if y is not None:
        out.update({"lufs_fit_minus_pro": round(lufs(y[:, sel]) - lufs(v[:, sel]), 2), "sm_fit_db": sm(y),
                    "tail_fit_db": rel(y, tsel)})
    return out


# ---------------------------------------------------------------- v7 comparison
V7 = {"エバ": ROOT / "data/outputs/エバ_auto", "シックスフィートアンダー": ROOT / "data/outputs/シックスフィートアンダー_auto"}
V7_OTHER = re.compile(r"^(0[34]|0[89]|1\d)_")  # real harm/aux/fx stems; generated Nectar harm (06/07) excluded


def stage_v7(name: str) -> dict:
    d = V7[name]
    med = d / "Media"
    bus = load_wav(next(d.glob("*_VocalBus.wav")))
    main_ = load_wav(med / "02_mainvocal_prepared.wav")
    inst_ = load_wav(med / "01_offvocal_unmasked.wav")
    oth = [load_wav(p) for p in sorted(med.glob("*.wav")) if V7_OTHER.match(p.name)]
    n = min(bus.shape[1], main_.shape[1], inst_.shape[1])
    other = sum(o[:, :n] for o in oth) if oth else np.zeros((2, n), np.float32)
    a = {"main": main_[:, :n], "other": other, "inst": inst_[:, :n]}
    mo = main_only(a)
    r = measure(name, a, bus[:, :n], None, mo)
    return {"main_only_s": round(mo.sum() * HOP / SR, 1), **r}


# ---------------------------------------------------------------- listen
def apply_master(x: np.ndarray, song: str) -> np.ndarray:
    """scalar master gain only: the per-bin H is biased (low coherence above 250 Hz) and would dull the highs.
    Output is loudness-normalized afterwards, so this only documents the domain."""
    return x * 10 ** (load_json(song, "sep")["master_gain_lowband_db"] / 20)


def stage_listen(song: str) -> dict:
    import torch
    a = aligned(song)
    mo = np.load(cdir(song) / "main_only.npy")
    v = np.load(cdir(song) / ("vsep.npy" if TARGET == "sep" else "vpre.npy"))
    L = int(30 * SR / HOP)
    cs = np.convolve(mo.astype(float), np.ones(L), "valid")
    f0 = int(np.argmax(cs))
    s0, s1 = f0 * HOP, f0 * HOP + 30 * SR
    pre = 5 * SR
    ck = torch.load(cdir(song) / "fit.pt")
    model = build_model().cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    xin = (a["main"].mean(0)[max(0, s0 - pre): s1] * ck["k"]).astype(np.float32)
    with torch.no_grad():
        y = (model(torch.tensor(xin, device="cuda")[None, None]) * 10 ** (ck["gain_db"] / 20))[0].cpu().numpy()
    y = y[:, -(s1 - s0):]
    inst_, main_ = a["inst"][:, s0:s1], a["main"][:, s0:s1]
    sel = np.flatnonzero(np.repeat(mo, HOP)[s0:s1])
    gc = 10 ** ((lufs(v[:, s0:s1][:, sel]) - lufs(main_[:, sel])) / 20)
    A = a["cover"][:, s0:s1]
    B = apply_master(y + inst_, song)
    Cm = apply_master(gc * main_ + inst_, song)
    tgt = -16.0
    outs = {k: z * 10 ** ((tgt - lufs(z)) / 20) for k, z in (("a_pro", A), ("b_fitchain", B), ("c_gainonly", Cm))}
    pk = max(np.abs(z).max() for z in outs.values())
    att = min(1.0, 0.98 / pk)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, z in outs.items():
        p = OUT / f"{song}_{k}.wav"
        sf.write(str(p), (z * att).T, SR, subtype="PCM_24")
        paths.append(p.name)
    res = {"start_s": round(s0 / SR, 1), "main_only_frac": round(float(cs[f0] / L), 2),
           "lufs_out": round(tgt + 20 * np.log10(att), 1), "files": paths}
    save_json(song, "listen", res)
    return res


# ---------------------------------------------------------------- main
STAGES = {"align": stage_align, "sub": stage_sub, "take": stage_take, "sep": stage_sep, "fit": stage_fit,
          "listen": stage_listen}


def stage_stats(names: list[str]) -> None:
    rows, eqs, fx = [], [], []
    for s in names:
        p = cdir(s) / "fit.json"
        if not p.exists():
            continue
        f, sb = load_json(s, "fit"), load_json(s, "sub")
        if "skipped" in f:
            continue
        sp = load_json(s, "sep")
        pr = f["params"]
        rows.append({"song": s, "res_db": sb["res_lin_db"], "sep_leak": sp["sep_leak_db"], "n_chunk": f["n_chunks"], "loss_fit": f["loss_fit"],
                     "loss_dry": f["loss_dry"], "dLUFS": f["lufs_fit_minus_pro"], "voc-inst": f["voc_minus_inst_lu"],
                     "SM_pro": f["sm_pro_db"], "SM_fit": f["sm_fit_db"], "tail_pro": f["tail_pro_db"],
                     "tail_fit": f["tail_fit_db"], "floor_pro": f["floor_pro_db"]})
        e = pr["eq_db"]
        eqs.append({"song": s, **{k: round(v - e["1000"], 1) for k, v in e.items()}})
        fx.append({"song": s, "hpf": pr.get("5.freq"), "cmp_th": pr.get("6.cmp_th"), "ratio": pr.get("6.cmp_ratio"),
                   "at_rt": [pr.get("6.at"), pr.get("6.rt")], "delay_ms": pr.get("7.effects.0.delay"),
                   "delay_db": pr["delay_db"], "fdn_db": pr["fdn_db"], "t60": pr["fdn_t60_s"],
                   "wet_dry": pr["wet_to_dry_db"], "pan": pr.get("7.pan.pan")})
    v7 = [{"song": k, **stage_v7(k)} for k in V7]
    for title, t in (("fit / pro metrics", rows), ("EQ (dB rel 1 kHz)", eqs), ("dynamics / send FX", fx),
                     ("v7 (VocalBus vs prepared main)", v7)):
        print(f"== {title}")
        table(t)
    (OUT / "summary.json").write_text(json.dumps({"fit": rows, "eq": eqs, "fx": fx, "v7": v7}, indent=1,
                                                 ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["all", "stats"])
    ap.add_argument("--songs")
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()
    if a.stage == "stats":
        return stage_stats(songs(a.songs))
    for stage in (list(STAGES) if a.stage == "all" else [a.stage]):
        rows = []
        for s in songs(a.songs):
            r = STAGES[stage](s) if stage != "fit" else STAGES[stage](s, a.steps)
            rows.append({"song": s, **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}})
        print(f"== {stage}")
        table(rows)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
