"""R2: pro vocal targets by subtraction (cover - T(inst)), with a BS-RoFormer fallback.

Pairs (cover, inst, separated vocal of the cover):
    Amo_<song>     utattemita_practicesite_reference/Amo_<song>/{cover_*.webm, inst_*.wav}
    Ryo_<cover>    youtube_references/歌った曲references (covers matched at the original key in
                   docs/evaluations/cover_stats.json) vs the v7 input off-vocal
    v8_<song>      the R3 v8 render (Master vs raw off-vocal; VocalBus = known vocal) -> method check

Stages (cache: data/processed/v8r_targets/<pair>/, never committed):
    inv    lengths, key of inst vs cover, same take (Amo: practice main vs separated cover vocal)
    sep    BS-RoFormer vocal of the cover (GPU; Ryo/v8 reuse existing stems)
    drift  band-limited cross-correlation cover vs inst every 5 s -> lag curve, ppm, steps
    sub    time-varying 2x2 (M/S) per-bin least squares inst -> cover per 3 s block, held-out residual
    meas   pro processing of the inst (ducking per band, EQ trend, S/M) and target vocal metrics
    report -> docs/v8r_targets/summary.json + short tables

    .venv\\Scripts\\python.exe scripts\\targets\\extract_targets.py <stage|all> [pair,pair...|amo|ryo|v8]
"""

from __future__ import annotations

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
CACHE = ROOT / "data" / "processed" / "v8r_targets"
DOCS = ROOT / "docs" / "v8r_targets"
SR = 44100
NFFT, HOP = 4096, 1024
BLOCK_S = 3.0
RES_OK_DB = -15.0
AMO = ["ApocalypseTokyo", "Bibibibi", "CatLoving", "DayByDays", "Disbarhythm", "Kyuukurarin", "QUEEN",
       "StardustMedley"]
RYO_SONG = {"Eva": "エバ", "FallInLove": "Fall in Love", "LoveAndDestroy": "ラブ&デストロイ",
            "SixFeetUnder": "シックスフィートアンダー"}
V7_DIRS = {"エバ": "エバ_auto", "Fall in Love": "Fall_in_Love_auto", "シックスフィートアンダー": "シックスフィートアンダー_auto",
           "ラブ&デストロイ": "ラブandデストロイ_auto_20261126-135529"}
V8_DIRS = {"Eva": "エバ_v8", "SixFeetUnder": "シックスフィートアンダー_v8"}
POST = re.compile(r"補正(後|語|有|あり|済)|修正(後|有|あり)")


# ---------------------------------------------------------------- io
def cdir(pair: str) -> Path:
    d = CACHE / pair
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_wav(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.T
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g, axis=1).astype(np.float32)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    return x[:2]


def save_wav(p: Path, x: np.ndarray) -> None:
    sf.write(str(p), x.T.astype(np.float32), SR, subtype="FLOAT")


def jsave(pair: str, name: str, obj: dict) -> None:
    (cdir(pair) / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float),
                                             encoding="utf-8")


def jload(pair: str, name: str) -> dict:
    return json.loads((cdir(pair) / f"{name}.json").read_text(encoding="utf-8"))


def table(rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    w = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    print(" | ".join(k.ljust(w[k]) for k in keys))
    for r in rows:
        print(" | ".join(str(r.get(k, "")).ljust(w[k]) for k in keys))


def ryo_pairs() -> dict[str, dict]:
    st = json.loads((ROOT / "docs/evaluations/cover_stats.json").read_text(encoding="utf-8"))["identity"]
    out = {}
    for k, v in st.items():
        if v["kind"] == "cover" and v["status"] == "use" and v["key_shift"] == 0:
            out[f"Ryo_{k}"] = {"key": k, "song": v["song"]}
    return out


def pairs(arg: str | None) -> list[str]:
    allp = [f"Amo_{s}" for s in AMO] + list(ryo_pairs()) + [f"v8_{s}" for s in V8_DIRS]
    if not arg:
        return allp
    sel = []
    for a in arg.split(","):
        sel += [p for p in allp if p.lower().startswith(a.lower() + "_")] if a.lower() in ("amo", "ryo", "v8") \
            else [p for p in allp if p == a]
    return sel


def sources(pair: str) -> dict[str, Path]:
    """paths of cover, inst and (if already available) the separated cover vocal."""
    if pair.startswith("Amo_"):
        d = REF / pair
        cov = cdir(pair) / "cover.wav"
        if not cov.exists():
            src = next(d.glob("cover_*.webm"))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", str(SR),
                            "-c:a", "pcm_f32le", str(cov)], check=True)
        return {"cover": cov, "inst": next(d.glob("inst_*.wav")), "sep": CACHE / "_sep" / f"{pair}_vocals.wav"}
    if pair.startswith("Ryo_"):
        info = ryo_pairs()[pair]
        k = info["key"]
        cr = ROOT / "data/processed/v8_cover_refs"
        return {"cover": cr / "wav" / f"{k}.wav", "sep": cr / "stems" / f"{k}_vocals.wav",
                "inst": ROOT / "data/outputs" / V7_DIRS[info["song"]] / "Media" / "00_offvocal_raw_backup.wav"}
    s = pair.removeprefix("v8_")
    d = ROOT / "data/outputs_v8r3" / V8_DIRS[s]
    return {"cover": next(d.glob("*_Master.wav")), "sep": next(d.glob("*_VocalBus.wav")),
            "inst": ROOT / "data/outputs" / V7_DIRS[RYO_SONG[s]] / "Media" / "00_offvocal_raw_backup.wav"}


def practice_mains(pair: str) -> list[Path]:
    wavs = sorted((REF / pair / "practice").rglob("*.wav"))
    post = [w for w in wavs if POST.search(str(w.parent))]
    mains = [w for w in (post or wavs) if re.search(r"メイン|main", w.stem, re.Ii)]
    return mains


def practice_all(pair: str) -> list[Path]:
    wavs = sorted((REF / pair / "practice").rglob("*.wav"))
    post = [w for w in wavs if POST.search(str(w.parent))]
    return post or [w for w in wavs if w.parent == REF / pair / "practice"]


# ---------------------------------------------------------------- dsp helpers
def bp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def xcorr_lag(a: np.ndarray, b: np.ndarray, maxlag: int) -> tuple[float, float]:
    """lag (fractional samples) with a[t] ~ b[t - lag], and normalized peak."""
    n = len(a) + len(b)
    nf = 1 << int(np.ceil(np.log2(n)))
    c = np.fft.irfft(np.fft.rfft(a, nf) * np.conj(np.fft.rfft(b, nf)), nf)
    c = np.concatenate([c[-maxlag:], c[: maxlag + 1]])
    i = int(np.argmax(c))
    y0, y1, y2 = c[max(i - 1, 0)], c[i], c[min(i + 1, len(c) - 1)]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return i - maxlag + float(np.clip(d, -1, 1)), float(y1 / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def env(x: np.ndarray, hop: int = 441) -> np.ndarray:
    """log energy 300-4k at 10 ms, high-passed (onset-like) for take matching."""
    m = bp(x.mean(0) if x.ndim == 2 else x, 300, 4000)
    n = len(m) // hop
    e = 10 * np.log10((m[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10)
    return e - signal.medfilt(e, 51)


def warp(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """x sampled at fractional positions pos (windowed sinc, 32 taps); out of range -> 0."""
    taps = np.arange(-15, 17)
    win = np.kaiser(len(taps) + 1, 8.0)[:-1].astype(np.float32)
    out = np.zeros((x.shape[0], len(pos)), np.float32)
    xp = np.pad(x, ((0, 0), (32, 32)))
    for s in range(0, len(pos), 1 << 18):
        p = pos[s: s + (1 << 18)]
        i0 = np.floor(p).astype(np.int64)
        fr = (p - i0).astype(np.float32)
        idx = i0[:, None] + taps[None, :]
        w = np.sinc(fr[:, None] - taps[None, :]).astype(np.float32) * win[None, :]
        valid = (i0 >= 0) & (i0 < x.shape[1] - 1)
        idx = np.clip(idx + 32, 0, xp.shape[1] - 1)
        for ch in range(x.shape[0]):
            out[ch, s: s + len(p)] = np.where(valid, (xp[ch][idx] * w).sum(1), 0.0)
    return out


def stft(x: np.ndarray) -> np.ndarray:
    return signal.stft(x, SR, window="hann", nperseg=NFFT, noverlap=NFFT - HOP, boundary=None,
                       padded=True)[2].astype(np.complex64)


def istft(X: np.ndarray, n: int) -> np.ndarray:
    y = signal.istft(X, SR, window="hann", nperseg=NFFT, noverlap=NFFT - HOP, boundary=False)[1]
    return np.pad(y, ((0, 0), (0, max(0, n - y.shape[-1]))))[:, :n].astype(np.float32)


def ms(x: np.ndarray) -> np.ndarray:
    return np.stack([(x[0] + x[1]) / 2, (x[0] - x[1]) / 2])


def lr(x: np.ndarray) -> np.ndarray:
    return np.stack([x[0] + x[1], x[0] - x[1]])


def frame_db(X: np.ndarray) -> np.ndarray:
    return 10 * np.log10((np.abs(X) ** 2).mean((0, 1)) + 1e-12)


def active(db: np.ndarray, rel: float, floor: float) -> np.ndarray:
    return db > max(floor, np.percentile(db, 99.5) - rel)


def dilate(a: np.ndarray, before: int, after: int) -> np.ndarray:
    out = a.copy()
    idx = np.flatnonzero(a)
    for s in range(-before, after + 1):
        out[np.clip(idx + s, 0, len(a) - 1)] = True
    return out


def lufs(x: np.ndarray) -> float:
    import pyloudnorm as pyln
    if x.shape[-1] < SR // 2:
        return float("nan")
    return float(pyln.Meter(SR, block_size=0.4).integrated_loudness(x.T.astype(np.float64)))


def chroma(x: np.ndarray) -> np.ndarray:
    import librosa
    y = signal.resample_poly(x.mean(0), 1, 4)
    c = librosa.feature.chroma_cqt(y=y, sr=SR // 4, hop_length=2048)
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)


# ---------------------------------------------------------------- stage: sep
def stage_sep(pair: str) -> dict:
    src = sources(pair)
    if not src["sep"].exists():
        sys.path.insert(0, str(ROOT / "src"))
        from mix_automation.analysis.reference_extractor import BSRoFormerSeparator
        sep = BSRoFormerSeparator(cache_dir=CACHE / "_sep", model_dir=ROOT / "data/processed/roformer_models")
        sep.separate(src["cover"], slug=pair, target_sr=SR)
    return {"sep": str(src["sep"].relative_to(ROOT))}


# ---------------------------------------------------------------- stage: drift (a)
def global_lag(cov: np.ndarray, inst: np.ndarray) -> tuple[int, float]:
    ds = 11
    c = signal.decimate(cov.mean(0), ds)
    i = signal.decimate(inst.mean(0), ds)
    lag, pk = xcorr_lag(c, i, len(c) - 1)
    return int(round(lag * ds)), pk


def stage_drift(pair: str) -> dict:
    src = sources(pair)
    cov, inst = load_wav(src["cover"]), load_wav(src["inst"])
    lag0, pk0 = global_lag(cov, inst)
    res = {"lag0_ms": round(lag0 / SR * 1000, 2), "global_peak": round(pk0, 3)}
    for name, lo, hi in (("low", 50, 250), ("mid", 250, 2000)):
        cb, ib = bp(cov.mean(0), lo, hi), bp(inst.mean(0), lo, hi)
        pts = []
        win, hopw = 8 * SR, 5 * SR
        for t0 in range(0, len(cb) - win, hopw):
            s = t0 - lag0
            if s < 0 or s + win > len(ib) or np.std(ib[s: s + win]) < 1e-4:
                continue
            lg, p = xcorr_lag(cb[t0: t0 + win], ib[s: s + win], int(0.025 * SR))
            pts.append((t0 / SR + 4, lag0 + lg, p))
        res[name] = np.round(np.array(pts), 4).tolist()
    pts = np.array(res["low"])
    good = pts[:, 2] > 0.5
    if good.sum() < 4:
        good = pts[:, 2] > np.median(pts[:, 2])
    t, lg = pts[good, 0], pts[good, 1]
    med = signal.medfilt(lg, 5) if len(lg) >= 5 else lg
    robust = np.abs(lg - med) < 0.0015 * SR
    t, lg = t[robust], lg[robust]
    slope, icpt = np.polyfit(t, lg, 1)
    lin_res = lg - (slope * t + icpt)
    steps = int((np.abs(np.diff(signal.medfilt(lg, 3) if len(lg) >= 3 else lg)) > 0.001 * SR).sum())
    span_ms = float((lg.max() - lg.min()) / SR * 1000)
    # warp model: constant / linear / piecewise
    if span_ms < 0.1:
        mode, curve = "const", [[0.0, float(np.median(lg))], [1e4, float(np.median(lg))]]
    elif np.abs(lin_res).max() < 0.15e-3 * SR:
        mode, curve = "linear", [[0.0, float(icpt)], [1e4, float(icpt + slope * 1e4)]]
    else:
        mode, curve = "piecewise", np.stack([t, signal.medfilt(lg, 3) if len(lg) >= 3 else lg], 1).tolist()
    res.update({"n_win": int(len(pts)), "n_used": int(len(t)), "peak_med": round(float(np.median(pts[:, 2])), 3),
                "drift_ppm": round(float(slope / SR * 1e6), 2), "span_ms": round(span_ms, 3),
                "lin_resid_max_ms": round(float(np.abs(lin_res).max() / SR * 1000), 3), "steps": steps,
                "mode": mode, "curve": curve, "cover_s": round(cov.shape[1] / SR, 1),
                "inst_s": round(inst.shape[1] / SR, 1)})
    jsave(pair, "drift", res)
    return {k: v for k, v in res.items() if k not in ("low", "mid", "curve")}


def aligned_inst(pair: str, cov_n: int, inst: np.ndarray) -> np.ndarray:
    d = jload(pair, "drift")
    c = np.array(d["curve"])
    t = np.arange(cov_n) / SR
    lag = np.interp(t, c[:, 0], c[:, 1])
    return warp(inst, np.arange(cov_n) - lag)


# ---------------------------------------------------------------- stage: sub (b)
def smooth_bins(R: np.ndarray, axis: int) -> np.ndarray:
    """moving average over frequency, half width ~3% of f (>= 2 bins)."""
    nb = R.shape[axis]
    k = np.maximum(2, np.round(np.arange(nb) * 0.03)).astype(int)
    c = np.cumsum(np.moveaxis(R, axis, 0), axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c], 0)
    lo, hi = np.clip(np.arange(nb) - k, 0, nb), np.clip(np.arange(nb) + k + 1, 0, nb)
    out = (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (c.ndim - 1))
    return np.moveaxis(out, 0, axis)


def block_corr(X: np.ndarray, Y: np.ndarray, blk: np.ndarray, nb: int, w: np.ndarray):
    """per-block Rxx[b,f,i,j] = sum X_i X_j*, Ryx[b,f,i,j] = sum Y_i X_j* over frames with weight w."""
    F = X.shape[1]
    Rxx = np.zeros((nb, F, 2, 2), np.complex128)
    Ryx = np.zeros((nb, F, 2, 2), np.complex128)
    for b in range(nb):
        m = (blk == b) & (w > 0)
        if not m.any():
            continue
        Xb, Yb = X[:, :, m], Y[:, :, m]
        Rxx[b] = np.einsum("ift,jft->fij", Xb, np.conj(Xb))
        Ryx[b] = np.einsum("ift,jft->fij", Yb, np.conj(Xb))
    return Rxx, Ryx


def solve(Rxx: np.ndarray, Ryx: np.ndarray, time_smooth: bool = True) -> np.ndarray:
    if time_smooth and Rxx.shape[0] > 2:
        k = np.array([0.25, 0.5, 0.25])
        Rxx = sum(k[i] * np.roll(Rxx, i - 1, 0) for i in range(3))
        Ryx = sum(k[i] * np.roll(Ryx, i - 1, 0) for i in range(3))
    Rxx, Ryx = smooth_bins(Rxx, 1), smooth_bins(Ryx, 1)
    lam = 1e-3 * np.trace(Rxx, axis1=-2, axis2=-1).real[..., None, None] + 1e-12
    return (Ryx @ np.linalg.inv(Rxx + lam * np.eye(2))).astype(np.complex64)


def predict(H: np.ndarray, X: np.ndarray, blk: np.ndarray) -> np.ndarray:
    """per-frame H linearly interpolated between block centres."""
    nb, T = H.shape[0], X.shape[2]
    fb = np.clip((np.arange(T) + 0.5) / (BLOCK_S * SR / HOP) - 0.5, 0, nb - 1)
    b0 = np.floor(fb).astype(int)
    b1 = np.minimum(b0 + 1, nb - 1)
    a = (fb - b0).astype(np.float32)
    out = np.empty_like(X)
    for s in range(0, T, 512):
        sl = slice(s, s + 512)
        Hf = (1 - a[sl, None, None, None]) * H[b0[sl]] + a[sl, None, None, None] * H[b1[sl]]  # (t,f,2,2)
        out[:, :, sl] = np.einsum("tfij,jft->ift", Hf, X[:, :, sl])
    return out


GAIN_EDGES = (30, 150, 300, 600, 1200, 2400, 4800, 9600, 16000)


def track_gain(C: np.ndarray, P: np.ndarray, fb: np.ndarray, use: np.ndarray, smooth: int = 3) -> np.ndarray:
    """per-frame, per-band real gain g with C ~ g * P (LS over the bins in `use`), smoothed over `smooth`
    frames; unbiased under an uncorrelated vocal. Returns (F, T) gains interpolated over log f."""
    cen, gs = [], []
    k = np.ones(smooth) / smooth
    for lo, hi in zip(GAIN_EDGES[:-1], GAIN_EDGES[1:]):
        b = (fb >= lo) & (fb < hi) & use
        num = np.real(C[:, b] * np.conj(P[:, b])).sum((0, 1))
        den = (np.abs(P[:, b]) ** 2).sum((0, 1))
        g = np.convolve(num, k, "same") / (np.convolve(den, k, "same") + 1e-12)
        gs.append(np.clip(g, 0.25, 4.0))
        cen.append(np.sqrt(lo * hi))
    gs = np.array(gs)
    lf = np.log(np.clip(fb, 20, None))
    lc = np.log(cen)
    out = np.empty((len(fb), gs.shape[1]), np.float32)
    for t in range(0, gs.shape[1], 2048):
        sl = slice(t, t + 2048)
        gsl = gs[:, sl]
        out[:, sl] = np.array([np.interp(lf, lc, gsl[:, j]) for j in range(gsl.shape[1])]).T
    return out


def masks(pair: str, n: int, C: np.ndarray, Ii: np.ndarray) -> dict[str, np.ndarray]:
    v = load_wav(sources(pair)["sep"])
    v = np.pad(v[:, :n], ((0, 0), (0, max(0, n - v.shape[1]))))
    V = stft(ms(v))
    T = C.shape[2]
    vdb = frame_db(V)[:T]
    # inst coverage (some off-vocals are shorter than the cover): frames within 1 s of inst signal
    cover_ok = dilate(active(frame_db(Ii), rel=60, floor=-80), 43, 43)
    act = active(vdb, rel=35, floor=-55) & cover_ok
    anyv = active(vdb, rel=45, floor=-60)
    free = ~dilate(anyv, 4, 20) & active(frame_db(Ii), rel=40, floor=-60) & active(frame_db(C), rel=50, floor=-70)
    return {"act": act, "any": anyv, "free": free, "cover_ok": cover_ok}


def band_res(Y, P, m, fb, lo, hi) -> float:
    b = (fb >= lo) & (fb < hi)
    e = (np.abs(Y[:, b][:, :, m] - P[:, b][:, :, m]) ** 2).sum()
    return round(float(10 * np.log10(e / ((np.abs(Y[:, b][:, :, m]) ** 2).sum() + 1e-12))), 1)


def stage_sub(pair: str) -> dict:
    src = sources(pair)
    cov = load_wav(src["cover"])
    n = cov.shape[1]
    inst = aligned_inst(pair, n, load_wav(src["inst"]))
    C, Ii = stft(ms(cov)), stft(ms(inst))
    T = C.shape[2]
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    mk = masks(pair, n, C, Ii)
    free = mk["free"]
    fpb = BLOCK_S * SR / HOP
    blk = (np.arange(T) / fpb).astype(int)
    nb = blk.max() + 1
    half = ((np.arange(T) // int(0.5 * SR / HOP)) % 2).astype(bool)  # 0.5 s interleave for 2-fold
    inband = (fb >= 30) & (fb < 16000)
    # static H (one per song), 2-fold
    preds = {}
    for name, nblk in (("static", 1), ("tv", nb)):
        bidx = np.zeros(T, int) if nblk == 1 else blk
        P = np.zeros_like(C)
        for fold in (False, True):
            Rxx, Ryx = block_corr(Ii, C, bidx, nblk, (half == fold).astype(float))
            H = solve(Rxx, Ryx, time_smooth=nblk > 1)
            Pf = predict(H, Ii, bidx) if nblk > 1 else np.einsum("fij,jft->ift", H[0], Ii)
            P[:, :, half != fold] = Pf[:, :, half != fold]
        preds[name] = P
    # + fast per-band gain (limiter / ducking), 2-fold over even/odd bins
    par = (np.arange(len(fb)) % 2).astype(bool)
    Pg = preds["tv"].copy()
    for fold in (False, True):
        G = track_gain(C, preds["tv"], fb, par == fold)
        Pg[:, par != fold] = G[None, par != fold] * preds["tv"][:, par != fold]
    preds["tvg"] = Pg
    Rxx, Ryx = block_corr(Ii, C, blk, nb, np.ones(T))
    H = solve(Rxx, Ryx)
    np.save(cdir(pair) / "H.npy", H)
    res = {"free_s": round(float(free.sum() * HOP / SR), 1), "act_s": round(float(mk["act"].sum() * HOP / SR), 1)}
    for name, P in preds.items():
        res[f"res_{name}_db"] = band_res(C[:, inband], P[:, inband], free, fb[inband], 30, 16000)
        for lo, hi in ((30, 250), (250, 2000), (2000, 8000), (8000, 16000)):
            res[f"res_{name}_{lo}_{hi}"] = band_res(C, P, free, fb, lo, hi)
    # coherence: static (whole song) and per block, voice-free frames, mid channel
    for lo, hi in ((250, 2000), (2000, 8000)):
        b = (fb >= lo) & (fb < hi)
        Cf, If = C[0, b][:, free], Ii[0, b][:, free]
        coh = np.abs((Cf * np.conj(If)).sum(1)) ** 2 / ((np.abs(Cf) ** 2).sum(1) * (np.abs(If) ** 2).sum(1) + 1e-12)
        res[f"coh_static_{lo}_{hi}"] = round(float(np.median(coh)), 2)
        cb = []
        for k in range(nb):
            m = free & (blk == k)
            if m.sum() < 0.5 * SR / HOP:
                continue
            Cb, Ib = C[0, b][:, m], Ii[0, b][:, m]
            sxy = smooth_bins((Cb * np.conj(Ib)).sum(1), 0)
            sxx, syy = smooth_bins((np.abs(Ib) ** 2).sum(1), 0), smooth_bins((np.abs(Cb) ** 2).sum(1), 0)
            cb.append(np.median(np.abs(sxy) ** 2 / (sxx * syy + 1e-12)))
        res[f"coh_tv_{lo}_{hi}"] = round(float(np.median(cb)), 2) if cb else None
    # per-block held-out residual -> usable seconds (best of tv / tvg)
    best = "tvg" if res["res_tvg_db"] < res["res_tv_db"] else "tv"
    res["model"] = best
    res["res_best_db"] = res[f"res_{best}_db"]
    P = preds[best]
    bres = np.full(nb, np.nan)
    for k in range(nb):
        m = free & (blk == k)
        if m.sum() >= 0.5 * SR / HOP:
            e = (np.abs(C[:, inband][:, :, m] - P[:, inband][:, :, m]) ** 2).sum()
            bres[k] = 10 * np.log10(e / ((np.abs(C[:, inband][:, :, m]) ** 2).sum() + 1e-12))
    known = np.flatnonzero(~np.isnan(bres))
    good = np.zeros(nb, bool)
    for k in range(nb):
        if len(known):
            j = known[np.argmin(np.abs(known - k))]
            good[k] = abs(j - k) <= 2 and bres[j] < RES_OK_DB
    res["usable_s"] = round(float((mk["act"] & good[blk]).sum() * HOP / SR), 1)
    res["blocks_measured"] = int(len(known))
    res["blocks_good"] = int((bres[known] < RES_OK_DB).sum())
    method = "sub" if res["res_best_db"] < RES_OK_DB else "sep"
    res["method"] = method
    # write target vocal + pro-processed inst (cache only)
    PI = predict(H, Ii, blk)
    if best == "tvg":
        G = track_gain(C, PI, fb, np.ones(len(fb), bool))
        np.save(cdir(pair) / "G.npy", G[:: 8].astype(np.float16))
        PI = G[None] * PI
    inst_pro = lr(istft(PI, n))
    if method == "sub":  # outside the inst coverage the subtraction is undefined -> silence
        cm = np.repeat(mk["cover_ok"], HOP)[:n]
        voc = lr(istft(C - PI, n)) * np.pad(cm, (0, n - len(cm)))[None]
    else:
        v = load_wav(src["sep"])
        voc = np.pad(v[:, :n], ((0, 0), (0, max(0, n - v.shape[1]))))
        inst_pro = cov - voc
    save_wav(cdir(pair) / "target_vocal.wav", voc)
    save_wav(cdir(pair) / "inst_pro.wav", inst_pro)
    save_wav(cdir(pair) / "inst_aligned.wav", inst)
    np.savez(cdir(pair) / "masks.npz", **mk, blk=blk)
    if pair.startswith("v8_"):  # method check: extracted vs known vocal bus
        vb = load_wav(src["sep"])[:, :n]
        vs = lr(istft(C - PI, n))[:, : vb.shape[1]]
        res["v8_extract_err_db"] = round(float(10 * np.log10(((vs - vb) ** 2).sum() / (vb ** 2).sum())), 1)
    jsave(pair, "sub", res)
    return res


# ---------------------------------------------------------------- stage: inv (materials)
def stage_inv(pair: str) -> dict:
    src = sources(pair)
    cov, inst = load_wav(src["cover"]), load_wav(src["inst"])
    lag0, _ = global_lag(cov, inst)
    res = {"cover_s": round(cov.shape[1] / SR, 1), "inst_s": round(inst.shape[1] / SR, 1),
           "inst_lag_ms": round(lag0 / SR * 1000, 1)}
    # key: chroma of inst vs cover on the cover time axis
    ia = np.zeros_like(cov)
    s0, s1 = max(0, lag0), min(cov.shape[1], inst.shape[1] + lag0)
    ia[:, s0:s1] = inst[:, s0 - lag0: s1 - lag0]
    cc, ci = chroma(cov), chroma(ia)
    nfr = min(cc.shape[1], ci.shape[1])
    on = ci[:, :nfr].sum(0) > 0.5
    sims = [float((cc[:, :nfr] * np.roll(ci[:, :nfr], k, 0)).sum(0)[on].mean()) for k in range(12)]
    kb = int(np.argmax(sims))
    res.update({"key_shift": kb if kb <= 6 else kb - 12, "chroma_k0": round(sims[0], 3),
                "chroma_next": round(sorted(sims)[-2] if kb == 0 else sims[0], 3)})
    if pair.startswith("Amo_"):
        mains = practice_mains(pair)
        st =[load_wav(p) for p in mains]
        L = max(s.shape[1] for s in st)
        pm = sum(np.pad(s, ((0, 0), (0, L - s.shape[1]))) for s in st)
        res["practice_main"] = [str(p.relative_to(REF / pair)) for p in mains]
        res["practice_s"] = round(L / SR, 1)
        sep = load_wav(src["sep"]) if src["sep"].exists() else None
        if sep is not None:
            ep, ev = env(pm), env(sep)
            lg, pk = xcorr_lag(ev, ep, len(ev) - 1)
            res["take_lag_ms"] = round(lg * 10, 1)
            res["take_vs_inst_ms"] = round(lg * 10 - lag0 / SR * 1000, 1)
            res["take_env_corr"] = round(pk, 3)
            # local: 10 s windows, +-0.3 s
            loc = []
            L0 = int(round(lg))
            for t0 in range(0, len(ev) - 1000, 500):
                s = t0 - L0
                if s < 0 or s + 1000 > len(ep) or np.std(ep[s: s + 1000]) < 1:
                    continue
                ll, pp = xcorr_lag(ev[t0: t0 + 1000], ep[s: s + 1000], 30)
                if pp > 0.3:
                    loc.append((L0 + ll) * 10)
            res["take_local_dev_ms"] = round(float(np.max(np.abs(np.array(loc) - np.median(loc)))), 0) if loc else None
            res["take_n_local"] = len(loc)
    jsave(pair, "inv", res)
    return res


# ---------------------------------------------------------------- stage: meas (step 4)
BANDS_DUCK = ((40, 150), (300, 1000), (1000, 3000), (3000, 6000), (6000, 12000))
BANDS_EQ = (63, 125, 250, 500, 1000, 2000, 4000, 8000, 12000)


def class_gain(Ii, C, m_cls, sel, fb, lo, hi) -> float | None:
    """power gain of the LS fit on frames m_cls (mid row), evaluated on frames sel."""
    b = (fb >= lo) & (fb < hi)
    if m_cls.sum() < 1.0 * SR / HOP:
        return None
    X, Y = Ii[:, b][:, :, m_cls], C[0, b][:, m_cls]
    Rxx = np.einsum("ift,jft->fij", X, np.conj(X))
    Ryx = np.einsum("ft,jft->fj", Y, np.conj(X))
    Rxx, Ryx = smooth_bins(Rxx, 0), smooth_bins(Ryx, 0)
    lam = 1e-3 * np.trace(Rxx, axis1=-2, axis2=-1).real[:, None, None] + 1e-12
    h = np.einsum("fj,fjk->fk", Ryx, np.linalg.inv(Rxx + lam * np.eye(2)))
    Xs = Ii[:, b][:, :, sel]
    out = np.einsum("fj,jft->ft", h, Xs)
    return float((np.abs(out) ** 2).sum() / ((np.abs(Xs[0]) ** 2).sum() + 1e-12))


def tail_db(v: np.ndarray) -> float | None:
    """energy 100-500 ms after phrase ends (level < peak - 20 dB) relative to the phrase; phrases >= 0.4 s
    followed by a gap >= 0.6 s."""
    h = 441
    m = v.mean(0)
    nfr = len(m) // h
    e = 10 * np.log10((m[: nfr * h].reshape(nfr, h) ** 2).mean(1) + 1e-12)
    on = e > np.percentile(e, 99) - 20
    on = signal.medfilt(on.astype(float), 9) > 0.5
    vals = []
    edges = np.flatnonzero(np.diff(on.astype(int)) == -1)
    for k in edges:
        st = k
        while st > 0 and on[st - 1]:
            st -= 1
        nxt = k + 1 + np.argmax(on[k + 1:]) if on[k + 1:].any() else nfr
        if k - st < 40 or nxt - k < 60:
            continue
        ph = (10 ** (e[st:k + 1] / 10)).mean()
        tl = (10 ** (e[k + 10: k + 50] / 10)).mean()
        vals.append(10 * np.log10(tl / ph))
    return round(float(np.median(vals)), 1) if len(vals) >= 3 else None


def tilt(x: np.ndarray, sel: np.ndarray) -> float:
    X = stft(ms(x))[:, :, sel]
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    p = (np.abs(X) ** 2).sum((0, 2))
    return float(10 * np.log10(p[(fb >= 3000) & (fb < 10000)].sum() / (p[(fb >= 200) & (fb < 2000)].sum() + 1e-12)))


def sm_db(x: np.ndarray, sel_s: np.ndarray) -> float:
    z = ms(x)[:, sel_s]
    return round(float(10 * np.log10((z[1] ** 2).sum() / ((z[0] ** 2).sum() + 1e-12))), 1)


def stage_meas(pair: str) -> dict:
    src = sources(pair)
    sub = jload(pair, "sub")
    cov = load_wav(src["cover"])
    n = cov.shape[1]
    inst = load_wav(cdir(pair) / "inst_aligned.wav")
    voc = load_wav(cdir(pair) / "target_vocal.wav")
    ipro = load_wav(cdir(pair) / "inst_pro.wav")
    mk = np.load(cdir(pair) / "masks.npz")
    act, free = mk["act"], mk["free"]
    C, Ii = stft(ms(cov)), stft(ms(inst))
    fb = np.fft.rfftfreq(NFFT, 1 / SR)
    T = C.shape[2]
    res = {"method": sub["method"]}
    # ducking: 8 s windows, act vs free fits evaluated on all frames of the window
    w8, h4 = int(8 * SR / HOP), int(4 * SR / HOP)
    duck = {f"{lo}-{hi}": [] for lo, hi in BANDS_DUCK}
    for s in range(0, T - w8, h4) if sub["method"] == "sub" else []:
        win = np.zeros(T, bool)
        win[s: s + w8] = True
        a, f = act & win, free & win
        if a.sum() < 1.5 * SR / HOP or f.sum() < 1.5 * SR / HOP:
            continue
        for lo, hi in BANDS_DUCK:
            ga, gf = class_gain(Ii, C, a, win, fb, lo, hi), class_gain(Ii, C, f, win, fb, lo, hi)
            if ga and gf:
                duck[f"{lo}-{hi}"].append(10 * np.log10(ga / gf))
    res["duck_n_win"] = len(duck["40-150"])
    res["duck_med_db"] = {k: round(float(np.median(v)), 2) if v else None for k, v in duck.items()}
    res["duck_p90_db"] = {k: round(float(np.percentile(v, 10)), 2) if v else None for k, v in duck.items()}
    # EQ trend: whole-song power gain inst -> pro inst per octave band (normalised to 1 kHz), all frames
    PI = stft(ms(ipro)) if sub["method"] == "sub" else None
    if PI is not None:
        g = {}
        for fc in BANDS_EQ:
            b = (fb >= fc / np.sqrt(2)) & (fb < fc * np.sqrt(2))
            g[fc] = 10 * np.log10((np.abs(PI[:, b]) ** 2).sum() / ((np.abs(Ii[:, b]) ** 2).sum() + 1e-12))
        res["eq_rel1k_db"] = {str(k): round(v - g[1000], 1) for k, v in g.items()}
        res["inst_gain_1k_db"] = round(g[1000], 1)
        allf = np.ones(n, bool)
        res["inst_sm_change_db"] = round(sm_db(ipro, allf) - sm_db(inst, allf), 1)
        res["inst_sm_raw_db"] = sm_db(inst, allf)
    # vocal metrics
    sel = np.flatnonzero(np.repeat(act, HOP)[:n])
    res["voc_minus_inst_lu"] = round(lufs(voc[:, sel]) - lufs(ipro[:, sel]), 1)
    per = []
    w10 = int(10 * SR / HOP)
    for s in range(0, T - w10, w10 // 2):
        a = act[s: s + w10]
        if a.mean() < 0.5:
            continue
        ss = np.flatnonzero(np.repeat(np.r_[np.zeros(s, bool), a], HOP)[:n])
        per.append(lufs(voc[:, ss]) - lufs(ipro[:, ss]))
    per = np.array([p for p in per if np.isfinite(p)])
    if len(per):
        res["voc_minus_inst_win"] = {"p10": round(float(np.percentile(per, 10)), 1),
                                     "med": round(float(np.median(per)), 1),
                                     "p90": round(float(np.percentile(per, 90)), 1), "n": int(len(per))}
    res["voc_sm_db"] = sm_db(voc, sel)
    res["voc_tail_db"] = tail_db(voc)
    asel = act[: T]
    res["bright_voc_minus_inst_db"] = round(tilt(voc, asel) - tilt(ipro, asel), 1)
    jsave(pair, "meas", res)
    return res


# ---------------------------------------------------------------- report
def rnd(o):
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    return round(float(o), 2) if isinstance(o, (float, np.floating)) else o


def stage_report(names: list[str]) -> None:
    out = {"schema_version": 1, "res_ok_db": RES_OK_DB, "block_s": BLOCK_S, "pairs": {}}
    prev = DOCS / "summary.json"
    if prev.exists():
        out["pairs"] = json.loads(prev.read_text(encoding="utf-8")).get("pairs", {})
    for p in names:
        d = {}
        for st in ("inv", "drift", "sub", "meas"):
            f = cdir(p) / f"{st}.json"
            if f.exists():
                d[st] = rnd({k: v for k, v in jload(p, st).items() if k not in ("low", "mid", "curve")})
        out["pairs"][p] = d
    DOCS.mkdir(parents=True, exist_ok=True)
    prev.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    rows = []
    for p in names:
        d = out["pairs"][p]
        s, dr = d.get("sub", {}), d.get("drift", {})
        rows.append({"pair": p, "ppm": dr.get("drift_ppm"), "mode": dr.get("mode"), "steps": dr.get("steps"),
                     "coh250-2k": s.get("coh_static_250_2000"), "coh2-8k": s.get("coh_static_2000_8000"),
                     "cohTV": f"{s.get('coh_tv_250_2000')}/{s.get('coh_tv_2000_8000')}",
                     "res_static": s.get("res_static_db"), "res_tv": s.get("res_tv_db"),
                     "res_tvg": s.get("res_tvg_db"),
                     "free_s": s.get("free_s"), "usable_s": s.get("usable_s"), "method": s.get("method")})
    table(rows)


STAGES = {"inv": stage_inv, "sep": stage_sep, "drift": stage_drift, "sub": stage_sub, "meas": stage_meas}


def main() -> None:
    stage = sys.argv[1]
    names = pairs(sys.argv[2] if len(sys.argv) > 2 else None)
    if stage == "report":
        stage_report(names)
        return
    todo = ["sep", "drift", "sub", "meas"] if stage == "all" else [stage]
    for st in todo:
        rows = []
        for p in names:
            try:
                r = STAGES[st](p)
            except Exception as e:  # report and continue with the next pair
                r = {"error": f"{type(e).__name__}: {e}"}
            rows.append({"pair": p, **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}})
            print(st, p, "done", flush=True)
        table(rows)
    if stage == "all":
        stage_report(names)


if __name__ == "__main__":
    main()
