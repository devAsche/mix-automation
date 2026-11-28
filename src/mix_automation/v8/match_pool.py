"""R4b: evaluate knob candidates for SongMatch in parallel processes (docs/v8r_match/README.md).

Step 0 of R4b (scripts/match/parallel_safety.py, data/processed/v8r_match/parallel_safety/
20261128-032827.json): with 1/2/4/6 processes at once every plugin output was bit-identical to the
serial run (PhoenixVerb excepted, which is non-deterministic even serially), no errors or dialogs, and
the per-process time hardly changed at 6 -> default 6 workers.

Each worker receives the prepared SongMatch objects once (pool initializer), keeps chain.plugin_cache()
open for its whole life (iLok / UAD plugins load once per worker), keeps one calibration warm-start
store per song, and loads the AFx-Rep model on the GPU once (lazily, on the first evaluation).
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Self

import numpy as np

DEFAULT_WORKERS = 6  # R4b step 0 (see module docstring)

_SONGS: dict[str, Any] = {}
_HINTS: dict[str, dict[str, float]] = {}
_CACHE_CM: Any = None


def _init(songs: dict[str, Any]) -> None:
    global _SONGS, _CACHE_CM
    from mix_automation.v8 import chain

    _SONGS = songs
    _CACHE_CM = chain.plugin_cache()
    _CACHE_CM.__enter__()  # left open until the worker exits


def knobs_from_dict(d: dict):
    from mix_automation.v8.knobs import V8Knobs

    return replace(V8Knobs(), **{k: float(v) for k, v in d.items()})


def _run(task: tuple) -> dict:
    """task = (song, knobs dict, tone or None, return_tone) -> SongMatch.evaluate result (+ "tone")."""
    from mix_automation.v8 import chain

    song, kd, tone, want_tone = task
    sm = _SONGS[song]
    prev = chain.set_calib_hints(_HINTS.setdefault(song, {}))
    t0 = time.time()
    try:
        res, tone_out = sm.evaluate(knobs_from_dict(kd), tone)
        if want_tone:
            res["tone"] = tone_out
    except Exception as e:  # noqa: BLE001 - one failed candidate must not stop the search
        res = {"dist": float("nan"), "per_cover": {}, "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-1500:], "sec": round(time.time() - t0, 2)}
    finally:
        chain.set_calib_hints(prev)
    return res


class EvalPool:
    """Pool of spawn workers holding the prepared songs. map() keeps the task order."""

    def __init__(self, songs: dict[str, Any], workers: int = DEFAULT_WORKERS) -> None:
        self.workers = workers
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(workers, initializer=_init, initargs=(songs,))

    def map(self, tasks: Sequence[tuple]) -> list[dict]:
        tasks = [(s, kd if isinstance(kd, dict) else kd.to_dict(), tone, bool(rt)) for s, kd, tone, rt in tasks]
        return self._pool.map(_run, tasks, chunksize=1)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        if exc[0] is None:
            self.close()
        else:
            self._pool.terminate()
            self._pool.join()


def finite_or(v: float, bad: float = 1e6) -> float:
    return float(v) if np.isfinite(v) else bad


# Early stop: R4 Eva levelled off after about 20 stage-A and 60 stage-B evaluations (docs/v8r_match/
# README.md 6 章), i.e. 2 and 5 generations of 12; stop when the best value improved by less than
# STOP_MIN_DELTA over the last STOP_PATIENCE generations (R4b 指示). 0.02 is 2-3 % of the R4 optima
# (0.67 / 0.99) and 20x the ~0.001 re-evaluation noise of a cached-plugin run (README 4 章).
STOP_PATIENCE = 3
STOP_MIN_DELTA = 0.02


def should_stop(best_hist: Sequence[float], patience: int = STOP_PATIENCE, min_delta: float = STOP_MIN_DELTA) -> bool:
    """best_hist[i] = best value after generation i (best_hist[0] may be the starting point)."""
    if len(best_hist) <= patience:
        return False
    return (best_hist[-1 - patience] - best_hist[-1]) < min_delta


__all__ = ["DEFAULT_WORKERS", "STOP_MIN_DELTA", "STOP_PATIENCE", "EvalPool", "finite_or", "knobs_from_dict",
           "should_stop"]
