"""Per-stage debug taps (v8 Phase 3 analysis aid).

Processing code calls `tap(name, audio)` at stage boundaries. Taps are recorded only inside a
`capture()` block, so the many optimisation passes that call the same functions record nothing and
pay nothing. Recording copies the array; it never changes the audio that flows on.

`active()` lets a stage do extra work only for the taps (e.g. re-render the first N plugins of a
chain on their own to expose intermediate outputs) while leaving the real signal path untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

_store: dict[str, np.ndarray] = {}
_prefix: str | None = None


def active() -> bool:
    return _prefix is not None


def tap(name: str, audio: np.ndarray) -> None:
    if _prefix is None:
        return
    _store[f"{_prefix}{name}"] = np.array(audio, dtype=np.float32, copy=True)


@contextmanager
def capture(prefix: str, enabled: bool = True) -> Iterator[None]:
    """Record taps made inside this block under `prefix` (no-op when `enabled` is False)."""
    global _prefix
    if not enabled:
        yield
        return
    prev = _prefix
    _prefix = prefix
    try:
        yield
    finally:
        _prefix = prev


def drain() -> dict[str, np.ndarray]:
    """Return and clear everything recorded so far."""
    out = dict(_store)
    _store.clear()
    return out
