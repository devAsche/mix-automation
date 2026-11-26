"""v8 engine entry point (skeleton only).

The four processing layers described in docs/V8_PLAN.md (fixed chain, phrase loudness ride,
syllable-level consonant shaping, space/FX from the backing track) will be added phase by phase.
"""

from __future__ import annotations

from pathlib import Path

ENGINE_NAME = "v8"


class V8NotImplementedError(NotImplementedError):
    """Raised while the v8 engine has no rendering path yet."""


def run_v8_automix_for_song(
    song_folder: Path,
    template_song_path: Path,
    profiles_json_path: Path,
    output_root: Path,
    **_options: object,
) -> dict:
    raise V8NotImplementedError(
        "v8 エンジンはまだ未実装です（Phase 1: 骨格のみ）。"
        "現行のレンダーは --engine v7（既定）を使ってください。計画: docs/V8_PLAN.md"
    )
