"""v8 engine (under construction).

Plan and ordering: docs/V8_PLAN.md. Until a phase lands, `mix-auto mix --engine v8`
stops with V8NotImplementedError instead of silently falling back to v7.
"""

from mix_automation.v8.engine import V8NotImplementedError, run_v8_automix_for_song

__all__ = ["V8NotImplementedError", "run_v8_automix_for_song"]
