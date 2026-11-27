"""v8 engine (R3: fixed chain on real VST3 plugins).

Plan and ordering: docs/V8_PLAN.md. `mix-auto mix --engine v8` renders the lead vocal through
v8/chain.py and stops with an exception if any plugin fails (no fallback to v7).
"""

from mix_automation.v8.engine import run_v8_automix_for_song

__all__ = ["run_v8_automix_for_song"]
