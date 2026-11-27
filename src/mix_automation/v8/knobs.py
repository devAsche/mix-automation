"""v8 R3: the knobs that R4/R5 are allowed to move, defined in one place.

Every knob has a range and a default. Sources (rulebook `ID@m:ss` = docs/mix_knowledge/rulebook.md):
defaults are "R3 既定・rulebook 範囲の中ほど" unless stated otherwise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KnobSpec:
    name: str
    lo: float
    hi: float
    default: float
    unit: str
    doc: str


KNOB_SPECS: tuple[KnobSpec, ...] = (
    # Paired pilot: 4 pro covers measured -5.5 .. +0.2 LU (docs/paired_pilot/README.md); default = middle.
    KnobSpec("vocal_offset_lu", -5.5, 0.5, -2.5, "LU", "声−オケの目標 LU（ペア音源パイロット プロ 4 曲 -5.5〜+0.2 の中ほど）"),
    # R3 既定: per-phrase loudness ride only corrects the excess beyond this tolerance (-c0bHeO5ISs@4:39).
    KnobSpec("ride_tolerance_lu", 0.5, 4.0, 1.5, "LU", "フレーズ単位で目標からこの幅を超えた分だけ直す"),
    # 2afvdjOpVJQ@5:21 (4-5 dB) + @6:53 (~6 dB) vs -c0bHeO5ISs@17:06 (3-4 dB): total 6..11, middle 8.5.
    KnobSpec("comp_total_gr_db", 6.0, 11.0, 8.5, "dB", "リードのコンプ 2 段の合計 GR（VCA 45%、LA-2A 55%）"),
    # -c0bHeO5ISs@19:22 / heG8RtaRwN0@16:20: Pultec high boost; R3 既定・範囲の中ほど.
    KnobSpec("brightness", 0.0, 6.0, 2.5, "knob", "Pultec EQP-1A hf_boost（8 KCS）"),
    # 2afvdjOpVJQ@10:30, RTVPdDmWrwo@26:38 (1 台で大きく下げない): R3 既定.
    KnobSpec("deess_db", 0.0, 8.0, 3.0, "dB", "RX De-ess で歯擦音フレームの 5–10 kHz を下げる目標"),
    # 2afvdjOpVJQ@8:02: light saturation; R3 既定.
    KnobSpec("saturation", 0.0, 4.0, 1.5, "knob", "Saturation Knob の量"),
    # JUXJShxfzKQ@7:07: doubler send -20..-17 dB.
    KnobSpec("doubler_send_db", -24.0, -14.0, -18.5, "dB", "ダブラーのセンド"),
    # JUXJShxfzKQ@8:49: room send -15..-16 dB.
    KnobSpec("room_send_db", -22.0, -12.0, -16.0, "dB", "ルームのセンド"),
    # JUXJShxfzKQ@14:30: plate send about -18 dB.
    KnobSpec("plate_send_db", -26.0, -12.0, -18.0, "dB", "プレートのセンド"),
    # JUXJShxfzKQ@14:30: plate decay 1.2..1.5 s.
    KnobSpec("plate_decay_s", 0.9, 2.0, 1.35, "s", "プレートの Decay"),
    # 2afvdjOpVJQ@14:30, Mo_qfyNdYQQ@11:09: delay barely audible inside the mix; R3 既定.
    KnobSpec("delay_send_db", -34.0, -16.0, -24.0, "dB", "1/16 ピンポンディレイのセンド"),
    # JUXJShxfzKQ@37:34: duck reverb/delay returns while singing; R3 既定.
    KnobSpec("fx_duck_db", 0.0, 12.0, 6.0, "dB", "歌っている間にプレート・ディレイのリターンを下げる最大量"),
    # RTVPdDmWrwo@41:11: band-wise ducking of the backing track; R3 既定.
    KnobSpec("inst_duck_max_db", 0.0, 6.0, 3.0, "dB", "オケの帯域別ダッキングの最大量"),
    # RTVPdDmWrwo@41:31-43:25: parallel 2mix glue; R3 既定.
    KnobSpec("glue_wet", 0.0, 0.6, 0.35, "ratio", "2mix 並列グルーの Wet"),
)

KNOB_SPEC_MAP: dict[str, KnobSpec] = {k.name: k for k in KNOB_SPECS}


@dataclass(frozen=True)
class V8Knobs:
    vocal_offset_lu: float = -2.5
    ride_tolerance_lu: float = 1.5
    comp_total_gr_db: float = 8.5
    brightness: float = 2.5
    deess_db: float = 3.0
    saturation: float = 1.5
    doubler_send_db: float = -18.5
    room_send_db: float = -16.0
    plate_send_db: float = -18.0
    plate_decay_s: float = 1.35
    delay_send_db: float = -24.0
    fx_duck_db: float = 6.0
    inst_duck_max_db: float = 3.0
    glue_wet: float = 0.35

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def validate_knobs(knobs: V8Knobs) -> V8Knobs:
    for f in fields(knobs):
        spec = KNOB_SPEC_MAP[f.name]
        v = float(getattr(knobs, f.name))
        if not (spec.lo <= v <= spec.hi):
            raise ValueError(f"knob {f.name}={v} is outside [{spec.lo}, {spec.hi}] {spec.unit}")
    return knobs


def load_knobs(json_path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> V8Knobs:
    """Defaults <- JSON file (optional) <- overrides. Unknown names and out-of-range values raise ValueError."""
    values: dict[str, float] = {}
    if json_path is not None:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"knobs JSON must be an object: {json_path}")
        values.update(data)
    if overrides:
        values.update(overrides)
    unknown = sorted(set(values) - set(KNOB_SPEC_MAP))
    if unknown:
        raise ValueError(f"unknown knob(s): {unknown}")
    knobs = replace(V8Knobs(), **{k: float(v) for k, v in values.items()})
    return validate_knobs(knobs)
