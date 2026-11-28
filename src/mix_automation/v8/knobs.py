"""v8: the knobs that R4/R5 are allowed to move, defined in one place.

Every knob has a range and a default. Sources:
- rulebook `ID@m:ss` = docs/mix_knowledge/rulebook.md
- articles `[n]` = docs/mix_knowledge/articles.md の出典番号
- R2 = docs/v8r_targets/README.md（引き算で取り出したプロの声とオケの実測）
"R3 既定" = R3 で決めた値をそのまま持ち越したもの。R3 の値の組は `R3_DEFAULTS`。
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
    integer: bool = False  # categorical knob: only whole numbers inside [lo, hi] are valid


KNOB_SPECS: tuple[KnobSpec, ...] = (
    # R2 README 5.1: pro (subtraction, 8 covers, harmonies included) median +2.6 LU, Eva +2.7..+4.6;
    # range -3..+5 as proposed there. Default +1: v8 is lead-only, so below the pro median (R4 指示).
    KnobSpec("vocal_offset_lu", -3.0, 5.0, 1.0, "LU", "声−オケの目標 LU（R2 プロ中央値 +2.6、リードのみなので +1）"),
    # R3 既定: per-phrase loudness ride only corrects the excess beyond this tolerance (-c0bHeO5ISs@4:39).
    KnobSpec("ride_tolerance_lu", 0.5, 4.0, 1.5, "LU", "フレーズ単位で目標からこの幅を超えた分だけ直す"),
    # articles 追補: 3-6 dB at 2-4:1 in 3 articles, [4] 3-6 dB per stage; videos 6-11 dB
    # (2afvdjOpVJQ@5:21, @6:53). Range 3..11, default 6 = where articles and videos meet.
    KnobSpec("comp_total_gr_db", 3.0, 11.0, 6.0, "dB", "リードのコンプ 2 段の合計 GR（VCA 45%、LA-2A 55%）"),
    # -c0bHeO5ISs@19:22 / heG8RtaRwN0@16:20: Pultec high boost; R3 既定・範囲の中ほど.
    KnobSpec("brightness", 0.0, 6.0, 2.5, "knob", "Pultec EQP-1A hf_boost（周波数は air_hz）"),
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
    KnobSpec("delay_send_db", -34.0, -16.0, -24.0, "dB", "ディレイのセンド（種類は delay_type）"),
    # JUXJShxfzKQ@37:34: duck reverb/delay returns while singing; R3 既定.
    KnobSpec("fx_duck_db", 0.0, 12.0, 6.0, "dB", "歌っている間にプレート・ディレイのリターンを下げる最大量"),
    # R2 README 4 章: pro band ducking ~0 (-0.9..+1.9 dB); articles [2] 1-2 dB. Range 0..3, default 0.5.
    KnobSpec("inst_duck_max_db", 0.0, 3.0, 0.5, "dB", "オケの帯域別ダッキングの最大量"),
    # RTVPdDmWrwo@41:31-43:25: parallel 2mix glue; R3 既定.
    KnobSpec("glue_wet", 0.0, 0.6, 0.35, "ratio", "2mix 並列グルーの Wet"),
    # articles [1][3][4][8]: HPF 60-100 Hz most often (80-150); R3 used 140. Range 70..180, default 100.
    KnobSpec("hpf_hz", 70.0, 180.0, 100.0, "Hz", "Persistent Q band 1 の HPF（18 dB/oct）"),
    # articles [3] presence 1-3 kHz, [8] 2-5 kHz +1..2 dB. Range 2-5 kHz; default 3 kHz (middle).
    KnobSpec("presence_hz", 2000.0, 5000.0, 3000.0, "Hz", "Persistent Q band 3（Peaking, Q 1.0）の中心"),
    # articles [8] +1..2 dB; range 0..3, default 0 (off: R3 had no presence band).
    KnobSpec("presence_db", 0.0, 3.0, 0.0, "dB", "Persistent Q band 3 のブースト量（0 で band 3 無効）"),
    # articles [2] air 6-8 kHz and up, [8] 10-12 kHz. Pultec hf_freq is discrete (8/10/12 KCS here).
    KnobSpec("air_hz", 8000.0, 12000.0, 8000.0, "Hz", "Pultec hf_freq（8/10/12 KCS の最も近い値に丸める）"),
    # articles [1] 1/16 once, [4] 1/8 FB 50 % HPF 200 LPF 10k, [8] slap 80-120 ms. See DELAY_TYPES.
    KnobSpec("delay_type", 0.0, 2.0, 0.0, "type", "0 = 1/16 1 回、1 = 1/8 反復、2 = スラップ", integer=True),
    # articles [2]: lift the high end of the backing track's Side to open the centre. 0 = off.
    KnobSpec("inst_side_high_db", 0.0, 2.0, 0.0, "dB", "オケの Side の高域シェルフ（6 kHz）"),
    # articles [8]: blend a heavily squashed copy (GR 10-20 dB). 0 = off.
    KnobSpec("par_comp_mix", 0.0, 0.3, 0.0, "ratio", "リードの並列コンプ（GR 15 dB の複製）の混ぜる量"),
    # articles [2][8]: clip-gain / gain envelope before the compressor. 0 = off.
    KnobSpec("pre_level_strength", 0.0, 1.0, 0.0, "ratio", "コンプ前にフレーズ音量の差をこの割合だけ均す"),
)

KNOB_SPEC_MAP: dict[str, KnobSpec] = {k.name: k for k in KNOB_SPECS}

# delay_type -> what build_sends renders (chain.py). The R4 default 0 is the closest of the three to
# the R3 1/16 ping-pong: with feedback 0.3 the R3 second tap was already 10.5 dB below the first.
DELAY_TYPES: dict[int, str] = {0: "1/16 single tap", 1: "1/8 repeat FB 0.5", 2: "slap 120 ms"}

# Pultec EQP-1A high_freq choices reachable by air_hz (plugin valid_values: 3/4/5/8/10/12/16 KCS).
PULTEC_AIR_KCS: tuple[int, ...] = (8, 10, 12)


def pultec_air_kcs(air_hz: float) -> tuple[str, float]:
    """Round air_hz to the nearest Pultec hf_freq choice -> (plugin value string, rounded Hz)."""
    k = min(PULTEC_AIR_KCS, key=lambda c: (abs(c * 1000.0 - air_hz), c))
    return f"{k} KCS", k * 1000.0


@dataclass(frozen=True)
class V8Knobs:
    vocal_offset_lu: float = 1.0
    ride_tolerance_lu: float = 1.5
    comp_total_gr_db: float = 6.0
    brightness: float = 2.5
    deess_db: float = 3.0
    saturation: float = 1.5
    doubler_send_db: float = -18.5
    room_send_db: float = -16.0
    plate_send_db: float = -18.0
    plate_decay_s: float = 1.35
    delay_send_db: float = -24.0
    fx_duck_db: float = 6.0
    inst_duck_max_db: float = 0.5
    glue_wet: float = 0.35
    hpf_hz: float = 100.0
    presence_hz: float = 3000.0
    presence_db: float = 0.0
    air_hz: float = 8000.0
    delay_type: float = 0.0
    inst_side_high_db: float = 0.0
    par_comp_mix: float = 0.0
    pre_level_strength: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# R3 render values (data/outputs_v8r3/*/mix_report.json "knobs") plus the new knobs at their
# no-effect setting. delay_type -1 = the R3 1/16 ping-pong (FB 0.3, HP 400 / LP 4 kHz); it is kept
# only to reproduce R3 and is not offered to the optimiser (spec range is 0..2).
R3_DEFAULTS: dict[str, float] = {
    "vocal_offset_lu": -2.5,
    "comp_total_gr_db": 8.5,
    "inst_duck_max_db": 3.0,
    "hpf_hz": 140.0,
    "presence_hz": 3000.0,
    "presence_db": 0.0,
    "air_hz": 8000.0,
    "delay_type": -1.0,
    "inst_side_high_db": 0.0,
    "par_comp_mix": 0.0,
    "pre_level_strength": 0.0,
}
R3_DELAY_TYPE = -1  # allowed by validate_knobs in addition to the spec range


# R5 (補正 EQ ループ): the tone knobs are fixed, not searched; Ryo corrects the timbre by ear with the
# last Persistent Q (correction_eq.py). Reason: R4 matched the tone to the pro covers and came out too
# bright and thin (docs/evaluations, R4 聴き比べ). V8Knobs defaults stay as they are (v8 engine default).
R5_TONE: dict[str, float] = {
    # Mo_qfyNdYQQ@2:38: HPF 119 Hz. R3's 140 Hz is high against Ryo's "芯が細い" (R4 聴き比べ).
    "hpf_hz": 120.0,
    # presence band off: rulebook RTVPdDmWrwo@34:37, a hissy 3 kHz is what makes a vocal sound cheap.
    "presence_db": 0.0,
    "presence_hz": 3000.0,  # unused while presence_db = 0; spec default (articles [3][8] middle)
    # R4 matched 4.5-5.8 and Ryo heard it "シャリつく"; R3 had 2.5. One lower than R3: Ryo adds top end
    # in the correction EQ if needed.
    "brightness": 1.5,
    # articles [8]: air 10-12 kHz; 10 kHz = above the sibilance band's lower part (R5 指示).
    "air_hz": 10000.0,
    "saturation": 1.5,  # R3 既定, 2afvdjOpVJQ@8:02 (light)
    "deess_db": 3.0,  # R3 既定, RTVPdDmWrwo@26:38 (one de-esser, do not pull down hard)
}


def with_r5_tone(knobs: V8Knobs) -> V8Knobs:
    """`knobs` with the tone knobs replaced by R5_TONE."""
    return replace(knobs, **R5_TONE)


def has_r5_tone(knobs: V8Knobs | dict[str, float]) -> bool:
    d = knobs if isinstance(knobs, dict) else knobs.to_dict()
    return all(abs(float(d[k]) - v) < 1e-9 for k, v in R5_TONE.items())


def validate_knobs(knobs: V8Knobs) -> V8Knobs:
    for f in fields(knobs):
        spec = KNOB_SPEC_MAP[f.name]
        v = float(getattr(knobs, f.name))
        if spec.integer:
            if v != round(v):
                raise ValueError(f"knob {f.name}={v} must be a whole number")
            if f.name == "delay_type" and int(v) == R3_DELAY_TYPE:
                continue
        if not (spec.lo <= v <= spec.hi):
            raise ValueError(f"knob {f.name}={v} is outside [{spec.lo}, {spec.hi}] {spec.unit}")
    return knobs


def load_knobs(json_path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> V8Knobs:
    """Defaults <- JSON file (optional) <- overrides. Unknown names and out-of-range values raise ValueError.

    Knobs missing from the JSON (e.g. an R3 mix_report "knobs" block) take the current defaults."""
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


def r3_knobs() -> V8Knobs:
    """The R3 setting: R3_DEFAULTS over the dataclass defaults (other knobs had the same value in R3)."""
    return load_knobs(None, R3_DEFAULTS)
