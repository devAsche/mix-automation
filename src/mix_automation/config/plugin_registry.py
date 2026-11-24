"""Registry of verified local VST3 plugins mapped to vocal mixing roles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

VST3_ROOT = Path(r"C:\Program Files\Common Files\VST3")


class PluginRole(StrEnum):
    """Functional role within the automated vocal mixing pipeline."""

    MOUTH_DECLICK = "mouth_declick"
    DEPLOSIVE = "deplosive"
    VOICE_DENOISE = "voice_denoise"
    BREATH_CONTROL = "breath_control"
    DEESSER = "deesser"
    AUTO_GAIN = "auto_gain"
    SUBTRACTIVE_EQ = "subtractive_eq"
    DYNAMIC_UNMASK_EQ = "dynamic_unmask_eq"
    PEAK_COMPRESSOR = "peak_compressor"
    OPTICAL_COMPRESSOR = "optical_compressor"
    COLOR_EQ = "color_eq"
    VOCAL_ENHANCER = "vocal_enhancer"
    RESONANCE_SUPPRESSOR = "resonance_suppressor"
    SATURATOR = "saturator"
    DOUBLER = "doubler"
    PITCH_FX = "pitch_fx"
    PLATE_REVERB = "plate_reverb"
    SPACE_REVERB = "space_reverb"
    DELAY = "delay"
    MASTER_PROCESSOR = "master_processor"
    TRUE_PEAK_LIMITER = "true_peak_limiter"


@dataclass(frozen=True)
class PluginSpec:
    """Metadata and file location for a verified active local VST3 plugin."""

    key: str
    name: str
    role: PluginRole
    path: Path
    offline_render_only: bool = False
    notes: str = ""

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def loadable_path(self) -> Path:
        """Return inner x86_64-win binary path when VST3 is packaged as a directory bundle."""
        if self.path.is_dir():
            inner = self.path / "Contents" / "x86_64-win" / self.path.name
            if inner.exists():
                return inner
        return self.path


# Verified active & licensed VST3 plugins on this workstation
# (Excludes expired licenses such as VocalSynth 2 and VocAlign 6 Standard)
DEFAULT_PLUGIN_CATALOG: tuple[PluginSpec, ...] = (
    # --- Stage 1: iZotope RX 10 Pre-processing ---
    PluginSpec(
        key="rx10_mouth_declick",
        name="iZotope RX 10 Mouth De-click",
        role=PluginRole.MOUTH_DECLICK,
        path=VST3_ROOT / "iZotope" / "RX 10 Mouth De-click.vst3",
        offline_render_only=True,
        notes="Removes lip noise and saliva clicks transparently before compression.",
    ),
    PluginSpec(
        key="rx10_deplosive",
        name="iZotope RX 10 De-plosive",
        role=PluginRole.DEPLOSIVE,
        path=VST3_ROOT / "iZotope" / "RX 10 De-plosive.vst3",
        offline_render_only=True,
        notes="Tames low-end bursts on p/b/t consonants without thinning vocal body.",
    ),
    PluginSpec(
        key="rx10_voice_denoise",
        name="iZotope RX 10 Voice De-noise",
        role=PluginRole.VOICE_DENOISE,
        path=VST3_ROOT / "iZotope" / "RX 10 Voice De-noise.vst3",
        offline_render_only=True,
        notes="Gentle adaptive noise floor cleanup.",
    ),
    PluginSpec(
        key="rx10_breath_control",
        name="iZotope RX 10 Breath Control",
        role=PluginRole.BREATH_CONTROL,
        path=VST3_ROOT / "iZotope" / "RX 10 Breath Control.vst3",
        offline_render_only=True,
        notes="Target-level breath attenuation preserving natural phrasing.",
    ),
    PluginSpec(
        key="rx10_deess",
        name="iZotope RX 10 De-ess",
        role=PluginRole.DEESSER,
        path=VST3_ROOT / "iZotope" / "RX 10 De-ess.vst3",
        notes="Spectral de-essing for Japanese sa/ta rows and English sibilants.",
    ),
    # --- Stage 2: Core Tone, Glue & Resonance Control ---
    PluginSpec(
        key="persistent_q",
        name="Persistent Q",
        role=PluginRole.SUBTRACTIVE_EQ,
        path=VST3_ROOT / "Persistent Q.vst3",
        notes="Precision multi-band parametric & dynamic EQ (873 exposed parameters).",
    ),
    PluginSpec(
        key="zl_spectrum_eq",
        name="ZL Spectrum Equalizer",
        role=PluginRole.SUBTRACTIVE_EQ,
        path=VST3_ROOT / "ZL Spectrum Equalizer.vst3",
        notes="Multi-band surgical and dynamic spectrum equalizer.",
    ),
    PluginSpec(
        key="vca_compressor",
        name="Softube VCA Compressor",
        role=PluginRole.PEAK_COMPRESSOR,
        path=VST3_ROOT / "VCA Compressor.vst3",
        notes="Fast transient-catching first-stage serial compressor.",
    ),
    PluginSpec(
        key="uad_la2a",
        name="UAD Teletronix LA-2A Tube Compressor",
        role=PluginRole.OPTICAL_COMPRESSOR,
        path=VST3_ROOT / "uaudio_teletronix_la-2a_tc.vst3",
        notes="Second-stage optical leveling for vocal density and forward stickiness.",
    ),
    PluginSpec(
        key="uad_pultec_eqp1a",
        name="UAD Pultec EQP-1A",
        role=PluginRole.COLOR_EQ,
        path=VST3_ROOT / "uaudio_pultec_eqp-1a.vst3",
        notes="Silky high-frequency air boost and low-mid warmth.",
    ),
    PluginSpec(
        key="uad_pultec_meq5",
        name="UAD Pultec MEQ-5",
        role=PluginRole.COLOR_EQ,
        path=VST3_ROOT / "uaudio_pultec_meq-5.vst3",
        notes="Midrange vocal presence sculpting.",
    ),
    PluginSpec(
        key="slick_eq",
        name="TDR VOS SlickEQ",
        role=PluginRole.COLOR_EQ,
        path=VST3_ROOT / "TDR VOS SlickEQ.vst3",
        notes="Musical 3-band tonal shaping EQ with subtle saturation models.",
    ),
    PluginSpec(
        key="izotope_vea",
        name="iZotope VEA",
        role=PluginRole.VOCAL_ENHANCER,
        path=VST3_ROOT / "iZotope" / "VEA.vst3",
        notes="Clean/Shape/Boost vocal tone enhancer.",
    ),
    PluginSpec(
        key="soothe2",
        name="Oeksound Soothe2",
        role=PluginRole.RESONANCE_SUPPRESSOR,
        path=VST3_ROOT / "soothe2_x64.vst3",
        notes="Dynamic resonance suppressor eliminating harsh 2-5kHz vocal spikes.",
    ),
    PluginSpec(
        key="saturation_knob",
        name="Softube Saturation Knob",
        role=PluginRole.SATURATOR,
        path=VST3_ROOT / "Saturation Knob.vst3",
        notes="Harmonic thickening to cut through dense rock/Vocaloid backing tracks.",
    ),
    PluginSpec(
        key="izotope_trash2",
        name="iZotope Trash 2",
        role=PluginRole.SATURATOR,
        path=VST3_ROOT / "iZotope" / "iZotope Trash 2.vst3",
        notes="Multi-stage distortion, filter, and radio/megaphone creative saturation.",
    ),
    PluginSpec(
        key="initial_dynamic_eq",
        name="Initial Dynamic EQ",
        role=PluginRole.DYNAMIC_UNMASK_EQ,
        path=VST3_ROOT / "Initial Dynamic EQ.vst3",
        notes="Dynamic spectral unmasking on instrumental bus.",
    ),
    # --- Stage 3: Spatial, Doubling & Special Vocal Effects ---
    PluginSpec(
        key="rev_plate_140",
        name="Arturia Rev PLATE-140",
        role=PluginRole.PLATE_REVERB,
        path=VST3_ROOT / "Rev PLATE-140.vst3",
        notes="Classic EMT 140 plate reverb for lush vocal blend.",
    ),
    PluginSpec(
        key="exponential_r2",
        name="Exponential Audio R2",
        role=PluginRole.SPACE_REVERB,
        path=VST3_ROOT / "R2.vst3",
        notes=" lush musical algorithmic stereo reverb.",
    ),
    PluginSpec(
        key="phoenix_verb",
        name="Exponential Audio PhoenixVerb",
        role=PluginRole.SPACE_REVERB,
        path=VST3_ROOT / "PhoenixVerb.vst3",
        notes="Transparent natural room/hall reverb and ethereal tails.",
    ),
    PluginSpec(
        key="khs_dual_delay",
        name="Kilohearts Dual Delay",
        role=PluginRole.DELAY,
        path=VST3_ROOT / "Kilohearts" / "kHs Dual Delay.vst3",
        notes="Stereo/ping-pong delay throws.",
    ),
    PluginSpec(
        key="vocal_doubler",
        name="iZotope Vocal Doubler",
        role=PluginRole.DOUBLER,
        path=VST3_ROOT / "iZotope" / "Vocal Doubler.vst3",
        notes="Natural chorus/doubling width enhancer.",
    ),
    PluginSpec(
        key="mautopitch",
        name="MeldaProduction MAutoPitch",
        role=PluginRole.PITCH_FX,
        path=VST3_ROOT / "MeldaProduction" / "Pitch Shift" / "MAutoPitch.vst3",
        notes="Hard-tune / robotic pitch quantization & formant shift.",
    ),
    PluginSpec(
        key="izotope_nectar3",
        name="iZotope Nectar 3",
        role=PluginRole.PITCH_FX,
        path=VST3_ROOT / "iZotope" / "Nectar 3.vst3",
        offline_render_only=True,
        notes="Formant-preserving vocal harmony engine (solo harmony + chromatic interval selection).",
    ),
    # --- Stage 4: Mastering ---
    PluginSpec(
        key="ozone9_elements",
        name="iZotope Ozone 9 Elements",
        role=PluginRole.MASTER_PROCESSOR,
        path=VST3_ROOT / "iZotope" / "Ozone 9 Elements.vst3",
        notes="Master EQ, Imager & Maximizer (120 automatable parameters; Ozone 12 Elements exposes 0).",
    ),
    PluginSpec(
        key="wave_breaker",
        name="Press Play Wave Breaker",
        role=PluginRole.TRUE_PEAK_LIMITER,
        path=VST3_ROOT / "WaveBreaker.vst3",
        notes="Transparent true-peak limiter for final loudness target.",
    ),
)


def get_plugin_map() -> dict[str, PluginSpec]:
    """Return all registered plugins keyed by identifier."""
    return {spec.key: spec for spec in DEFAULT_PLUGIN_CATALOG}


def audit_installed_plugins() -> list[tuple[PluginSpec, bool]]:
    """Verify availability of each registered VST3 bundle on disk."""
    return [(spec, spec.exists) for spec in DEFAULT_PLUGIN_CATALOG]
