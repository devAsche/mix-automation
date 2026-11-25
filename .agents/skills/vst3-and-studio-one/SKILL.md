---
name: vst3-and-studio-one
description: >-
  Empirical VST3 plugin parameter quirks, Pedalboard adapter contracts, and
  PreSonus Studio One 5.5 native .song / .vstpreset engineering rules. Use this
  skill before editing vst3_adapter.py, vst3_optimizer.py, plugin_registry.py,
  or song_builder.py.
---

# VST3 & Studio One 5.5 Engineering Runbook

Full reference table and plugin ClassIDs live in [docs/VST3_AND_STUDIO_ONE_HANDBOOK.md](../../../docs/VST3_AND_STUDIO_ONE_HANDBOOK.md).

## 1. Strict Parameter Write Contract (`vst3_adapter.py`)

Never set Pedalboard VST3 attributes blindly (`setattr(plugin, k, v)` can silently fail or coerce wrongly on discrete/string params). Always route through `resolve_vst3_value` / verified helpers in `src/mix_automation/dsp/vst3_adapter.py` and verify via `tests/test_vst3_contracts.py`.

## 2. Proven Local VST3 Quirks (Do NOT Regress)

- **Directory-bundle `.vst3` paths**: Use `PluginSpec.loadable_path` (`Contents/x86_64-win/<name>.vst3`) when loading via Pedalboard.
- **`Oeksound Soothe2`**:
  - Frequency parameters expect kHz string tokens such as `'2k5'`, `'6k5'`, `'10k'` (parsed by `_KHZ_RE` in `vst3_adapter.py`).
  - Use `resolution="eco"` inside CMA-ES iterations for speed, and `resolution="high"` for final render and Studio One preset export.
- **`Softube VCA Compressor`**:
  - `attack` (`0.25..4.0`) is **smaller = faster**. Fast clean peak-taming uses `attack=0.52–0.72` and `output_drive=0.1`. Do NOT raise to `1.75` + `drive=1.0` (that acts as a harsh transient expander).
  - For main-harmony onset crushing ("語頭潰し"), use ultra-fast attack (`0.25`) + slow release (`3.40`) + high ratio (`6:1`).
- **`UAD Teletronix LA-2A`**:
  - Always bisect `peak_reduct` to measured optical GR (`2.0–3.6 dB`) AND bisect `gain` to exact unity RMS so staging never drops or jumps across songs.
- **`Softube Saturation Knob`**:
  - Always band-limit strictly to `240 Hz–4.0 kHz` (Midrange-Only) BOTH pre- and post-saturation in DSP, and prepend a `240 Hz HPF + 4.0 kHz LPF` `Persistent Q` on the `Saturation Parallel` bus in Studio One 5 to guarantee zero HF sizzle.
- **`Persistent Q` (16-Band v6 Layout)**:
  - Band 1: F0-adaptive HPF.
  - Bands 2–11: 10-band ERB/Octave differentiable EQ (use `z_db.detach()` and `cur_core_margin.detach()` on `sibilance`/`air_sheen` losses so HF targets never scoop lower mids).
  - Bands 12–13: Automated narrow-Q resonance notches (`Q=6.5–7.2`).
  - Bands 14–16: Real-time Dynamic mud (`290 Hz`), ultra-fast `1.2 ms` Dynamic bite (`3.25 kHz`), Spectral HF silkener (`7.8 kHz`).
- **`iZotope VocalSynth 2` (Licensed, 317 params)**:
  - Outputs **silence** unless either `input_lvl_dry_corrected = 100.0` (for pure FX chains like Radio Filter + Distort) OR an active synth module is enabled (`bvx_enable=True, bvx_lvl=100.0` for Biovox gloss).
- **`iZotope Ozone 9 Elements` vs `Ozone 12 Elements`**:
  - Use `Ozone 9 Elements` (120 automatable params); `Ozone 12 Elements` exposes `0` automatable params via Pedalboard.
- **Excluded / Avoid**:
  - `VocAlign 6 Standard` (expired license — excluded from registry).
  - `Waves MV2` upward compression (`low_level` pulls up room/HF noise floor — replaced by `LA-2A`).

## 3. Studio One 5.5 Native `.song` / `.vstpreset` Rules (`song_builder.py`)

- Studio One `5.5.2.86528` does NOT support `.dawproject`. Always build native `.song` ZIP archives from `data/templates/test20261124/test20261124.song`.
- **Critical JUCE Unwrapping**: Pedalboard's `plugin.raw_state` wraps VST3 state inside `<VST3PluginState><IComponent>...</IComponent><IEditController>...</IEditController></VST3PluginState>` using JUCE's custom Base64 alphabet (`.A-Za-z0-9+`). Always unwrap via `extract_steinberg_comp_and_cont()` into native Steinberg `Comp` and `Cont` chunks inside `.vstpreset`.
- When adding a new VST3 plugin to Studio One export, verify its exact ClassID in `%APPDATA%\PreSonus\Studio One 5\x64\Plugins-*.settings` and add it to `DEFAULT_S1_PLUGIN_SPECS` in `song_builder.py`.
- Always verify after changes by running `pytest tests/test_vst3_contracts.py -v` and checking exported `.song` archives for 0 `<VST3PluginState>` occurrences.
