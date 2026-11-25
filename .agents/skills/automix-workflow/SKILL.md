---
name: automix-workflow
description: >-
  Run, audit, and test the MIX Automation v6 pipeline (mix-auto CLI, ruff,
  pytest, Studio One .song verification, and acoustic report inspection). Use
  this skill whenever executing mixes, validating DSP changes, or auditing
  output quality across benchmark songs.
---

# MIX Automation Execution & Verification Workflow

## 1. Environment & Quick Commands

Always use the project virtual environment at `C:\MIX_automation\.venv`:

```powershell
# Lint (0 errors required)
C:\MIX_automation\.venv\Scripts\ruff.exe check --select E,F,W,I --ignore E501 C:\MIX_automation\src C:\MIX_automation\tests

# Full test suite (15 unit + 6 live VST3 contract tests = 21 tests)
C:\MIX_automation\.venv\Scripts\pytest.exe C:\MIX_automation\tests -v

# Check local Studio One 5 + 27 VST3 plugins
C:\MIX_automation\.venv\Scripts\mix-auto.exe doctor
```

## 2. Running Full Auto-MIX on Songs

**Important stability rule**: Run each song in its own clean CLI invocation rather than chaining multiple full songs inside a single long-lived Python process (repeated `Soothe2` + `LA-2A` C++ plugin resets across back-to-back songs in one process can occasionally trigger a silent native crash).

```powershell
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\ラブ&デストロイ" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\シックスフィートアンダー" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\Fall in Love" --style auto
```

Key CLI flags:
- `--style auto`: 44-song continuous 9-NN acoustic synthesis (default).
- `--taste azusagawa --taste-weight 0.25`: Default personalized MIXer aesthetic blend (`--taste none` disables).
- `--auto-pitch` / `--auto-align`: Opt-in ONLY for un-Melodyne'd raw vocal takes (keep OFF by default to protect groove and lead/harmony sync).
- `--enable-nectar-mix`: Force synthetic Nectar 3 harmonies into 2mix even when real harmonies exist.

## 3. Mandatory Post-Render Audit Checklist

After modifying DSP or `song_builder.py`, inspect `data/outputs/<song>_auto/mix_report.json` and the generated `.song` archive:

1. **Acoustic balance targets**:
   - `k_lufs_vi_ratio_db` and `mid_k_lufs_vi_ratio_db` close to resolved target (no double-stacking vocal float).
   - `wet_tail_ratio_db` aligned bidirectionally (neither bone-dry nor washed out).
   - `band_margins_db` (`core_presence`, `sibilance`, `air_sheen`) controlled without scooping `body_warmth` / `core_mid`.
   - Quiet HF floor (`qHF`) clean (no upward-compression noise hiss) and 2–6 kHz transient punch (`atk26`) around pro baseline (`~13.5`).
2. **Studio One 5.5 `.song` integrity**:
   - All XML files inside the `.song` ZIP parse cleanly.
   - Zero occurrences of `<VST3PluginState>` (JUCE leak count MUST be `0`).
   - Expected `.vstpreset` count matches console rack (16–17 presets; `choir_wide_bus` present when choir stems exist; `VocalSynth 2` present when verified `radio_voice` span is detected).
   - Quick CLI inspection: `C:\MIX_automation\.venv\Scripts\mix-auto.exe inspect-song "<path>.song"`

## 4. PowerShell Scripting Caution

When running ad-hoc Python verification in PowerShell, avoid complex nested double quotes inside `python -c "..."`. For multi-line audits, write a temporary script under `<ArtifactDirectory>/scratch/` and execute it with `C:\MIX_automation\.venv\Scripts\python.exe`.
