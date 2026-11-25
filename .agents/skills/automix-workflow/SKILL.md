---
name: automix-workflow
description: >-
  Run, audit, and test the MIX Automation v7 pipeline (mix-auto CLI, ruff,
  pytest, Studio One .song verification, A/B blind harness, preference learning,
  and acoustic report inspection). Use this skill whenever executing mixes,
  validating DSP changes, or auditing output quality across benchmark songs.
---

# MIX Automation Execution & Verification Workflow (`v7.0`)

## 1. Environment & Quick Commands

Always use the project virtual environment at `C:\MIX_automation\.venv`:

```powershell
# Lint (0 errors required)
C:\MIX_automation\.venv\Scripts\ruff.exe check --select E,F,W,I --ignore E501 C:\MIX_automation\src C:\MIX_automation\tests

# Full test suite (18 unit + 7 live VST3 contract tests = 25 tests)
C:\MIX_automation\.venv\Scripts\pytest.exe C:\MIX_automation\tests -v

# Check local Studio One 5 + 27 VST3 plugins
C:\MIX_automation\.venv\Scripts\mix-auto.exe doctor
```

## 2. Running Full Auto-MIX & Auxiliary Workflows

**Important stability rule**: Run each song in its own clean CLI invocation rather than chaining multiple full songs inside a single long-lived Python process (repeated `Soothe2` + `LA-2A` C++ plugin resets across back-to-back songs in one process can occasionally trigger a silent native crash).

```powershell
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\ラブ&デストロイ" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\シックスフィートアンダー" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\Fall in Love" --style auto
C:\MIX_automation\.venv\Scripts\mix-auto.exe mix "C:\MIX_automation\data\raw\エバ" --style auto

# Extract -14 LUFS loudness-normalized A/B clips, run Do-Not-Regress checks & build HTML console
C:\MIX_automation\.venv\Scripts\mix-auto.exe benchmark-ab

# Learn user aesthetic offsets from a manually tweaked Studio One 5 .song
C:\MIX_automation\.venv\Scripts\mix-auto.exe learn-preferences "<auto_song>.song" "<user_tweaked>.song"

# Scan & validate newly collected muse tracks in youtube_references
C:\MIX_automation\.venv\Scripts\mix-auto.exe scan-references
```

## 3. Mandatory Post-Render Audit Checklist

After modifying DSP or `song_builder.py`, run `mix-auto.exe benchmark-ab` and verify:

1. **Acoustic balance & anti-regression gates**:
   - `klufs_error_db <= 0.85 dB` across all `*_auto` benchmark songs (including extreme unmastered off-vocals like `エバ`).
   - `no_lower_mid_scoop` (`body_warmth >= target - 1.8 dB`) and `no_hf_sizzle` (`air_sheen <= -17.5 dB`).
   - Bidirectional `wet_tail_ratio_db` cohesion.
2. **Studio One 5.5 `.song` & MIDI integrity**:
   - All XML files inside `.song` parse cleanly and JUCE `<VST3PluginState>` leak count is `0`.
   - Expected `.vstpreset` count (16–19 presets including Stage 3 live FX tracks when detected) and valid 4-track `*_Director_Harmony_Guide.mid`.

## 4. PowerShell Scripting Caution

In Windows PowerShell 5.1, avoid nested double quotes inside `python -c "..."` (PowerShell strips inner quotes). Instead pipe multi-line Python over stdin using single-quoted here-strings: `@' ... '@ | & 'C:\MIX_automation\.venv\Scripts\python.exe' -`.
