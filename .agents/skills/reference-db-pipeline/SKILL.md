---
name: reference-db-pipeline
description: >-
  Manage, audit, and rebuild the Utattemita reference database
  (profiles/pro_target_profiles.json) including large-scale batches collected
  via muse or manual downloads. Use this skill when adding new reference tracks,
  validating off-vocal pairs, or rebuilding acoustic target profiles.
---

# Reference Database & Target Profile Pipeline

## 1. Current Database Baseline (`schema_version: 4.0`)

- Location: `C:\MIX_automation\data\references\`
- Output JSON: `C:\MIX_automation\profiles\pro_target_profiles.json`
- Separator: `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (cached under `data/processed/roformer_stems/`)
- Current coverage: 44 songs across 8 cover artists (`AokiYo`, `aoru`, `Azusagawa`, `HoshirubeSho`, `NanadakiIma`, `Saburou`, `Sou`, `TOY`), all 44 paired with official off-vocals.
- Archetype distribution: `dense_rock_edm` (28), `groovy_mid_pop` (11), `intimate_ballad_sparse` (5).

## 2. Quality Gates for Adding New References (Especially `muse` Batches)

Before ingesting large batches of Utattemita tracks into `data/references/`, enforce these quality gates to prevent reference target degradation:

1. **Audio bandwidth & integrity**: Reject heavily crushed low-bitrate rips with <16 kHz brickwall cutoffs, severe clipping, or mono-collapsed masters.
2. **Official off-vocal alignment**: Verify that the paired official off-vocal matches the cover's key, tempo, and arrangement (cross-correlation & chroma agreement). Exclude key-shifted or custom-arranged covers from paired calibration unless separately tagged.
3. **Deduplication**: Avoid adding duplicate copies of the exact same cover release under different artist/info folders.
4. **Archetype & vocal stratification**: Prioritize under-represented archetypes (`intimate_ballad_sparse`, acoustic/jazz, whisper/shout, wider male/female F0 registers) so 9-NN interpolation and future learned regressors stay balanced.
5. **Metadata hygiene**: Preserve artist/MIXer folder grouping so `--taste <artist>` style anchors remain accurate.

## 3. Rebuilding Profiles

```powershell
C:\MIX_automation\.venv\Scripts\mix-auto.exe build-references
```

After rebuilding:
1. Verify `profiles/pro_target_profiles.json` reports expected `num_reference_songs`, `official_offvocal_calibration`, and non-empty archetype/artist entries.
2. Check that `pytest C:\MIX_automation\tests -v` passes with 0 regressions.
3. Commit updated `profiles/pro_target_profiles.json` (source audio stays gitignored) to `devAsche/mix-automation`.
