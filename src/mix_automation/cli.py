"""v2 Command-line interface for MIX Automation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from mix_automation.analysis.relative_profile import compute_section_relative_profile
from mix_automation.config.plugin_registry import audit_installed_plugins, get_plugin_map
from mix_automation.io.studio_one import extract_vst3_processor_cid, inspect_studio_one_song

app = typer.Typer(help="v2 Reference-grade automated vocal mixing & Studio One 5 session engine.")
console = Console()

PROJECT_ROOT = Path(r"C:\MIX_automation")
STUDIO_ONE_EXE = Path(r"C:\Program Files\PreSonus\Studio One 5\Studio One.exe")
DEFAULT_TEMPLATE_SONG = PROJECT_ROOT / "data" / "templates" / "test20261124" / "test20261124.song"
DEFAULT_PROFILES_JSON = PROJECT_ROOT / "profiles" / "pro_target_profiles.json"
ENGINES = ("v7", "v8")


@app.command()
def doctor() -> None:
    """Verify local Studio One 5 installation and registered VST3 plugins."""
    s1_status = "[green]FOUND[/green]" if STUDIO_ONE_EXE.exists() else "[red]MISSING[/red]"
    console.print(f"\n[bold]Studio One 5 Executable:[/bold] {STUDIO_ONE_EXE} ({s1_status})\n")

    table = Table(title="Registered VST3 Mixing Chain Audit (v2)")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Role", style="magenta")
    table.add_column("Plugin Name")
    table.add_column("Status", justify="center")
    table.add_column("CID (moduleinfo)", justify="center")

    found_count = 0
    results = audit_installed_plugins()
    for spec, exists in results:
        if exists:
            found_count += 1
        cid = extract_vst3_processor_cid(spec.path) if exists else None
        status_str = "[green]OK[/green]" if exists else "[red]MISSING[/red]"
        cid_str = cid[:8] + "…" if cid else "-"
        table.add_row(spec.key, spec.role.value, spec.name, status_str, cid_str)

    console.print(table)
    console.print(
        f"\n[bold]Verified {found_count} / {len(results)} registered VST3 modules.[/bold]\n"
    )


@app.command("inspect-plugin")
def inspect_plugin(plugin_key: str) -> None:
    """Load a registered VST3 plugin via Pedalboard and dump its controllable parameters."""
    from pedalboard import load_plugin

    plugin_map = get_plugin_map()
    if plugin_key not in plugin_map:
        console.print(f"[red]Unknown plugin key: {plugin_key}[/red]")
        raise typer.Exit(code=1)

    spec = plugin_map[plugin_key]
    if not spec.exists:
        console.print(f"[red]Plugin file not found on disk: {spec.path}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Loading [bold]{spec.name}[/bold] ({spec.loadable_path}) ...")
    plugin = load_plugin(str(spec.loadable_path))
    raw_len = len(plugin.raw_state)
    console.print(f"Loaded successfully. Raw VST3 binary state size: [cyan]{raw_len} bytes[/cyan]")

    table = Table(title=f"Exposed Parameters — {spec.name}")
    table.add_column("Parameter Name", style="cyan")
    table.add_column("Current Value", style="green")
    table.add_column("Range / Valid Values")

    for name, param in plugin.parameters.items():
        val = getattr(plugin, name, None)
        valid = getattr(param, "valid_values", None)
        range_info = str(valid[:5]) if valid else str(getattr(param, "range", ""))
        table.add_row(name, str(val), range_info)

    console.print(table)


@app.command("build-references")
def build_references(
    references_dir: Path = typer.Option(PROJECT_ROOT / "data" / "references", help="References root"),
    inbox_dir: Path = typer.Option(PROJECT_ROOT / "youtube_references", help="Optional muse inbox root to quality-gate & include"),
    cache_dir: Path = typer.Option(PROJECT_ROOT / "data" / "processed" / "roformer_stems", help="BS-RoFormer stem cache"),
    out_json: Path = typer.Option(DEFAULT_PROFILES_JSON, help="Output target profiles JSON"),
) -> None:
    """Separate all curated + quality-gated inbox pro reference tracks on GPU via BS-RoFormer and build target profiles."""
    from mix_automation.analysis.reference_extractor import build_all_reference_profiles

    extra_roots = [inbox_dir] if inbox_dir.exists() else None
    res = build_all_reference_profiles(references_dir, cache_dir, out_json, extra_inbox_roots=extra_roots)
    console.print(f"[green]Built psychoacoustic target profiles from {res['num_reference_songs']} songs -> {out_json}[/green]")


@app.command("mix")
def mix_song(
    song_folder: Path = typer.Argument(..., exists=True, help="Raw song folder containing stems/ (or flat WAVs) and off-vocal"),
    style: str = typer.Option(
        "auto",
        "--style",
        "-s",
        help="Target style: auto (44-song acoustic k-NN) | consensus | azusagawa | toy | saburou | aokiyo | aoru | hoshirubesho | nanadakiima | sou",
    ),
    taste: str = typer.Option(
        "azusagawa",
        "--taste",
        help="Preferred MIXer aesthetic blended onto acoustic k-NN (pass 'none' for pure 44-song k-NN)",
    ),
    taste_weight: float = typer.Option(
        0.25,
        "--taste-weight",
        help="Blend ratio (0.0-0.85) for --taste when enabled",
    ),
    auto_pitch: bool = typer.Option(
        False,
        "--auto-pitch",
        help="Force smooth Praat PitchTier micro-tuning even if stem appears Melodyne-edited",
    ),
    auto_align: bool = typer.Option(
        False,
        "--auto-align",
        help="Opt-in safe phrase timing slip alignment on untuned raw vocals (OFF by default to protect groove)",
    ),
    enable_nectar_mix: bool = typer.Option(
        False,
        "--enable-nectar-mix",
        help="Force mixing synthetic Nectar 3 harmonies into 2mix even when user recorded real harmonies",
    ),
    template_song: Path = typer.Option(DEFAULT_TEMPLATE_SONG, "--template", "-t", help="Base Studio One 5 .song"),
    profiles_json: Path = typer.Option(DEFAULT_PROFILES_JSON, "--profiles", "-p", help="Target profiles JSON"),
    output_root: Path = typer.Option(PROJECT_ROOT / "data" / "outputs", "--out-dir", "-o", help="Output directory"),
    engine: str = typer.Option(
        "v7",
        "--engine",
        help="Mixing engine: v7 (current, default) | v8 (under construction, see docs/V8_PLAN.md)",
    ),
    export_debug_stems: bool = typer.Option(
        False,
        "--export-debug-stems",
        help="v7: also write lead dry / wet returns / harmony bus / aux layers to <out>/debug_stems (analysis only; audio unchanged)",
    ),
    v8_knobs: Annotated[
        Path | None,
        typer.Option("--v8-knobs", help="v8: JSON file overriding knob values (names/ranges: mix_automation/v8/knobs.py)"),
    ] = None,
) -> None:
    """Run full v6 ultra-precision automated vocal mixing (Stages ①-④) and generate true 1:1 Studio One 5 .song + Master WAV."""
    engine = engine.strip().lower()
    if engine not in ENGINES:
        console.print(f"[red]Unknown --engine '{engine}'. Choose one of: {', '.join(ENGINES)}[/red]")
        raise typer.Exit(code=2)

    if engine == "v8":
        from mix_automation.v8 import run_v8_automix_for_song

        v8_report = run_v8_automix_for_song(
            song_folder=song_folder,
            output_root=output_root,
            knobs_json=v8_knobs,
        )
        m = v8_report["metrics"]
        console.print("\n[bold green]v8 (R3) Auto-MIX Complete![/bold green]")
        console.print(f"  Output dir : [cyan]{v8_report['output_dir']}[/cyan]")
        console.print(
            f"  Metrics    : voc-inst {m['voc_minus_inst_lu']} LU | side/mid {m['vocal_side_mid_db']} dB | "
            f"master {m['master_lufs']} LUFS | tempo {v8_report['tempo_bpm']} BPM"
        )
        return

    from mix_automation.pipeline import run_full_automix_for_song

    report = run_full_automix_for_song(
        song_folder=song_folder,
        template_song_path=template_song,
        profiles_json_path=profiles_json,
        output_root=output_root,
        style=style,
        force_auto_pitch=auto_pitch,
        enable_auto_align=auto_align,
        taste_style=taste,
        taste_weight=taste_weight,
        enable_nectar_in_mix=enable_nectar_mix,
        export_debug_stems=export_debug_stems,
    )
    console.print("\n[bold green]v7 Auto-MIX Complete![/bold green]")
    console.print(f"  Archetype & Target   : [magenta]{report['optimized_chain_params']['archetype_chain']} -> {report['resolved_target_profile']}[/magenta]")
    console.print(f"  Studio One 5 Session : [cyan]{report['studio_one_song']}[/cyan] ({len(report.get('studio_one_injected_presets', []))} native VST3 presets)")
    console.print(f"  Mastered 2mix WAV    : [cyan]{report['master_wav']}[/cyan]")
    console.print(f"  Processed Vocal Bus  : [cyan]{report['vocal_bus_wav']}[/cyan]")
    if report.get("director_midi_guide"):
        console.print(f"  Director MIDI Guide  : [cyan]{report['director_midi_guide']}[/cyan]\n")


@app.command("propagate-anchor")
def propagate_anchor(
    song_path: Path = typer.Argument(..., exists=True, help="User-adjusted Studio One 5 .song file"),
    anchor_index: int = typer.Option(0, "--anchor", "-a", help="0-based section index tweaked by the user"),
) -> None:
    """Propagate user's manual tweak on one anchor section across all sections in a Studio One 5 .song."""
    from mix_automation.pipeline import repropagate_anchor_in_song

    out_song = repropagate_anchor_in_song(song_path, anchor_event_index=anchor_index)
    console.print(f"[green]Propagated anchor balance across all sections -> {out_song}[/green]")


@app.command("analyze-pair")
def analyze_pair(
    vocal_path: Path = typer.Argument(..., exists=True, help="Vocal stem WAV/FLAC"),
    inst_path: Path = typer.Argument(..., exists=True, help="Instrumental (off-vocal) WAV/FLAC"),
    output_json: Path | None = typer.Option(None, "--out", "-o", help="Optional JSON output path"),
) -> None:
    """Compute the v2 song-invariant relative vocal-to-instrumental acoustic profile."""
    vocal, sr_v = sf.read(str(vocal_path), always_2d=True)
    inst, sr_i = sf.read(str(inst_path), always_2d=True)
    if sr_v != sr_i:
        console.print(f"[red]Sample rate mismatch: vocal={sr_v}Hz vs instrumental={sr_i}Hz.[/red]")
        raise typer.Exit(code=1)

    profile = compute_section_relative_profile(vocal.T, inst.T, sr=sr_v)
    formatted = json.dumps(profile.to_dict(), indent=2, ensure_ascii=False)
    console.print_json(formatted)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(formatted, encoding="utf-8")


@app.command("inspect-song")
def inspect_song_cmd(
    song_path: Path = typer.Argument(..., exists=True, help="Studio One 5 .song file"),
) -> None:
    """Inspect XML structure, preset chunks, and media layout inside a Studio One 5 .song file."""
    summary = inspect_studio_one_song(song_path)
    console.print(f"\n[bold]Song Archive:[/bold] {summary.song_path}")
    console.print(f"[cyan]XML/XStream entries ({len(summary.xml_entries)}):[/cyan] {summary.xml_entries}")
    console.print(f"[magenta]Preset entries ({len(summary.preset_entries)}):[/magenta] {summary.preset_entries}")
    console.print(f"[green]Media entries ({len(summary.media_entries)}):[/green] {summary.media_entries}")
    console.print(f"[yellow]Detected Devices:[/yellow] {summary.detected_plugin_names}\n")


@app.command("learn-preferences")
def learn_preferences(
    auto_song: Path = typer.Argument(..., exists=True, help="Auto-generated Studio One 5 .song file"),
    user_song: Path = typer.Argument(..., exists=True, help="User-adjusted Studio One 5 .song file"),
    offsets: Path = typer.Option(
        DEFAULT_PROFILES_JSON.parent / "user_preference_offsets.json",
        "--offsets",
        "-o",
        help="Path to save/update bounded EMA user_preference_offsets.json",
    ),
    alpha: float = typer.Option(0.35, "--alpha", help="EMA learning rate (0.05-1.0)"),
) -> None:
    """Learn persistent user mixing preference offsets from DAW tweaks between auto & user .song files."""
    from mix_automation.analysis.preference_learner import update_user_preference_offsets

    record = update_user_preference_offsets(
        auto_song_path=auto_song,
        user_song_path=user_song,
        offsets_json_path=offsets,
        alpha=alpha,
    )

    table = Table(title=f"Learned User Preference Offsets (Sessions: {record['num_sessions']}, alpha={record['alpha']})")
    table.add_column("Target Dimension", style="cyan")
    table.add_column("EMA Offset (dB)", justify="right", style="green")
    table.add_column("Last Session Delta (dB)", justify="right", style="magenta")

    last_d = record.get("last_session", {}).get("extracted_deltas", {})
    last_sends = last_d.get("send_deltas_db", {})
    last_pq = last_d.get("persistent_q_mix_band_deltas_db", {})

    table.add_row(
        "Vocal Level (K-LUFS V/I)",
        f"{record['k_lufs_vi_offset_db']:+.3f}",
        f"{last_d.get('mainvocal_fader_delta_db', 0.0) + last_d.get('main_bus_fader_delta_db', 0.0) + last_d.get('mean_clip_gain_delta_db', 0.0):+.3f}",
    )
    table.add_row(
        "Spatial Wet-Tail",
        f"{record['wet_tail_offset_db']:+.3f}",
        f"{last_sends.get('plate_reverb', 0.0):+.3f} (Plate)",
    )
    table.add_row(
        "Harmony / Width (Side-Mid)",
        f"{record['vocal_side_mid_offset_db']:+.3f}",
        f"{last_d.get('harmvocal_fader_delta_db', 0.0):+.3f} (Harm)",
    )
    for band_name, val in record.get("intrinsic_band_offsets_db", {}).items():
        table.add_row(
            f"EQ Band: {band_name}",
            f"{val:+.3f}",
            f"{last_pq.get(band_name, 0.0):+.3f}",
        )

    console.print(table)
    console.print(f"[green]Updated preference offsets -> {offsets}[/green]\n")


@app.command("scan-references")
def scan_references(
    inbox: Path = typer.Option(
        PROJECT_ROOT / "youtube_references",
        "--inbox",
        "-i",
        help="Candidate reference inbox root (containing artist/song folders)",
    ),
) -> None:
    """Scan youtube_references inbox and enforce all 4 acoustic/alignment quality gates."""
    from mix_automation.analysis.reference_extractor import scan_youtube_references_inbox

    summary = scan_youtube_references_inbox(inbox_root=inbox)

    table = Table(title=f"Reference Inbox Quality Gate Audit — {inbox}")
    table.add_column("Candidate", style="cyan")
    table.add_column("Cover ID", style="dim")
    table.add_column("Archetype", style="magenta")
    table.add_column("Key Shift", justify="center")
    table.add_column("Paired Calib", justify="center")
    table.add_column("Gate", justify="center")
    table.add_column("Rejection Reasons")

    for cand in summary.get("candidates", []):
        gate_str = "[green]PASS[/green]" if cand["passed"] else "[red]FAIL[/red]"
        ks_str = f"{cand['key_shift_semitones']:+d}st" if cand["key_shifted"] else "0st"
        calib_str = "[green]YES[/green]" if cand["paired_calibration_eligible"] else "[yellow]NO[/yellow]"
        reasons = "; ".join(cand.get("rejection_reasons", [])) or "-"
        table.add_row(
            f"{cand['group']}/{cand['slug']}",
            str(cand.get("cover_id") or "-"),
            str(cand.get("backing_archetype", "-")),
            ks_str,
            calib_str,
            gate_str,
            reasons,
        )

    console.print(table)
    console.print(
        f"[bold]Evaluated {summary['total_candidates']} candidates: "
        f"[green]{summary['passed_count']} passed[/green], "
        f"[red]{summary['rejected_count']} rejected[/red], "
        f"key-shifted={summary['key_shifted_count']}, "
        f"paired-calib-eligible={summary['paired_calibration_eligible_count']}[/bold]\n"
    )


@app.command("benchmark-ab")
def benchmark_ab(
    outputs_dir: Path = typer.Option(
        PROJECT_ROOT / "data" / "outputs",
        "--outputs-dir",
        help="Completed AutoMIX outputs root",
    ),
    out_dir: Path = typer.Option(
        PROJECT_ROOT / "docs" / "reports" / "ab_blind_test",
        "--out-dir",
        help="Destination folder for -14 LUFS clips & interactive A/B console HTML",
    ),
) -> None:
    """Extract loudness-normalized (-14 LUFS) A/B listening clips, run Do-Not-Regress checks & build HTML console."""
    from mix_automation.analysis.ab_harness import (
        extract_loudness_normalized_ab_clips,
        generate_ab_blind_console_html,
    )

    summary = extract_loudness_normalized_ab_clips(
        outputs_root=outputs_dir,
        out_ab_dir=out_dir,
        raw_root=PROJECT_ROOT / "data" / "raw",
    )
    html_path = generate_ab_blind_console_html(summary, out_dir / "index.html")

    table = Table(title="Do-Not-Regress Multi-Song Acoustic & Studio One 5 Audit")
    table.add_column("Song / Run", style="cyan")
    table.add_column("K-LUFS Err", justify="center")
    table.add_column("No Scoop", justify="center")
    table.add_column("No Muffling", justify="center")
    table.add_column("No HF Sizzle", justify="center")
    table.add_column("S1 Sync+Presets", justify="center")
    table.add_column("Overall", justify="center")

    def _badge(ok: bool, detail: str) -> str:
        return f"[green]PASS[/green] ({detail})" if ok else f"[red]FAIL[/red] ({detail})"

    for s in summary.get("songs", []):
        hc = s.get("health_checks", {})
        hm = s.get("health_metrics", {})
        overall = "[green]PASS[/green]" if s.get("all_health_checks_passed") else "[red]FAIL[/red]"
        table.add_row(
            s["run_id"],
            _badge(bool(hc.get("klufs_error_db")), f"{hm.get('klufs_abs_error_db', 0.0):.2f}dB"),
            _badge(bool(hc.get("no_lower_mid_scoop")), f"{hm.get('achieved_body_warmth', 0.0):.1f}"),
            _badge(bool(hc.get("no_vocal_muffling", True)), f"sib={hm.get('achieved_sibilance', 0.0):.1f}"),
            _badge(bool(hc.get("no_hf_sizzle")), f"{hm.get('achieved_air_sheen', 0.0):.1f}dB"),
            _badge(bool(hc.get("studio_one_zero_juce_leak")), f"leak={hm.get('juce_leak_count', '?')},sync={hm.get('song_sync_locked', '?')}"),
            overall,
        )

    console.print(table)
    console.print(
        f"[bold]Audited {summary['songs_evaluated']} songs — "
        f"Passed: {summary['songs_passed']} / {summary['songs_evaluated']}[/bold]"
    )
    console.print(f"[cyan]Interactive Blind A/B Console -> {html_path}[/cyan]\n")


if __name__ == "__main__":
    app()
