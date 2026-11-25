"""v2 Command-line interface for MIX Automation."""

from __future__ import annotations

import json
from pathlib import Path

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
    cache_dir: Path = typer.Option(PROJECT_ROOT / "data" / "processed" / "roformer_stems", help="BS-RoFormer stem cache"),
    out_json: Path = typer.Option(DEFAULT_PROFILES_JSON, help="Output target profiles JSON"),
) -> None:
    """Separate all 35 pro reference tracks on GPU via BS-RoFormer and build v2 stratified profiles."""
    from mix_automation.analysis.reference_extractor import build_all_reference_profiles

    res = build_all_reference_profiles(references_dir, cache_dir, out_json)
    console.print(f"[green]Built v3 psychoacoustic target profiles from {res['num_reference_songs']} songs -> {out_json}[/green]")


@app.command("mix")
def mix_song(
    song_folder: Path = typer.Argument(..., exists=True, help="Raw song folder containing stems/ and off-vocal"),
    style: str = typer.Option(
        "auto",
        "--style",
        "-s",
        help="Target style: auto (44-song acoustic k-NN) | consensus | azusagawa | toy | saburou | aokiyo | aoru | hoshirubesho | nanadakiima | sou",
    ),
    taste: str | None = typer.Option(
        None,
        "--taste",
        help="Optional preferred MIXer aesthetic to blend onto acoustic k-NN (e.g. azusagawa, sou, aoru, toy)",
    ),
    taste_weight: float = typer.Option(
        0.35,
        "--taste-weight",
        help="Blend ratio (0.0-0.85) for --taste when specified",
    ),
    auto_pitch: bool = typer.Option(
        False,
        "--auto-pitch",
        help="Force Praat TD-PSOLA micro-pitch tuning even if stem appears Melodyne-edited",
    ),
    template_song: Path = typer.Option(DEFAULT_TEMPLATE_SONG, "--template", "-t", help="Base Studio One 5 .song"),
    profiles_json: Path = typer.Option(DEFAULT_PROFILES_JSON, "--profiles", "-p", help="Target profiles JSON"),
    output_root: Path = typer.Option(PROJECT_ROOT / "data" / "outputs", "--out-dir", "-o", help="Output directory"),
) -> None:
    """Run full v4 automated vocal mixing (Stages ①-④) and generate true 1:1 Studio One 5 .song + Master WAV."""
    from mix_automation.pipeline import run_full_automix_for_song

    report = run_full_automix_for_song(
        song_folder=song_folder,
        template_song_path=template_song,
        profiles_json_path=profiles_json,
        output_root=output_root,
        style=style,
        force_auto_pitch=auto_pitch,
        taste_style=taste,
        taste_weight=taste_weight if taste else 0.0,
    )
    console.print("\n[bold green]v4 Auto-MIX Complete![/bold green]")
    console.print(f"  Archetype & Target   : [magenta]{report['optimized_chain_params']['archetype_chain']} -> {report['resolved_target_profile']}[/magenta]")
    console.print(f"  Studio One 5 Session : [cyan]{report['studio_one_song']}[/cyan] ({len(report.get('studio_one_injected_presets', []))} VST3 presets)")
    console.print(f"  Mastered 2mix WAV    : [cyan]{report['master_wav']}[/cyan]")
    console.print(f"  Processed Vocal Bus  : [cyan]{report['vocal_bus_wav']}[/cyan]\n")


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


if __name__ == "__main__":
    app()
