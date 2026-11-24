"""Command-line interface for MIX Automation."""

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

app = typer.Typer(help="Reference-grade automated vocal mixing engine for Studio One 5.")
console = Console()

STUDIO_ONE_EXE = Path(r"C:\Program Files\PreSonus\Studio One 5\Studio One.exe")


@app.command()
def doctor() -> None:
    """Verify local Studio One 5 installation and registered VST3 plugins."""
    s1_status = "[green]FOUND[/green]" if STUDIO_ONE_EXE.exists() else "[red]MISSING[/red]"
    console.print(f"\n[bold]Studio One 5 Executable:[/bold] {STUDIO_ONE_EXE} ({s1_status})\n")

    table = Table(title="Registered VST3 Mixing Chain Audit")
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


@app.command("analyze-pair")
def analyze_pair(
    vocal_path: Path = typer.Argument(..., exists=True, help="Vocal stem WAV/FLAC"),
    inst_path: Path = typer.Argument(..., exists=True, help="Instrumental (off-vocal) WAV/FLAC"),
    output_json: Path | None = typer.Option(None, "--out", "-o", help="Optional JSON output path"),
) -> None:
    """Compute the song-invariant relative vocal-to-instrumental acoustic profile."""
    vocal, sr_v = sf.read(str(vocal_path), always_2d=False)
    inst, sr_i = sf.read(str(inst_path), always_2d=False)
    if sr_v != sr_i:
        console.print(
            f"[red]Sample rate mismatch: vocal={sr_v}Hz vs instrumental={sr_i}Hz. Resample first.[/red]"
        )
        raise typer.Exit(code=1)

    profile = compute_section_relative_profile(vocal, inst, sr=sr_v)
    payload = profile.to_dict()
    formatted = json.dumps(payload, indent=2, ensure_ascii=False)
    console.print_json(formatted)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(formatted, encoding="utf-8")
        console.print(f"[green]Saved profile to {output_json}[/green]")


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
