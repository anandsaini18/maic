from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import typer
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from maic.core.model_registry import (
    CURATED_MODELS,
    CuratedModel,
    list_installed_models,
    resolve_hf_id,
)

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("maic")
except Exception:
    _VERSION = "1.0.0"

_ASCII_ART = """\
███╗   ███╗ █████╗ ██╗ ██████╗
████╗ ████║██╔══██╗██║██╔════╝
██╔████╔██║███████║██║██║
██║╚██╔╝██║██╔══██║██║██║
██║ ╚═╝ ██║██║  ██║██║╚██████╗
╚═╝     ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝"""

# Disable built-in --help at the top level so the banner callback owns it.
# Sub-commands retain their own --help via their own Click contexts.
app = typer.Typer(
    name="maic",
    help="Maic — local LLM inference for Apple Silicon.",
    no_args_is_help=False,
    context_settings={"help_option_names": []},
)
_HELP_CTX = {"help_option_names": ["--help", "-h"]}

models_app = typer.Typer(help="Model management commands.", context_settings=_HELP_CTX)
app.add_typer(models_app, name="models")

console = Console()


# ── banner ───────────────────────────────────────────────────────────────────


def _print_banner() -> None:
    content = Text()
    content.append(_ASCII_ART + "\n", style="bold white")
    content.append(f"\nv{_VERSION}  ·  Local LLM inference for Apple Silicon\n", style="dim white")
    content.append("OpenAI-compatible · MLX-powered · OpenCode-ready", style="dim cyan")
    console.print(Panel(Align.center(content), border_style="bright_black", padding=(1, 4)))
    console.print()

    # Quick-start suggestions
    table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    table.add_column("cmd", style="cyan", no_wrap=True)
    table.add_column("desc", style="dim white")
    steps = [
        ("maic run",                       "Start the inference server (interactive model picker)"),
        ("maic models list",               "Browse curated models with RAM requirements"),
        ("maic models list --tool-calling","Only show tool-capable models"),
        ("maic models pull qwen-coder-7b", "Download a model by alias"),
        ("maic setup opencode",            "Auto-configure OpenCode to use Maic"),
        ("maic doctor",                    "Check system health and installed models"),
    ]
    for cmd, desc in steps:
        table.add_row(cmd, desc)

    console.print("  [bold white]Quick start[/bold white]\n")
    console.print(table)
    console.print()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"maic v{_VERSION}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    help: bool = typer.Option(False, "--help", "-h", help="Show this message and exit."),
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """Maic — local LLM inference for Apple Silicon."""
    if help:
        _print_banner()
        typer.echo(ctx.get_help())
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _print_banner()
        from maic.tui import run_tui

        run_tui()
        raise typer.Exit()


# ── helpers ──────────────────────────────────────────────────────────────────


def _available_ram_gb() -> float:
    """Return total system RAM in GB (best-effort)."""
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024**3)
    except Exception:
        return 0.0


def _opencode_config_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "opencode" / "opencode.json"


def _is_server_running(port: int) -> bool:
    try:
        import urllib.request

        urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def _model_downloaded(hf_id: str) -> bool:
    """Check if a model exists in the HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_id == hf_id:
                return True
    except Exception:
        pass
    return False


# ── maic run ─────────────────────────────────────────────────────────────────


def _pick_model(installed: list[str]) -> str | None:
    """Show a model selection prompt and return the chosen HF ID, or None to cancel."""
    alias_map: dict[str, CuratedModel] = {m["hf_id"]: m for m in CURATED_MODELS}

    try:
        import questionary

        from maic.tui import STYLE

        choices = []
        for hf_id in installed:
            cm: CuratedModel | None = alias_map.get(hf_id)
            alias = cm["alias"] if cm else hf_id.split("/")[-1]
            size = f"{cm['size_gb']} GB" if cm else "?"
            tools = "⚙ tools" if cm and cm["tool_calling"] else ""
            label = f"{alias:<22} {size:<8} {tools}"
            choices.append(questionary.Choice(title=label, value=hf_id))

        return cast(
            "str | None",
            questionary.select("Select a model:", choices=choices, style=STYLE).ask(),
        )

    except ImportError:
        # Fallback: numbered text menu (questionary unavailable)
        console.print("\n[bold]Installed models:[/bold]")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Num", style="cyan", justify="right")
        table.add_column("Model", style="white")
        table.add_column("Alias", style="dim")

        for i, hf_id in enumerate(installed, 1):
            entry = alias_map.get(hf_id)
            alias = entry["alias"] if entry else ""
            table.add_row(str(i), hf_id, f"({alias})" if alias else "")

        console.print(table)
        console.print()

        while True:
            choice = input("Select model [1]: ").strip() or "1"
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(installed):
                    return installed[idx]
            except ValueError:
                pass
            console.print(f"[red]Enter a number between 1 and {len(installed)}[/red]")


def _launch_server(hf_id: str | None, host: str, port: int) -> None:
    """Launch the MLX inference server using the best available strategy."""
    # Resolve to local path if the model is stored in ~/models/ (Maic's download dir).
    # mlx_lm.server looks in HuggingFace cache by default; we bypass that by passing
    # the absolute local path when available.
    if hf_id:
        local_dir = Path.home() / "models" / hf_id.replace("/", "--")
        if local_dir.exists() and any(local_dir.glob("*.safetensors")):
            hf_id = str(local_dir)

    # Strategy 1: mlx-openai-server subprocess (preferred)
    # Skip if the binary is installed inside our own venv — it collides with Maic's `app` package.
    mlx_server_bin = shutil.which("mlx_openai_server") or shutil.which("mlx-openai-server")
    _in_our_venv = mlx_server_bin and str(Path(sys.executable).parent) in (mlx_server_bin or "")
    if mlx_server_bin and not _in_our_venv:
        cmd = [mlx_server_bin, "--host", host, "--port", str(port)]
        if hf_id:
            cmd += ["--model", hf_id]
        console.print(f"[green]→[/green] Launching mlx-openai-server on http://{host}:{port}")
        if hf_id:
            console.print(f"  Model: [cyan]{hf_id}[/cyan]")
        try:
            os.execvp(cmd[0], cmd)  # replace process
        except Exception as exc:
            console.print(f"[yellow]⚠ exec failed ({exc}), trying subprocess...[/yellow]")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as sub_exc:
                console.print(f"[red]✗ Server launch failed: {sub_exc}[/red]")
                raise typer.Exit(1) from sub_exc
        return

    # Strategy 2: mlx_lm.server (if mlx-lm is installed)
    try:
        import mlx_lm  # noqa: F401  # type: ignore[import]

        cmd = [sys.executable, "-m", "mlx_lm", "server", "--host", host, "--port", str(port)]
        if hf_id:
            cmd += ["--model", hf_id]
        console.print(f"[green]→[/green] Launching mlx_lm.server on http://{host}:{port}")
        if hf_id:
            console.print(f"  Model: [cyan]{hf_id}[/cyan]")
        # Skip HuggingFace revision check on startup — model is already local.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            os.execvp(cmd[0], cmd)
        except Exception:
            subprocess.run(cmd, check=True)
        return
    except ImportError:
        pass

    # Strategy 3: Maic FastAPI fallback
    console.print(
        "[yellow]⚠ mlx-openai-server not found — falling back to Maic FastAPI server.[/yellow]"
    )
    console.print(
        "  Install the full stack: [bold]pip install maic\\[mlx][/bold] or "
        "[bold]pip install mlx-openai-server[/bold]"
    )
    import uvicorn

    if hf_id:
        os.environ["DEFAULT_MODEL"] = hf_id
    uvicorn.run(
        "maic.api.routes:router",
        host=host,
        port=port,
        log_level="info",
    )


@app.command(context_settings=_HELP_CTX)
def run(
    model: str | None = typer.Option(None, "--model", "-m", help="Model alias or HF repo ID"),
    port: int = typer.Option(8001, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
) -> None:
    """Start the Maic inference server."""
    hf_id: str | None = None
    if model:
        hf_id = resolve_hf_id(model)
    else:
        installed = list_installed_models()
        if not installed:
            console.print("[yellow]No models installed.[/yellow]")
            console.print("  Run: [cyan]maic models pull <alias>[/cyan]")
            console.print("  See: [cyan]maic models list[/cyan]")
            raise typer.Exit(0)
        elif len(installed) == 1:
            hf_id = installed[0]
            console.print(f"[dim]Auto-selected: {hf_id}[/dim]")
        else:
            picked = _pick_model(installed)
            if picked is None:
                raise typer.Exit(0)
            hf_id = picked

    _launch_server(hf_id=hf_id, host=host, port=port)


# ── maic models list ──────────────────────────────────────────────────────────


@models_app.command("list")
def models_list(
    tool_calling: bool = typer.Option(False, "--tool-calling", help="Show only tool-calling models"),
    max_ram: int | None = typer.Option(None, "--max-ram", help="Filter by max RAM (GB)"),
) -> None:
    """Browse curated models for your Mac."""
    ram_gb = _available_ram_gb()
    detected = f"{ram_gb:.0f} GB" if ram_gb > 0 else "unknown"
    installed_set = set(list_installed_models())

    table = Table(title=f"Maic model registry  (detected RAM: {detected})", show_lines=True)
    table.add_column("", justify="center", no_wrap=True)  # installed indicator
    table.add_column("Alias", style="cyan", no_wrap=True)
    table.add_column("HF ID", style="dim")
    table.add_column("Size", justify="right")
    table.add_column("Min RAM", justify="right")
    table.add_column("Tools", justify="center")
    table.add_column("Description")

    for m in CURATED_MODELS:
        if tool_calling and not m["tool_calling"]:
            continue
        if max_ram is not None and m["min_ram_gb"] > max_ram:
            continue

        fits = ram_gb == 0 or ram_gb >= m["min_ram_gb"]

        def cell(text: str, _d: bool = fits) -> str:
            return text if _d else f"[dim]{text}[/dim]"

        tool_icon = "✓" if m["tool_calling"] else "✗"
        tool_style = "green" if m["tool_calling"] else "red"
        dl_icon = "[green]↓[/green]" if m["hf_id"] in installed_set else ""
        table.add_row(
            dl_icon,
            cell(m["alias"]),
            cell(m["hf_id"]),
            cell(f"{m['size_gb']} GB"),
            cell(f"{m['min_ram_gb']} GB"),
            f"[{tool_style}]{tool_icon}[/{tool_style}]" if fits else f"[dim]{tool_icon}[/dim]",
            cell(m["description"]),
        )

    console.print(table)
    if ram_gb > 0:
        console.print(
            f"[dim]Dimmed rows require more RAM than detected ({detected}). "
            "They may still work on unified memory Macs.[/dim]"
        )
    if installed_set:
        console.print(f"[dim][green]↓[/green] = installed locally ({len(installed_set)} model(s))[/dim]")


# ── maic models pull ──────────────────────────────────────────────────────────


@models_app.command("pull")
def models_pull(
    alias_or_id: str = typer.Argument(..., help="Model alias (e.g. qwen-coder-7b) or HF repo ID"),
    workers: int = typer.Option(4, "--workers", "-j", help="Parallel workers (fallback path only)"),
) -> None:
    """Download a model by alias or HuggingFace ID (fast parallel download)."""
    from maic.core.downloader import ensure_hf_cache_symlink, fast_download

    hf_id = resolve_hf_id(alias_or_id)
    alias = alias_or_id  # keep the human-friendly name for display

    # Detect whether hf-transfer (Layer 1) is available so we can pick the right UX.
    _has_hf_transfer = False
    try:
        import hf_transfer  # noqa: F401

        _has_hf_transfer = True
    except ImportError:
        pass

    console.print(
        f"[green]→[/green] Downloading [cyan]{hf_id}[/cyan] "
        + ("[dim](hf-transfer)[/dim]" if _has_hf_transfer else f"[dim]({workers} workers)[/dim]")
    )

    try:
        if _has_hf_transfer:
            # hf-transfer is opaque; show a spinner while it runs.
            with console.status(f"[cyan]Downloading {alias}…[/cyan]"):
                path = fast_download(hf_id, max_workers=workers)
        else:
            # Pure-Python fallback: show per-file progress.
            from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

            with Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(f"[cyan]{alias}[/cyan]", total=None)

                def _cb(done: int, total: int, filename: str) -> None:
                    progress.update(task_id, completed=done, total=total)

                path = fast_download(hf_id, max_workers=workers, progress_callback=_cb)

        ensure_hf_cache_symlink(hf_id, path)
        console.print(f"[green]✓[/green] Saved to {path}")
    except Exception as exc:
        console.print(f"[red]✗ Download failed: {exc}[/red]")
        raise typer.Exit(1) from exc


# ── maic setup opencode ───────────────────────────────────────────────────────

setup_app = typer.Typer(help="Setup integrations.", context_settings=_HELP_CTX)
app.add_typer(setup_app, name="setup")


@setup_app.command("opencode")
def setup_opencode(
    port: int = typer.Option(8001, "--port", "-p", help="Port Maic listens on"),
) -> None:
    """Auto-configure OpenCode to use Maic."""
    config_path = _opencode_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config (deep-merge)
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            backup = config_path.with_suffix(".json.bak")
            config_path.rename(backup)
            console.print(f"[yellow]⚠ Malformed JSON — backed up to {backup}[/yellow]")

    # OpenCode uses provider as a dict keyed by name
    providers: dict[str, Any] = existing.get("provider", {})
    providers["maic"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Maic (local MLX)",
        "options": {
            "baseURL": f"http://localhost:{port}/v1",
            "apiKey": "maic-local",
        },
    }
    existing.setdefault("$schema", "https://opencode.ai/config.json")
    existing["provider"] = providers

    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    console.print(f"[green]✓[/green] OpenCode config written to [cyan]{config_path}[/cyan]")
    console.print(
        f"  Maic provider: [dim]http://localhost:{port}/v1[/dim]  "
        "(restart OpenCode to pick up changes)"
    )


# ── maic doctor ───────────────────────────────────────────────────────────────


@app.command(context_settings=_HELP_CTX)
def doctor(
    port: int = typer.Option(8001, "--port", "-p", help="Port to check for running server"),
) -> None:
    """Check system health."""
    table = Table(title="Maic doctor", show_header=True, show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    def row(label: str, ok: bool, detail: str = "") -> None:
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(label, icon, detail)

    # Python version
    vi = sys.version_info
    py_ok = vi >= (3, 10)
    row("Python ≥ 3.10", py_ok, f"{vi.major}.{vi.minor}.{vi.micro}")

    # MLX — probe via subprocess to avoid crashing if no Metal GPU is present
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mlx.core; print('ok')"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        mlx_ok = result.returncode == 0 and "ok" in result.stdout
        row("MLX available", mlx_ok, "yes" if mlx_ok else "pip install maic[mlx]")
    except Exception:
        row("MLX available", False, "pip install maic[mlx]")

    # mlx-openai-server
    mlx_server_bin = shutil.which("mlx_openai_server") or shutil.which("mlx-openai-server")
    row(
        "mlx-openai-server",
        bool(mlx_server_bin),
        mlx_server_bin or "pip install mlx-openai-server",
    )

    # Server running
    running = _is_server_running(port)
    row("Server running", running, f"http://localhost:{port}/v1/models")

    # OpenCode config
    oc_path = _opencode_config_path()
    oc_ok = oc_path.exists()
    has_maic_provider = False
    if oc_ok:
        try:
            cfg = json.loads(oc_path.read_text())
            has_maic_provider = "maic" in cfg.get("provider", {})
        except Exception:
            pass
    row("OpenCode config", oc_ok, str(oc_path) if oc_ok else "run: maic setup opencode")
    row("OpenCode maic provider", has_maic_provider, "" if has_maic_provider else "run: maic setup opencode")

    # Installed models
    n_installed = len(list_installed_models())
    row(
        "Models installed",
        n_installed > 0,
        str(n_installed) if n_installed > 0 else "run: maic models pull <alias>",
    )

    # RAM
    ram = _available_ram_gb()
    row("RAM detected", ram > 0, f"{ram:.1f} GB" if ram > 0 else "psutil unavailable")

    console.print(table)

    # North-star hint
    if not running:
        console.print("\n[bold]Quick start:[/bold]")
        console.print("  [cyan]maic models pull qwen-coder-7b[/cyan]")
        console.print("  [cyan]maic run --model qwen-coder-7b[/cyan]")
        console.print("  [cyan]maic setup opencode[/cyan]")


if __name__ == "__main__":
    app()
