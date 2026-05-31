from __future__ import annotations

import sys

import questionary
from questionary import Style
from rich.console import Console

from maic.core.model_registry import CURATED_MODELS, CuratedModel, list_installed_models

STYLE = Style(
    [
        ("qmark", "fg:#00d7ff bold"),
        ("question", "bold"),
        ("answer", "fg:#00d7ff bold"),
        ("pointer", "fg:#00d7ff bold"),
        ("highlighted", "fg:#00d7ff bold"),
        ("selected", "fg:#00d7ff"),
        ("separator", "fg:#555555"),
        ("instruction", "fg:#555555"),
    ]
)

console = Console()

_MAIN_CHOICES = [
    questionary.Choice("▶▶  Start inference server", value="run"),
    questionary.Choice("■■  Model manager", value="models"),
    questionary.Choice("⚙⚙  Setup integrations", value="setup"),
    questionary.Choice("◆◆  System health check", value="doctor"),
    questionary.Choice("××  Exit", value="exit"),
]


def _installed_model_choices(include_back: bool = True) -> list[questionary.Choice]:
    """Choices for locally installed models, with optional Back entry at the end."""
    installed = list_installed_models()
    alias_map: dict[str, CuratedModel] = {m["hf_id"]: m for m in CURATED_MODELS}
    choices: list[questionary.Choice] = []
    for hf_id in installed:
        cm = alias_map.get(hf_id)
        alias = cm["alias"] if cm else hf_id.split("/")[-1]
        size = f"{cm['size_gb']} GB" if cm else "?"
        tools = "[TOOLS]" if cm and cm["tool_calling"] else ""
        label = f"{alias:<22} {size:<8} {tools}"
        choices.append(questionary.Choice(title=label, value=hf_id))
    if include_back:
        choices.append(questionary.Choice(title="◄◄  Back", value=None))
    return choices


def _not_installed_model_choices() -> list[questionary.Choice]:
    """Choices for curated models not yet downloaded, always ending with Back."""
    installed = set(list_installed_models())
    choices: list[questionary.Choice] = []
    for m in CURATED_MODELS:
        if m["hf_id"] not in installed:
            size = f"{m['size_gb']} GB"
            tools = "[TOOLS]" if m["tool_calling"] else "      "
            desc = m["description"][:42]
            label = f"{m['alias']:<22} {size:<8} {tools:<10} {desc}"
            choices.append(questionary.Choice(title=label, value=m["hf_id"]))
    choices.append(questionary.Choice(title="◄◄  Back", value=None))
    return choices


def _flow_run_server() -> None:
    """Interactive flow: pick an installed model and launch the server."""
    from maic.cli import _launch_server  # lazy — cli.py imports tui lazily too

    choices = _installed_model_choices(include_back=True)
    no_models = all(c.value is None for c in choices)
    if no_models:
        console.print("\n[yellow]╭─ No models installed ─╮[/yellow]")
        console.print("[yellow]│[/yellow]  Go to [cyan]Model manager → Download[/cyan] first.  [yellow]│[/yellow]")
        console.print("[yellow]╰──────────────────────────╯[/yellow]\n")
        return

    selected = questionary.select(
        "Select a model to serve:", choices=choices, style=STYLE
    ).ask()
    if selected is None:
        return

    port_raw = questionary.text("Port to listen on:", default="8001", style=STYLE).ask()
    if port_raw is None:
        return
    try:
        port = int(port_raw)
    except ValueError:
        console.print("[red]Invalid port number.[/red]")
        return

    ok = questionary.confirm(f"Start server on port {port}?", default=True, style=STYLE).ask()
    if not ok:
        return

    console.print()
    _launch_server(hf_id=selected, host="127.0.0.1", port=port)
    # os.execvp replaces the process; if we reach here the strategy fell through


def _flow_manage_models() -> None:
    """Interactive flow: list all models or download a new one."""
    import typer

    while True:
        action = questionary.select(
            "Model management:",
            choices=[
                questionary.Choice("▼▼  List all models", value="list"),
                questionary.Choice("↓↓  Download a model", value="download"),
                questionary.Choice("◄◄  Back", value=None),
            ],
            style=STYLE,
        ).ask()

        if action is None:
            return

        if action == "list":
            console.print()
            console.print("[cyan]─── Available Models ───[/cyan]")
            console.print()
            try:
                from maic.cli import models_list

                models_list(tool_calling=False, max_ram=None)
            except typer.Exit:
                pass
            console.print()

        elif action == "download":
            choices = _not_installed_model_choices()
            all_installed = all(c.value is None for c in choices)
            if all_installed:
                console.print("\n[green]✓ All curated models are already installed![/green]\n")
                continue

            selected = questionary.select(
                "Select a model to download:", choices=choices, style=STYLE
            ).ask()
            if selected is None:
                continue

            ok = questionary.confirm(f"Download {selected}?", default=True, style=STYLE).ask()
            if not ok:
                continue

            console.print()
            console.print("[cyan]─── Downloading ───[/cyan]")
            console.print()
            try:
                from maic.cli import models_pull

                models_pull(alias_or_id=selected)
            except typer.Exit:
                pass
            console.print()


def _flow_setup() -> None:
    """Interactive flow: configure integrations (OpenCode)."""
    import typer

    while True:
        action = questionary.select(
            "Setup integrations:",
            choices=[
                questionary.Choice("◈◈  Configure OpenCode", value="opencode"),
                questionary.Choice("◄◄  Back", value=None),
            ],
            style=STYLE,
        ).ask()

        if action is None:
            return

        if action == "opencode":
            port_raw = questionary.text(
                "Port Maic listens on:", default="8001", style=STYLE
            ).ask()
            if port_raw is None:
                continue
            try:
                port = int(port_raw)
            except ValueError:
                console.print("[red]✗ Invalid port.[/red]")
                continue

            console.print()
            console.print("[cyan]─── Configuring OpenCode ───[/cyan]")
            console.print()
            try:
                from maic.cli import setup_opencode

                setup_opencode(port=port)
            except typer.Exit:
                pass
            console.print()


def _flow_doctor() -> None:
    """Run the system health check and display the rich table."""
    import typer

    console.print()
    console.print("[cyan]─── System Health Check ───[/cyan]")
    console.print()
    try:
        from maic.cli import doctor

        doctor(port=8001)
    except typer.Exit:
        pass
    console.print()


def run_tui() -> None:
    """Launch the interactive Maic TUI. Loops until the user selects Exit."""
    try:
        while True:
            action = questionary.select(
                "What do you want to do?",
                choices=_MAIN_CHOICES,
                style=STYLE,
            ).ask()

            if action is None or action == "exit":
                console.print("[dim]▬▬▬ Goodbye! ▬▬▬[/dim]")
                return

            if action == "run":
                _flow_run_server()
            elif action == "models":
                _flow_manage_models()
            elif action == "setup":
                _flow_setup()
            elif action == "doctor":
                _flow_doctor()

    except KeyboardInterrupt:
        console.print("\n[dim]▬▬▬ Interrupted ▬▬▬[/dim]")
        sys.exit(0)
    except OSError:
        # stdin is not a terminal (e.g. piped input) — exit quietly
        sys.exit(0)
