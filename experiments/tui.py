from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

try:
    import questionary

    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

from core.eval.config import EvaluationConfig

EXPERIMENTS_DIR = Path(__file__).parent
TRAIN_CONFIGS_DIR = EXPERIMENTS_DIR / "configs" / "train"
EVAL_CONFIGS_DIR = EXPERIMENTS_DIR / "configs" / "eval"

console = Console()
app = typer.Typer(add_completion=False, help="Interactive CLI Launcher for Training & Evaluation")


class _BackSentinel:
    """Sentinel class for handling Back navigation in menus."""

    pass


BACK = _BackSentinel()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Main interactive menu when invoked without subcommands."""
    if ctx.invoked_subcommand is not None:
        return

    while True:
        console.clear()
        if HAS_QUESTIONARY:
            choice = questionary.select(
                "Interactive Experiment Launcher",
                choices=[
                    questionary.Choice(title="1) Train", value="1"),
                    questionary.Choice(title="2) Evaluate", value="2"),
                    questionary.Choice(title="0) Exit", value="0"),
                ],
            ).ask()
        else:
            console.print(
                Panel(
                    "[bold cyan]Interactive Experiment Launcher[/bold cyan]\n"
                    "[dim]Select a workflow using number options[/dim]",
                    title="Menu",
                    expand=False,
                )
            )
            console.print("  [bold yellow]1[/bold yellow]) Train")
            console.print("  [bold yellow]2[/bold yellow]) Evaluate")
            console.print("  [bold yellow]0[/bold yellow]) Exit\n")
            choice = Prompt.ask("Select option", choices=["0", "1", "2"], default="1")

        if choice == "0" or choice is None:
            console.print("[yellow]Goodbye![/yellow]")
            break
        elif choice == "1":
            completed = train()
            if completed:
                _pause()
        elif choice == "2":
            completed = evaluate()
            if completed:
                _pause()


@app.command()
def train(
    device: str | None = typer.Option(
        None, "--device", help="Torch device override (e.g. cpu, cuda)"
    ),
) -> bool:
    """Interactively configure and run Training (with optional Evaluation). Returns True if completed."""
    console.print(Panel("[bold green]Workflow: Training (+ Optional Evaluation)[/bold green]"))

    train_config = _select_config_file("Select Training Config", TRAIN_CONFIGS_DIR)
    if not train_config:
        console.print("[yellow]Training cancelled.[/yellow]")
        return False

    # Check for matching eval config
    matched_eval = _resolve_matching_eval_config(train_config)
    eval_config: Path | None = None
    selected_settings: list[str] = []
    run_first_only: bool = False

    if matched_eval:
        rel_matched = matched_eval.relative_to(EVAL_CONFIGS_DIR)
        console.print(f"\n[bold cyan]Matched Evaluation Config Found:[/bold cyan] {rel_matched}")

        if HAS_QUESTIONARY:
            action_choice = questionary.select(
                "Run evaluation after training?",
                choices=[
                    questionary.Choice(title=f"Yes (use matched config: {rel_matched})", value="y"),
                    questionary.Choice(title="No (train only)", value="n"),
                    questionary.Choice(title="Choose a different evaluation config", value="c"),
                    questionary.Choice(title="x Cancel to Main Menu", value="cancel"),
                ],
            ).ask()
            action = action_choice or "cancel"
        else:
            action = Prompt.ask(
                "Run evaluation after training?",
                choices=["y", "n", "c", "cancel"],
                default="y",
                show_choices=True,
            )

        if action == "y":
            eval_config = matched_eval
        elif action == "c":
            eval_config = _select_config_file("Select Evaluation Config", EVAL_CONFIGS_DIR)
            if not eval_config:
                if _ask_confirm(
                    "Evaluation selection cancelled. Continue with Training only (no eval)?",
                    default=True,
                ):
                    eval_config = None
                else:
                    console.print("[yellow]Workflow cancelled.[/yellow]")
                    return False
        elif action == "cancel":
            console.print("[yellow]Workflow cancelled.[/yellow]")
            return False
    else:
        want_eval = _ask_confirm(
            "No exact matching eval config found. Would you like to select an evaluation config?",
            default=False,
        )
        if want_eval:
            eval_config = _select_config_file("Select Evaluation Config", EVAL_CONFIGS_DIR)
            if not eval_config and not _ask_confirm(
                "Evaluation selection cancelled. Continue with Training only (no eval)?",
                default=True,
            ):
                console.print("[yellow]Workflow cancelled.[/yellow]")
                return False

    # Configure eval settings if eval is requested
    if eval_config:
        available_settings = _extract_eval_settings(eval_config)
        if available_settings:
            selected_settings, run_first_only = _select_eval_settings(available_settings)

    # Display execution summary
    summary_table = Table(title="Execution Plan", show_header=True, header_style="bold magenta")
    summary_table.add_column("Step", style="cyan")
    summary_table.add_column("Details", style="white")

    rel_train = train_config.relative_to(EXPERIMENTS_DIR)
    summary_table.add_row(
        "1. Train", f"[green]python experiments/train.py --config {rel_train}[/green]"
    )

    if eval_config:
        rel_eval = eval_config.relative_to(EXPERIMENTS_DIR)
        eval_detail = f"[green]python experiments/eval.py --config {rel_eval}[/green]"
        if selected_settings:
            eval_detail += f" [yellow]--setting {' '.join(selected_settings)}[/yellow]"
        elif run_first_only:
            eval_detail += " [yellow]--first[/yellow]"
        summary_table.add_row("2. Eval", eval_detail)
    else:
        summary_table.add_row("2. Eval", "[dim]Disabled[/dim]")

    console.print("\n", summary_table)

    if not _ask_confirm("Proceed with execution?", default=True):
        console.print("[yellow]Execution cancelled.[/yellow]")
        return False

    # Execute Training
    cmd_train = [sys.executable, str(EXPERIMENTS_DIR / "train.py"), "--config", str(train_config)]
    if device:
        cmd_train.extend(["--device", device])

    console.print(f"\n[bold green]>>> Running Training ({train_config.stem})...[/bold green]")
    train_res = subprocess.run(cmd_train)

    if train_res.returncode != 0:
        console.print(
            "[bold red]Training failed with non-zero exit code. Skipping Evaluation.[/bold red]"
        )
        raise typer.Exit(code=train_res.returncode)

    console.print("[bold green]Training completed successfully![/bold green]")

    # Execute Evaluation if requested
    if eval_config:
        cmd_eval = [sys.executable, str(EXPERIMENTS_DIR / "eval.py"), "--config", str(eval_config)]
        if device:
            cmd_eval.extend(["--device", device])
        if run_first_only:
            cmd_eval.append("--first")
        elif selected_settings:
            for s in selected_settings:
                cmd_eval.extend(["--setting", s])

        console.print(f"\n[bold green]>>> Running Evaluation ({eval_config.stem})...[/bold green]")
        eval_res = subprocess.run(cmd_eval)
        if eval_res.returncode != 0:
            console.print("[bold red]Evaluation failed.[/bold red]")
            raise typer.Exit(code=eval_res.returncode)
        console.print("[bold green]Evaluation completed successfully![/bold green]")

    return True


@app.command(
    name="eval",
)
def evaluate(
    device: str | None = typer.Option(
        None, "--device", help="Torch device override (e.g. cpu, cuda)"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force recomputation of existing evaluation results"
    ),
) -> bool:
    """Interactively configure and run Evaluation. Returns True if completed."""
    console.print(Panel("[bold green]Workflow: Standalone Evaluation[/bold green]"))

    eval_config = _select_config_file("Select Evaluation Config", EVAL_CONFIGS_DIR)
    if not eval_config:
        console.print("[yellow]Evaluation cancelled.[/yellow]")
        return False

    available_settings = _extract_eval_settings(eval_config)
    selected_settings: list[str] = []
    run_first_only: bool = False

    if available_settings:
        selected_settings, run_first_only = _select_eval_settings(available_settings)

    rel_eval = eval_config.relative_to(EXPERIMENTS_DIR)
    cmd_eval = [sys.executable, str(EXPERIMENTS_DIR / "eval.py"), "--config", str(rel_eval)]
    if device:
        cmd_eval.extend(["--device", device])
    if force:
        cmd_eval.append("--force")
    if run_first_only:
        cmd_eval.append("--first")
    elif selected_settings:
        for s in selected_settings:
            cmd_eval.extend(["--setting", s])

    console.print(f"\n[bold cyan]Command to run:[/bold cyan] [green]{' '.join(cmd_eval)}[/green]")

    if not _ask_confirm("Proceed with evaluation?", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return False

    console.print(f"\n[bold green]>>> Running Evaluation ({eval_config.stem})...[/bold green]")
    eval_res = subprocess.run(cmd_eval)
    if eval_res.returncode != 0:
        console.print("[bold red]Evaluation failed.[/bold red]")
        raise typer.Exit(code=eval_res.returncode)
    console.print("[bold green]Evaluation completed successfully![/bold green]")
    return True


def _select_config_file(title: str, base_dir: Path) -> Path | None:
    """Two-stage config file selector: 1) Select problem directory, 2) Select config file."""
    if not base_dir.exists():
        console.print(f"[bold red]Base directory {base_dir} does not exist.[/bold red]")
        return None

    # Discover problem directories that contain config files
    prob_dirs = sorted(
        [
            d
            for d in base_dir.iterdir()
            if d.is_dir() and d.name not in ("problems", "search") and _get_yaml_files(d)
        ],
        key=lambda d: d.name,
    )

    if not prob_dirs:
        files = _get_yaml_files(base_dir)
        res = _display_file_menu(title, files, base_dir, allow_back=False)
        return res if isinstance(res, Path) else None

    while True:
        # Stage 1: Select Problem Directory
        if HAS_QUESTIONARY:
            prob_choices = [
                questionary.Choice(
                    title=f"📁 {d.name.upper()} ({len(_get_yaml_files(d))} configs)", value=d
                )
                for d in prob_dirs
            ]
            prob_choices.append(
                questionary.Choice(title="🌐 All problems (browse all configs)", value="ALL")
            )
            prob_choices.append(questionary.Choice(title="x Cancel to Main Menu", value=None))

            selected_dir = questionary.select(
                f"{title} -- Choose Problem:",
                choices=prob_choices,
                use_indicator=True,
            ).ask()
        else:
            table = Table(
                title=f"{title} -- Choose Problem", show_lines=False, header_style="bold cyan"
            )
            table.add_column("#", style="yellow", justify="right")
            table.add_column("Problem Directory", style="magenta")
            table.add_column("Config Count", style="green")

            for idx, d in enumerate(prob_dirs, start=1):
                table.add_row(str(idx), d.name.upper(), str(len(_get_yaml_files(d))))

            table.add_row(str(len(prob_dirs) + 1), "ALL (browse all problems)", "-")
            console.print(table)
            console.print("  [bold yellow]0[/bold yellow]: Cancel to Main Menu")

            choices_str = [str(i) for i in range(len(prob_dirs) + 2)]
            selection = Prompt.ask("\nSelect problem number", choices=choices_str, default="1")
            sel_int = int(selection)
            if sel_int == 0:
                selected_dir = None
            elif sel_int == len(prob_dirs) + 1:
                selected_dir = "ALL"
            else:
                selected_dir = prob_dirs[sel_int - 1]

        if selected_dir is None:
            return None

        # Determine target files for Stage 2
        if selected_dir == "ALL":
            target_files = _get_yaml_files(base_dir)
            menu_title = f"{title} (All Problems)"
        else:
            target_files = _get_yaml_files(selected_dir)
            menu_title = f"{title} ({selected_dir.name.upper()})"

        # Stage 2: Select Config File inside selected problem directory
        result = _display_file_menu(menu_title, target_files, base_dir, allow_back=True)

        if result is BACK:
            continue  # Go back to Stage 1 (Problem Selection)
        elif isinstance(result, Path):
            return result  # User selected a config file
        else:
            return None  # Cancelled to Main Menu


def _display_file_menu(
    title: str, files: list[Path], base_dir: Path, allow_back: bool = True
) -> Path | _BackSentinel | None:
    """Display a menu of config files with arrow key navigation or Rich fallback."""
    if not files:
        console.print(f"[bold red]No config files found in {base_dir}[/bold red]")
        return None

    if HAS_QUESTIONARY:
        choices = [questionary.Choice(title=str(f.relative_to(base_dir)), value=f) for f in files]
        if allow_back:
            choices.append(questionary.Choice(title="< Back to Problem Selection", value=BACK))
        choices.append(questionary.Choice(title="x Cancel to Main Menu", value=None))

        return questionary.select(
            title,
            choices=choices,
            use_indicator=True,
        ).ask()
    else:
        table = Table(title=title, show_lines=False, header_style="bold cyan")
        table.add_column("#", style="yellow", justify="right")
        table.add_column("Problem", style="magenta")
        table.add_column("Config File", style="green")

        for idx, path in enumerate(files, start=1):
            rel = path.relative_to(base_dir)
            problem = rel.parts[0] if len(rel.parts) > 1 else "-"
            table.add_row(str(idx), problem, str(rel))

        console.print(table)
        if allow_back:
            console.print("  [bold yellow]B[/bold yellow]: Back to Problem Selection")
        console.print("  [bold yellow]0[/bold yellow]: Cancel to Main Menu")

        choices_str = [str(i) for i in range(len(files) + 1)]
        if allow_back:
            choices_str.extend(["b", "B"])

        selection = Prompt.ask("\nSelect config number", choices=choices_str, default="1")
        sel_upper = selection.strip().upper()

        if sel_upper == "B" and allow_back:
            return BACK
        if sel_upper == "0":
            return None

        return files[int(selection) - 1]


def _resolve_matching_eval_config(train_config: Path) -> Path | None:
    """Find corresponding eval config path for a given train config."""
    try:
        rel_path = train_config.relative_to(TRAIN_CONFIGS_DIR)
        candidate = EVAL_CONFIGS_DIR / rel_path
        if candidate.exists():
            return candidate
    except ValueError:
        pass

    # Fallback search by stem name under eval
    stem_matches = list(EVAL_CONFIGS_DIR.rglob(f"{train_config.stem}.*"))
    if stem_matches:
        return stem_matches[0]

    return None


def _extract_eval_settings(eval_config_path: Path) -> list[str]:
    """Load evaluation config and extract setting labels."""
    try:
        eval_config = EvaluationConfig.from_file(eval_config_path)
        return [run.setting.label for run in eval_config.runs()]
    except Exception as exc:
        console.print(
            f"[yellow]Could not parse eval settings from {eval_config_path.name}: {exc}[/yellow]"
        )
        return []


def _select_eval_settings(settings: list[str]) -> tuple[list[str], bool]:
    """Prompt user to select specific setting labels or run all / first using arrow keys.

    Returns (selected_settings, run_first_only).
    """
    if not settings:
        return [], False

    if HAS_QUESTIONARY:
        choices = [
            questionary.Choice(title="Run ALL settings", value="ALL"),
            questionary.Choice(title="Run FIRST setting only (--first)", value="FIRST"),
        ]
        for label in settings:
            choices.append(questionary.Choice(title=f"Setting: {label}", value=label))

        selected = questionary.checkbox(
            "Select setting(s) to evaluate (Use Arrow keys, Space to toggle, Enter to confirm):",
            choices=choices,
        ).ask()

        if not selected or "ALL" in selected:
            return [], False
        if "FIRST" in selected:
            return [], True

        return [s for s in selected if s not in ("ALL", "FIRST")], False
    else:
        table = Table(
            title="Available Evaluation Settings", show_lines=False, header_style="bold cyan"
        )
        table.add_column("#", style="yellow", justify="right")
        table.add_column("Setting Label", style="green")

        table.add_row("0", "[bold white]Run ALL settings[/bold white]")
        table.add_row("F", "[bold magenta]Run FIRST setting only (--first)[/bold magenta]")
        for idx, label in enumerate(settings, start=1):
            table.add_row(str(idx), label)

        console.print(table)
        raw = Prompt.ask(
            "Select setting number(s) (e.g. '1', '1,2', '0' for all, 'F' for first)",
            default="0",
        )

        raw_clean = raw.strip().upper()
        if raw_clean == "F":
            return [], True
        if raw_clean == "0" or not raw_clean:
            return [], False

        selected_indices: list[int] = []
        for item in raw_clean.replace(" ", "").split(","):
            if item.isdigit():
                val = int(item)
                if 1 <= val <= len(settings):
                    selected_indices.append(val - 1)

        return [settings[i] for i in selected_indices], False


def _get_yaml_files(root: Path) -> list[Path]:
    """Find all .yml and .yaml files under root directory recursively."""
    if not root.exists():
        return []
    return sorted(
        [
            p
            for p in root.rglob("*.yml")
            if p.is_file() and p.parent.name not in ("problems", "search")
        ]
    ) + sorted(
        [
            p
            for p in root.rglob("*.yaml")
            if p.is_file() and p.parent.name not in ("problems", "search")
        ]
    )


def _ask_confirm(prompt_msg: str, default: bool = True) -> bool:
    """Helper to ask yes/no confirmation with arrow key support if questionary is available."""
    if HAS_QUESTIONARY:
        res = questionary.confirm(prompt_msg, default=default).ask()
        return bool(res)
    return Confirm.ask(prompt_msg, default=default)


def _pause() -> None:
    """Pause before clearing screen in menu loop."""
    if HAS_QUESTIONARY:
        questionary.press_any_key("Press any key to return to main menu...").ask()
    else:
        Prompt.ask("\nPress Enter to return to main menu...")


if __name__ == "__main__":
    app()
