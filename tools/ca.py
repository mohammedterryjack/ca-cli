#! /usr/bin/env python3

from pyfiglet import Figlet
import click
from pathlib import Path

from tools.utils import (
    CellularAutomata,
    render_halfblock,
    save_evolution,
    save_evolutions_as_parquet,
)

f = Figlet(font="smslant")
banner = f.renderText("CellularAutomata")


# -----------------------------
# Parse the -ic flag
# -----------------------------
def parse_ic(ic_value: str | None):
    """
    Supported formats for -ic:

    - "10312"      → [10312]
    - "100:200"    → [100..200]
    - "x10"        → ('random', 10)
    - None         → caller handles → [None] (one random IC)
    """
    if ic_value is None:
        return None  # no IC provided → one random IC

    ic_value = ic_value.strip().lower()

    # Random count, example: x10
    if ic_value.startswith("x") and ic_value[1:].isdigit():
        count = int(ic_value[1:])
        return ("random", count)

    # Range: "100:200"
    if ":" in ic_value:
        a, b = ic_value.split(":")
        start = int(a)
        end = int(b)
        return list(range(start, end + 1))

    # Single integer
    return [int(ic_value)]


@click.command()
@click.option(
    "-ic",
    "--initial-configuration",
    type=str,
    help="Initial configuration: single val (101), range (100:120), or xN for N random ICs",
)
@click.option("-r", "--rule", type=int, help="Local update rule number")
@click.option("-n", "--neighbourhood", type=int, default=1)
@click.option("-w", "--width", type=int, default=30)
@click.option("-t", "--timesteps", type=int, default=20)
@click.option("-s", "--states", type=int, default=2)
@click.option("--display/--nodisplay", default=True)
@click.option("-ct", "--configuration-timestep", type=int)
@click.option("-cc", "--cell-change", type=int)
@click.option("-o", "--output", type=Path, help="Directory to save evolution images")
def cellular_automata(
    initial_configuration,
    rule,
    neighbourhood,
    width,
    timesteps,
    states,
    display,
    configuration_timestep,
    cell_change,
    output,
):
    evolutions = {}
    # -----------------------------
    # EXPAND THE -IC VALUE INTO A LIST OF ICs TO RUN
    # -----------------------------
    parsed = parse_ic(initial_configuration)

    if parsed is None:
        # No -ic passed → 1 random IC
        ic_list = [None]
    elif isinstance(parsed, tuple) and parsed[0] == "random":
        # xN random ICs
        _, count = parsed
        ic_list = [None] * count
    else:
        # Single or range list
        ic_list = parsed

    # Ensure output directory exists
    if output:
        output.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # RUN ONE SIMULATION PER IC
    # -----------------------------
    for ic in ic_list:

        ca = CellularAutomata(
            cell_states=states,
            neighbourhood_radius=neighbourhood,
            lattice_width=width,
            time_steps=timesteps,
            initial_state=ic,  # None → random IC
            transition_rule_number=rule,
        )

        # Extract actual IC and rule used by CA
        actual_ic = ca.info.lattice_evolution[0]
        actual_rule = ca.info.local_transition_rule

        # Display block (optional)
        if display:
            click.secho(banner, fg="green", bold=True)
            click.echo("Welcome to the Cellular Automata CLI 🚀")
            click.echo(click.style("Running with parameters:", bold=True))
            click.echo(f"  Rule: {actual_rule}")
            click.echo(f"  Initial configuration: {actual_ic}")
            click.echo(f"  Neighbourhood: {neighbourhood}")
            click.echo(f"  Width: {width}")
            click.echo(f"  Time steps: {timesteps}")
            click.echo(f"  States: {states}")
            click.echo(render_halfblock(values=ca.evolution))

        # Single-time-step / cell-change output
        if configuration_timestep is not None:
            click.echo(" ".join(map(str, ca.evolution[configuration_timestep])))

        if cell_change is not None:
            click.echo(" ".join(map(str, ca.evolution[:, cell_change])))

        # Save output image
        if output:
            filename = f"{actual_rule}-{actual_ic}"[:100]
            (output / filename).mkdir(parents=True, exist_ok=True)
            path = output / filename / "evolution.png"
            save_evolution(values=ca.evolution, path=path)
            evolutions[filename] = ca.evolution
            if display:
                click.echo(f"Saved → {path}")
    if output:
        save_evolutions_as_parquet(evolutions=evolutions, output_dir=output)


if __name__ == "__main__":
    cellular_automata()
