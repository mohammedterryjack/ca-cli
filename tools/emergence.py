#!/usr/bin/env python3

from pathlib import Path

import click
from pyfiglet import Figlet

from tools.utils import (
    render_halfblock,
    save_evolution,
    save_evolutions_as_parquet,
    filter_spacetime_by_delay_embedding_density,
    load_evolutions_from_parquet,
)

f = Figlet(font="smslant")
banner = f.renderText("Emergence")


@click.command(name="emergence")
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=Path,
    help="Path to an input parquet file.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=Path,
    help="Path to save the output image (default: rareness_output.png).",
)
@click.option(
    "--threshold",
    "-th",
    type=float,
    help="Optional threshold (e.g., 0.3) to binarize the result.",
)
@click.option(
    "--time-delay",
    "-tau",
    type=int,
    default=1,
    help="Time delay (tau) for delay embedding (default: 1).",
)
@click.option(
    "--embedding-dimension",
    "-d",
    type=int,
    default=3,
    help="Embedding dimension for delay embedding (default: 3).",
)
def complexity_cli(
    input_path: Path,
    output_path: Path,
    threshold: float | None,
    time_delay: int,
    embedding_dimension: int,
) -> None:
    emergence_evolutions = {}

    click.secho(banner, fg="yellow", bold=True)

    if not input_path.is_file() or input_path.suffix != ".parquet":
        raise click.ClickException(f"Input must be a Parquet file, got {input_path}")

    loaded_evolutions = load_evolutions_from_parquet(input_path)

    if output_path is None:
        if input_path.is_dir():
            output_path = input_path.parent / f"{input_path.name}_emergence"
        else:
            output_path = input_path.parent / f"{input_path.stem}_emergence"

    output_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Output directory → {output_path}")

    for name, data in loaded_evolutions.items():

        click.echo(
            f"Loaded input: {data.shape} (min={data.min():.3f}, max={data.max():.3f})"
        )
        click.echo(
            f"Parameters → tau={time_delay}, dim={embedding_dimension}, threshold={threshold}"
        )

        rareness = filter_spacetime_by_delay_embedding_density(
            spacetime=data,
            time_delay=time_delay,
            embedding_dimension=embedding_dimension,
        )
        if threshold is not None:
            binary = (rareness >= threshold).astype(int)
            click.echo(f"Applied threshold: {threshold}")
            binary.tolist()
            click.echo(render_halfblock(values=binary))
            rareness = binary

        out_file = output_path / f"{name}.png"
        save_evolution(values=rareness, path=out_file)
        click.echo(f"Saved output image → {out_file}")
        emergence_evolutions[name] = rareness

    save_evolutions_as_parquet(evolutions=emergence_evolutions, output_dir=output_path)
    click.echo(f"Saved all to {output_path}.parquet")


if __name__ == "__main__":
    complexity_cli()
