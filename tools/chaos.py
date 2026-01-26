#! /usr/bin/env python3

from pyfiglet import Figlet
import click
from pathlib import Path
import matplotlib.pyplot as plt
from json import dump

from tools.utils import (
    O1TestForChaos,
    evolution_to_1d_sequence_cosine,
    evolution_to_1d_sequence_gray,
    load_evolutions_from_parquet,
)


f = Figlet(font="smslant")
banner = f.renderText("01-Test for Chaos")


@click.command("chaos")
@click.option(
    "-i",
    "--input",
    type=Path,
    required=True,
    help="Parquet file containing saved evolutions.",
)
@click.option(
    "-o", "--output", type=Path, help="Directory to save the trajectory image."
)
@click.option(
    "-e",
    "--encoding",
    type=str,
    default="gray",
    help="The method used to encode a CA evolution into a 1d timeseries",
)
@click.option(
    "-a", "--angles", type=int, default=10, help="Number of angles for the chaos test."
)
@click.option(
    "-c",
    "--compare",
    is_flag=True,
    default=False
)
def chaos_cli(input: Path, encoding: str, output: Path, angles: int, compare:bool) -> None:
    click.secho(banner, fg="blue", bold=True)
    click.echo(f"encoding: {encoding}")
    click.echo(f"angles: {angles}")
    if encoding == "gray":
        encode_evolution = evolution_to_1d_sequence_gray
    elif encoding == "cosine":
        encode_evolution = evolution_to_1d_sequence_cosine
    else:
        raise ValueError(f"Unrecognised encoding {encoding}")

    evolutions = load_evolutions_from_parquet(input)
    chaoticities = {}
    if compare:
        phis={}
    for name, evolution in evolutions.items():
        if output:
            path_save = output / name
            path_save.mkdir(parents=True, exist_ok=True)
        else:
            path_save = None

        phi = encode_evolution(evolution)
        if compare: 
            phis[name] = phi
        click.echo(f"{name} trajctory len {len(phi)}")
        if output:
            plt.clf()
            plt.figure()
            plt.figure(figsize=(8, 4))
            plt.plot(phi, c="orange")
            plt.ylim(0, 1)

            plt.savefig(path_save / "trajectory.png", dpi=300, bbox_inches="tight")
            plt.close('all') 
        chaoticity = O1TestForChaos.test_for_chaos(
            observables=phi, n_angles=angles, path_save=path_save
        )
        click.echo(f"Chaoticity: {chaoticity}")
        chaoticities[name] = chaoticity
    if output:
        with (output / "chaos.json").open("w") as f:
            dump(chaoticities, f, indent=2)
        if compare: 
            plt.clf()
            for name, phi in phis.items():
                plt.plot(phi, label=name)

            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(output / "trajectories.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    chaos_cli()
