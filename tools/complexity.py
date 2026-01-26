from pyfiglet import Figlet
import click
from pathlib import Path
from json import dump
import numpy as np
from tools.utils import (
    learn_filter_by_gradient_descent,
    ft2d_compression,
    ft2d_decompression,
    combine_side_by_side,
    load_evolutions_from_parquet,
    line_plot,
    render_halfblock,
    plot_mask_loss,
    save_evolution,
)

f = Figlet(font="smslant")
banner = f.renderText("Fourier Transform Kolmogorov Complexity")


@click.command("complexity")
@click.option("-i", "--input", type=Path, required=True)
@click.option("-o", "--output", type=Path)
@click.option("-lr", "--learning-rate", type=float, default=1e-2)
@click.option("-th", "--quantisation-threshold", type=float, default=0.5)
@click.option("-s", "--sigmoid-sharpness", type=int, default=1)
@click.option("-it", "--iterations", type=int, default=500)
@click.option("-w", "--sparsity-weight", type=float, default=1.0)
def complexity_cli(
    input: Path,
    output: Path | None,
    learning_rate: float,
    quantisation_threshold: float,
    sigmoid_sharpness: int,
    iterations: int,
    sparsity_weight: float
) -> None:
    click.secho(banner, fg="blue", bold=True)
    click.echo(f"--learning-rate={learning_rate}")
    click.echo(f"--iterations={iterations}")
    click.echo(f"--quantisation-threshold={quantisation_threshold}")
    click.echo(f"--sigmoid-sharpness={sigmoid_sharpness}")
    click.echo(f"--sparsity-weight={sparsity_weight}")
 

    complexities = {}
    evolutions = load_evolutions_from_parquet(input)
    for name, evolution in evolutions.items():
        click.echo(name)
        m, losses = learn_filter_by_gradient_descent(
              binary_image=evolution,
              quantisation_threshold=quantisation_threshold,
              sigmoid_sharpness=sigmoid_sharpness,
              learning_rate=learning_rate,
              iterations=iterations,
              sparsity_loss_weight=sparsity_weight  
        ) 
        mask = [list(map(int, row)) for row in m]
        click.echo(render_halfblock(values=mask))

        exactness = [loss.exactness for loss in losses]
        sparsity = [loss.sparsity for loss in losses]
        click.echo(f"iterations: {iterations}")
        click.echo(f"exactness: {line_plot(exactness)}")
        click.echo(f"sparsity: {line_plot(sparsity)}")

        z = ft2d_compression(s=evolution, m=m)
        s_hat = ft2d_decompression(z=z, θ=quantisation_threshold)

        reconstructed_evolution = [list(map(int, row)) for row in s_hat]
        if output:
            path_save = output / name
            path_save.mkdir(parents=True, exist_ok=True)
            save_evolution(
                values=evolution,
                path=path_save / "evolution.png",
            )
            save_evolution(
                values=reconstructed_evolution,
                path=path_save / "reconstructed_evolution.png",
            )
            save_evolution(values=mask, path=path_save / "fourier_transform_mask.png")
            plot_mask_loss(losses, path_save / "loss.png")

        left = render_halfblock(evolution)
        right = render_halfblock(reconstructed_evolution)
        click.echo(combine_side_by_side(left, right))

        score = m.sum() / m.size
        click.echo(f"Complexity: {score}")
        complexities[name] = float(score)
        
        reconstruction_is_lossless = np.all(np.isclose(s_hat, np.array(evolution)))
        click.echo(f"Is lossless? {reconstruction_is_lossless}")
    if output:
        with (output / "complexity.json").open("w") as f:
            dump(complexities, f, indent=2)


if __name__ == "__main__":
    complexity_cli()
