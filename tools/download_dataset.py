#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from datasets import load_dataset
from numpy import array
from tqdm import tqdm
import click

from tools.utils.shared_utils.io import save_evolutions_as_parquet


EVOLUTION_DEPTH = 20


def convert_dataset_to_dict(dataset, split_name):
    """
    Convert HuggingFace dataset split → dict{name: ndarray}
    """
    evolutions = {}

    for idx, item in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
        evolution = [item[f"input_ids_{i}"] for i in range(EVOLUTION_DEPTH)]
        evolutions[f"{split_name}_{idx}"] = array(evolution, dtype=np.uint8)

    return evolutions


@click.command()
@click.option(
    "--output",
    "-o",
    type=Path,
    required=True,
    help="Directory to store the generated parquet dataset.",
)
@click.option("--notrain", is_flag=True, help="Skip downloading the train split.")
@click.option("--notest", is_flag=True, help="Skip downloading the test split.")
@click.option("--noval", is_flag=True, help="Skip downloading the validation split.")
def main(output: Path, notrain: bool, notest: bool, noval: bool):
    """
    Download the irodkin/1dCA_r2s20T20 dataset (or selected splits)
    and save it in the training parquet format expected by the Emergence Lab tools.
    """
    output.mkdir(parents=True, exist_ok=True)

    click.echo("Loading dataset from HuggingFace…")
    ds = load_dataset("irodkin/1dCA_r2s20T20")

    # Process splits
    if not notrain:
        click.echo("\nConverting TRAIN split...")
        train_evos = convert_dataset_to_dict(ds["train"], "train")
        save_evolutions_as_parquet(train_evos, output / "train.parquet")
        click.secho("✓ Saved train.parquet", fg="green")

    if not notest:
        click.echo("\nConverting TEST split...")
        test_evos = convert_dataset_to_dict(ds["test"], "test")
        save_evolutions_as_parquet(test_evos, output / "test.parquet")
        click.secho("✓ Saved test.parquet", fg="green")

    if not noval:
        click.echo("\nConverting VALIDATION split...")
        val_evos = convert_dataset_to_dict(ds["validation"], "val")
        save_evolutions_as_parquet(val_evos, output / "val.parquet")
        click.secho("✓ Saved val.parquet", fg="green")

    click.echo("\nAll selected splits saved.")
    click.secho(f"Output directory: {output}", fg="blue", bold=True)


if __name__ == "__main__":
    main()
