#! /usr/bin/env python3

import click
from pyfiglet import Figlet
from pathlib import Path

from tools.utils import (
    train,
    load_model,
    generate_text,
    line_plot,
)


f = Figlet(font="smslant")
banner = f.renderText("CharGPT")


@click.group()
def gpt():
    """Character-level GPT CLI"""
    click.secho(banner, fg="magenta", bold=True)
    click.secho("Welcome to character-GPT 🔡🤖", fg="magenta")
    click.secho(
        "Train your own character-level GPT model (with just a CPU).\n", fg="magenta"
    )


# =====================================================================
#  gpt learn
# =====================================================================
@gpt.command(name="train")
@click.option(
    "--input-dir",
    "-i",
    type=Path,
    required=True,
    help="Directory containing training parquet files.",
)
@click.option(
    "--vocab",
    "-v",
    type=str,
    default=(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " .,!?;:'\"()[]{}<>-_/\\\n"
    ),
    help="String of characters forming the vocabulary.",
)
@click.option("--sequence-len", "-s", type=int, default=1024)
@click.option("--layers", "-l", type=int, default=12)
@click.option("--heads", "-h", type=int, default=6)
@click.option("--embedding-dimension", "-d", type=int, default=768)
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--learning-rate", "-lr", type=float, default=3e-4)
@click.option("--epochs", "-e", type=int, default=2000)
@click.option(
    "--output-dir",
    "-o",
    type=Path,
    required=True,
    help="Where to save checkpoints + vocab.",
)
@click.option(
    "--resume",
    "-r",
    type=str,
    default=None,
    help="Optional checkpoint path to resume training.",
)
def train_gpt(
    input_dir: Path,
    vocab: str | None,
    sequence_len: int,
    layers: int,
    heads: int,
    embedding_dimension: int,
    batch_size: int,
    learning_rate: int,
    epochs: int,
    output_dir: Path,
    resume: str | None,
) -> None:
    """Train a character-level GPT model."""
    click.secho("\n📦 Training Configuration:", fg="cyan", bold=True)

    params = {
        "Input directory": str(input_dir),
        "Output directory": str(output_dir),
        "Resume checkpoint": resume,
        "Vocabulary": vocab,
        "Vocab size": len(vocab) + 2,
        "Sequence length": sequence_len,
        "Layers": layers,
        "Heads": heads,
        "Embedding dimension": embedding_dimension,
        "Batch size": batch_size,
        "Learning rate": learning_rate,
        "Training epochs": epochs,
    }

    for key, val in params.items():
        click.secho(f"  • {key:20} ", fg="bright_magenta", nl=False)
        click.echo(f"{val}")

    click.secho("🚀 Starting GPT Training", fg="green", bold=True)
    loss = train(
        parquet_dir=input_dir,
        vocab_string=vocab,
        batch_size=batch_size,
        lr=learning_rate,
        steps=epochs,
        out_dir=output_dir,
        resume_checkpoint=resume,
        sequence_len=sequence_len,
        layers=layers,
        heads=heads,
        embedding_dimension=embedding_dimension,
    )
    loss_curve = line_plot(loss)
    click.secho(loss_curve, fg="green")
    click.secho("🎉 Training complete!", fg="green", bold=True)


# =====================================================================
#  gpt generate
# =====================================================================
@gpt.command(name="run")
@click.option(
    "--input-dir",
    "-i",
    type=Path,
    required=True,
    help="Directory with vocab.json and checkpoint.",
)
@click.option("--checkpoint", "-ckp", type=str, default="final.pt")
@click.option("--prompt", "-p", type=str, default="§")
@click.option("--temperature", "-t", type=float, default=1.0)
@click.option("--top-k", "-k", type=int, default=None)
@click.option("--output-dir", "-o", type=Path, help="Directory to store predictions")
def run_gpt(
    input_dir: Path,
    checkpoint: str,
    prompt: str,
    temperature: float,
    top_k: int | None,
    output_dir: Path | None,
) -> None:
    """Generate text from a trained character-level GPT model."""
    click.secho("✨ Loading model…", fg="yellow")

    model, vocab = load_model(input_dir, checkpoint)

    config = model.config  # ← grab the config directly from the model
    click.secho("\n📦 Model Configuration:", fg="cyan", bold=True)
    params = {
        "Input": str(input_dir),
        "Sequence length": config.sequence_len,
        "Vocab size": config.vocab_size,
        "Layers": config.n_layer,
        "Heads": config.n_head,
        "KV Heads": config.n_kv_head,
        "Embedding dim": config.n_embd,
        "Checkpoint": checkpoint,
        "Vocabulary": vocab.vocab_list,
        "BOS": vocab.bos,
        "Wildcard": vocab.wildcard,
    }

    for key, val in params.items():
        click.secho(f"  • {key:18}", fg="bright_magenta", nl=False)
        click.echo(val)

    click.secho("📝 Generating text…\n", fg="green")

    output = generate_text(
        model=model,
        vocab=vocab,
        prompt=prompt,
        length=config.sequence_len,
        temperature=temperature,
        top_k=top_k,
    )

    click.secho("\n=== Generated Text ===\n", fg="magenta", bold=True)
    click.echo(output)


# ======================
# Entry Point
# ======================
if __name__ == "__main__":
    gpt()
