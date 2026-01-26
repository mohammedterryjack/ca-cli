from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from tools.utils.gpt_utils.vocab import CharVocab
from tools.utils.gpt_utils.dataloader import char_stream_loader_with_state
from tools.utils.gpt_utils.model import GPT, GPTConfig


def train(
    parquet_dir: Path,
    vocab_string: str,
    batch_size: int,
    lr: float,
    steps: int,
    out_dir: Path,
    resume_checkpoint: str | None,
    sequence_len: int,
    layers: int,
    heads: int,
    embedding_dimension: int,
) -> list[float]:
    loss_history = []

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"
    config_path = out_dir / "config.json"
    if resume_checkpoint and vocab_path.exists():
        vocab = CharVocab.load(vocab_path)
        config = GPTConfig.load(config_path)
    else:
        vocab = CharVocab(vocab_string)
        vocab.save(vocab_path)
        config = GPTConfig(
            sequence_len=sequence_len,
            n_layer=layers,
            n_head=heads,
            n_kv_head=heads,
            n_embd=embedding_dimension,
            vocab_size=vocab.size,
        )
        config.save(config_path)

    model = GPT(config).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    start_step = 0
    loader_state = None
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        loader_state = ckpt["loader_state"]
        start_step = ckpt["step"]

    loader = char_stream_loader_with_state(
        vocab=vocab,
        B=batch_size,
        T=config.sequence_len,
        parquet_dir=parquet_dir,
        split="train",
        device=device,
        resume_state_dict=loader_state,
    )

    model.train()
    pbar = trange(start_step, steps, desc="Training", unit="step")
    for step in pbar:
        x, y, loader_state = next(loader)

        logits = model(x)  # (B, T, vocab)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        loss_history.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

        if step % 1000 == 0 and step > 0:
            save_path = out_dir / f"ckpt_{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "loader_state": loader_state,
                    "step": step,
                },
                save_path,
            )
            tqdm.write(f"Saved {save_path}")

    final_path = out_dir / "final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "loader_state": loader_state,
            "step": steps,
        },
        final_path,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("GPT Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_path = out_dir / "loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return loss_history
