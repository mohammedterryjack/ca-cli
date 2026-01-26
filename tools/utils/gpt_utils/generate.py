from pathlib import Path

import torch

from tools.utils.gpt_utils.model import GPT, GPTConfig
from tools.utils.gpt_utils.vocab import CharVocab


def load_model(model_dir: Path, checkpoint: str = "final.pt") -> tuple[GPT, CharVocab]:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    vocab = CharVocab.load(model_dir / "vocab.json")
    config = GPTConfig.load(model_dir / "config.json")
    model = GPT(config).to(device)

    ckpt_path = model_dir / checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, vocab


@torch.no_grad()
def generate_text(
    model: GPT,
    vocab: CharVocab,
    prompt: str,
    length: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.eval()
    tokens = [vocab.stoi[vocab.bos]] + vocab.encode_text(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(
        0
    )  # (1, seq_len)

    generated = tokens.copy()
    for _ in range(length):
        input_trimmed = input_ids[:, -model.config.sequence_len :]
        logits = model(input_trimmed)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            min_v = v[:, -1].unsqueeze(-1)
            logits[logits < min_v] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=device)], dim=1
        )

    return vocab.decode_ids(generated)
