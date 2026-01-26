from pathlib import Path
import time

from pyarrow.parquet import ParquetFile
from collections import deque

import torch


def char_stream_loader_with_state(
    vocab,
    B,
    T,
    parquet_dir,
    split="train",
    device="cpu",
    resume_state_dict=None,
    mini_batch_size=16,
    log_every=1000,
):
    assert split in ["train", "val"]

    parquet_paths = list(parquet_dir.glob("*.parquet"))
    if split == "train":
        parquet_paths = parquet_paths[:-1]
    else:
        parquet_paths = parquet_paths[-1:]

    if not parquet_paths:
        raise ValueError(f"No parquet files available for split '{split}'.")

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict else None

    def document_batches():
        pq_idx = resume_pq_idx
        while True:
            while pq_idx < len(parquet_paths):
                pf = ParquetFile(parquet_paths[pq_idx])
                rg_idx = resume_rg_idx if resume_rg_idx is not None else 0
                resume_rg_idx_local = None

                while rg_idx < pf.num_row_groups:
                    start_rg = time.time()
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column("text").to_pylist()
                    for i in range(0, len(texts), mini_batch_size):
                        mini_texts = texts[i : i + mini_batch_size]
                        for txt in mini_texts:
                            ids = [vocab.stoi[vocab.bos]] + vocab.encode_text(txt)
                            yield ids, (pq_idx, rg_idx)

                    rg_idx += 1
                pq_idx += 1
            pq_idx = 0

    stream = document_batches()
    needed = B * T + 1
    buffer = deque()
    total_tokens_processed = 0

    while True:
        start_fill = time.time()
        while len(buffer) < needed:
            try:
                tokens, (pq_idx, rg_idx) = next(stream)
            except Exception as e:
                print(f"[ERROR] Exception in stream: {e}")
                raise
            buffer.extend(tokens)
            total_tokens_processed += len(tokens)
        arr = [buffer.popleft() for _ in range(needed)]
        scratch = torch.tensor(arr, dtype=torch.long)

        x = scratch[:-1].reshape(B, T).to(device)
        y = scratch[1:].reshape(B, T).to(device)

        state = {"pq_idx": pq_idx, "rg_idx": rg_idx}
        yield x, y, state
