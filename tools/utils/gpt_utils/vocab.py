from pathlib import Path
from json import dump, load


class CharVocab:
    def __init__(self, vocab: str, bos: str = "§", wildcard: str = "*") -> None:
        self.bos = bos
        self.wildcard = wildcard
        self.vocab_list = bos + vocab + wildcard
        assert len(self.vocab_list) == len(
            set(self.vocab_list)
        ), "duplicate characters detected"
        self.stoi = {ch: i for i, ch in enumerate(self.vocab_list)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab_list)}

    def encode_text(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                ids.append(self.stoi[self.wildcard])
        return ids

    def decode_ids(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        obj = {
            "vocab": self.vocab_list[1:-1],
            "bos": self.bos,
            "wildcard": self.wildcard,
        }
        with path.open("w", encoding="utf-8") as f:
            dump(obj, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        with path.open(encoding="utf-8") as f:
            obj = load(f)
        return cls(**obj)

    @property
    def size(self) -> int:
        return len(self.vocab_list)
